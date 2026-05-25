"""6-stage pipeline driver.

Walks the stages in order, validates each artifact, handles the review-revision
loop, and surfaces a final recommendation. Knows nothing about Claude itself —
that's `sdk_runner`. Knows nothing about vectorbt — that's `backtest/`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .mcp_servers import build_phd_mcp_server
from .models import Decision, ReviewDecision, RunManifest
from .persistence import (
    STAGE_DIRS,
    append_transcript,
    ensure_run_layout,
    mint_run_id,
    now,
    read_artifact,
    run_dir,
    stage_dir,
)
from .sdk_runner import StageInvocation, StageRunResult, run_stage
from .stages import STAGE_ORDER, STAGES, ModelConfig, render_prompt

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input / config
# ---------------------------------------------------------------------------


@dataclass
class PipelineInput:
    thesis: str
    universe: str
    strategy_type: str
    research_questions: list[str]
    model_config: ModelConfig
    cwd: Path
    runs_root: Path | None = None  # default → cwd / "runs"
    max_review_rounds: int = 3
    ceiling_pct: float = 50.0


@dataclass
class StageOutcome:
    stage_name: str
    run_result: StageRunResult
    artifact_paths: list[Path]
    validated_models: list[BaseModel]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run_pipeline(inp: PipelineInput) -> dict[str, Any]:
    """Execute all 6 stages. Returns a summary dict suitable for CLI printing."""
    run_id = mint_run_id()
    ensure_run_layout(run_id, root=inp.runs_root)

    manifest = RunManifest(
        run_id=run_id,
        created_at=now(),
        thesis=inp.thesis,
        universe=inp.universe,
        strategy_type=inp.strategy_type,
        stage_models={s: inp.model_config.get(s).model for s in STAGE_ORDER},
        stage_effort={s: (inp.model_config.get(s).effort or "none") for s in STAGE_ORDER},
    )
    _write_manifest(manifest, root=inp.runs_root)

    outcomes: dict[str, StageOutcome] = {}

    # --- Research / Review revision loop ----------------------------------
    revision_round = 0
    review_decision: ReviewDecision | None = None
    while True:
        outcomes["research"] = await _run_one(
            "research",
            inp,
            run_id,
            extra_ctx={
                "revision_round": revision_round,
                "revision_block": _revision_block(review_decision),
                "research_questions": _format_research_questions(inp.research_questions),
            },
        )

        outcomes["review"] = await _run_one(
            "review",
            inp,
            run_id,
            extra_ctx={"revision_round": revision_round},
        )

        review_decision = next(
            m for m in outcomes["review"].validated_models if isinstance(m, ReviewDecision)
        )
        _update_manifest(run_id, root=inp.runs_root, last_decision=review_decision.decision)

        if review_decision.decision == Decision.PASS:
            break
        if review_decision.decision == Decision.REJECT:
            return _summarize(
                run_id, outcomes, halted_at="review", reason="review REJECT"
            )
        revision_round += 1
        if revision_round >= inp.max_review_rounds:
            return _summarize(
                run_id,
                outcomes,
                halted_at="review",
                reason=f"max {inp.max_review_rounds} revision rounds exceeded",
            )

    # --- Synthesis → Backtest → Risk → Report -----------------------------
    for stage_name in ("strategy_synthesis", "backtest", "risk_check", "report"):
        outcomes[stage_name] = await _run_one(stage_name, inp, run_id, extra_ctx={})

    return _summarize(run_id, outcomes)


# ---------------------------------------------------------------------------
# One stage (with checkpoint-respawn on ceiling)
# ---------------------------------------------------------------------------


async def _run_one(
    stage_name: str,
    inp: PipelineInput,
    run_id: str,
    *,
    extra_ctx: dict[str, Any],
    max_respawns: int = 2,
) -> StageOutcome:
    stage = STAGES[stage_name]
    sm = inp.model_config.get(stage_name)

    mcp_servers = {"phd": build_phd_mcp_server()} if stage.uses_phd_mcp else {}

    ctx = {
        "run_id": run_id,
        "thesis": inp.thesis,
        "universe": inp.universe,
        "strategy_type": inp.strategy_type,
        "today": now().date().isoformat(),
        "now_iso": now().isoformat(),
        "revision_round": 0,
        "revision_block": "",
        "research_questions": "",
        **extra_ctx,
    }
    system_prompt = render_prompt(stage.prompt_template, **ctx)
    user_prompt = _user_prompt_for(stage_name, ctx)

    respawns = 0
    last_result: StageRunResult | None = None
    while True:
        inv = StageInvocation(
            run_id=run_id,
            stage_name=stage_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=list(stage.allowed_tools),
            mcp_servers=mcp_servers,
            model=sm.model,
            effort=sm.effort,
            cwd=str(inp.cwd),
        )
        result = await run_stage(inv, ceiling_pct=inp.ceiling_pct)
        last_result = result

        if result.exit_reason == "error":
            raise RuntimeError(f"stage {stage_name} errored: {result.error}")

        artifact_paths, models = _try_load_outputs(stage_name, run_id, inp.runs_root)
        if artifact_paths:
            return StageOutcome(stage_name, result, artifact_paths, models)

        if result.exit_reason == "ceiling_hit" and respawns < max_respawns:
            respawns += 1
            log.warning(
                "stage %s hit context ceiling — respawning (round %d)",
                stage_name,
                respawns,
            )
            checkpoint_path = stage_dir(run_id, stage_name, root=inp.runs_root) / "checkpoint.json"
            checkpoint_text = checkpoint_path.read_text() if checkpoint_path.exists() else "{}"
            user_prompt = (
                "You are resuming a stage that was paused at the context-window "
                "ceiling. Below is the checkpoint you wrote. Continue from there "
                "and produce the final artifact.\n\n"
                f"--- checkpoint.json ---\n{checkpoint_text}\n--- end checkpoint ---"
            )
            continue

        raise RuntimeError(
            f"stage {stage_name} produced no usable artifact "
            f"(exit_reason={result.exit_reason}, peak_pct={result.peak_context_pct:.1f})"
        )


def _try_load_outputs(
    stage_name: str, run_id: str, root: Path | None
) -> tuple[list[Path], list[BaseModel]]:
    stage = STAGES[stage_name]
    filenames = (
        (stage.output_filename,) if isinstance(stage.output_filename, str) else stage.output_filename
    )
    cls_seq = (
        (stage.output_model_cls,)
        if isinstance(stage.output_model_cls, type)
        else stage.output_model_cls
    )
    paths: list[Path] = []
    models: list[BaseModel] = []
    for fname, cls in zip(filenames, cls_seq):
        path = stage_dir(run_id, stage_name, root=root) / fname
        if not path.exists():
            return [], []
        try:
            model = read_artifact(run_id, stage_name, fname, cls, root=root)
        except (ValidationError, json.JSONDecodeError) as exc:
            append_transcript(
                run_id, stage_name, {"validation_error": str(exc), "file": fname}, root=root
            )
            return [], []
        paths.append(path)
        models.append(model)
    return paths, models


# ---------------------------------------------------------------------------
# Prompt context builders
# ---------------------------------------------------------------------------


def _format_research_questions(qs: list[str]) -> str:
    if not qs:
        return "(use your judgment to derive 5–10 questions from the thesis)"
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))


def _revision_block(prev: ReviewDecision | None) -> str:
    if prev is None or prev.decision == Decision.PASS:
        return ""
    bullets = "\n".join(f"- {b}" for b in prev.must_address)
    return (
        "\n## Revision required\n\n"
        f"Prior decision: **{prev.decision.value}**.\n"
        f"Reviewer rationale: {prev.rationale}\n\n"
        "You MUST address each item below in your new findings:\n"
        f"{bullets}\n"
    )


def _user_prompt_for(stage_name: str, ctx: dict) -> str:
    """The kickoff message for each stage. Short — the system prompt does the heavy lifting."""
    run_id = ctx["run_id"]
    if stage_name == "research":
        return (
            f"Begin Research for run {run_id}. Universe: {ctx['universe']}. "
            f"Strategy type: {ctx['strategy_type']}. Thesis: {ctx['thesis']}."
        )
    if stage_name == "review":
        return (
            f"Begin Review for run {run_id}. Read "
            f"runs/{run_id}/01_research/findings.json and verify every claim."
        )
    if stage_name == "strategy_synthesis":
        return (
            f"Begin Strategy Synthesis for run {run_id}. Read the validated "
            f"findings and produce a StrategySpec."
        )
    if stage_name == "backtest":
        return (
            f"Begin Backtest for run {run_id}. Spec at "
            f"runs/{run_id}/03_strategy/strategy_spec.json."
        )
    if stage_name == "risk_check":
        return (
            f"Begin Risk Check for run {run_id}. Run walk-forward, "
            f"parameter sensitivity, and Monte Carlo."
        )
    if stage_name == "report":
        return (
            f"Begin Report for run {run_id}. Synthesize all upstream artifacts "
            f"and produce both the markdown and JSON report."
        )
    raise ValueError(f"unknown stage {stage_name!r}")


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def _write_manifest(manifest: RunManifest, *, root: Path | None) -> None:
    path = run_dir(manifest.run_id, root=root) / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))


def _update_manifest(
    run_id: str,
    *,
    root: Path | None,
    completed: str | None = None,
    last_decision: Decision | None = None,
) -> None:
    path = run_dir(run_id, root=root) / "manifest.json"
    data = json.loads(path.read_text())
    manifest = RunManifest.model_validate(data)
    if completed and completed not in manifest.completed_stages:
        manifest.completed_stages.append(completed)
    if last_decision is not None:
        manifest.last_decision = last_decision
    path.write_text(manifest.model_dump_json(indent=2))


def _summarize(
    run_id: str,
    outcomes: dict[str, StageOutcome],
    *,
    halted_at: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "halted_at": halted_at,
        "halt_reason": reason,
        "stages_completed": list(outcomes.keys()),
        "tokens_per_stage": {
            k: {
                "input": v.run_result.total_input_tokens,
                "output": v.run_result.total_output_tokens,
                "peak_pct": round(v.run_result.peak_context_pct, 1),
            }
            for k, v in outcomes.items()
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
