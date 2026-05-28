"""6-stage pipeline driver.

Walks the stages in order, validates each artifact, handles the review-revision
loop, and surfaces a final recommendation. Knows nothing about Claude itself —
that's `sdk_runner`. Knows nothing about vectorbt — that's `backtest/`.

Three structural guarantees live here:

  * **Deterministic handoff** — each stage receives the *validated* upstream
    artifacts injected inline into its prompt, not "please read this file".
    The agent never has to guess a path and cannot drift onto the wrong input.
  * **Integrity checks** — every artifact is checked for run-id match (and the
    backtest result for spec-hash match) before it's accepted as a handoff.
  * **Resumability** — completion is recorded in the manifest after each stage,
    and a run can be resumed: stages with a valid artifact are skipped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .backtest.runner import spec_hash
from .mcp_servers import build_phd_mcp_server
from .models import (
    BacktestResult,
    Decision,
    ReviewDecision,
    RunManifest,
    StageStat,
    StrategySpec,
)
from .persistence import (
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
    soft_ceiling_pct: float = 50.0
    hard_ceiling_pct: float = 65.0
    # Resume controls.
    resume_run_id: str | None = None
    from_stage: str | None = None  # force re-run from this stage onward


@dataclass
class StageOutcome:
    stage_name: str
    artifact_paths: list[Path]
    validated_models: list[BaseModel]
    run_result: StageRunResult | None = None  # None when loaded from disk (skipped)
    respawns: int = 0
    skipped: bool = False


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run_pipeline(inp: PipelineInput) -> dict[str, Any]:
    """Execute (or resume) all 6 stages. Returns a summary dict for the CLI."""
    if inp.resume_run_id:
        run_id = inp.resume_run_id
        if not (run_dir(run_id, root=inp.runs_root) / "manifest.json").exists():
            raise FileNotFoundError(f"cannot resume: no manifest for run {run_id}")
        ensure_run_layout(run_id, root=inp.runs_root)
        log.info("resuming run %s (from_stage=%s)", run_id, inp.from_stage)
    else:
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
    from_idx = _stage_idx(inp.from_stage) if inp.from_stage else None

    # --- Research / Review revision loop ----------------------------------
    revision_round = 0
    review_decision: ReviewDecision | None = None
    while True:
        force_loop = revision_round > 0  # a revision must always re-run the agents
        outcomes["research"] = await _run_one(
            "research", inp, run_id, from_idx=from_idx, force=force_loop,
            extra_ctx={
                "revision_round": revision_round,
                "revision_block": _revision_block(review_decision),
                "research_questions": _format_research_questions(inp.research_questions),
            },
        )
        outcomes["review"] = await _run_one(
            "review", inp, run_id, from_idx=from_idx, force=force_loop,
            extra_ctx={"revision_round": revision_round},
        )

        review_decision = next(
            m for m in outcomes["review"].validated_models if isinstance(m, ReviewDecision)
        )
        _update_manifest(
            run_id, root=inp.runs_root, last_decision=review_decision.decision,
            revision_rounds=revision_round,
        )

        if review_decision.decision == Decision.PASS:
            break
        if review_decision.decision == Decision.REJECT:
            return _halt(run_id, inp.runs_root, outcomes, "review", "review REJECT")
        revision_round += 1
        if revision_round >= inp.max_review_rounds:
            return _halt(
                run_id, inp.runs_root, outcomes, "review",
                f"max {inp.max_review_rounds} revision rounds exceeded",
            )

    # --- Synthesis → Backtest → Risk → Report -----------------------------
    for stage_name in ("strategy_synthesis", "backtest", "risk_check", "report"):
        outcomes[stage_name] = await _run_one(
            stage_name, inp, run_id, from_idx=from_idx, force=False, extra_ctx={}
        )

    return _summarize(run_id, inp.runs_root, outcomes)


# ---------------------------------------------------------------------------
# One stage: resume-skip, ceiling-respawn, integrity-check, manifest-record
# ---------------------------------------------------------------------------


async def _run_one(
    stage_name: str,
    inp: PipelineInput,
    run_id: str,
    *,
    extra_ctx: dict[str, Any],
    from_idx: int | None,
    force: bool,
    max_respawns: int = 2,
) -> StageOutcome:
    stage = STAGES[stage_name]

    # Resume-skip: if a valid artifact already exists and we're not forced to
    # re-run this stage, load it and move on.
    if not force and _may_skip(stage_name, from_idx):
        paths, models = _try_load_outputs(stage_name, run_id, inp.runs_root)
        if paths and _integrity_ok(stage_name, models, run_id, inp.runs_root):
            log.info("stage %s: existing valid artifact found — skipping", stage_name)
            return StageOutcome(stage_name, paths, models, skipped=True)

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
    base_user_prompt = _user_prompt_for(stage_name, ctx)
    handoff = _prior_artifacts_block(stage_name, run_id, inp.runs_root)
    user_prompt = base_user_prompt + handoff

    respawns = 0
    result: StageRunResult | None = None
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
            soft_ceiling_pct=inp.soft_ceiling_pct,
            hard_ceiling_pct=inp.hard_ceiling_pct,
        )
        result = await run_stage(inv)
        if result.exit_reason == "error":
            raise RuntimeError(f"stage {stage_name} errored: {result.error}")

        paths, models = _try_load_outputs(stage_name, run_id, inp.runs_root)
        if paths and _integrity_ok(stage_name, models, run_id, inp.runs_root):
            _record_stage(run_id, inp.runs_root, stage_name, result, respawns)
            return StageOutcome(stage_name, paths, models, run_result=result, respawns=respawns)

        # No usable artifact yet. If we paused at the ceiling, respawn with the
        # checkpoint as the sole prior context (fresh window again).
        if result.exit_reason == "ceiling_hit" and respawns < max_respawns:
            respawns += 1
            log.warning("stage %s hit ceiling — respawn %d/%d", stage_name, respawns, max_respawns)
            checkpoint = stage_dir(run_id, stage_name, root=inp.runs_root) / "checkpoint.json"
            checkpoint_text = checkpoint.read_text() if checkpoint.exists() else "{}"
            user_prompt = (
                base_user_prompt
                + "\n\nYou are RESUMING this stage after pausing at the context "
                "ceiling. Below is the checkpoint you wrote; continue from it and "
                "produce the final artifact.\n\n--- checkpoint.json ---\n"
                + checkpoint_text
                + "\n--- end checkpoint ---"
                + handoff
            )
            continue

        raise RuntimeError(
            f"stage {stage_name} produced no valid artifact "
            f"(exit_reason={result.exit_reason}, peak_pct={result.peak_context_pct:.0f})"
        )


# ---------------------------------------------------------------------------
# Deterministic handoff: inject validated upstream artifacts inline
# ---------------------------------------------------------------------------

# Which upstream artifacts each stage should receive, as (stage, filename) pairs.
_HANDOFF_INPUTS: dict[str, list[tuple[str, str]]] = {
    "research": [],
    "review": [("research", "findings.json")],
    "strategy_synthesis": [("research", "findings.json"), ("review", "critique.json")],
    "backtest": [("strategy_synthesis", "strategy_spec.json")],
    "risk_check": [("strategy_synthesis", "strategy_spec.json"), ("backtest", "result.json")],
    "report": [
        ("research", "findings.json"),
        ("review", "critique.json"),
        ("review", "decision.json"),
        ("strategy_synthesis", "strategy_spec.json"),
        ("backtest", "result.json"),
        ("risk_check", "assessment.json"),
    ],
}


def _prior_artifacts_block(stage_name: str, run_id: str, root: Path | None) -> str:
    """Read each upstream artifact and embed its JSON inline. This is the
    deterministic handoff: the agent does not need a Read tool call and cannot
    drift onto the wrong file. Paths are still given so tool-driven stages
    (backtest/risk) can pass them to phd-mcp."""
    inputs = _HANDOFF_INPUTS.get(stage_name, [])
    if not inputs:
        return ""
    parts = [
        "\n\n## Validated upstream artifacts (authoritative — use these, do NOT "
        "re-derive or read transcripts)\n"
    ]
    for up_stage, fname in inputs:
        path = stage_dir(run_id, up_stage, root=root) / fname
        if not path.exists():
            continue
        parts.append(f"\n### {up_stage}/{fname}\n`path: {path}`\n```json\n{path.read_text()}\n```\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Integrity checks on handoff
# ---------------------------------------------------------------------------


def _integrity_ok(
    stage_name: str, models: list[BaseModel], run_id: str, root: Path | None
) -> bool:
    """Guard against stale / mismatched artifacts crossing a boundary."""
    for m in models:
        model_run_id = getattr(m, "run_id", None)
        if model_run_id is not None and model_run_id != run_id:
            log.error("stage %s artifact run_id mismatch: %s != %s",
                      stage_name, model_run_id, run_id)
            return False

    # The backtest result must have been produced from the current spec.
    if stage_name == "backtest":
        result = next((m for m in models if isinstance(m, BacktestResult)), None)
        spec_path = stage_dir(run_id, "strategy_synthesis", root=root) / "strategy_spec.json"
        if result is not None and spec_path.exists():
            spec = StrategySpec.model_validate_json(spec_path.read_text())
            if result.strategy_spec_hash != spec_hash(spec):
                log.error("backtest result spec-hash mismatch — stale backtest")
                return False
    return True


# ---------------------------------------------------------------------------
# Output loading
# ---------------------------------------------------------------------------


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
# Resume helpers
# ---------------------------------------------------------------------------


def _stage_idx(stage_name: str) -> int:
    try:
        return STAGE_ORDER.index(stage_name)
    except ValueError as exc:
        raise ValueError(f"unknown stage {stage_name!r}; valid: {STAGE_ORDER}") from exc


def _may_skip(stage_name: str, from_idx: int | None) -> bool:
    """A stage may be skipped (loaded from disk) only if it sits *before* the
    forced re-run point. With no from_stage, all stages are skippable."""
    if from_idx is None:
        return True
    return _stage_idx(stage_name) < from_idx


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
    run_id = ctx["run_id"]
    prompts = {
        "research": (
            f"Begin Research for run {run_id}. Universe: {ctx['universe']}. "
            f"Strategy type: {ctx['strategy_type']}. Thesis: {ctx['thesis']}."
        ),
        "review": (
            f"Begin Review for run {run_id}. Verify every claim in the findings "
            f"provided below. Write critique.json and decision.json."
        ),
        "strategy_synthesis": (
            f"Begin Strategy Synthesis for run {run_id}. Convert the validated "
            f"findings below into a frozen StrategySpec at "
            f"runs/{run_id}/03_strategy/strategy_spec.json."
        ),
        "backtest": (
            f"Begin Backtest for run {run_id}. The frozen spec is below; its path "
            f"is runs/{run_id}/03_strategy/strategy_spec.json. Call run_backtest."
        ),
        "risk_check": (
            f"Begin Risk Check for run {run_id}. Run walk-forward, parameter "
            f"sensitivity, and Monte Carlo using the spec and result below."
        ),
        "report": (
            f"Begin Report for run {run_id}. Synthesize the artifacts below into "
            f"final_report.md and final_report.json."
        ),
    }
    return prompts[stage_name]


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def _write_manifest(manifest: RunManifest, *, root: Path | None) -> None:
    path = run_dir(manifest.run_id, root=root) / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))


def _load_manifest(run_id: str, *, root: Path | None) -> RunManifest:
    path = run_dir(run_id, root=root) / "manifest.json"
    return RunManifest.model_validate_json(path.read_text())


def _update_manifest(run_id: str, *, root: Path | None, **fields: Any) -> RunManifest:
    manifest = _load_manifest(run_id, root=root)
    for k, v in fields.items():
        setattr(manifest, k, v)
    _write_manifest(manifest, root=root)
    return manifest


def _record_stage(
    run_id: str, root: Path | None, stage_name: str, result: StageRunResult, respawns: int
) -> None:
    manifest = _load_manifest(run_id, root=root)
    if stage_name not in manifest.completed_stages:
        manifest.completed_stages.append(stage_name)
    manifest.stage_stats[stage_name] = StageStat(
        peak_context_pct=round(result.peak_context_pct, 1),
        output_tokens=result.total_output_tokens,
        respawns=respawns,
        ceiling_hit=result.exit_reason == "ceiling_hit",
        completed_at=now(),
    )
    _write_manifest(manifest, root=root)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _halt(
    run_id: str, root: Path | None, outcomes: dict[str, StageOutcome],
    halted_at: str, reason: str,
) -> dict[str, Any]:
    _update_manifest(run_id, root=root, halted_at=halted_at, halt_reason=reason)
    return _summarize(run_id, root, outcomes, halted_at=halted_at, reason=reason)


def _summarize(
    run_id: str, root: Path | None, outcomes: dict[str, StageOutcome],
    *, halted_at: str | None = None, reason: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "halted_at": halted_at,
        "halt_reason": reason,
        "stages_completed": [k for k, v in outcomes.items() if not v.skipped],
        "stages_skipped": [k for k, v in outcomes.items() if v.skipped],
        "stage_stats": {
            k: {
                "peak_pct": round(v.run_result.peak_context_pct, 1) if v.run_result else None,
                "output_tokens": v.run_result.total_output_tokens if v.run_result else None,
                "respawns": v.respawns,
                "skipped": v.skipped,
            }
            for k, v in outcomes.items()
        },
        "completed_at": now().isoformat(),
    }
