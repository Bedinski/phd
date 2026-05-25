"""Typer CLI: `phd research | inspect | deploy`."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .orchestrator import PipelineInput, run_pipeline
from .persistence import STAGE_DIRS, run_dir
from .stages._models import ModelConfig

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
console = Console()


def _bootstrap_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _build_model_config(
    cheap: bool,
    all_sonnet: bool,
    overrides: dict[str, str | None],
    effort_override: str | None,
) -> ModelConfig:
    cfg = ModelConfig()
    if cheap:
        cfg.cheap()
    elif all_sonnet:
        cfg.all_sonnet()
    for stage_name, model_alias in overrides.items():
        if model_alias:
            cfg.override_stage(stage_name, model=model_alias)
        if effort_override is not None:
            cfg.override_stage(stage_name, effort=effort_override)
    return cfg


@app.command()
def research(
    universe: str = typer.Option(..., help="Universe identifier: sp500, sp500_top10, nasdaq100, custom"),
    strategy_type: str = typer.Option(..., help='Strategy type, e.g. "post-earnings drift mean reversion"'),
    thesis: str = typer.Option(
        None,
        help="Optional one-sentence thesis; defaults to a templated form of strategy-type + universe",
    ),
    questions: Optional[list[str]] = typer.Option(
        None, "--question", "-q", help="Specific research questions (repeatable)"
    ),
    cheap: bool = typer.Option(False, "--cheap", help="Sonnet everywhere, no thinking"),
    all_sonnet: bool = typer.Option(False, "--all-sonnet", help="Sonnet everywhere, thinking on"),
    effort: Optional[str] = typer.Option(
        None, help="Override effort for all stages (low|medium|high|xhigh|max)"
    ),
    model_research: Optional[str] = typer.Option(None),
    model_review: Optional[str] = typer.Option(None),
    model_synthesis: Optional[str] = typer.Option(None),
    model_backtest: Optional[str] = typer.Option(None),
    model_risk: Optional[str] = typer.Option(None),
    model_report: Optional[str] = typer.Option(None),
    ceiling_pct: float = typer.Option(50.0, help="Context-window ceiling per stage (%%)"),
    max_review_rounds: int = typer.Option(3),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Run the full 6-stage pipeline."""
    _bootstrap_logging(verbose)
    model_cfg = _build_model_config(
        cheap=cheap,
        all_sonnet=all_sonnet,
        overrides={
            "research": model_research,
            "review": model_review,
            "strategy_synthesis": model_synthesis,
            "backtest": model_backtest,
            "risk_check": model_risk,
            "report": model_report,
        },
        effort_override=effort,
    )
    thesis_str = thesis or f"Identify and validate a {strategy_type} edge on {universe}."
    inp = PipelineInput(
        thesis=thesis_str,
        universe=universe,
        strategy_type=strategy_type,
        research_questions=list(questions or []),
        model_config=model_cfg,
        cwd=Path.cwd(),
        max_review_rounds=max_review_rounds,
        ceiling_pct=ceiling_pct,
    )
    summary = asyncio.run(run_pipeline(inp))
    console.print_json(json.dumps(summary, default=str))


@app.command()
def inspect(
    run_id: str = typer.Argument(...),
):
    """Pretty-print artifacts for a finished (or in-progress) run."""
    root = run_dir(run_id)
    if not root.exists():
        console.print(f"[red]no such run:[/red] {root}")
        raise typer.Exit(1)

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        console.print_json(json.dumps(manifest, indent=2, default=str))

    table = Table(title=f"Stages for {run_id}")
    table.add_column("Stage")
    table.add_column("Artifacts")
    for stage_name, sub in STAGE_DIRS.items():
        sd = root / sub
        if not sd.exists():
            continue
        files = sorted(p.name for p in sd.glob("*"))
        table.add_row(stage_name, ", ".join(files) or "(empty)")
    console.print(table)


@app.command()
def deploy(
    run_id: str = typer.Argument(...),
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Default dry-run; pass --live to submit"),
):
    """Manual gate: deploy a finished run's StrategySpec to Alpaca paper trading."""
    from .deploy.alpaca_paper import deploy as do_deploy

    result = do_deploy(run_id, dry_run=dry_run)
    console.print_json(json.dumps(result, default=str))


if __name__ == "__main__":
    app()
