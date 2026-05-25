"""Run-id minting, artifact directories, and pydantic-validated I/O.

Every artifact that crosses a stage boundary goes through `read_artifact` /
`write_artifact` here — that's the only path that touches `runs/<run_id>/*`.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

RUNS_ROOT = Path("runs")

STAGE_DIRS = {
    "research": "01_research",
    "review": "02_review",
    "strategy_synthesis": "03_strategy",
    "backtest": "04_backtest",
    "risk_check": "05_risk",
    "report": "06_report",
}


def mint_run_id() -> str:
    """A sortable, human-friendly run id."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(3)}"


def run_dir(run_id: str, *, root: Path | None = None) -> Path:
    return (root or RUNS_ROOT) / run_id


def stage_dir(run_id: str, stage_name: str, *, root: Path | None = None) -> Path:
    if stage_name not in STAGE_DIRS:
        raise KeyError(f"unknown stage {stage_name!r}")
    return run_dir(run_id, root=root) / STAGE_DIRS[stage_name]


def ensure_run_layout(run_id: str, *, root: Path | None = None) -> Path:
    """Create the runs/<run_id>/0N_*/ tree."""
    base = run_dir(run_id, root=root)
    base.mkdir(parents=True, exist_ok=True)
    for d in STAGE_DIRS.values():
        (base / d).mkdir(exist_ok=True)
    return base


def write_artifact(
    run_id: str,
    stage_name: str,
    filename: str,
    model: BaseModel,
    *,
    root: Path | None = None,
) -> Path:
    path = stage_dir(run_id, stage_name, root=root) / filename
    path.write_text(model.model_dump_json(indent=2))
    return path


def read_artifact(
    run_id: str,
    stage_name: str,
    filename: str,
    model_cls: type[T],
    *,
    root: Path | None = None,
) -> T:
    path = stage_dir(run_id, stage_name, root=root) / filename
    data = json.loads(path.read_text())
    return model_cls.model_validate(data)


def append_transcript(run_id: str, stage_name: str, event: dict, *, root: Path | None = None) -> None:
    """Append one JSON line to runs/<run_id>/0N_*/transcript.jsonl.

    Transcripts exist for human debugging only — they MUST NOT be loaded into
    a downstream agent's context.
    """
    path = stage_dir(run_id, stage_name, root=root) / "transcript.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def now() -> datetime:
    return datetime.now(timezone.utc)
