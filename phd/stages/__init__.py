"""Stage definitions. Each stage exposes:

- name
- allowed_tools (per the anti-dilution rule: each stage stays in its lane)
- output model class
- system_prompt / user_prompt builders
- mcp_servers (only Backtest + Risk Check use phd-mcp)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import (
    BacktestResult,
    FinalReport,
    ResearchFindings,
    ReviewCritique,
    ReviewDecision,
    RiskAssessment,
    StrategySpec,
)
from ._models import ModelConfig, StageModel

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass(frozen=True)
class StageDef:
    """A pure data class describing one stage."""

    name: str
    prompt_template: str
    allowed_tools: tuple[str, ...]
    output_filename: str | tuple[str, ...]
    output_model_cls: type | tuple[type, ...]
    uses_phd_mcp: bool = False


def _load(prompt_filename: str) -> str:
    return (PROMPTS_DIR / prompt_filename).read_text()


STAGES: dict[str, StageDef] = {
    "research": StageDef(
        name="research",
        prompt_template=_load("research.md"),
        allowed_tools=("WebSearch", "WebFetch", "Read", "Write"),
        output_filename="findings.json",
        output_model_cls=ResearchFindings,
    ),
    "review": StageDef(
        name="review",
        prompt_template=_load("review.md"),
        allowed_tools=("WebFetch", "Read", "Write"),
        output_filename=("critique.json", "decision.json"),
        output_model_cls=(ReviewCritique, ReviewDecision),
    ),
    "strategy_synthesis": StageDef(
        name="strategy_synthesis",
        prompt_template=_load("strategy_synthesis.md"),
        allowed_tools=("Read", "Write"),
        output_filename="strategy_spec.json",
        output_model_cls=StrategySpec,
    ),
    "backtest": StageDef(
        name="backtest",
        prompt_template=_load("backtest.md"),
        allowed_tools=(
            "Read",
            "Write",
            "mcp__phd__run_backtest",
            "mcp__phd__fetch_ohlcv",
        ),
        output_filename="result.json",
        output_model_cls=BacktestResult,
        uses_phd_mcp=True,
    ),
    "risk_check": StageDef(
        name="risk_check",
        prompt_template=_load("risk_check.md"),
        allowed_tools=(
            "Read",
            "Write",
            "mcp__phd__walk_forward",
            "mcp__phd__param_sensitivity",
            "mcp__phd__monte_carlo",
        ),
        output_filename="assessment.json",
        output_model_cls=RiskAssessment,
        uses_phd_mcp=True,
    ),
    "report": StageDef(
        name="report",
        prompt_template=_load("report.md"),
        allowed_tools=("Read", "Write"),
        output_filename="final_report.json",
        output_model_cls=FinalReport,
    ),
}


STAGE_ORDER = (
    "research",
    "review",
    "strategy_synthesis",
    "backtest",
    "risk_check",
    "report",
)


__all__ = [
    "STAGES",
    "STAGE_ORDER",
    "StageDef",
    "ModelConfig",
    "StageModel",
    "render_prompt",
]


def render_prompt(template: str, **context: Any) -> str:
    """Render a markdown template that uses {placeholders}.

    The templates use double braces ``{{`` / ``}}`` for any literal JSON braces,
    so plain str.format works.
    """
    return template.format(**context)
