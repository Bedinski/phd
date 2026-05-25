"""Pydantic models defining the inter-stage contracts.

Every artifact that crosses an agent boundary is one of these. The orchestrator
validates each stage's output against the corresponding model before passing it
to the next stage — this is the structural anti-dilution mechanism.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Decision(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"
    REJECT = "REJECT"


class Recommendation(str, Enum):
    DEPLOY = "DEPLOY"
    HOLD = "HOLD"
    DISCARD = "DISCARD"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"


class RiskVerdict(str, Enum):
    PASS = "PASS"
    CAUTION = "CAUTION"
    FAIL = "FAIL"


class Source(BaseModel):
    url: HttpUrl
    title: str | None = None
    publisher: str | None = None
    accessed_at: date


class Claim(BaseModel):
    """A single citable assertion produced by the Research stage."""

    id: str = Field(description="Stable claim id, e.g. C001")
    statement: str = Field(min_length=10, max_length=600)
    sources: list[Source] = Field(min_length=1)
    confidence: Confidence
    relevance_to_thesis: str = Field(min_length=10, max_length=400)


# ---------------------------------------------------------------------------
# Stage 01 — Research
# ---------------------------------------------------------------------------


class ResearchFindings(BaseModel):
    """Output of the Research stage."""

    run_id: str
    thesis: str
    universe: str
    strategy_type: str
    research_questions: list[str]
    claims: list[Claim]
    open_questions: list[str] = Field(default_factory=list)
    generated_at: datetime
    revision_round: int = 0


# ---------------------------------------------------------------------------
# Stage 02 — Review
# ---------------------------------------------------------------------------


class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Verdict
    notes: str = Field(max_length=600)
    re_checked_urls: list[HttpUrl] = Field(default_factory=list)


class ReviewCritique(BaseModel):
    """Per-claim review of the research findings."""

    run_id: str
    revision_round: int
    claim_verdicts: list[ClaimVerdict]
    systemic_concerns: list[str] = Field(default_factory=list)
    generated_at: datetime


class ReviewDecision(BaseModel):
    """PASS / REVISE / REJECT gate. Drives the orchestrator's revision loop."""

    run_id: str
    revision_round: int
    decision: Decision
    rationale: str
    must_address: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 03 — Strategy Synthesis (the FROZEN spec)
# ---------------------------------------------------------------------------


class RuleExpression(BaseModel):
    """A single boolean rule expressed in a small DSL.

    The DSL is intentionally tiny: the backtest runner knows how to interpret
    these, so the agent can't smuggle arbitrary code in via the spec.
    """

    indicator: str = Field(description="e.g. sma, ema, rsi, atr, close, volume")
    params: dict[str, float | int | str] = Field(default_factory=dict)
    operator: Literal["gt", "lt", "ge", "le", "eq", "cross_above", "cross_below"]
    rhs_indicator: str | None = None
    rhs_params: dict[str, float | int | str] = Field(default_factory=dict)
    rhs_value: float | None = None

    @model_validator(mode="after")
    def _rhs_exclusive(self) -> RuleExpression:
        if (self.rhs_indicator is None) == (self.rhs_value is None):
            raise ValueError("exactly one of rhs_indicator / rhs_value must be set")
        return self


class UniverseFilter(BaseModel):
    base: Literal["sp500", "sp500_top10", "nasdaq100", "custom"]
    custom_tickers: list[str] = Field(default_factory=list)
    min_avg_dollar_volume: float | None = None
    min_price: float | None = None
    max_price: float | None = None


class Sizing(BaseModel):
    method: Literal["equal_weight", "fixed_fraction", "vol_target"]
    fraction: float | None = Field(default=None, ge=0, le=1)
    target_vol_annual: float | None = Field(default=None, gt=0)
    max_positions: int = Field(default=20, ge=1, le=500)


class StrategySpec(BaseModel):
    """The frozen handoff between research/review and execution.

    Downstream stages MUST NOT re-interpret raw research. They consume only this.
    """

    run_id: str
    name: str
    description: str
    universe: UniverseFilter
    timeframe: Literal["1d", "1h", "30m", "15m"] = "1d"
    entry_rules: list[RuleExpression] = Field(min_length=1)
    exit_rules: list[RuleExpression] = Field(min_length=1)
    sizing: Sizing
    stop_loss_pct: float | None = Field(default=None, gt=0, lt=1)
    take_profit_pct: float | None = Field(default=None, gt=0)
    max_hold_days: int | None = Field(default=None, ge=1)
    params: dict[str, float | int | str] = Field(default_factory=dict)
    backtest_start: date
    backtest_end: date


# ---------------------------------------------------------------------------
# Stage 04 — Backtest
# ---------------------------------------------------------------------------


class BacktestResult(BaseModel):
    """Numbers come from vectorbt, not the LLM."""

    run_id: str
    strategy_spec_hash: str
    total_return: float
    cagr: float
    sharpe: float
    sortino: float | None = None
    max_drawdown: float
    win_rate: float
    profit_factor: float | None = None
    num_trades: int
    avg_trade_return: float
    equity_curve_path: str
    trades_csv_path: str
    monthly_returns_csv_path: str | None = None
    backtest_start: date
    backtest_end: date
    universe_size: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# Stage 05 — Risk Check
# ---------------------------------------------------------------------------


class DimensionAssessment(BaseModel):
    dimension: Literal["walk_forward", "param_sensitivity", "monte_carlo"]
    verdict: RiskVerdict
    summary: str = Field(max_length=500)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    run_id: str
    dimensions: list[DimensionAssessment]
    overall_verdict: RiskVerdict
    overall_summary: str = Field(max_length=1000)
    generated_at: datetime


# ---------------------------------------------------------------------------
# Stage 06 — Final Report
# ---------------------------------------------------------------------------


class FinalReport(BaseModel):
    run_id: str
    strategy_name: str
    executive_summary: str = Field(max_length=2000)
    evidence_chain_summary: str = Field(max_length=2000)
    headline_metrics: dict[str, float]
    recommendation: Recommendation
    recommendation_rationale: str = Field(min_length=20, max_length=2000)
    caveats: list[str] = Field(default_factory=list)
    generated_at: datetime


# ---------------------------------------------------------------------------
# Pipeline manifest
# ---------------------------------------------------------------------------


class RunManifest(BaseModel):
    """Top-level metadata for a pipeline run, written first to runs/<run_id>/."""

    run_id: str
    created_at: datetime
    thesis: str
    universe: str
    strategy_type: str
    stage_models: dict[str, str]
    stage_effort: dict[str, str]
    completed_stages: list[str] = Field(default_factory=list)
    last_decision: Decision | None = None
