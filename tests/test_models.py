"""Pydantic round-trip + validation tests for the inter-stage contracts."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from phd.models import (
    BacktestResult,
    Claim,
    Confidence,
    Decision,
    DimensionAssessment,
    FinalReport,
    Recommendation,
    ResearchFindings,
    ReviewCritique,
    ReviewDecision,
    RiskAssessment,
    RiskVerdict,
    RuleExpression,
    Sizing,
    Source,
    StrategySpec,
    UniverseFilter,
    Verdict,
)


def _now() -> datetime:
    return datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)


def test_claim_requires_at_least_one_source() -> None:
    with pytest.raises(ValidationError):
        Claim(
            id="C001",
            statement="A statement long enough.",
            sources=[],
            confidence=Confidence.MEDIUM,
            relevance_to_thesis="x" * 20,
        )


def test_research_findings_roundtrip() -> None:
    rf = ResearchFindings(
        run_id="r-1",
        thesis="mean rev edge",
        universe="sp500_top10",
        strategy_type="rsi reversal",
        research_questions=["does it work?"],
        claims=[
            Claim(
                id="C001",
                statement="RSI(2) < 10 has historically preceded a 5d positive return on average.",
                sources=[
                    Source(
                        url="https://example.com/study",
                        title="A Study",
                        publisher="Pub",
                        accessed_at=date(2026, 5, 25),
                    )
                ],
                confidence=Confidence.MEDIUM,
                relevance_to_thesis="directly supports the entry rule",
            )
        ],
        generated_at=_now(),
    )
    serialized = rf.model_dump_json()
    loaded = ResearchFindings.model_validate_json(serialized)
    assert loaded.claims[0].id == "C001"


def test_strategy_spec_rule_validation() -> None:
    # rhs_indicator XOR rhs_value
    with pytest.raises(ValidationError):
        RuleExpression(
            indicator="sma",
            params={"window": 50},
            operator="gt",
            rhs_indicator="sma",
            rhs_params={"window": 200},
            rhs_value=100.0,  # both set -> invalid
        )

    rule_ok = RuleExpression(
        indicator="sma",
        params={"window": 50},
        operator="cross_above",
        rhs_indicator="sma",
        rhs_params={"window": 200},
    )
    assert rule_ok.operator == "cross_above"


def test_strategy_spec_roundtrip() -> None:
    spec = StrategySpec(
        run_id="r-1",
        name="sma-50-200",
        description="Classic crossover.",
        universe=UniverseFilter(base="sp500_top10"),
        entry_rules=[
            RuleExpression(
                indicator="sma", params={"window": 50}, operator="cross_above",
                rhs_indicator="sma", rhs_params={"window": 200},
            )
        ],
        exit_rules=[
            RuleExpression(
                indicator="sma", params={"window": 50}, operator="cross_below",
                rhs_indicator="sma", rhs_params={"window": 200},
            )
        ],
        sizing=Sizing(method="equal_weight", max_positions=10),
        backtest_start=date(2020, 1, 1),
        backtest_end=date(2024, 12, 31),
    )
    blob = spec.model_dump_json()
    loaded = StrategySpec.model_validate_json(blob)
    assert loaded.entry_rules[0].operator == "cross_above"


def test_review_decision_enum() -> None:
    rd = ReviewDecision(
        run_id="r-1",
        revision_round=0,
        decision=Decision.PASS,
        rationale="all clear",
    )
    assert rd.decision == Decision.PASS


def test_review_critique_per_claim() -> None:
    rc = ReviewCritique(
        run_id="r-1",
        revision_round=0,
        claim_verdicts=[
            {
                "claim_id": "C001",
                "verdict": Verdict.SUPPORTED,
                "notes": "source clearly supports it",
            }
        ],
        generated_at=_now(),
    )
    assert rc.claim_verdicts[0].verdict == Verdict.SUPPORTED


def test_backtest_result_required_fields() -> None:
    br = BacktestResult(
        run_id="r-1",
        strategy_spec_hash="abc123",
        total_return=0.25,
        cagr=0.06,
        sharpe=0.85,
        max_drawdown=-0.12,
        win_rate=0.55,
        num_trades=42,
        avg_trade_return=0.005,
        equity_curve_path="x.png",
        trades_csv_path="t.csv",
        backtest_start=date(2020, 1, 1),
        backtest_end=date(2024, 12, 31),
        universe_size=10,
        generated_at=_now(),
    )
    assert br.sharpe == pytest.approx(0.85)


def test_risk_assessment_aggregates_dimensions() -> None:
    ra = RiskAssessment(
        run_id="r-1",
        dimensions=[
            DimensionAssessment(
                dimension="walk_forward", verdict=RiskVerdict.PASS,
                summary="ok", metrics={"oos_to_is_ratio": 0.8},
            ),
            DimensionAssessment(
                dimension="param_sensitivity", verdict=RiskVerdict.CAUTION,
                summary="moderate", metrics={"sharpe_cv": 0.7},
            ),
            DimensionAssessment(
                dimension="monte_carlo", verdict=RiskVerdict.PASS,
                summary="ok", metrics={"p5_terminal": 1.1},
            ),
        ],
        overall_verdict=RiskVerdict.CAUTION,
        overall_summary="walk_forward=PASS; param_sensitivity=CAUTION; monte_carlo=PASS",
        generated_at=_now(),
    )
    assert ra.overall_verdict == RiskVerdict.CAUTION


def test_final_report_recommendation() -> None:
    fr = FinalReport(
        run_id="r-1",
        strategy_name="sma-50-200",
        executive_summary="x" * 50,
        evidence_chain_summary="y" * 50,
        headline_metrics={"sharpe": 0.85, "cagr": 0.06, "max_drawdown": -0.12,
                          "total_return": 0.25, "num_trades": 42},
        recommendation=Recommendation.DEPLOY,
        recommendation_rationale="x" * 30,
        generated_at=_now(),
    )
    assert fr.recommendation == Recommendation.DEPLOY
