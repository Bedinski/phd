"""Tests for orchestrator handoff, integrity checks, and resume-skip logic.

These cover the pure functions — no Claude calls. The stage runner is exercised
in test_sdk_runner.py.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from phd import orchestrator as orch
from phd.models import (
    BacktestResult,
    Claim,
    Confidence,
    ResearchFindings,
    RuleExpression,
    Sizing,
    Source,
    StrategySpec,
    UniverseFilter,
)
from phd.persistence import ensure_run_layout, write_artifact


def _now() -> datetime:
    return datetime(2026, 5, 25, tzinfo=timezone.utc)


def _findings(run_id: str) -> ResearchFindings:
    return ResearchFindings(
        run_id=run_id, thesis="t", universe="u", strategy_type="s",
        research_questions=["q?"],
        claims=[
            Claim(
                id="C001",
                statement="A specific, cited claim about the strategy edge.",
                sources=[Source(url="https://e.com/x", accessed_at=date(2026, 5, 25))],
                confidence=Confidence.HIGH,
                relevance_to_thesis="x" * 25,
            )
        ],
        generated_at=_now(),
    )


def _spec(run_id: str) -> StrategySpec:
    return StrategySpec(
        run_id=run_id, name="sma", description="d",
        universe=UniverseFilter(base="sp500_top10"),
        entry_rules=[RuleExpression(indicator="sma", params={"window": 50},
                                    operator="cross_above", rhs_indicator="sma",
                                    rhs_params={"window": 200})],
        exit_rules=[RuleExpression(indicator="sma", params={"window": 50},
                                   operator="cross_below", rhs_indicator="sma",
                                   rhs_params={"window": 200})],
        sizing=Sizing(method="equal_weight", max_positions=10),
        backtest_start=date(2020, 1, 1), backtest_end=date(2024, 12, 31),
    )


# --- handoff -----------------------------------------------------------------


def test_prior_artifacts_block_injects_upstream(tmp_path) -> None:
    rid = "r1"
    ensure_run_layout(rid, root=tmp_path)
    write_artifact(rid, "research", "findings.json", _findings(rid), root=tmp_path)

    block = orch._prior_artifacts_block("review", rid, tmp_path)
    assert "findings.json" in block
    assert "C001" in block  # actual artifact content is embedded inline
    assert "do NOT re-derive" in block


def test_prior_artifacts_block_empty_for_research() -> None:
    assert orch._prior_artifacts_block("research", "r1", None) == ""


# --- integrity ---------------------------------------------------------------


def test_integrity_rejects_run_id_mismatch(tmp_path) -> None:
    rid = "r1"
    ensure_run_layout(rid, root=tmp_path)
    wrong = _findings("DIFFERENT")
    assert orch._integrity_ok("research", [wrong], rid, tmp_path) is False
    right = _findings(rid)
    assert orch._integrity_ok("research", [right], rid, tmp_path) is True


def test_integrity_rejects_stale_backtest_spec_hash(tmp_path) -> None:
    rid = "r1"
    ensure_run_layout(rid, root=tmp_path)
    spec = _spec(rid)
    write_artifact(rid, "strategy_synthesis", "strategy_spec.json", spec, root=tmp_path)

    stale = BacktestResult(
        run_id=rid, strategy_spec_hash="deadbeef",  # does not match the spec
        total_return=0.1, cagr=0.03, sharpe=0.5, max_drawdown=-0.1, win_rate=0.5,
        num_trades=10, avg_trade_return=0.01, equity_curve_path="e.png",
        trades_csv_path="t.csv", backtest_start=date(2020, 1, 1),
        backtest_end=date(2024, 12, 31), universe_size=10, generated_at=_now(),
    )
    assert orch._integrity_ok("backtest", [stale], rid, tmp_path) is False

    from phd.backtest.runner import spec_hash

    fresh = stale.model_copy(update={"strategy_spec_hash": spec_hash(spec)})
    assert orch._integrity_ok("backtest", [fresh], rid, tmp_path) is True


# --- resume-skip -------------------------------------------------------------


def test_may_skip_without_from_stage() -> None:
    assert orch._may_skip("research", None) is True
    assert orch._may_skip("report", None) is True


def test_may_skip_respects_from_stage_boundary() -> None:
    idx = orch._stage_idx("backtest")
    # Stages before backtest are skippable; backtest and after are not.
    assert orch._may_skip("research", idx) is True
    assert orch._may_skip("strategy_synthesis", idx) is True
    assert orch._may_skip("backtest", idx) is False
    assert orch._may_skip("risk_check", idx) is False


def test_stage_idx_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        orch._stage_idx("nonexistent")


# --- required artifacts -----------------------------------------------------


def test_required_artifact_paths_single_file() -> None:
    paths = orch._required_artifact_paths("research", "r1", None)
    assert len(paths) == 1
    assert paths[0].name == "findings.json"


def test_required_artifact_paths_dual_file() -> None:
    paths = orch._required_artifact_paths("review", "r1", None)
    names = sorted(p.name for p in paths)
    assert names == ["critique.json", "decision.json"]
