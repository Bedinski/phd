"""Backtest runner smoke test using synthetic OHLCV (no network).

We monkeypatch `phd.backtest.data.load_universe` to return synthetic data so
the test runs offline and deterministically.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from phd.backtest import runner
from phd.models import RuleExpression, Sizing, StrategySpec, UniverseFilter


def _synth(ticker: str) -> pd.DataFrame:
    """Trending series so a SMA crossover produces real trades."""
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    drift = np.linspace(0, 50, n) + 5 * np.sin(np.arange(n) / 80)
    noise = rng.normal(0, 0.5, n)
    close = 100 + drift + np.cumsum(noise)
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.1, n),
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


@pytest.fixture
def patched_universe(monkeypatch):
    def _fake_load_universe(tickers, start, end, *, source="yfinance", interval="1d"):
        out = {}
        for t in tickers:
            df = _synth(t)
            df = df[(df.index.date >= start) & (df.index.date <= end)]
            out[t] = df
        return out

    monkeypatch.setattr(
        "phd.backtest.data.load_universe", _fake_load_universe
    )


def _sma_crossover_spec(run_id: str = "test-run") -> StrategySpec:
    return StrategySpec(
        run_id=run_id,
        name="sma-50-200",
        description="Classic SMA crossover on synthetic data",
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
        backtest_end=date(2022, 12, 31),
    )


def test_run_backtest_produces_finite_metrics(tmp_path, patched_universe) -> None:
    spec = _sma_crossover_spec()
    out_dir = tmp_path / "bt"
    result = runner.run_backtest(spec, out_dir=out_dir)

    assert result.universe_size > 0
    assert np.isfinite(result.sharpe)
    assert np.isfinite(result.cagr)
    assert -1.0 <= result.max_drawdown <= 0.0
    assert (out_dir / "equity_curve.png").exists()
    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "trades.csv").exists()


def test_spec_hash_stable(patched_universe) -> None:
    spec1 = _sma_crossover_spec("run-A")
    spec2 = _sma_crossover_spec("run-A")
    assert runner.spec_hash(spec1) == runner.spec_hash(spec2)


def test_spec_hash_changes_with_params(patched_universe) -> None:
    spec1 = _sma_crossover_spec()
    spec2 = spec1.model_copy(
        update={
            "entry_rules": [
                RuleExpression(
                    indicator="sma", params={"window": 20}, operator="cross_above",
                    rhs_indicator="sma", rhs_params={"window": 200},
                )
            ]
        }
    )
    assert runner.spec_hash(spec1) != runner.spec_hash(spec2)
