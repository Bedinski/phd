"""Indicator dispatch tests against synthetic OHLCV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phd.backtest import indicators as ind


@pytest.fixture
def synth_ohlcv() -> pd.DataFrame:
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.2, n),
            "high": close + rng.uniform(0.1, 1.0, n),
            "low": close - rng.uniform(0.1, 1.0, n),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def test_close_is_passthrough(synth_ohlcv) -> None:
    s = ind.compute("close", synth_ohlcv, {})
    assert s.equals(synth_ohlcv["close"])


def test_sma_window(synth_ohlcv) -> None:
    s = ind.compute("sma", synth_ohlcv, {"window": 20})
    assert s.iloc[19] == pytest.approx(synth_ohlcv["close"].iloc[:20].mean())
    assert pd.isna(s.iloc[18])


def test_ema_smoothing(synth_ohlcv) -> None:
    s = ind.compute("ema", synth_ohlcv, {"window": 10})
    assert not s.tail(50).isna().any()


def test_rsi_bounded(synth_ohlcv) -> None:
    s = ind.compute("rsi", synth_ohlcv, {"window": 14}).dropna()
    assert (s >= 0).all() and (s <= 100).all()


def test_atr_non_negative(synth_ohlcv) -> None:
    s = ind.compute("atr", synth_ohlcv, {"window": 14}).dropna()
    assert (s >= 0).all()


def test_unknown_indicator_raises(synth_ohlcv) -> None:
    with pytest.raises(ValueError):
        ind.compute("nonexistent", synth_ohlcv, {})
