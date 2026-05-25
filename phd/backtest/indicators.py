"""Indicator dispatch from the StrategySpec rule DSL to pandas Series.

The dispatch is deliberately small: each indicator name maps to one function
that takes (df, params) and returns a Series. This keeps the agent's spec
output bounded — they can't smuggle arbitrary code in.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd


def _close(df: pd.DataFrame, _params: dict) -> pd.Series:
    return df["close"]


def _open(df: pd.DataFrame, _params: dict) -> pd.Series:
    return df["open"]


def _high(df: pd.DataFrame, _params: dict) -> pd.Series:
    return df["high"]


def _low(df: pd.DataFrame, _params: dict) -> pd.Series:
    return df["low"]


def _volume(df: pd.DataFrame, _params: dict) -> pd.Series:
    return df["volume"]


def _sma(df: pd.DataFrame, params: dict) -> pd.Series:
    n = int(params.get("window", params.get("n", 20)))
    return df["close"].rolling(n).mean()


def _ema(df: pd.DataFrame, params: dict) -> pd.Series:
    n = int(params.get("window", params.get("n", 20)))
    return df["close"].ewm(span=n, adjust=False).mean()


def _rsi(df: pd.DataFrame, params: dict) -> pd.Series:
    n = int(params.get("window", params.get("n", 14)))
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, params: dict) -> pd.Series:
    n = int(params.get("window", params.get("n", 14)))
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def _bbands_upper(df: pd.DataFrame, params: dict) -> pd.Series:
    n = int(params.get("window", 20))
    mult = float(params.get("mult", 2.0))
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    return ma + mult * sd


def _bbands_lower(df: pd.DataFrame, params: dict) -> pd.Series:
    n = int(params.get("window", 20))
    mult = float(params.get("mult", 2.0))
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    return ma - mult * sd


def _return_n(df: pd.DataFrame, params: dict) -> pd.Series:
    n = int(params.get("n", 1))
    return df["close"].pct_change(n)


REGISTRY: dict[str, Callable[[pd.DataFrame, dict], pd.Series]] = {
    "close": _close,
    "open": _open,
    "high": _high,
    "low": _low,
    "volume": _volume,
    "sma": _sma,
    "ema": _ema,
    "rsi": _rsi,
    "atr": _atr,
    "bbands_upper": _bbands_upper,
    "bbands_lower": _bbands_lower,
    "return_n": _return_n,
}


def compute(name: str, df: pd.DataFrame, params: dict) -> pd.Series:
    fn = REGISTRY.get(name)
    if fn is None:
        raise ValueError(
            f"unknown indicator {name!r}. supported: {sorted(REGISTRY)}"
        )
    return fn(df, params or {})
