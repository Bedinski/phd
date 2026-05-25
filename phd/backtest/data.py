"""OHLCV data loading with on-disk pickle cache.

yfinance is the free default. Alpaca is the upgrade path once the user wires
APCA credentials. Cache files live under data_cache/ (gitignored).
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE_ROOT = Path("data_cache")


def _cache_path(ticker: str, start: date, end: date, source: str) -> Path:
    CACHE_ROOT.mkdir(exist_ok=True)
    key = f"{source}-{ticker}-{start.isoformat()}-{end.isoformat()}".encode()
    digest = hashlib.sha1(key).hexdigest()[:12]
    return CACHE_ROOT / f"{source}-{ticker}-{digest}.pkl"


def load_ohlcv(
    ticker: str,
    start: date,
    end: date,
    *,
    source: str = "yfinance",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns: open, high, low, close, volume."""
    if use_cache:
        path = _cache_path(ticker, start, end, source)
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)

    if source == "yfinance":
        df = _load_yfinance(ticker, start, end, interval)
    elif source == "alpaca":
        df = _load_alpaca(ticker, start, end, interval)
    else:
        raise ValueError(f"unknown data source: {source!r}")

    if use_cache:
        with _cache_path(ticker, start, end, source).open("wb") as f:
            pickle.dump(df, f)
    return df


def _load_yfinance(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} {start}..{end}")
    # yfinance returns columns as MultiIndex when downloading multiple symbols;
    # for a single ticker we still get plain columns.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _load_alpaca(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    import os

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca credentials missing — set APCA_API_KEY_ID and APCA_API_SECRET_KEY")

    tf = {
        "1d": TimeFrame.Day,
        "1h": TimeFrame.Hour,
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
    }.get(interval)
    if tf is None:
        raise ValueError(f"unsupported interval: {interval!r}")

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    req = StockBarsRequest(symbol_or_symbols=ticker, timeframe=tf, start=start, end=end)
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        raise RuntimeError(f"Alpaca returned no data for {ticker} {start}..{end}")
    bars = bars.reset_index().set_index("timestamp")
    df = bars.rename(
        columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
    )[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_universe(
    tickers: list[str],
    start: date,
    end: date,
    *,
    source: str = "yfinance",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Load OHLCV for many tickers. Skips tickers that fail rather than aborting the run."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = load_ohlcv(t, start, end, source=source, interval=interval)
        except Exception as exc:  # noqa: BLE001 — universe scan tolerates partial failure
            log.warning("skipping %s: %s", t, exc)
    if not out:
        raise RuntimeError(f"no usable tickers in universe of size {len(tickers)}")
    return out
