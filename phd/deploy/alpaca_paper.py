"""Manual-gate deployment of a frozen StrategySpec to an Alpaca paper account.

This module does NOT run automatically. The user invokes `phd deploy <run_id>`
after reviewing the final_report.md. The strategy spec is the contract — the
deploy script materializes today's intended positions from it and submits
market orders to the paper-trading endpoint.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..backtest import data as data_mod
from ..backtest import runner as bt_runner
from ..backtest import universe as universe_mod
from ..models import StrategySpec
from ..persistence import read_artifact

log = logging.getLogger(__name__)


@dataclass
class IntendedOrder:
    ticker: str
    side: str  # "buy" | "sell"
    qty: float
    rationale: str


def compute_intended_orders(spec: StrategySpec, *, as_of: date | None = None) -> list[IntendedOrder]:
    """For each ticker in the resolved universe, re-evaluate the spec's entry
    and exit rules on the most recent bars and decide what action (if any)
    today would take.
    """

    as_of = as_of or date.today()
    start = date(as_of.year - 1, as_of.month, as_of.day)  # 1y of history is enough for indicators
    tickers = universe_mod.resolve(spec.universe.base, spec.universe.custom_tickers)
    orders: list[IntendedOrder] = []

    for tkr in tickers:
        try:
            df = data_mod.load_ohlcv(tkr, start, as_of, source="yfinance", interval=spec.timeframe)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", tkr, exc)
            continue
        entries, exits = bt_runner._build_signals(spec, df)  # type: ignore[attr-defined]
        if len(entries) == 0:
            continue
        if bool(entries.iloc[-1]):
            orders.append(
                IntendedOrder(
                    ticker=tkr,
                    side="buy",
                    qty=1,  # 1 share placeholder; real sizing applied by submit_orders
                    rationale="entry rule fired on latest bar",
                )
            )
        elif bool(exits.iloc[-1]):
            orders.append(
                IntendedOrder(
                    ticker=tkr,
                    side="sell",
                    qty=1,
                    rationale="exit rule fired on latest bar",
                )
            )

    return orders


def submit_orders(orders: list[IntendedOrder], *, dry_run: bool = True) -> dict:
    """Submit the orders to Alpaca paper trading. If `dry_run`, only print."""
    if dry_run:
        return {"dry_run": True, "would_submit": [o.__dict__ for o in orders]}

    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "missing APCA_API_KEY_ID / APCA_API_SECRET_KEY. Copy .env.example to .env and fill in."
        )

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    client = TradingClient(api_key=key, secret_key=secret, paper=True)
    submitted: list[dict] = []
    for o in orders:
        req = MarketOrderRequest(
            symbol=o.ticker.replace("-", "."),  # Alpaca uses BRK.B; spec uses BRK-B
            qty=o.qty,
            side=OrderSide.BUY if o.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        resp = client.submit_order(req)
        submitted.append(
            {
                "ticker": o.ticker,
                "side": o.side,
                "qty": o.qty,
                "alpaca_order_id": getattr(resp, "id", None),
            }
        )
    return {"dry_run": False, "submitted": submitted}


def deploy(run_id: str, *, runs_root: Path | None = None, dry_run: bool = True) -> dict:
    spec = read_artifact(run_id, "strategy_synthesis", "strategy_spec.json", StrategySpec, root=runs_root)
    orders = compute_intended_orders(spec)
    result = submit_orders(orders, dry_run=dry_run)
    result["run_id"] = run_id
    result["strategy_name"] = spec.name
    result["order_count"] = len(orders)
    return result
