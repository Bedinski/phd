"""The phd-mcp server. Five tools the Backtest and Risk-Check stages use:

  fetch_ohlcv         — OHLCV with on-disk cache
  run_backtest        — strategy_spec.json → BacktestResult
  walk_forward        — rolling in-sample vs out-of-sample Sharpe
  param_sensitivity   — grid sweep, report CV of Sharpe across params
  monte_carlo         — block-bootstrap robustness on equity curve

All heavy computation lives in phd.backtest.runner; these are thin adapters
that the LLM can call by name and get back JSON summaries.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from ..backtest import data as data_mod
from ..backtest import runner
from ..models import StrategySpec

log = logging.getLogger(__name__)


def _text(payload: Any) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]}


def _err(message: str) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps({"error": message})}],
        "isError": True,
    }


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------


@tool(
    "fetch_ohlcv",
    "Download OHLCV bars for a ticker between two dates. Results are cached "
    "on disk so subsequent calls for the same ticker/range are instant. "
    "Returns a JSON summary with row count, date span, and the cached file path.",
    {
        "ticker": str,
        "start": str,   # ISO date
        "end": str,
        "source": str,  # "yfinance" | "alpaca"
        "interval": str,
    },
)
async def fetch_ohlcv(args: dict) -> dict:
    try:
        df = data_mod.load_ohlcv(
            ticker=args["ticker"],
            start=date.fromisoformat(args["start"]),
            end=date.fromisoformat(args["end"]),
            source=args.get("source", "yfinance"),
            interval=args.get("interval", "1d"),
        )
        return _text(
            {
                "ticker": args["ticker"],
                "rows": int(len(df)),
                "first_date": df.index[0].date().isoformat(),
                "last_date": df.index[-1].date().isoformat(),
                "columns": list(df.columns),
            }
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("fetch_ohlcv failed")
        return _err(f"fetch_ohlcv failed: {exc}")


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------


@tool(
    "run_backtest",
    "Execute a frozen strategy_spec.json across its universe and return a "
    "structured BacktestResult JSON. Writes equity_curve.png, trades.csv, and "
    "monthly_returns.csv into the given out_dir. Numbers are computed by "
    "vectorbt, never by the LLM.",
    {
        "spec_path": str,    # path to strategy_spec.json
        "out_dir": str,      # where to write result artifacts
        "data_source": str,  # optional
    },
)
async def run_backtest(args: dict) -> dict:
    try:
        spec = StrategySpec.model_validate_json(Path(args["spec_path"]).read_text())
        result = runner.run_backtest(
            spec,
            out_dir=Path(args["out_dir"]),
            data_source=args.get("data_source", "yfinance"),
        )
        return _text(result.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        log.exception("run_backtest failed")
        return _err(f"run_backtest failed: {exc}")


# ---------------------------------------------------------------------------
# walk_forward
# ---------------------------------------------------------------------------


@tool(
    "walk_forward",
    "Roll the strategy_spec forward in folds (in-sample / out-of-sample) and "
    "compare Sharpe across folds. Returns a DimensionAssessment JSON with a "
    "PASS / CAUTION / FAIL verdict.",
    {
        "spec_path": str,
        "out_dir": str,
        "in_sample_days": int,
        "out_sample_days": int,
    },
)
async def walk_forward(args: dict) -> dict:
    try:
        spec = StrategySpec.model_validate_json(Path(args["spec_path"]).read_text())
        assessment = runner.walk_forward(
            spec,
            out_dir=Path(args["out_dir"]),
            in_sample_days=int(args.get("in_sample_days", 252)),
            out_sample_days=int(args.get("out_sample_days", 63)),
        )
        return _text(assessment.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        log.exception("walk_forward failed")
        return _err(f"walk_forward failed: {exc}")


# ---------------------------------------------------------------------------
# param_sensitivity
# ---------------------------------------------------------------------------


@tool(
    "param_sensitivity",
    "Sweep a parameter grid over the spec's params; report mean / std / "
    "coefficient-of-variation of Sharpe across combinations. A low CV with "
    "positive mean Sharpe suggests the strategy is not over-fit to one set "
    "of parameters. Grid is a JSON dict of param_name -> list of values.",
    {
        "spec_path": str,
        "grid_json": str,
        "out_dir": str,
    },
)
async def param_sensitivity(args: dict) -> dict:
    try:
        spec = StrategySpec.model_validate_json(Path(args["spec_path"]).read_text())
        grid = json.loads(args["grid_json"])
        if not isinstance(grid, dict) or not all(isinstance(v, list) for v in grid.values()):
            raise ValueError("grid_json must be a JSON object mapping name -> list of values")
        assessment = runner.param_sensitivity(spec, grid, out_dir=Path(args["out_dir"]))
        return _text(assessment.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        log.exception("param_sensitivity failed")
        return _err(f"param_sensitivity failed: {exc}")


# ---------------------------------------------------------------------------
# monte_carlo
# ---------------------------------------------------------------------------


@tool(
    "monte_carlo",
    "Block-bootstrap the equity curve (CSV at equity_csv_path) and report "
    "the distribution of terminal value and worst drawdown. Useful for "
    "judging whether the headline result depends on a lucky sequence.",
    {
        "equity_csv_path": str,
        "out_dir": str,
        "n": int,
        "block_size": int,
    },
)
async def monte_carlo(args: dict) -> dict:
    try:
        assessment = runner.monte_carlo(
            equity_csv_path=args["equity_csv_path"],
            out_dir=Path(args["out_dir"]),
            n=int(args.get("n", 1000)),
            block_size=int(args.get("block_size", 10)),
        )
        return _text(assessment.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        log.exception("monte_carlo failed")
        return _err(f"monte_carlo failed: {exc}")


# ---------------------------------------------------------------------------
# Server builder
# ---------------------------------------------------------------------------


def build_phd_mcp_server():
    """Return the McpSdkServerConfig to plug into ClaudeAgentOptions.mcp_servers."""
    return create_sdk_mcp_server(
        name="phd",
        version="0.1.0",
        tools=[fetch_ohlcv, run_backtest, walk_forward, param_sensitivity, monte_carlo],
    )
