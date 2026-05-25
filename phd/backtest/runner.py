"""StrategySpec → vectorbt Portfolio → BacktestResult.

This is the deterministic, agent-free core. The Backtest stage's agent calls
this via the phd-mcp MCP server; it never writes vectorbt code itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..models import (
    BacktestResult,
    DimensionAssessment,
    RiskAssessment,
    RiskVerdict,
    RuleExpression,
    StrategySpec,
)
from ..persistence import now
from . import data as data_mod
from . import indicators as ind
from . import universe as universe_mod

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def spec_hash(spec: StrategySpec) -> str:
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _evaluate_rule(rule: RuleExpression, df: pd.DataFrame) -> pd.Series:
    lhs = ind.compute(rule.indicator, df, dict(rule.params))
    if rule.rhs_indicator is not None:
        rhs = ind.compute(rule.rhs_indicator, df, dict(rule.rhs_params))
    else:
        rhs = pd.Series(rule.rhs_value, index=df.index)
    op = rule.operator
    if op == "gt":
        return lhs > rhs
    if op == "lt":
        return lhs < rhs
    if op == "ge":
        return lhs >= rhs
    if op == "le":
        return lhs <= rhs
    if op == "eq":
        return lhs == rhs
    if op == "cross_above":
        return (lhs.shift(1) <= rhs.shift(1)) & (lhs > rhs)
    if op == "cross_below":
        return (lhs.shift(1) >= rhs.shift(1)) & (lhs < rhs)
    raise ValueError(f"unknown operator {op!r}")


def _and_all(series_list: Iterable[pd.Series]) -> pd.Series:
    series = list(series_list)
    if not series:
        raise ValueError("expected at least one rule series")
    out = series[0].astype(bool)
    for s in series[1:]:
        out = out & s.astype(bool)
    return out.fillna(False)


def _build_signals(spec: StrategySpec, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    entries = _and_all(_evaluate_rule(r, df) for r in spec.entry_rules)
    exits = _and_all(_evaluate_rule(r, df) for r in spec.exit_rules)
    return entries, exits


# ---------------------------------------------------------------------------
# Public API — invoked by the phd-mcp server
# ---------------------------------------------------------------------------


def run_backtest(
    spec: StrategySpec,
    *,
    out_dir: Path,
    data_source: str = "yfinance",
) -> BacktestResult:
    """Execute the spec across the resolved universe; return a BacktestResult."""
    import vectorbt as vbt  # local import — heavy

    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = universe_mod.resolve(spec.universe.base, spec.universe.custom_tickers)
    panel = data_mod.load_universe(
        tickers,
        spec.backtest_start,
        spec.backtest_end,
        source=data_source,
        interval=spec.timeframe,
    )

    closes: dict[str, pd.Series] = {}
    entry_signals: dict[str, pd.Series] = {}
    exit_signals: dict[str, pd.Series] = {}

    for tkr, df in panel.items():
        if spec.universe.min_price is not None and df["close"].mean() < spec.universe.min_price:
            continue
        if spec.universe.max_price is not None and df["close"].mean() > spec.universe.max_price:
            continue
        if spec.universe.min_avg_dollar_volume is not None:
            adv = (df["close"] * df["volume"]).mean()
            if adv < spec.universe.min_avg_dollar_volume:
                continue
        entries, exits = _build_signals(spec, df)
        closes[tkr] = df["close"]
        entry_signals[tkr] = entries
        exit_signals[tkr] = exits

    if not closes:
        raise RuntimeError("no tickers passed universe filters")

    close_df = pd.concat(closes, axis=1)
    entries_df = pd.concat(entry_signals, axis=1).reindex(close_df.index).fillna(False)
    exits_df = pd.concat(exit_signals, axis=1).reindex(close_df.index).fillna(False)

    sizing = spec.sizing
    if sizing.method == "equal_weight":
        size = 1.0 / max(sizing.max_positions, 1)
    elif sizing.method == "fixed_fraction":
        size = float(sizing.fraction or 0.05)
    else:
        size = 0.05  # vol_target — placeholder; proper sizing comes later

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries_df,
        exits=exits_df,
        size=size,
        size_type="percent",
        init_cash=100_000,
        fees=0.0005,
        slippage=0.0005,
        freq=spec.timeframe,
        sl_stop=spec.stop_loss_pct,
        tp_stop=spec.take_profit_pct,
    )

    equity = pf.value().sum(axis=1) if pf.value().ndim > 1 else pf.value()
    total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1)
    n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)
    sharpe = _sharpe(equity)
    sortino = _sortino(equity)
    max_dd = _max_drawdown(equity)
    trades = pf.trades.records_readable
    num_trades = int(len(trades))
    win_rate = float((trades["Return"] > 0).mean()) if num_trades > 0 else 0.0
    avg_trade_return = float(trades["Return"].mean()) if num_trades > 0 else 0.0
    pf_factor = _profit_factor(trades)

    # Persist artifacts.
    trades_csv = out_dir / "trades.csv"
    trades.to_csv(trades_csv, index=False)

    equity_csv = out_dir / "equity_curve.csv"
    equity.rename("equity").to_csv(equity_csv)

    equity_png = out_dir / "equity_curve.png"
    _save_equity_png(equity, equity_png)

    monthly_csv = out_dir / "monthly_returns.csv"
    monthly = equity.resample("ME").last().pct_change().dropna()
    monthly.rename("monthly_return").to_csv(monthly_csv)

    return BacktestResult(
        run_id=spec.run_id,
        strategy_spec_hash=spec_hash(spec),
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=pf_factor,
        num_trades=num_trades,
        avg_trade_return=avg_trade_return,
        equity_curve_path=str(equity_png),
        trades_csv_path=str(trades_csv),
        monthly_returns_csv_path=str(monthly_csv),
        backtest_start=spec.backtest_start,
        backtest_end=spec.backtest_end,
        universe_size=len(closes),
        generated_at=now(),
    )


# ---------------------------------------------------------------------------
# Risk / robustness
# ---------------------------------------------------------------------------


def walk_forward(
    spec: StrategySpec,
    *,
    out_dir: Path,
    in_sample_days: int = 252,
    out_sample_days: int = 63,
) -> DimensionAssessment:
    """Simple expanding-window WFO: split the spec's backtest range into folds,
    compare in-sample vs out-of-sample Sharpe per fold.

    No re-optimization of params here (we treat the spec params as fixed);
    the goal is to detect performance decay across rolling windows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    span_days = (spec.backtest_end - spec.backtest_start).days
    total_window = in_sample_days + out_sample_days
    if span_days < total_window * 1.5:
        return DimensionAssessment(
            dimension="walk_forward",
            verdict=RiskVerdict.CAUTION,
            summary=f"insufficient history ({span_days}d) for {total_window}d folds",
            metrics={"span_days": float(span_days)},
            artifact_paths=[],
        )

    rows: list[dict] = []
    cur = spec.backtest_start
    fold = 0
    from datetime import timedelta

    while True:
        is_end = cur + timedelta(days=in_sample_days)
        oos_end = is_end + timedelta(days=out_sample_days)
        if oos_end > spec.backtest_end:
            break

        is_spec = spec.model_copy(update={"backtest_start": cur, "backtest_end": is_end})
        oos_spec = spec.model_copy(update={"backtest_start": is_end, "backtest_end": oos_end})
        try:
            is_res = run_backtest(is_spec, out_dir=out_dir / f"fold_{fold:02d}_is")
            oos_res = run_backtest(oos_spec, out_dir=out_dir / f"fold_{fold:02d}_oos")
        except Exception as exc:  # noqa: BLE001
            log.warning("fold %d skipped: %s", fold, exc)
            cur = is_end
            fold += 1
            continue

        rows.append(
            {
                "fold": fold,
                "is_start": is_spec.backtest_start.isoformat(),
                "is_end": is_spec.backtest_end.isoformat(),
                "oos_end": oos_spec.backtest_end.isoformat(),
                "is_sharpe": is_res.sharpe,
                "oos_sharpe": oos_res.sharpe,
                "is_return": is_res.total_return,
                "oos_return": oos_res.total_return,
            }
        )
        cur = is_end
        fold += 1

    if not rows:
        return DimensionAssessment(
            dimension="walk_forward",
            verdict=RiskVerdict.FAIL,
            summary="no walk-forward folds completed",
            metrics={},
            artifact_paths=[],
        )

    df = pd.DataFrame(rows)
    csv = out_dir / "wfo.csv"
    df.to_csv(csv, index=False)

    mean_is = float(df["is_sharpe"].mean())
    mean_oos = float(df["oos_sharpe"].mean())
    ratio = mean_oos / mean_is if abs(mean_is) > 1e-9 else 0.0
    if mean_oos >= 0.5 * mean_is and mean_oos > 0:
        v = RiskVerdict.PASS
    elif mean_oos > 0:
        v = RiskVerdict.CAUTION
    else:
        v = RiskVerdict.FAIL
    return DimensionAssessment(
        dimension="walk_forward",
        verdict=v,
        summary=(
            f"{len(df)} folds; mean IS Sharpe={mean_is:.2f}, OOS={mean_oos:.2f}, "
            f"ratio={ratio:.2f}"
        ),
        metrics={"mean_is_sharpe": mean_is, "mean_oos_sharpe": mean_oos, "oos_to_is_ratio": ratio},
        artifact_paths=[str(csv)],
    )


def param_sensitivity(
    spec: StrategySpec,
    grid: dict[str, list[float | int | str]],
    *,
    out_dir: Path,
) -> DimensionAssessment:
    """Sweep `grid` over `spec.params`, run a backtest per combo, report
    Sharpe / max-DD coefficient of variation across the grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import itertools as it

    keys = list(grid.keys())
    combos = list(it.product(*(grid[k] for k in keys)))
    if not combos:
        return DimensionAssessment(
            dimension="param_sensitivity",
            verdict=RiskVerdict.CAUTION,
            summary="empty grid",
            metrics={},
            artifact_paths=[],
        )

    rows: list[dict] = []
    for i, values in enumerate(combos):
        new_params = dict(spec.params)
        for k, v in zip(keys, values):
            new_params[k] = v
        sub_spec = spec.model_copy(update={"params": new_params})
        try:
            res = run_backtest(sub_spec, out_dir=out_dir / f"combo_{i:03d}")
        except Exception as exc:  # noqa: BLE001
            log.warning("combo %s skipped: %s", values, exc)
            continue
        row = {k: v for k, v in zip(keys, values)}
        row.update(
            {"sharpe": res.sharpe, "max_dd": res.max_drawdown, "total_return": res.total_return}
        )
        rows.append(row)

    if not rows:
        return DimensionAssessment(
            dimension="param_sensitivity",
            verdict=RiskVerdict.FAIL,
            summary="no parameter combinations completed",
            metrics={},
            artifact_paths=[],
        )

    df = pd.DataFrame(rows)
    csv = out_dir / "param_grid.csv"
    df.to_csv(csv, index=False)
    sharpe_mean = float(df["sharpe"].mean())
    sharpe_std = float(df["sharpe"].std() or 0.0)
    cv = sharpe_std / abs(sharpe_mean) if abs(sharpe_mean) > 1e-9 else float("inf")
    if cv < 0.5 and sharpe_mean > 0:
        v = RiskVerdict.PASS
    elif cv < 1.0 and sharpe_mean > 0:
        v = RiskVerdict.CAUTION
    else:
        v = RiskVerdict.FAIL
    return DimensionAssessment(
        dimension="param_sensitivity",
        verdict=v,
        summary=(
            f"{len(df)} combos; Sharpe mean={sharpe_mean:.2f} std={sharpe_std:.2f} "
            f"CV={cv:.2f}"
        ),
        metrics={"sharpe_mean": sharpe_mean, "sharpe_std": sharpe_std, "sharpe_cv": cv},
        artifact_paths=[str(csv)],
    )


def monte_carlo(
    equity_csv_path: str,
    *,
    out_dir: Path,
    n: int = 1000,
    block_size: int = 10,
) -> DimensionAssessment:
    """Block-bootstrap the equity curve's returns, report percentile distribution
    of terminal value and worst-drawdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.read_csv(equity_csv_path, index_col=0, parse_dates=True).squeeze()
    returns = equity.pct_change().dropna().values
    if len(returns) < block_size * 5:
        return DimensionAssessment(
            dimension="monte_carlo",
            verdict=RiskVerdict.CAUTION,
            summary=f"too few return observations ({len(returns)}) for block_size={block_size}",
            metrics={},
            artifact_paths=[],
        )

    rng = np.random.default_rng(42)
    n_blocks = len(returns) // block_size
    sim_terminal: list[float] = []
    sim_maxdd: list[float] = []
    for _ in range(n):
        starts = rng.integers(0, len(returns) - block_size, size=n_blocks)
        sample = np.concatenate([returns[s : s + block_size] for s in starts])
        eq = np.cumprod(1 + sample)
        sim_terminal.append(float(eq[-1]))
        running_max = np.maximum.accumulate(eq)
        dd = (eq / running_max - 1).min()
        sim_maxdd.append(float(dd))

    df = pd.DataFrame({"terminal": sim_terminal, "max_dd": sim_maxdd})
    csv = out_dir / "monte_carlo.csv"
    df.to_csv(csv, index=False)
    p5_terminal = float(np.percentile(sim_terminal, 5))
    p95_maxdd = float(np.percentile(sim_maxdd, 5))  # 5th percentile of (negative) DD = worst tail
    median_terminal = float(np.median(sim_terminal))

    if p5_terminal > 1.0 and p95_maxdd > -0.30:
        v = RiskVerdict.PASS
    elif p5_terminal > 0.85:
        v = RiskVerdict.CAUTION
    else:
        v = RiskVerdict.FAIL
    return DimensionAssessment(
        dimension="monte_carlo",
        verdict=v,
        summary=(
            f"n={n} block={block_size}; median terminal={median_terminal:.3f}, "
            f"p5={p5_terminal:.3f}, worst-tail DD p5={p95_maxdd:.3f}"
        ),
        metrics={
            "median_terminal": median_terminal,
            "p5_terminal": p5_terminal,
            "p5_worst_dd": p95_maxdd,
        },
        artifact_paths=[str(csv)],
    )


def overall_assessment(dims: list[DimensionAssessment]) -> RiskAssessment:
    if not dims:
        raise ValueError("need at least one dimension")
    order = {RiskVerdict.PASS: 0, RiskVerdict.CAUTION: 1, RiskVerdict.FAIL: 2}
    worst = max(dims, key=lambda d: order[d.verdict])
    summary = "; ".join(f"{d.dimension}={d.verdict.value}" for d in dims)
    return RiskAssessment(
        run_id=dims[0].metrics.get("run_id", ""),
        dimensions=dims,
        overall_verdict=worst.verdict,
        overall_summary=summary,
        generated_at=now(),
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _sharpe(equity: pd.Series, periods_per_year: int = 252) -> float:
    returns = equity.pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _sortino(equity: pd.Series, periods_per_year: int = 252) -> float | None:
    returns = equity.pct_change().dropna()
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return None
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = (equity / running_max) - 1
    return float(dd.min())


def _profit_factor(trades: pd.DataFrame) -> float | None:
    if len(trades) == 0 or "Return" not in trades.columns:
        return None
    wins = trades.loc[trades["Return"] > 0, "Return"].sum()
    losses = -trades.loc[trades["Return"] < 0, "Return"].sum()
    if losses == 0:
        return None
    return float(wins / losses)


def _save_equity_png(equity: pd.Series, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
