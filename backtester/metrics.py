"""
Trade performance metrics: winrate, RR, drawdown, Sharpe, Calmar.
"""

from __future__ import annotations

import math

import numpy as np

from pineforge_ai.backtester.simulator import TradeResult


def compute_metrics(trades: list[TradeResult]) -> dict:
    """
    Compute aggregate metrics from a list of TradeResult.
    """
    if not trades:
        return {
            "n_trades": 0, "n_wins": 0, "n_losses": 0,
            "winrate": 0.0, "avg_rr": 0.0, "expectancy_r": 0.0,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe": 0.0, "calmar": 0.0,
            "result_breakdown": {},
        }

    finished = [t for t in trades if t.result not in ("invalid", "never_triggered")]
    n = len(finished)
    if n == 0:
        return {
            "n_trades":         len(trades),
            "n_wins":           0,
            "n_losses":         0,
            "winrate":          0.0,
            "avg_rr":           0.0,
            "expectancy_r":     0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe":           0.0,
            "calmar":           0.0,
            "result_breakdown": _breakdown(trades),
        }

    wins = [t for t in finished if t.pnl_pct > 0]
    losses = [t for t in finished if t.pnl_pct <= 0]
    winrate = len(wins) / n * 100.0

    rrs = [t.rr_achieved for t in finished]
    avg_rr = float(np.mean(rrs)) if rrs else 0.0
    expectancy_r = float(np.mean(rrs))  # avg R per trade

    # Equity curve compounded (1% risk per trade hypothesis)
    risk_per_trade = 0.01
    equity = [1.0]
    for t in finished:
        ret = risk_per_trade * t.rr_achieved
        equity.append(equity[-1] * (1.0 + ret))
    total_return_pct = (equity[-1] - 1.0) * 100.0

    # Max drawdown
    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    max_dd = float(dd.min()) * 100.0  # negative

    # Sharpe (per-trade, annualized assuming 1 trade/day for simplicity)
    rets = [risk_per_trade * t.rr_achieved for t in finished]
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = float(np.mean(rets) / np.std(rets) * math.sqrt(252))
    else:
        sharpe = 0.0

    # Calmar = total_return / |max_dd|
    calmar = float(total_return_pct / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "n_trades":         len(trades),
        "n_finished":       n,
        "n_wins":           len(wins),
        "n_losses":         len(losses),
        "winrate":          round(winrate, 2),
        "avg_rr":           round(avg_rr, 3),
        "expectancy_r":     round(expectancy_r, 3),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe":           round(sharpe, 3),
        "calmar":           round(calmar, 3),
        "result_breakdown": _breakdown(trades),
    }


def _breakdown(trades: list[TradeResult]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in trades:
        out[t.result] = out.get(t.result, 0) + 1
    return out
