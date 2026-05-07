"""
Per-iteration report writer + accumulated context summarizer for prompts.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from pineforge_ai.backtester.simulator import TradeResult, trades_to_dicts


def write_iteration_report(
    output_dir: str,
    symbol: str,
    iteration: int,
    start_date: str,
    end_date: str,
    signals_json: dict,
    trades: list[TradeResult],
    metrics: dict,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "-").replace("^", "").replace("=", "")
    fp = os.path.join(output_dir, f"{safe}_iter_{iteration:02d}.json")
    payload = {
        "symbol":     symbol,
        "iteration":  iteration,
        "start_date": start_date,
        "end_date":   end_date,
        "signals":    signals_json,
        "trades":     trades_to_dicts(trades),
        "metrics":    metrics,
        "generated":  datetime.utcnow().isoformat() + "Z",
    }
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return os.path.abspath(fp)


def summarize_iteration(
    iteration: int,
    start_date: str,
    end_date: str,
    metrics: dict,
    signals_json: dict,
) -> str:
    """Build a single-line summary for the next iteration's pretrain context."""
    n_sigs = len(signals_json.get("entries", []))
    wr = metrics.get("winrate", 0)
    rr = metrics.get("avg_rr", 0)
    dd = metrics.get("max_drawdown_pct", 0)
    breakdown = metrics.get("result_breakdown", {})
    bd_str = ", ".join(f"{k}={v}" for k, v in breakdown.items())
    return (
        f"Iter {iteration} ({start_date} → {end_date}): {n_sigs} señales, "
        f"WR={wr}%, avgRR={rr}, DD={dd}%. Breakdown: {bd_str or '—'}"
    )
