"""
Walk-forward training engine.

For each iteration:
    1. Slice OHLCV up to iteration's "end" date.
    2. Compute all indicators.
    3. Build prompt (with prior iteration summaries as pretrain context).
    4. Send to AI -> JSON signals.
    5. Simulate signals on lookahead bars beyond "end" date.
    6. Compute metrics + write report.
    7. Add summary to pretrain context for next iteration.

Final iteration uses the present moment (end = now) and emits the LIVE prompt + analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import pandas as pd

from pineforge_ai.backtester.metrics import compute_metrics
from pineforge_ai.backtester.report import summarize_iteration, write_iteration_report
from pineforge_ai.backtester.simulator import simulate_signals


@dataclass
class IterationResult:
    iteration: int
    start_date: str
    end_date: str
    prompt: str
    signals: dict
    metrics: dict
    report_path: str
    summary_line: str


def slice_dfs_up_to(
    dfs: dict[str, pd.DataFrame],
    cutoff_utc: datetime,
) -> dict[str, pd.DataFrame]:
    """Return shallow copies of each TF DF with rows up to (inclusive) cutoff."""
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        out[tf] = df[df.index <= cutoff_utc].copy()
    return out


def find_idx_at_or_before(df: pd.DataFrame, dt: datetime) -> int:
    """Index of the last bar with timestamp <= dt."""
    if df.empty:
        return 0
    mask = df.index <= dt
    if not mask.any():
        return 0
    return int(mask.sum()) - 1


def split_window(start: datetime, end: datetime, n_iters: int) -> list[tuple[datetime, datetime]]:
    """
    Divide [start, end] into n_iters equal sub-windows. Each yields (window_start, window_end).
    The window_end is what the AI "sees as present" in that iteration.
    """
    total_seconds = (end - start).total_seconds()
    step = total_seconds / n_iters
    out = []
    for i in range(n_iters):
        ws = start + timedelta(seconds=step * i)
        we = start + timedelta(seconds=step * (i + 1))
        out.append((ws, we))
    return out


def run_walk_forward(
    symbol: str,
    dfs_full: dict[str, pd.DataFrame],
    timeframes: list[str],
    start: datetime,
    end: datetime,
    iterations: int,
    output_dir: str,
    build_prompt_fn: Callable,
    send_fn: Callable | None,
    indicator_fn: Callable,
    correlations_fn: Callable | None = None,
    volatility_fn: Callable | None = None,
    lookahead_bars: int = 200,
    source: str = "auto",
    exchange: str = "binance",
    dry_run: bool = False,
) -> list[IterationResult]:
    """
    Run iterative walk-forward training.

    Args:
        symbol:           Trading symbol
        dfs_full:         Full-range OHLCV dict (must cover start to end + lookahead)
        timeframes:       Ordered list of TFs to use
        start, end:       UTC datetimes
        iterations:       Number of walk-forward iterations
        output_dir:       Where to write per-iteration reports
        build_prompt_fn:  (symbol, dfs_slice, tfs, summaries..., pretrain_summary, dt_utc) -> str
        send_fn:          (prompt) -> JSON dict.  If None or dry_run, skip API call.
        indicator_fn:     (dfs_slice) -> dict of summaries (wt, lux, smc, tq, cc, ict, tl)
        lookahead_bars:   Bars after iteration cutoff to wait for trade outcomes
    """
    windows = split_window(start, end, iterations)
    pretrain: list[str] = []
    results: list[IterationResult] = []

    for i, (ws, we) in enumerate(windows, start=1):
        print(f"\n── Iteration {i}/{iterations}: {ws.date()} → {we.date()} ──")

        dfs_slice = slice_dfs_up_to(dfs_full, we)
        if not any(len(df) > 0 for df in dfs_slice.values()):
            print(f"  WARN: empty slice at {we}, skipping.")
            continue

        summaries = indicator_fn(dfs_slice)

        corr = correlations_fn() if correlations_fn else None
        vol  = volatility_fn(dfs_slice) if volatility_fn else None

        prompt = build_prompt_fn(
            symbol=symbol,
            dfs=dfs_slice,
            timeframes=timeframes,
            wt_summary=summaries.get("wt"),
            lux_summary=summaries.get("lux"),
            smc_sum=summaries.get("smc"),
            tq_summary=summaries.get("tq"),
            cc_summary=summaries.get("cc"),
            ict_sum=summaries.get("ict"),
            tl_summary=summaries.get("tl"),
            correlations=corr,
            volatility=vol,
            pretrain_summary=pretrain.copy() if pretrain else None,
            source=source,
            exchange=exchange,
            dt_utc=we,
        )

        # AI call
        signals: dict = {"entries": [], "_dry_run": True}
        if not dry_run and send_fn is not None:
            try:
                signals = send_fn(prompt)
            except Exception as e:
                print(f"  AI call failed: {e}")
                signals = {"entries": [], "_error": str(e)}

        # Simulate
        start_idx_per_tf = {
            tf: find_idx_at_or_before(dfs_full[tf], we)
            for tf in timeframes if tf in dfs_full
        }
        trades = simulate_signals(
            signals_json=signals,
            dfs=dfs_full,
            start_idx_per_tf=start_idx_per_tf,
            lookahead_bars=lookahead_bars,
        )
        metrics = compute_metrics(trades)

        # Write report
        report_path = write_iteration_report(
            output_dir=output_dir,
            symbol=symbol,
            iteration=i,
            start_date=ws.strftime("%Y-%m-%d"),
            end_date=we.strftime("%Y-%m-%d"),
            signals_json=signals,
            trades=trades,
            metrics=metrics,
        )

        summary_line = summarize_iteration(
            iteration=i,
            start_date=ws.strftime("%Y-%m-%d"),
            end_date=we.strftime("%Y-%m-%d"),
            metrics=metrics,
            signals_json=signals,
        )
        pretrain.append(summary_line)
        print(f"  {summary_line}")
        print(f"  Report: {report_path}")

        results.append(IterationResult(
            iteration=i,
            start_date=ws.strftime("%Y-%m-%d"),
            end_date=we.strftime("%Y-%m-%d"),
            prompt=prompt,
            signals=signals,
            metrics=metrics,
            report_path=report_path,
            summary_line=summary_line,
        ))

    return results
