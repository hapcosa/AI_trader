"""
EMAs — Exponential Moving Averages (20/50/200).

Simple trend-following overlay: price vs EMAs and EMA stacking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classics import _ema


DEFAULT_PERIODS = (20, 50, 200)
SLOPE_LOOKBACK = 3
FLAT_PCT_THRESHOLD = 0.01


def emas(
    df: pd.DataFrame,
    periods: tuple[int, ...] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Add EMA columns for each period.

    Columns: ``close`` plus ``ema{period}`` for every requested period.
    Uses Pine's ``ta.ema`` formula (``ewm(span=length, adjust=False)``).
    """
    close = df["close"].astype(float)
    out = pd.DataFrame({"close": close}, index=df.index)
    for length in periods:
        out[f"ema{length}"] = _ema(close, length)
    return out


def emas_all_timeframes(
    dfs: dict[str, pd.DataFrame],
    periods: tuple[int, ...] = DEFAULT_PERIODS,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        results[tf] = emas(df, periods=periods)
    return results


def _safe(val, decimals: int = 2):
    try:
        f = float(val)
        return round(f, decimals) if not np.isnan(f) else "—"
    except Exception:
        return "—"


def _price_vs(price: float, ema: float) -> str:
    if np.isnan(price) or np.isnan(ema):
        return "—"
    return "above" if price >= ema else "below"


def _slope_label(current: float, previous: float) -> str:
    if np.isnan(current) or np.isnan(previous) or previous == 0.0:
        return "—"
    pct = (current - previous) / abs(previous) * 100.0
    if abs(pct) < FLAT_PCT_THRESHOLD:
        return "flat"
    return "up" if pct > 0 else "down"


def emas_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Latest-bar state per timeframe for prompt generation."""
    summary: dict[str, dict] = {}
    for tf, df in results.items():
        if df is None or df.empty:
            continue
        valid = df.dropna(subset=["close"])
        if valid.empty:
            continue
        last = valid.iloc[-1]
        prev = valid.iloc[-(SLOPE_LOOKBACK + 1)] if len(valid) > SLOPE_LOOKBACK else last

        close = float(last["close"])
        ema20 = float(last.get("ema20", np.nan))
        ema50 = float(last.get("ema50", np.nan))
        ema200 = float(last.get("ema200", np.nan))

        if not np.isnan(ema20) and not np.isnan(ema50) and not np.isnan(ema200):
            if ema20 > ema50 > ema200:
                stack = "bull"
            elif ema20 < ema50 < ema200:
                stack = "bear"
            else:
                stack = "mixed"
        else:
            stack = "—"

        summary[tf] = {
            "close": _safe(close),
            "ema20": _safe(ema20),
            "ema50": _safe(ema50),
            "ema200": _safe(ema200),
            "price_vs_ema20": _price_vs(close, ema20),
            "price_vs_ema50": _price_vs(close, ema50),
            "price_vs_ema200": _price_vs(close, ema200),
            "stack": stack,
            "slope20": _slope_label(ema20, float(prev.get("ema20", np.nan))),
        }
    return summary
