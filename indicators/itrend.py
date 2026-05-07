"""
Ehlers Instantaneous Trendline (iTrend) — standalone adaptive filter.

Port of the iTrend recurrence from Ehlers' "Cybernetic Analysis for Stocks and Futures".
Not a signal generator — pure adaptive trend filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FLAT_THRESHOLD = 0.01  # % slope magnitude below which trend is "Flat"


def itrend(
    df: pd.DataFrame,
    alpha: float = 0.09,
) -> pd.DataFrame:
    """
    Ehlers iTrend adaptive moving average.

    Uses recurrence:
        it[i] = (a - a²/4)*close[i] + 0.5*a²*close[i-1]
                - (a - 0.75*a²)*close[i-2] + 2*(1-a)*it[i-1] - (1-a)²*it[i-2]

    Outputs:
        itrend       float  — adaptive trend value (price units)
        itrend_bull  bool   — it[i] > it[i-2]  (rising)
        itrend_bear  bool   — it[i] < it[i-2]  (falling)
        itrend_slope float  — (it[i] - it[i-2]) / it[i-2] * 100  (% per 2 bars)
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    a = alpha
    it = np.zeros(n)
    for i in range(n):
        if np.isnan(close[i]):
            it[i] = it[i - 1] if i > 0 else 0.0
            continue
        if i < 2:
            it[i] = close[i]
        else:
            it[i] = (
                (a - a * a / 4.0) * close[i]
                + 0.5 * a * a * close[i - 1]
                - (a - 0.75 * a * a) * close[i - 2]
                + 2.0 * (1.0 - a) * it[i - 1]
                - (1.0 - a) ** 2 * it[i - 2]
            )

    slope = np.full(n, np.nan)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if it[i - 2] != 0.0 and not np.isnan(it[i - 2]):
            slope[i] = (it[i] - it[i - 2]) / abs(it[i - 2]) * 100.0
        bull[i] = bool(it[i] > it[i - 2])
        bear[i] = bool(it[i] < it[i - 2])

    return pd.DataFrame({
        "itrend":       it,
        "itrend_bull":  bull,
        "itrend_bear":  bear,
        "itrend_slope": slope,
        "close":        close,
    }, index=df.index)


def itrend_all_timeframes(
    dfs: dict[str, pd.DataFrame],
    alpha: float = 0.09,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        out[tf] = itrend(df, alpha=alpha)
    return out


def itrend_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for tf, df in results.items():
        if df is None or df.empty:
            continue
        valid = df.dropna(subset=["itrend"])
        if valid.empty:
            continue
        last = valid.iloc[-1]
        val = float(last["itrend"])
        slope = float(last["itrend_slope"]) if not np.isnan(last["itrend_slope"]) else 0.0

        if abs(slope) < FLAT_THRESHOLD:
            trend = "→ Flat"
        elif bool(last["itrend_bull"]):
            trend = "↑ Bull"
        else:
            trend = "↓ Bear"

        # Check if close crossed itrend in last 3 bars
        tail = df.tail(3)
        sig = "—"
        if "close" not in df.columns:
            pass
        else:
            closes = tail.get("close", pd.Series(dtype=float))
            its = tail["itrend"]
            if len(closes) >= 2:
                prev_above = (closes.iloc[-2] > its.iloc[-2])
                curr_above = (closes.iloc[-1] > its.iloc[-1])
                if not prev_above and curr_above:
                    sig = "CrossUp (close > iTrend)"
                elif prev_above and not curr_above:
                    sig = "CrossDn (close < iTrend)"

        summary[tf] = {
            "value": round(val, 4),
            "slope": round(slope, 4),
            "trend": trend,
            "signal": sig,
        }
    return summary
