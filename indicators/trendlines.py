"""
Trendlines with Breaks — Python port of trendlinesbreakout.pine (LuxAlgo).

Dynamic diagonal trendlines anchored to pivots, with breakout detection.
Slope methods: ATR, Stdev, Linreg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()  # Pine ta.atr default = RMA, but for slope use ~SMA


def _pivot_high(series: pd.Series, length: int) -> pd.Series:
    """Pine `ta.pivothigh(length, length)` — center-detected pivot."""
    arr = series.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(length, len(arr) - length):
        wl = arr[i - length:i]
        wr = arr[i + 1:i + length + 1]
        if np.isnan(arr[i]) or np.all(np.isnan(wl)) or np.all(np.isnan(wr)):
            continue
        if arr[i] > np.nanmax(wl) and arr[i] > np.nanmax(wr):
            out[i] = arr[i]
    return pd.Series(out, index=series.index)


def _pivot_low(series: pd.Series, length: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(length, len(arr) - length):
        wl = arr[i - length:i]
        wr = arr[i + 1:i + length + 1]
        if np.isnan(arr[i]) or np.all(np.isnan(wl)) or np.all(np.isnan(wr)):
            continue
        if arr[i] < np.nanmin(wl) and arr[i] < np.nanmin(wr):
            out[i] = arr[i]
    return pd.Series(out, index=series.index)


# ─── Trendlines Engine ────────────────────────────────────────────────────────

def trendlines(
    df: pd.DataFrame,
    length: int = 14,
    mult: float = 1.0,
    calc_method: str = "atr",  # 'atr' | 'stdev' | 'linreg'
) -> pd.DataFrame:
    """
    Returns:
        upper_tl       upper (down-sloping) trendline value at each bar
        lower_tl       lower (up-sloping) trendline value at each bar
        slope_ph       slope of upper trendline
        slope_pl       slope of lower trendline
        break_up       bool — close broke upper trendline upward
        break_dn       bool — close broke lower trendline downward
    """
    n = len(df)
    close = df["close"].to_numpy(dtype=float)

    ph = _pivot_high(df["high"], length).to_numpy(dtype=float)
    pl = _pivot_low(df["low"], length).to_numpy(dtype=float)

    if calc_method == "atr":
        slope_base = _atr(df, length).to_numpy(dtype=float) / length * mult
    elif calc_method == "stdev":
        slope_base = df["close"].rolling(length).std(ddof=0).to_numpy() / length * mult
    else:  # linreg
        # Slope from rolling linear regression
        def _slope(y):
            if np.isnan(y).any():
                return np.nan
            x = np.arange(len(y), dtype=float)
            return abs(np.polyfit(x, y, 1)[0]) * mult
        slope_base = df["close"].rolling(length).apply(_slope, raw=True).to_numpy()

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    slope_ph = np.zeros(n)
    slope_pl = np.zeros(n)

    upos = np.zeros(n, dtype=int)
    dnos = np.zeros(n, dtype=int)

    cur_upper = np.nan
    cur_lower = np.nan
    cur_sph = 0.0
    cur_spl = 0.0

    for i in range(n):
        s = slope_base[i] if not np.isnan(slope_base[i]) else 0.0
        # On pivot detection, anchor trendline
        if not np.isnan(ph[i]):
            cur_upper = ph[i]
            cur_sph = s
        else:
            if not np.isnan(cur_upper):
                cur_upper = cur_upper - cur_sph

        if not np.isnan(pl[i]):
            cur_lower = pl[i]
            cur_spl = s
        else:
            if not np.isnan(cur_lower):
                cur_lower = cur_lower + cur_spl

        upper[i] = cur_upper
        lower[i] = cur_lower
        slope_ph[i] = cur_sph
        slope_pl[i] = cur_spl

        # Breakout state
        prev_upos = upos[i - 1] if i > 0 else 0
        prev_dnos = dnos[i - 1] if i > 0 else 0

        if not np.isnan(ph[i]):
            upos[i] = 0
        elif not np.isnan(cur_upper) and close[i] > (cur_upper - cur_sph * length):
            upos[i] = 1
        else:
            upos[i] = prev_upos

        if not np.isnan(pl[i]):
            dnos[i] = 0
        elif not np.isnan(cur_lower) and close[i] < (cur_lower + cur_spl * length):
            dnos[i] = 1
        else:
            dnos[i] = prev_dnos

    break_up = np.zeros(n, dtype=bool)
    break_dn = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if upos[i] > upos[i - 1]:
            break_up[i] = True
        if dnos[i] > dnos[i - 1]:
            break_dn[i] = True

    return pd.DataFrame({
        "upper_tl": upper,
        "lower_tl": lower,
        "slope_ph": slope_ph,
        "slope_pl": slope_pl,
        "break_up": break_up,
        "break_dn": break_dn,
    }, index=df.index)


# ─── MTF wrappers ─────────────────────────────────────────────────────────────

def trendlines_all_timeframes(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        out[tf] = trendlines(df)
    return out


def trendlines_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for tf, df in results.items():
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        tail = df.tail(5)

        sigs = []
        if tail["break_up"].any(): sigs.append("Break Up")
        if tail["break_dn"].any(): sigs.append("Break Dn")

        def _f(v):
            return None if pd.isna(v) else float(v)

        # Slope direction: positive slope_pl = uptrend support, negative = none
        upper_v = _f(last["upper_tl"])
        lower_v = _f(last["lower_tl"])

        summary[tf] = {
            "upper_tl":  upper_v,
            "lower_tl":  lower_v,
            "slope_up":  round(float(last["slope_pl"]), 6) if not pd.isna(last["slope_pl"]) else 0.0,
            "slope_dn":  round(float(last["slope_ph"]), 6) if not pd.isna(last["slope_ph"]) else 0.0,
            "signals":   ", ".join(sigs) if sigs else "—",
        }
    return summary
