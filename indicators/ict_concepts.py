"""
ICT Concepts — Liquidity zones, IFVG, Volume Imbalance, Displacement, MSS.

Source: ictconcepts.pine (LuxAlgo ICT Concepts).
Provides ICT perspective complementing classical SMC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        wl = arr[i - left:i]
        wr = arr[i + 1:i + right + 1]
        if np.isnan(arr[i]) or np.all(np.isnan(wl)) or np.all(np.isnan(wr)):
            continue
        if arr[i] > np.nanmax(wl) and arr[i] > np.nanmax(wr):
            out[i] = arr[i]
    return pd.Series(out, index=series.index)


def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        wl = arr[i - left:i]
        wr = arr[i + 1:i + right + 1]
        if np.isnan(arr[i]) or np.all(np.isnan(wl)) or np.all(np.isnan(wr)):
            continue
        if arr[i] < np.nanmin(wl) and arr[i] < np.nanmin(wr):
            out[i] = arr[i]
    return pd.Series(out, index=series.index)


# ─── ICT Concepts Engine ──────────────────────────────────────────────────────

def ict_concepts(
    df: pd.DataFrame,
    pivot_left: int = 5,
    pivot_right: int = 1,
    displacement_atr_mult: float = 1.5,
    fvg_min_atr_pct: float = 0.10,
    max_active: int = 5,
) -> pd.DataFrame:
    """
    Return ICT structure events per bar.

    Outputs (per bar):
        bsl_level         most recent unswept buy-side liquidity (pivot high)
        ssl_level         most recent unswept sell-side liquidity (pivot low)
        bsl_swept         bool — last bar swept BSL (high > last BSL)
        ssl_swept         bool — last bar swept SSL (low < last SSL)
        bos_bull          bool — close > prior swing high
        bos_bear          bool — close < prior swing low
        mss_bull          bool — bos_bull while prior trend was bear
        mss_bear          bool — bos_bear while prior trend was bull
        fvg_bull_top/btm  active bullish FVG (top/bottom levels)
        fvg_bear_top/btm  active bearish FVG (top/bottom levels)
        ifvg_bull         bool — bullish FVG that was subsequently filled (inverted)
        ifvg_bear         bool — bearish FVG inverted
        vol_imbalance_up  bool — bullish volume imbalance (high[i-2] < low[i])
        vol_imbalance_dn  bool — bearish volume imbalance (low[i-2] > high[i])
        displacement      bool — bar range > displacement_atr_mult * ATR
    """
    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)

    atr = _atr(df).to_numpy(dtype=float)

    ph = _pivot_high(df["high"], pivot_left, pivot_right).to_numpy(dtype=float)
    pl = _pivot_low(df["low"], pivot_left, pivot_right).to_numpy(dtype=float)

    bsl_level = np.full(n, np.nan)
    ssl_level = np.full(n, np.nan)
    bsl_swept = np.zeros(n, dtype=bool)
    ssl_swept = np.zeros(n, dtype=bool)
    bos_bull = np.zeros(n, dtype=bool)
    bos_bear = np.zeros(n, dtype=bool)
    mss_bull = np.zeros(n, dtype=bool)
    mss_bear = np.zeros(n, dtype=bool)

    fvg_bull_top = np.full(n, np.nan)
    fvg_bull_btm = np.full(n, np.nan)
    fvg_bear_top = np.full(n, np.nan)
    fvg_bear_btm = np.full(n, np.nan)
    ifvg_bull = np.zeros(n, dtype=bool)
    ifvg_bear = np.zeros(n, dtype=bool)

    vol_imb_up = np.zeros(n, dtype=bool)
    vol_imb_dn = np.zeros(n, dtype=bool)
    displacement = np.zeros(n, dtype=bool)

    # State
    active_bsl: list[tuple[int, float]] = []   # (bar, level)
    active_ssl: list[tuple[int, float]] = []
    active_bull_fvg: list[tuple[int, float, float]] = []  # (bar, top, btm)
    active_bear_fvg: list[tuple[int, float, float]] = []
    last_trend = 0  # +1 bull, -1 bear

    for i in range(n):
        # Pivot detection lags by `pivot_right` bars
        # When ph[i - pivot_right] is non-nan, we register the pivot at bar i-right
        pi = i - pivot_right
        if pi >= 0:
            if not np.isnan(ph[pi]):
                active_bsl.append((pi, ph[pi]))
                if len(active_bsl) > max_active:
                    active_bsl.pop(0)
            if not np.isnan(pl[pi]):
                active_ssl.append((pi, pl[pi]))
                if len(active_ssl) > max_active:
                    active_ssl.pop(0)

        # Liquidity sweeps
        if active_bsl:
            top_lvl = active_bsl[-1][1]
            bsl_level[i] = top_lvl
            if high[i] > top_lvl:
                bsl_swept[i] = True
                # BOS bull = close above sweep
                if close[i] > top_lvl:
                    bos_bull[i] = True
                    if last_trend < 0:
                        mss_bull[i] = True
                    last_trend = 1
                active_bsl.pop()  # remove swept

        if active_ssl:
            btm_lvl = active_ssl[-1][1]
            ssl_level[i] = btm_lvl
            if low[i] < btm_lvl:
                ssl_swept[i] = True
                if close[i] < btm_lvl:
                    bos_bear[i] = True
                    if last_trend > 0:
                        mss_bear[i] = True
                    last_trend = -1
                active_ssl.pop()

        # FVG detection (3-bar): bullish if low[i] > high[i-2], bearish if high[i] < low[i-2]
        if i >= 2:
            if low[i] > high[i - 2]:
                top = low[i]
                btm = high[i - 2]
                if (top - btm) > atr[i] * fvg_min_atr_pct:
                    active_bull_fvg.append((i, top, btm))
                    if len(active_bull_fvg) > max_active:
                        active_bull_fvg.pop(0)
            if high[i] < low[i - 2]:
                top = low[i - 2]
                btm = high[i]
                if (top - btm) > atr[i] * fvg_min_atr_pct:
                    active_bear_fvg.append((i, top, btm))
                    if len(active_bear_fvg) > max_active:
                        active_bear_fvg.pop(0)

        # Mitigate / invert FVGs
        keep_bull = []
        for bar, top, btm in active_bull_fvg:
            if close[i] < btm:
                ifvg_bull[i] = True  # inverted (filled below)
            else:
                keep_bull.append((bar, top, btm))
        active_bull_fvg = keep_bull

        keep_bear = []
        for bar, top, btm in active_bear_fvg:
            if close[i] > top:
                ifvg_bear[i] = True
            else:
                keep_bear.append((bar, top, btm))
        active_bear_fvg = keep_bear

        # Active FVG (most recent)
        if active_bull_fvg:
            _, t, b = active_bull_fvg[-1]
            fvg_bull_top[i] = t
            fvg_bull_btm[i] = b
        if active_bear_fvg:
            _, t, b = active_bear_fvg[-1]
            fvg_bear_top[i] = t
            fvg_bear_btm[i] = b

        # Volume Imbalance: gap between current candle body and 2-bar-ago body
        if i >= 2:
            # Bullish VI: low[i] > high[i-2] (no body overlap)
            if low[i] > high[i - 2]:
                vol_imb_up[i] = True
            if high[i] < low[i - 2]:
                vol_imb_dn[i] = True

        # Displacement: bar range > 1.5 × ATR
        if not np.isnan(atr[i]) and atr[i] > 0:
            if (high[i] - low[i]) > displacement_atr_mult * atr[i]:
                displacement[i] = True

    return pd.DataFrame({
        "bsl_level":      bsl_level,
        "ssl_level":      ssl_level,
        "bsl_swept":      bsl_swept,
        "ssl_swept":      ssl_swept,
        "bos_bull":       bos_bull,
        "bos_bear":       bos_bear,
        "mss_bull":       mss_bull,
        "mss_bear":       mss_bear,
        "fvg_bull_top":   fvg_bull_top,
        "fvg_bull_btm":   fvg_bull_btm,
        "fvg_bear_top":   fvg_bear_top,
        "fvg_bear_btm":   fvg_bear_btm,
        "ifvg_bull":      ifvg_bull,
        "ifvg_bear":      ifvg_bear,
        "vol_imb_up":     vol_imb_up,
        "vol_imb_dn":     vol_imb_dn,
        "displacement":   displacement,
    }, index=df.index)


# ─── MTF wrappers ─────────────────────────────────────────────────────────────

def ict_all_timeframes(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        out[tf] = ict_concepts(df)
    return out


def ict_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for tf, df in results.items():
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        tail = df.tail(5)

        # Last MSS / BOS event
        last_event = "—"
        for ev, label in [
            ("mss_bull", "MSS Bull"),
            ("mss_bear", "MSS Bear"),
            ("bos_bull", "BOS Bull"),
            ("bos_bear", "BOS Bear"),
        ]:
            if tail[ev].any():
                last_event = label
                break

        recent_signals = []
        if tail["bsl_swept"].any(): recent_signals.append("BSL Swept")
        if tail["ssl_swept"].any(): recent_signals.append("SSL Swept")
        if tail["ifvg_bull"].any(): recent_signals.append("IFVG Bull")
        if tail["ifvg_bear"].any(): recent_signals.append("IFVG Bear")
        if tail["displacement"].any(): recent_signals.append("Displacement")
        if tail["vol_imb_up"].any(): recent_signals.append("VolImb Up")
        if tail["vol_imb_dn"].any(): recent_signals.append("VolImb Dn")

        def _f(v):
            return None if pd.isna(v) else float(v)

        summary[tf] = {
            "last_event":     last_event,
            "bsl_level":      _f(last["bsl_level"]),
            "ssl_level":      _f(last["ssl_level"]),
            "fvg_bull_top":   _f(last["fvg_bull_top"]),
            "fvg_bull_btm":   _f(last["fvg_bull_btm"]),
            "fvg_bear_top":   _f(last["fvg_bear_top"]),
            "fvg_bear_btm":   _f(last["fvg_bear_btm"]),
            "signals":        ", ".join(recent_signals) if recent_signals else "—",
        }
    return summary
