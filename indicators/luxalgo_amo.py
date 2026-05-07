"""
LuxAlgo Adaptive Momentum Oscillator — exact Python port of luxalgooscilator.pine

Pine Script source: Adaptive Momentum Oscillator [LuxAlgo]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    """Local max pivot — NaN where no pivot. Signal placed at pivot bar (no lookahead in use)."""
    arr = series.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        window_l = arr[i - left:i]
        window_r = arr[i + 1:i + right + 1]
        if (np.isnan(arr[i]) or np.all(np.isnan(window_l)) or np.all(np.isnan(window_r))):
            continue
        if arr[i] > np.nanmax(window_l) and arr[i] > np.nanmax(window_r):
            result[i] = arr[i]
    return pd.Series(result, index=series.index)


def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        window_l = arr[i - left:i]
        window_r = arr[i + 1:i + right + 1]
        if (np.isnan(arr[i]) or np.all(np.isnan(window_l)) or np.all(np.isnan(window_r))):
            continue
        if arr[i] < np.nanmin(window_l) and arr[i] < np.nanmin(window_r):
            result[i] = arr[i]
    return pd.Series(result, index=series.index)


def _linreg_rolling(series: pd.Series, length: int) -> pd.Series:
    """
    Rolling linear regression — value at end of window (offset=0).
    Equivalent to ta.linreg(src, length, 0) in Pine Script.
    """
    def _lr(y: np.ndarray) -> float:
        if np.isnan(y).any() or len(y) < 2:
            return np.nan
        x = np.arange(len(y), dtype=float)
        m, b = np.polyfit(x, y, 1)
        return m * (len(y) - 1) + b

    return series.rolling(length).apply(_lr, raw=True)


# ─── AMO Calculation (port of amo() function in Pine) ────────────────────────

def _amo_series(series: pd.Series, length: int) -> pd.Series:
    """
    Adaptive Momentum — for each bar, find the index in [1..length] that
    maximizes |delta|, then return that delta (with sign).

    Pine Script:
        amo(float data, int length) =>
            float max = 0.
            float amo = 0.
            for index = 1 to length
                delta = data - data[index]
                absoluteMomentum = math.abs(delta)
                max := math.max(max, absoluteMomentum)
                amo := max == absoluteMomentum ? delta : amo
            amo
    """
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(length, n):
        max_abs = 0.0
        amo_val = 0.0
        for idx in range(1, length + 1):
            delta = arr[i] - arr[i - idx]
            abs_mom = abs(delta)
            if abs_mom >= max_abs:
                max_abs = abs_mom
                amo_val = delta
        result[i] = amo_val

    return pd.Series(result, index=series.index)


# ─── AMA Calculation (port of ama() function in Pine) ────────────────────────

def _ama_series(series: pd.Series, length: int) -> pd.Series:
    """
    Adaptive Moving Average using Efficiency Ratio.

    Pine Script:
        ama(float data, int length) =>
            var float ama = 0.
            efficiencyRatio = math.abs(data) / math.sum(math.abs(ta.change(data)), length)
            ama += nz(efficiencyRatio * (data - ama))

    Note: This is an iterative accumulator — cannot be fully vectorized.
    """
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)

    # Precompute |change| for efficiency ratio
    abs_change = np.abs(np.diff(arr, prepend=arr[0]))

    ama_val = 0.0
    started = False
    for i in range(length, n):
        if np.isnan(arr[i]):
            continue
        # Efficiency ratio = |data| / sum(|change|, length)
        window_changes = abs_change[max(0, i - length + 1):i + 1]
        sum_abs_change = np.nansum(window_changes)
        if sum_abs_change < 1e-10:
            er = 0.0
        else:
            er = abs(arr[i]) / sum_abs_change
        if not started:
            ama_val = arr[i]  # initialize to first valid value
            started = True
        ama_val += er * (arr[i] - ama_val)
        result[i] = ama_val

    return pd.Series(result, index=series.index)


# ─── Divergence Detection ─────────────────────────────────────────────────────

def _detect_divergences(
    price: pd.Series,
    osc: pd.Series,
    div_length: int = 4,
) -> tuple[pd.Series, pd.Series]:
    """
    Detect bullish and bearish divergences between price and oscillator.

    Bullish div: price makes lower low, oscillator makes higher low
    Bearish div: price makes higher high, oscillator makes lower high

    Returns:
        (bull_div, bear_div) — boolean Series
    """
    left = right = div_length

    price_ph = _pivot_high(price, left, right)
    price_pl = _pivot_low(price, left, right)
    osc_ph   = _pivot_high(osc, left, right)
    osc_pl   = _pivot_low(osc, left, right)

    n = len(price)
    bull_div = pd.Series(False, index=price.index)
    bear_div = pd.Series(False, index=price.index)

    price_arr = price.to_numpy(dtype=float)
    osc_arr   = osc.to_numpy(dtype=float)
    price_pl_arr = price_pl.to_numpy(dtype=float)
    price_ph_arr = price_ph.to_numpy(dtype=float)
    osc_pl_arr   = osc_pl.to_numpy(dtype=float)
    osc_ph_arr   = osc_ph.to_numpy(dtype=float)

    for i in range(left * 2 + right, n):
        # Find previous pivot lows (bullish divergence)
        prev_pl_indices = [j for j in range(max(0, i - left * 5), i) if not np.isnan(price_pl_arr[j])]
        if len(prev_pl_indices) >= 1:
            j = prev_pl_indices[-1]
            if (not np.isnan(price_pl_arr[i]) and not np.isnan(osc_pl_arr[i]) and
                    not np.isnan(price_pl_arr[j]) and not np.isnan(osc_pl_arr[j])):
                if price_pl_arr[i] < price_pl_arr[j] and osc_pl_arr[i] > osc_pl_arr[j]:
                    bull_div.iloc[i] = True

        # Find previous pivot highs (bearish divergence)
        prev_ph_indices = [j for j in range(max(0, i - left * 5), i) if not np.isnan(price_ph_arr[j])]
        if len(prev_ph_indices) >= 1:
            j = prev_ph_indices[-1]
            if (not np.isnan(price_ph_arr[i]) and not np.isnan(osc_ph_arr[i]) and
                    not np.isnan(price_ph_arr[j]) and not np.isnan(osc_ph_arr[j])):
                if price_ph_arr[i] > price_ph_arr[j] and osc_ph_arr[i] < osc_ph_arr[j]:
                    bear_div.iloc[i] = True

    return bull_div, bear_div


# ─── Main Function ────────────────────────────────────────────────────────────

def adaptive_momentum(
    df: pd.DataFrame,
    length: int = 14,
    smoothing: int = 9,
    divergence_length: int = 4,
    detect_divergences: bool = True,
) -> pd.DataFrame:
    """
    LuxAlgo Adaptive Momentum Oscillator — port of luxalgooscilator.pine

    Args:
        df:                  OHLCV DataFrame (UTC index)
        length:              AMO/AMA window length (Pine: lengthInput=14)
        smoothing:           linreg smoothing for AMO (Pine: smoothingInput=9)
        divergence_length:   Pivot lookback for divergences (Pine: divergencesLengthInput=4)
        detect_divergences:  Whether to compute divergences (expensive for large datasets)

    Returns:
        pd.DataFrame with columns:
            amo         float  Raw Adaptive Momentum (before linreg)
            amo_smooth  float  Smoothed AMO via linreg (main oscillator line)
            ama         float  Adaptive Moving Average of amo_smooth
            bull_div    bool   Bullish divergence detected
            bear_div    bool   Bearish divergence detected
            direction   int    +1 (bull) / -1 (bear) / 0 (neutral)
    """
    src = df["close"]

    # Step 1: AMO
    amo = _amo_series(src, length)

    # Step 2: Smooth AMO with linreg
    amo_smooth = _linreg_rolling(amo, smoothing)

    # Step 3: AMA of smoothed AMO
    ama = _ama_series(amo_smooth, length)

    # Step 4: Direction
    direction = pd.Series(0, index=df.index)
    direction[amo_smooth > 0] =  1
    direction[amo_smooth < 0] = -1

    # Step 5: Divergences
    if detect_divergences:
        bull_div, bear_div = _detect_divergences(src, amo_smooth, divergence_length)
    else:
        bull_div = pd.Series(False, index=df.index)
        bear_div = pd.Series(False, index=df.index)

    return pd.DataFrame(
        {
            "amo":        amo,
            "amo_smooth": amo_smooth,
            "ama":        ama,
            "bull_div":   bull_div,
            "bear_div":   bear_div,
            "direction":  direction,
        },
        index=df.index,
    )


def adaptive_momentum_all_timeframes(
    dfs: dict[str, pd.DataFrame],
    length: int = 14,
    smoothing: int = 9,
    divergence_length: int = 4,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        results[tf] = adaptive_momentum(
            df, length=length, smoothing=smoothing, divergence_length=divergence_length
        )
    return results


# ─── Summary for Prompt ───────────────────────────────────────────────────────

def luxalgo_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Extract latest bar values per timeframe for prompt generation."""
    summary: dict[str, dict] = {}
    for tf, df in results.items():
        if df is None or df.empty:
            continue
        last = df.iloc[-1]

        amo_val = float(last["amo_smooth"])
        ama_val = float(last["ama"])
        direction = int(last["direction"])

        dir_str = "↑ Bull" if direction > 0 else ("↓ Bear" if direction < 0 else "→ Neutral")

        # Check last N bars for divergences
        tail = df.tail(5)
        div_str = []
        if tail["bull_div"].any():
            div_str.append("Bull Div")
        if tail["bear_div"].any():
            div_str.append("Bear Div")

        summary[tf] = {
            "amo":       round(amo_val, 4),
            "ama":       round(ama_val, 4),
            "direction": dir_str,
            "divergence": ", ".join(div_str) if div_str else "—",
        }
    return summary
