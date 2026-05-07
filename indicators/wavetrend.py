"""
WaveTrend Oscillator — exact Python port of oscilador_v26.pine / oscilador_v31.pine

Pine Script source: CryptoProofit® - Oscillator 2.6
Core algorithm: WaveTrend (EMA-based CCI variant) normalized to 0-100
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── EMA / SMA helpers ────────────────────────────────────────────────────────

def _ema(series: pd.Series, length: int) -> pd.Series:
    """EMA with adjust=False — identical to Pine Script ta.ema()."""
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


# ─── Core WaveTrend Calculation ───────────────────────────────────────────────

def wavetrend(
    df: pd.DataFrame,
    hyperwave_len: int = 6,
    trigger_len: int = 2,
    kernel_factor1: float = 0.8,
    kernel_factor2: float = 0.3,
    norm_len: int = 100,
    vol_lookback: int = 30,
) -> pd.DataFrame:
    """
    WaveTrend Oscillator — port of oscilador_v26.pine

    Args:
        df:             OHLCV DataFrame (UTC index)
        hyperwave_len:  Base length for HyperWave EMA (Pine: hyperWaveLen=6)
        trigger_len:    SMA length for trigger line (Pine: triggerLen=2)
        kernel_factor1: Fast kernel factor (Pine: kernelFactor1=0.8)
        kernel_factor2: Slow kernel factor (Pine: kernelFactor2=0.3)
        norm_len:       Rolling window for 0-100 normalization (Pine: 100)
        vol_lookback:   SMA period for relative volume (Pine: volLookback=30)

    Returns:
        pd.DataFrame with columns:
            osc_norm      float [0-100]  Main oscillator
            trig_norm     float [0-100]  Trigger line
            hyper_norm    float [0-100]  HyperWave
            sma_osc       float          SMA(21) of osc_norm
            is_bull_mom   bool           osc > trigger
            mom_cross_up  bool           crossover(osc, trigger)
            mom_cross_dn  bool           crossunder(osc, trigger)
            rvol          float          Relative volume
            pressure_pct  float          Buy pressure %
            wt1           float          Raw WaveTrend 1 (before normalization)
            wt2           float          Raw WaveTrend 2
    """
    channel_len = max(hyperwave_len * 2, 10)
    avg_len     = max(hyperwave_len, 4)

    # src = HLC3
    src = (df["high"] + df["low"] + df["close"]) / 3.0

    # WaveTrend core
    esa = _ema(src, channel_len)
    d   = _ema((src - esa).abs(), channel_len)
    ci  = (src - esa) / (0.015 * d)
    wt1 = _ema(ci, avg_len)
    wt2 = _sma(wt1, trigger_len)

    # HyperWave: double EMA smoothing
    hw_len1 = max(int(hyperwave_len * kernel_factor1), 2)
    hw_len2 = max(int(hyperwave_len * kernel_factor2), 2)
    hyper   = _ema(_ema(wt1, hw_len1), hw_len2)

    # Fast oscillator (not plotted separately but calculated)
    fast_ci = ci * kernel_factor1
    fast_len = max(int(avg_len * kernel_factor2), 2)
    _fast_wt = _ema(fast_ci, fast_len)  # noqa: F841 — reserved for future use

    # Normalization 0-100 using rolling window (equivalent to Pine Script)
    hi_norm = wt1.rolling(norm_len).max()
    lo_norm = wt1.rolling(norm_len).min()
    rng     = hi_norm - lo_norm
    safe    = rng.where(rng != 0, other=1.0)

    osc_norm   = (wt1 - lo_norm) / safe * 100.0
    trig_norm  = (wt2 - lo_norm) / safe * 100.0
    hyper_norm = (hyper - lo_norm) / safe * 100.0

    # SMA of oscillator
    sma_osc = _sma(osc_norm, 21)

    # Momentum signals
    is_bull_mom  = osc_norm > trig_norm
    mom_cross_up = (~is_bull_mom.shift(1).fillna(False)) & is_bull_mom
    mom_cross_dn = is_bull_mom.shift(1).fillna(False) & (~is_bull_mom)

    # Relative Volume
    avg_vol  = _sma(df["volume"], vol_lookback)
    rvol     = df["volume"] / avg_vol.where(avg_vol > 0, other=1.0)

    # Buy/Sell Pressure
    hl_range   = (df["high"] - df["low"]).where(
        (df["high"] - df["low"]) > 0, other=df["close"].diff().abs().fillna(1e-8)
    )
    buy_vol    = df["volume"] * (df["close"] - df["low"]) / hl_range
    pressure_pct = (buy_vol / df["volume"].where(df["volume"] > 0, other=1.0)) * 100.0

    # ─── EOT (End of Trend) — diff fast - slow EMA, smoothed ─────────────────
    eot_fast_len = max(int(hyperwave_len * 0.5), 2)
    eot_slow_len = max(int(hyperwave_len * 2.0), 8)
    eot_raw = _ema(src, eot_fast_len) - _ema(src, eot_slow_len)
    eot = _ema(eot_raw, max(int(hyperwave_len * 0.5), 2))
    # Normalize EOT to 0-100
    eot_hi = eot.rolling(norm_len).max()
    eot_lo = eot.rolling(norm_len).min()
    eot_rng = (eot_hi - eot_lo).where(eot_hi - eot_lo != 0, other=1.0)
    eot_norm = (eot - eot_lo) / eot_rng * 100.0

    # ─── CP Bounce — osc cross from OS zone (<25) up + rvol > 1, or OB zone (>75) down ─
    in_os = osc_norm < 25
    in_ob = osc_norm > 75
    cp_bounce_long = (
        in_os.shift(1).fillna(False)
        & (osc_norm > osc_norm.shift(1))
        & (rvol > 1.0)
    )
    cp_bounce_short = (
        in_ob.shift(1).fillna(False)
        & (osc_norm < osc_norm.shift(1))
        & (rvol > 1.0)
    )

    # ─── Volatility Regime — BB width vs 100-bar baseline ────────────────────
    bb_len = 20
    bb_mean = df["close"].rolling(bb_len).mean()
    bb_std = df["close"].rolling(bb_len).std(ddof=0)
    bb_width = (bb_std * 4.0) / bb_mean.where(bb_mean > 0, other=1.0) * 100.0  # %
    bb_width_baseline = bb_width.rolling(100).mean()
    vol_ratio = bb_width / bb_width_baseline.where(bb_width_baseline > 0, other=1.0)

    result = pd.DataFrame(
        {
            "osc_norm":    osc_norm,
            "trig_norm":   trig_norm,
            "hyper_norm":  hyper_norm,
            "sma_osc":     sma_osc,
            "is_bull_mom": is_bull_mom,
            "mom_cross_up": mom_cross_up,
            "mom_cross_dn": mom_cross_dn,
            "rvol":        rvol,
            "pressure_pct": pressure_pct,
            "wt1":         wt1,
            "wt2":         wt2,
            "eot":         eot,
            "eot_norm":    eot_norm,
            "cp_bounce_long":  cp_bounce_long,
            "cp_bounce_short": cp_bounce_short,
            "bb_width_pct": bb_width,
            "vol_ratio":   vol_ratio,
        },
        index=df.index,
    )
    return result


# ─── MTF Trend (equivalent to f_mtfTrend in Pine) ────────────────────────────

def wavetrend_mtf_trend(
    df: pd.DataFrame,
    hyperwave_len: int = 6,
    trigger_len: int = 2,
) -> pd.Series:
    """
    Compute MTF trend direction per bar.

    Returns pd.Series with values:
        +1.0 = Bullish  (osc > 50 and osc > trigger)
        -1.0 = Bearish  (osc < 50 and osc <= trigger)
         0.0 = Neutral

    Equivalent to f_mtfTrend() in oscilador_v26.pine
    """
    res = wavetrend(df, hyperwave_len=hyperwave_len, trigger_len=trigger_len)
    osc  = res["osc_norm"]
    bull = res["is_bull_mom"]

    trend = pd.Series(0.0, index=df.index)
    trend[(osc > 50) & bull]             =  1.0
    trend[(osc < 50) & (~bull)]          = -1.0
    return trend


def wavetrend_all_timeframes(
    dfs: dict[str, pd.DataFrame],
    hyperwave_len: int = 6,
    trigger_len: int = 2,
) -> dict[str, pd.DataFrame]:
    """
    Run wavetrend on multiple timeframes.

    Args:
        dfs: dict[timeframe_str, OHLCV_DataFrame]

    Returns:
        dict[timeframe_str, wavetrend_result_DataFrame]
    """
    results: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        results[tf] = wavetrend(df, hyperwave_len=hyperwave_len, trigger_len=trigger_len)
    return results


# ─── Summary for Prompt ───────────────────────────────────────────────────────

def wavetrend_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Extract latest bar values per timeframe for prompt generation.

    Returns:
        dict[tf] = {
            'osc': float, 'trigger': float, 'hyper': float,
            'trend': str,  'signal': str, 'rvol': float
        }
    """
    def _safe(val, decimals=1):
        try:
            f = float(val)
            return round(f, decimals) if not np.isnan(f) else "—"
        except Exception:
            return "—"

    summary: dict[str, dict] = {}
    for tf, df in results.items():
        if df is None or df.empty:
            continue
        # Use last non-NaN row for osc_norm
        valid = df.dropna(subset=["osc_norm"])
        if valid.empty:
            continue
        last = valid.iloc[-1]

        # Trend string
        osc_val = last["osc_norm"]
        is_bull = last["is_bull_mom"]
        if not np.isnan(osc_val):
            if osc_val > 50 and is_bull:
                trend_str = "↑ Bull"
            elif osc_val < 50 and not is_bull:
                trend_str = "↓ Bear"
            else:
                trend_str = "→ Neutral"
        else:
            trend_str = "→ Neutral"

        # Signal string (last 3 bars for recency)
        signals = []
        tail = df.tail(3)
        if tail["mom_cross_up"].any():
            signals.append("MomCrossUp")
        if tail["mom_cross_dn"].any():
            signals.append("MomCrossDn")

        # Extra signals from upgraded features
        if tail["cp_bounce_long"].any():  signals.append("CP Bounce↑")
        if tail["cp_bounce_short"].any(): signals.append("CP Bounce↓")

        # Vol regime
        vol_r = float(last.get("vol_ratio", 1.0)) if not np.isnan(last.get("vol_ratio", 1.0)) else 1.0
        if vol_r > 1.4:
            vol_regime = "high"
        elif vol_r < 0.7:
            vol_regime = "low"
        else:
            vol_regime = "normal"

        summary[tf] = {
            "osc":     _safe(last["osc_norm"],  1),
            "trigger": _safe(last["trig_norm"], 1),
            "hyper":   _safe(last["hyper_norm"], 1),
            "trend":   trend_str,
            "signal":  ", ".join(signals) if signals else "—",
            "rvol":    _safe(last["rvol"], 2),
            "pressure": _safe(last["pressure_pct"], 1),
            "eot":     _safe(last.get("eot_norm"), 1),
            "vol_regime": vol_regime,
            "vol_ratio": _safe(vol_r, 2),
        }
    return summary
