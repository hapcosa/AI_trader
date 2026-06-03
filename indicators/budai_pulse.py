"""
BudAI Pulse Flow Oscillator — exact Python port of
pineforge/Osciladores/budai_pulse.pine (mirrors KryptoLab strategies/budai_pulse.py).

Oscillator = blend of WaveTrend (cyclic center of gravity) + COG (Ehlers) +
Momentum, each normalized 0-100 over `norm_len`. Trigger = normalized sma(wt1).
Signal = osc/trig crossover, optionally read alongside Chaikin Money Flow.

`osc_source` is the oscillator feed (default 'hlc3', the Pine's hardcoded
source). It feeds WaveTrend + COG; momentum stays on close and money-flow on
HLC/volume, exactly like the Pine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, length: int) -> pd.Series:
    """EMA with adjust=False — identical to Pine ta.ema()."""
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def _resolve_source(df: pd.DataFrame, osc_source: str) -> pd.Series:
    if osc_source == "close":
        return df["close"].astype(float)
    if osc_source == "hl2":
        return (df["high"] + df["low"]) / 2.0
    if osc_source == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3.0
    if osc_source == "ohlc4":
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    raise ValueError(f"unknown osc_source {osc_source!r}")


def _f_norm(s: pd.Series, length: int) -> pd.Series:
    """Normalize to 0-100 over a rolling window (Pine f_norm). Partial windows
    (min_periods=1) match ta.lowest/highest's available-bar behavior."""
    lo = s.rolling(length, min_periods=1).min()
    hi = s.rolling(length, min_periods=1).max()
    rng = hi - lo
    return (100.0 * (s - lo) / rng).where(rng != 0.0, 50.0)


def _cog(src: pd.Series, cog_len: int) -> pd.Series:
    """Ehlers Center of Gravity, nz(src[i]) → 0 before history."""
    s = src.to_numpy(dtype=float)
    n = len(s)
    out = np.zeros(n)
    for t in range(n):
        num = den = 0.0
        for i in range(cog_len):
            p = s[t - i] if t - i >= 0 else 0.0
            num += (1 + i) * p
            den += p
        out[t] = (-num / den + (cog_len + 1) / 2.0) if den != 0.0 else 0.0
    return pd.Series(out, index=src.index)


def budai_pulse(
    df: pd.DataFrame,
    chan_len: int = 9,
    avg_len: int = 12,
    trig_len: int = 3,
    use_cog: bool = True,
    cog_len: int = 9,
    mom_len: int = 10,
    mf_len: int = 14,
    norm_len: int = 100,
    ob_level: float = 80.0,
    os_level: float = 20.0,
    osc_source: str = "hlc3",
) -> pd.DataFrame:
    """BudAI Pulse oscillator.

    Outputs (columns added to a copy of df's index):
        osc            blended oscillator, 0-100
        trig           normalized trigger, 0-100
        cmf            Chaikin Money Flow (-1..1)
        bull_cross     osc crosses above trig
        bear_cross     osc crosses below trig
        is_ob, is_os   osc >= ob_level / osc <= os_level
        is_bull        osc >= trig
    """
    src = _resolve_source(df, osc_source)
    close, high, low = df["close"], df["high"], df["low"]
    volume = df["volume"]

    esa = _ema(src, chan_len)
    de = _ema((src - esa).abs(), chan_len)
    ci = ((src - esa) / (0.015 * de)).where(de != 0.0, 0.0)
    wt1 = _ema(ci, avg_len)
    wt2 = _sma(wt1, trig_len)

    cog = _cog(src, cog_len)
    mom = close - close.shift(mom_len)

    hl = high - low
    mfm = (((close - low) - (high - close)) / hl).where(hl != 0.0, 0.0)
    vol = volume.where(volume != 0.0, 1.0)
    cmf = ((mfm * vol).rolling(mf_len, min_periods=1).sum()
           / vol.rolling(mf_len, min_periods=1).sum())

    wt_n = _f_norm(wt1, norm_len)
    cog_n = _f_norm(cog, norm_len)
    mom_n = _f_norm(mom, norm_len)
    if use_cog:
        osc = wt_n * 0.6 + cog_n * 0.25 + mom_n * 0.15
    else:
        osc = wt_n * 0.8 + mom_n * 0.2
    trig = _f_norm(wt2, norm_len)

    out = pd.DataFrame(index=df.index)
    out["osc"] = osc
    out["trig"] = trig
    out["cmf"] = cmf
    out["bull_cross"] = (osc > trig) & (osc.shift(1) <= trig.shift(1))
    out["bear_cross"] = (osc < trig) & (osc.shift(1) >= trig.shift(1))
    out["is_ob"] = osc >= ob_level
    out["is_os"] = osc <= os_level
    out["is_bull"] = osc >= trig
    return out


def budai_pulse_all_timeframes(
    dfs: dict[str, pd.DataFrame], **kwargs
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        results[tf] = budai_pulse(df, **kwargs)
    return results


def budai_pulse_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Latest-bar state per timeframe for prompt generation."""
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
        valid = df.dropna(subset=["osc"])
        if valid.empty:
            continue
        last = valid.iloc[-1]

        osc_v = float(last["osc"])
        is_bull = bool(last["is_bull"])
        if bool(last["is_ob"]):
            zone = "OverBought"
        elif bool(last["is_os"]):
            zone = "OverSold"
        else:
            zone = "Neutral"
        trend = "↑ Bull" if is_bull else "↓ Bear"

        tail = df.tail(3)
        signals = []
        if tail["bull_cross"].any():
            signals.append("BullCross")
        if tail["bear_cross"].any():
            signals.append("BearCross")

        cmf_v = float(last["cmf"]) if not np.isnan(last["cmf"]) else 0.0
        flow = "Accumulation" if cmf_v > 0 else "Distribution"

        summary[tf] = {
            "osc": _safe(osc_v),
            "trig": _safe(last["trig"]),
            "zone": zone,
            "trend": trend,
            "flow": flow,
            "cmf": _safe(cmf_v, 3),
            "signal": ", ".join(signals) if signals else "—",
        }
    return summary
