"""
BudAI Abyss Wave Oscillator — exact Python port of
pineforge/Osciladores/budai_abyss.pine (mirrors KryptoLab strategies/budai_abyss.py).

Oscillator = raw WaveTrend (wt1 = ema(ci, avg_len), wt2 = sma(wt1, trig_len));
NOT normalized (the ±100 clamp in the Pine is display-only). The strong signal
is a wt1/wt2 crossover inside an EXTREME zone (wt1 <= os_x long / wt1 >= ob_x
short) read alongside Chaikin Money Flow direction.

`osc_source` is the oscillator feed (default 'hlc3', the Pine's default source).
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


def budai_abyss(
    df: pd.DataFrame,
    chan_len: int = 9,
    avg_len: int = 12,
    trig_len: int = 3,
    mf_len: int = 14,
    ob_x: float = 53.0,
    os_x: float = -53.0,
    osc_source: str = "hlc3",
) -> pd.DataFrame:
    """BudAI Abyss oscillator.

    Outputs:
        wt1, wt2       raw WaveTrend + trigger
        cmf            Chaikin Money Flow
        bull_cross     wt1 crosses above wt2
        bear_cross     wt1 crosses below wt2
        strong_up      bull_cross in extreme OS zone + accumulation
        strong_dn      bear_cross in extreme OB zone + distribution
        is_ob, is_os   wt1 >= ob_x / wt1 <= os_x
        is_bull        wt1 >= wt2
    """
    src = _resolve_source(df, osc_source)
    close, high, low = df["close"], df["high"], df["low"]
    volume = df["volume"]

    esa = _ema(src, chan_len)
    de = _ema((src - esa).abs(), chan_len)
    ci = ((src - esa) / (0.015 * de)).where(de != 0.0, 0.0)
    wt1 = _ema(ci, avg_len)
    wt2 = _sma(wt1, trig_len)

    hl = high - low
    mfm = (((close - low) - (high - close)) / hl).where(hl != 0.0, 0.0)
    vol = volume.where(volume != 0.0, 1.0)
    cmf = ((mfm * vol).rolling(mf_len, min_periods=1).sum()
           / vol.rolling(mf_len, min_periods=1).sum())

    bull_cross = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    bear_cross = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

    out = pd.DataFrame(index=df.index)
    out["wt1"] = wt1
    out["wt2"] = wt2
    out["cmf"] = cmf
    out["bull_cross"] = bull_cross
    out["bear_cross"] = bear_cross
    out["strong_up"] = bull_cross & (wt1 <= os_x) & (cmf > 0.0)
    out["strong_dn"] = bear_cross & (wt1 >= ob_x) & (cmf < 0.0)
    out["is_ob"] = wt1 >= ob_x
    out["is_os"] = wt1 <= os_x
    out["is_bull"] = wt1 >= wt2
    return out


def budai_abyss_all_timeframes(
    dfs: dict[str, pd.DataFrame], **kwargs
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        results[tf] = budai_abyss(df, **kwargs)
    return results


def budai_abyss_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
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
        valid = df.dropna(subset=["wt1"])
        if valid.empty:
            continue
        last = valid.iloc[-1]

        is_bull = bool(last["is_bull"])
        if bool(last["is_ob"]):
            zone = "OverBought-Extreme"
        elif bool(last["is_os"]):
            zone = "OverSold-Extreme"
        else:
            zone = "Neutral"
        trend = "↑ Bull" if is_bull else "↓ Bear"

        tail = df.tail(3)
        signals = []
        if tail["strong_up"].any():
            signals.append("StrongUp")
        elif tail["bull_cross"].any():
            signals.append("BullCross")
        if tail["strong_dn"].any():
            signals.append("StrongDn")
        elif tail["bear_cross"].any():
            signals.append("BearCross")

        cmf_v = float(last["cmf"]) if not np.isnan(last["cmf"]) else 0.0
        flow = "Accumulation" if cmf_v > 0 else "Distribution"

        summary[tf] = {
            "wt1": _safe(last["wt1"]),
            "wt2": _safe(last["wt2"]),
            "zone": zone,
            "trend": trend,
            "flow": flow,
            "cmf": _safe(cmf_v, 3),
            "signal": ", ".join(signals) if signals else "—",
        }
    return summary
