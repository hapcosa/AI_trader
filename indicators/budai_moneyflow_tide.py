"""
BudAI Smart Money Flow Tide — exact Python port of
pineforge/Osciladores/budai_moneyflow_tide.pine (mirrors KryptoLab
strategies/budai_moneyflow_tide.py).

Two Money-Flow-Index lines centered on 0 (±100): fast/slow = ema((mfi(src, len)
-50)*2, smooth). Strong signal = fast/slow crossover inside an OB/OS zone
(fast<=os_lvl long / fast>=ob_lvl short) read alongside Chaikin Money Flow.

The Pine has no `src` input (mfi reads hlc3); `osc_source` (default 'hlc3') is
still exposed for the optimizable MFI feed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, length: int) -> pd.Series:
    """EMA with adjust=False — identical to Pine ta.ema()."""
    return series.ewm(span=length, adjust=False).mean()


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


def _mfi(src: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Pine ta.mfi(src, length).

        upper = sum(volume * (change(src) <= 0 ? 0 : src), length)
        lower = sum(volume * (change(src) >= 0 ? 0 : src), length)
        mfi   = 100 - 100/(1 + upper/lower)
    """
    change = src.diff()
    raw = volume * src
    pos = raw.where(change > 0.0, 0.0)            # change<=0 (incl. NaN) → 0
    neg = raw.where(change < 0.0, 0.0)            # change>=0 (incl. NaN) → 0
    upper = pos.rolling(length, min_periods=1).sum()
    lower = neg.rolling(length, min_periods=1).sum()
    mfi = 100.0 - 100.0 / (1.0 + upper / lower)
    # lower == 0 → ratio +inf → mfi 100 (if upper>0) else 50 (0/0)
    mfi = mfi.where(lower != 0.0, other=np.where(upper > 0.0, 100.0, 50.0))
    return mfi


def budai_moneyflow_tide(
    df: pd.DataFrame,
    fast_len: int = 14,
    slow_len: int = 28,
    cmf_len: int = 20,
    smooth: int = 2,
    ob_lvl: float = 60.0,
    os_lvl: float = -60.0,
    osc_source: str = "hlc3",
) -> pd.DataFrame:
    """BudAI Money Flow Tide oscillator.

    Outputs:
        fast, slow     dual MFI lines centered on 0 (±100)
        cmf            Chaikin Money Flow
        bull_cross     fast crosses above slow
        bear_cross     fast crosses below slow
        strong_up      bull_cross in OS zone + accumulation
        strong_dn      bear_cross in OB zone + distribution
        is_ob, is_os   fast >= ob_lvl / fast <= os_lvl
        is_bull        fast >= slow
    """
    src = _resolve_source(df, osc_source)
    close, high, low = df["close"], df["high"], df["low"]
    volume = df["volume"]

    mfi_fast = (_mfi(src, volume, fast_len) - 50.0) * 2.0
    mfi_slow = (_mfi(src, volume, slow_len) - 50.0) * 2.0
    fast = _ema(mfi_fast.clip(-100.0, 100.0), smooth)
    slow = _ema(mfi_slow.clip(-100.0, 100.0), smooth)

    hl = high - low
    mfm = (((close - low) - (high - close)) / hl).where(hl != 0.0, 0.0)
    vol = volume.where(volume != 0.0, 1.0)
    cmf = ((mfm * vol).rolling(cmf_len, min_periods=1).sum()
           / vol.rolling(cmf_len, min_periods=1).sum())

    bull_cross = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    bear_cross = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    out = pd.DataFrame(index=df.index)
    out["fast"] = fast
    out["slow"] = slow
    out["cmf"] = cmf
    out["bull_cross"] = bull_cross
    out["bear_cross"] = bear_cross
    out["strong_up"] = bull_cross & (fast <= os_lvl) & (cmf > 0.0)
    out["strong_dn"] = bear_cross & (fast >= ob_lvl) & (cmf < 0.0)
    out["is_ob"] = fast >= ob_lvl
    out["is_os"] = fast <= os_lvl
    out["is_bull"] = fast >= slow
    return out


def budai_moneyflow_tide_all_timeframes(
    dfs: dict[str, pd.DataFrame], **kwargs
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        results[tf] = budai_moneyflow_tide(df, **kwargs)
    return results


def budai_moneyflow_tide_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
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
        valid = df.dropna(subset=["fast"])
        if valid.empty:
            continue
        last = valid.iloc[-1]

        is_bull = bool(last["is_bull"])
        if bool(last["is_ob"]):
            zone = "Distribution-OB"
        elif bool(last["is_os"]):
            zone = "Accumulation-OS"
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
            "fast": _safe(last["fast"]),
            "slow": _safe(last["slow"]),
            "zone": zone,
            "trend": trend,
            "flow": flow,
            "cmf": _safe(cmf_v, 3),
            "signal": ", ".join(signals) if signals else "—",
        }
    return summary
