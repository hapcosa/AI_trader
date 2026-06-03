"""
BudAI Athenea Oscillator — exact Python port of
pineforge/Osciladores/budai_athenea_oscillator.pine (mirrors KryptoLab
strategies/budai_athenea.py).

Hybrid oscillator = WaveTrend + Linear-Regression slope + COG (Ehlers), each
normalized 0-100 and blended (0.5/0.3/0.2 with COG, else 0.6/0.4). Trigger =
normalized sma(wt1). Two entry families read alongside Chaikin Money Flow:
  - cyclic cross: osc/trig crossover with osc<os_lvl OR Vix-Fix panic (long),
    crossunder with osc>ob_lvl (short);
  - squeeze release: BB-inside-KC squeeze turns off with osc on the right side
    of 50.

`osc_source` (default 'hlc3') feeds WaveTrend + slope + COG + squeeze + Vix-Fix
(the Pine's `src`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def _stdev(series: pd.Series, length: int) -> pd.Series:
    """Population stdev — Pine ta.stdev default (biased, ddof=0)."""
    return series.rolling(length).std(ddof=0)


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
    lo = s.rolling(length, min_periods=1).min()
    hi = s.rolling(length, min_periods=1).max()
    rng = hi - lo
    return (100.0 * (s - lo) / rng).where(rng != 0.0, 50.0)


def _linreg_slope(s: pd.Series, length: int) -> pd.Series:
    """Slope of the OLS line over `length` bars = ta.linreg(s,len,0)-linreg(s,len,1)."""
    x = np.arange(length, dtype=float)
    sx = x.sum()
    sxx = (x * x).sum()
    denom = length * sxx - sx * sx

    def _slope(y: np.ndarray) -> float:
        sy = y.sum()
        sxy = (x * y).sum()
        return (length * sxy - sx * sy) / denom if denom != 0 else 0.0

    return s.rolling(length).apply(_slope, raw=True)


def _cog(src: pd.Series, cog_len: int) -> pd.Series:
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


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def budai_athenea(
    df: pd.DataFrame,
    chan_len: int = 9,
    avg_len: int = 12,
    trig_len: int = 3,
    slope_len: int = 20,
    use_cog: bool = True,
    cog_len: int = 9,
    norm_len: int = 100,
    mf_len: int = 14,
    ob_lvl: float = 80.0,
    os_lvl: float = 20.0,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    kc_len: int = 20,
    kc_mult: float = 1.5,
    use_tr: bool = True,
    use_vix: bool = True,
    vix_len: int = 22,
    bb_len_v: int = 20,
    bb_mult_v: float = 2.0,
    ph_len: int = 50,
    ph_up: float = 0.85,
    osc_source: str = "hlc3",
) -> pd.DataFrame:
    """BudAI Athenea hybrid oscillator.

    Outputs:
        osc, trig      blended oscillator + trigger, 0-100
        cmf            Chaikin Money Flow
        bull_cross, bear_cross
        released       squeeze just turned off
        is_panic       Vix-Fix panic
        entry_buy/entry_sell      cyclic-cross signals
        sqz_buy/sqz_sell          squeeze-release signals
        signal_buy/signal_sell    entry | squeeze (the webhook signal)
        is_ob, is_os, is_bull
    """
    src = _resolve_source(df, osc_source)
    close, high, low = df["close"], df["high"], df["low"]
    volume = df["volume"]

    # WaveTrend
    esa = _ema(src, chan_len)
    de = _ema((src - esa).abs(), chan_len)
    ci = ((src - esa) / (0.015 * de)).where(de != 0.0, 0.0)
    wt1 = _ema(ci, avg_len)
    wt2 = _sma(wt1, trig_len)

    slope_raw = _linreg_slope(src, slope_len)
    cog = _cog(src, cog_len)

    wt_n = _f_norm(wt1, norm_len)
    slope_n = _f_norm(slope_raw, norm_len)
    cog_n = _f_norm(cog, norm_len)
    if use_cog:
        osc = wt_n * 0.5 + slope_n * 0.3 + cog_n * 0.2
    else:
        osc = wt_n * 0.6 + slope_n * 0.4
    trig = _f_norm(wt2, norm_len)

    hl = high - low
    mfm = (((close - low) - (high - close)) / hl).where(hl != 0.0, 0.0)
    vol = volume.where(volume != 0.0, 1.0)
    cmf = ((mfm * vol).rolling(mf_len, min_periods=1).sum()
           / vol.rolling(mf_len, min_periods=1).sum())

    # Squeeze (Bollinger inside Keltner)
    basis_bb = _sma(src, bb_len)
    dev_bb = bb_mult * _stdev(src, bb_len)
    bb_upper, bb_lower = basis_bb + dev_bb, basis_bb - dev_bb
    kc_ma = _sma(src, kc_len)
    kc_range = _true_range(high, low, close) if use_tr else (high - low)
    kc_range_ma = _sma(kc_range, kc_len)
    kc_upper, kc_lower = kc_ma + kc_range_ma * kc_mult, kc_ma - kc_range_ma * kc_mult
    sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    released = sqz_on.shift(1, fill_value=False) & (~sqz_on)

    # Williams Vix-Fix panic
    hhv = src.rolling(vix_len, min_periods=1).max()
    vix_raw = ((hhv - low) / hhv * 100.0).where(hhv != 0.0, 0.0)
    vix_fix = _ema(vix_raw, 2)
    b_up_v = _sma(vix_fix, bb_len_v) + bb_mult_v * _stdev(vix_fix, bb_len_v)
    range_hi_v = vix_fix.rolling(ph_len, min_periods=1).max() * ph_up
    is_panic = ((vix_fix >= b_up_v) | (vix_fix >= range_hi_v)) if use_vix else pd.Series(False, index=df.index)

    bull_cross = (osc > trig) & (osc.shift(1) <= trig.shift(1))
    bear_cross = (osc < trig) & (osc.shift(1) >= trig.shift(1))

    entry_buy = bull_cross & ((osc < os_lvl) | is_panic) & (cmf > 0.0)
    entry_sell = bear_cross & (osc > ob_lvl) & (cmf < 0.0)
    sqz_buy = released & (osc > 50.0) & (cmf > 0.0)
    sqz_sell = released & (osc < 50.0) & (cmf < 0.0)

    out = pd.DataFrame(index=df.index)
    out["osc"] = osc
    out["trig"] = trig
    out["cmf"] = cmf
    out["bull_cross"] = bull_cross
    out["bear_cross"] = bear_cross
    out["released"] = released
    out["is_panic"] = is_panic
    out["entry_buy"] = entry_buy
    out["entry_sell"] = entry_sell
    out["sqz_buy"] = sqz_buy
    out["sqz_sell"] = sqz_sell
    out["signal_buy"] = entry_buy | sqz_buy
    out["signal_sell"] = entry_sell | sqz_sell
    out["is_ob"] = osc >= ob_lvl
    out["is_os"] = osc <= os_lvl
    out["is_bull"] = osc >= trig
    return out


def budai_athenea_all_timeframes(
    dfs: dict[str, pd.DataFrame], **kwargs
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        results[tf] = budai_athenea(df, **kwargs)
    return results


def budai_athenea_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
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
        if tail["signal_buy"].any():
            signals.append("BuyEntry")
        if tail["signal_sell"].any():
            signals.append("SellEntry")

        cmf_v = float(last["cmf"]) if not np.isnan(last["cmf"]) else 0.0
        flow = "Accumulation" if cmf_v > 0 else "Distribution"

        summary[tf] = {
            "osc": _safe(last["osc"]),
            "trig": _safe(last["trig"]),
            "zone": zone,
            "trend": trend,
            "flow": flow,
            "panic": bool(last["is_panic"]),
            "squeeze_release": bool(df["released"].tail(3).any()),
            "signal": ", ".join(signals) if signals else "—",
        }
    return summary
