"""
SMC Buda — Python port of BUDA MONEY CONCEPTS v2.0
Based on BigBeluga - Smart Money Concepts [1.0.0]

Signal logic ported exactly from BudaSMC.pine:
  Type A: CHoCH + trend aligned + MTF + demand/supply zone
  Type B: BOS continuation + trend + MTF + OB pullback
  Type C: Liquidity sweep + MTF + zone + candle direction
  Confluence 0-5: CTF trend, MTF trend, in OB, in FVG, sweep
  SL: OB.btm/top ± ATR(14)*0.5  |  TP: entry ± risk*RR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# ─── Data Classes ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MTFState:
    """Higher-TF state snapshot — equiv. Pine MTFState UDT."""
    trend: int = 0               # 1 bull / -1 bear / 0 neutral
    bos_level: float = np.nan
    choch_level: float = np.nan
    has_bull_ob: bool = False
    has_bear_ob: bool = False
    has_bull_fvg: bool = False
    has_bear_fvg: bool = False


@dataclass
class TradeSignal:
    """Trade signal — equiv. Pine TradeSignal UDT."""
    is_long: bool = False
    signal_type: str = ""        # "A" | "B" | "C"
    entry: float = np.nan
    stop_loss: float = np.nan
    take_profit: float = np.nan
    bar_idx: int = 0
    confluence: int = 0          # 0-5
    reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# ─── Helpers ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _atr14(df: pd.DataFrame) -> np.ndarray:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _ema(tr, 14).to_numpy(dtype=float)


def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        if arr[i] > np.nanmax(arr[i - left:i]) and arr[i] > np.nanmax(arr[i + 1:i + right + 1]):
            result[i] = arr[i]
    return pd.Series(result, index=series.index)


def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)
    for i in range(left, len(arr) - right):
        if arr[i] < np.nanmin(arr[i - left:i]) and arr[i] < np.nanmin(arr[i + 1:i + right + 1]):
            result[i] = arr[i]
    return pd.Series(result, index=series.index)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 1. Market Structure ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def market_structure(
    df: pd.DataFrame,
    ms_len: int = 5,
    build_sweep: bool = True,
) -> pd.DataFrame:
    """
    Port of Pine structure() state machine.

    Columns:
        swing_high / swing_low   float   pivot values
        bos_bull / bos_bear      bool    BOS event bar
        choch_bull / choch_bear  bool    CHoCH event bar
        dnsweep / upsweep        bool    liquidity sweep bar
        ms_txt                   str     "bos" | "choch" | "" — event on this bar
        ms_trend                 int     +1/-1/0
        last_bos_level           float   level of most recent BOS/CHoCH
        last_event               str     forward-filled event label
    """
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    n = len(df)

    swing_high = _pivot_high(df["high"], ms_len, ms_len).to_numpy(dtype=float)
    swing_low  = _pivot_low(df["low"],   ms_len, ms_len).to_numpy(dtype=float)

    bos_bull   = np.zeros(n, dtype=bool)
    bos_bear   = np.zeros(n, dtype=bool)
    choch_bull = np.zeros(n, dtype=bool)
    choch_bear = np.zeros(n, dtype=bool)
    dnsweep    = np.zeros(n, dtype=bool)
    upsweep    = np.zeros(n, dtype=bool)
    ms_trend   = np.zeros(n, dtype=int)
    ms_txt     = np.full(n, "", dtype=object)
    last_bos   = np.full(n, np.nan)

    last_sh = np.nan
    last_sl = np.nan
    trend = 0
    bos_level = np.nan
    choch_level = np.nan

    for i in range(ms_len * 2, n):
        if not np.isnan(swing_high[i]):
            last_sh = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_sl = swing_low[i]

        c = close[i]
        h = high[i]
        l = low[i]

        # ── Sweep detection ──────────────────────────────────────────────────
        # dnsweep: wick below BOS/swing low, close back above (stop hunt → bull)
        # upsweep: wick above BOS/swing high, close back below (stop hunt → bear)
        if build_sweep:
            ref_low  = bos_level if not np.isnan(bos_level) and trend == -1 else last_sl
            ref_high = bos_level if not np.isnan(bos_level) and trend ==  1 else last_sh
            if not np.isnan(ref_low)  and l <= ref_low  and c > ref_low:
                dnsweep[i] = True
            if not np.isnan(ref_high) and h >= ref_high and c < ref_high:
                upsweep[i] = True

        # ── BOS / CHoCH ──────────────────────────────────────────────────────
        if not np.isnan(last_sh) and c > last_sh:
            if trend == -1:
                choch_bull[i] = True
                ms_txt[i] = "choch"
            else:
                bos_bull[i] = True
                ms_txt[i] = "bos"
            last_bos[i] = last_sh
            bos_level = last_sh
            choch_level = np.nan
            trend = 1
            last_sh = np.nan

        elif not np.isnan(last_sl) and c < last_sl:
            if trend == 1:
                choch_bear[i] = True
                ms_txt[i] = "choch"
            else:
                bos_bear[i] = True
                ms_txt[i] = "bos"
            last_bos[i] = last_sl
            bos_level = last_sl
            choch_level = np.nan
            trend = -1
            last_sl = np.nan

        ms_trend[i] = trend

    # Forward-fill last_event
    last_ev_ff = []
    cur = "—"
    for i in range(n):
        ev = None
        if choch_bull[i]:   ev = "CHoCH Bull"
        elif choch_bear[i]: ev = "CHoCH Bear"
        elif bos_bull[i]:   ev = "BOS Bull"
        elif bos_bear[i]:   ev = "BOS Bear"
        elif dnsweep[i]:    ev = "Sweep Bull"
        elif upsweep[i]:    ev = "Sweep Bear"
        if ev:
            cur = ev
        last_ev_ff.append(cur)

    return pd.DataFrame({
        "swing_high":     pd.Series(swing_high,  index=df.index),
        "swing_low":      pd.Series(swing_low,   index=df.index),
        "bos_bull":       pd.Series(bos_bull,    index=df.index),
        "bos_bear":       pd.Series(bos_bear,    index=df.index),
        "choch_bull":     pd.Series(choch_bull,  index=df.index),
        "choch_bear":     pd.Series(choch_bear,  index=df.index),
        "dnsweep":        pd.Series(dnsweep,     index=df.index),
        "upsweep":        pd.Series(upsweep,     index=df.index),
        "ms_txt":         pd.Series(ms_txt,      index=df.index),
        "ms_trend":       pd.Series(ms_trend,    index=df.index),
        "last_bos_level": pd.Series(last_bos,    index=df.index),
        "last_event":     pd.Series(last_ev_ff,  index=df.index),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 2. Volumetric Order Blocks ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def order_blocks(
    df: pd.DataFrame,
    ms_df: pd.DataFrame,
    ob_len: int = 5,
    max_obs: int = 5,
    mitigation: str = "Close",
) -> pd.DataFrame:
    """
    Port of fnOB() + mitigated() logic.

    In-zone check (matches Pine):
        Bull OB: low <= blk.top AND close >= blk.btm
        Bear OB: high >= blk.btm AND close <= blk.top

    Mitigation (equiv. obmiti):
        'Close': min(c,o) < btm  /  max(c,o) > top
        'Wick':  low < btm       /  high > top
        'Avg':   low < avg       /  high > avg

    Columns:
        ob_bull_top/btm     float   most recent active bull OB
        ob_bear_top/btm     float   most recent active bear OB
        ob_bull_avg         float   midpoint of bull OB
        ob_bear_avg         float   midpoint of bear OB
        ob_bull_sl          float   SL level for bull OB signal (btm - ATR*0.5)
        ob_bear_sl          float   SL level for bear OB signal (top + ATR*0.5)
        ob_bull_vol         float   volume at formation
        ob_bear_vol         float
        ob_bull_mitigated   bool
        ob_bear_mitigated   bool
        in_bull_ob          bool    price currently inside bull OB zone
        in_bear_ob          bool    price currently inside bear OB zone
    """
    close  = df["close"].to_numpy(dtype=float)
    open_  = df["open"].to_numpy(dtype=float)
    high   = df["high"].to_numpy(dtype=float)
    low    = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    atr    = _atr14(df)
    n = len(df)

    bos_bull   = ms_df["bos_bull"].to_numpy(dtype=bool)
    bos_bear   = ms_df["bos_bear"].to_numpy(dtype=bool)
    choch_bull = ms_df["choch_bull"].to_numpy(dtype=bool)
    choch_bear = ms_df["choch_bear"].to_numpy(dtype=bool)

    ob_bt  = np.full(n, np.nan)
    ob_bb  = np.full(n, np.nan)
    ob_ba  = np.full(n, np.nan)
    ob_bsl = np.full(n, np.nan)
    ob_bv  = np.full(n, np.nan)
    ob_ert = np.full(n, np.nan)
    ob_erb = np.full(n, np.nan)
    ob_bea = np.full(n, np.nan)
    ob_bsl2 = np.full(n, np.nan)
    ob_bev = np.full(n, np.nan)
    ob_bm  = np.zeros(n, dtype=bool)
    ob_erm = np.zeros(n, dtype=bool)
    in_bull = np.zeros(n, dtype=bool)
    in_bear = np.zeros(n, dtype=bool)

    # Each zone: (top, btm, avg, vol)
    active_bull: list[tuple[float, float, float, float]] = []
    active_bear: list[tuple[float, float, float, float]] = []

    def _mit_bull(i: int, top: float, btm: float) -> bool:
        avg = (top + btm) / 2
        if mitigation == "Close":   return min(close[i], open_[i]) < btm
        if mitigation == "Wick":    return low[i] < btm
        return low[i] < avg

    def _mit_bear(i: int, top: float, btm: float) -> bool:
        avg = (top + btm) / 2
        if mitigation == "Close":   return max(close[i], open_[i]) > top
        if mitigation == "Wick":    return high[i] > top
        return high[i] > avg

    for i in range(ob_len + 1, n):
        # Register new OBs on BOS/CHoCH
        if bos_bull[i] or choch_bull[i]:
            for k in range(i - 1, max(i - ob_len - 1, 0), -1):
                if close[k] < open_[k]:  # last bearish candle before impulse
                    top = max(open_[k], close[k])
                    btm = min(open_[k], close[k])
                    active_bull.append((top, btm, (top + btm) / 2, volume[k]))
                    if len(active_bull) > max_obs:
                        active_bull.pop(0)
                    break

        if bos_bear[i] or choch_bear[i]:
            for k in range(i - 1, max(i - ob_len - 1, 0), -1):
                if close[k] > open_[k]:  # last bullish candle before impulse
                    top = max(open_[k], close[k])
                    btm = min(open_[k], close[k])
                    active_bear.append((top, btm, (top + btm) / 2, volume[k]))
                    if len(active_bear) > max_obs:
                        active_bear.pop(0)
                    break

        # Mitigation
        new_bull = []
        for z in active_bull:
            top, btm, avg, vol = z
            if _mit_bull(i, top, btm):
                ob_bm[i] = True
            else:
                new_bull.append(z)
        active_bull = new_bull

        new_bear = []
        for z in active_bear:
            top, btm, avg, vol = z
            if _mit_bear(i, top, btm):
                ob_erm[i] = True
            else:
                new_bear.append(z)
        active_bear = new_bear

        # Record most recent active zone + in-zone check (Pine exact)
        if active_bull:
            top, btm, avg, vol = active_bull[-1]
            ob_bt[i]  = top
            ob_bb[i]  = btm
            ob_ba[i]  = avg
            ob_bsl[i] = btm - atr[i] * 0.5
            ob_bv[i]  = vol
            # Bull OB entry: low <= top AND close >= btm
            if low[i] <= top and close[i] >= btm:
                in_bull[i] = True

        if active_bear:
            top, btm, avg, vol = active_bear[-1]
            ob_ert[i]  = top
            ob_erb[i]  = btm
            ob_bea[i]  = avg
            ob_bsl2[i] = top + atr[i] * 0.5
            ob_bev[i]  = vol
            # Bear OB entry: high >= btm AND close <= top
            if high[i] >= btm and close[i] <= top:
                in_bear[i] = True

    return pd.DataFrame({
        "ob_bull_top":       ob_bt,
        "ob_bull_btm":       ob_bb,
        "ob_bull_avg":       ob_ba,
        "ob_bull_sl":        ob_bsl,
        "ob_bull_vol":       ob_bv,
        "ob_bear_top":       ob_ert,
        "ob_bear_btm":       ob_erb,
        "ob_bear_avg":       ob_bea,
        "ob_bear_sl":        ob_bsl2,
        "ob_bear_vol":       ob_bev,
        "ob_bull_mitigated": ob_bm,
        "ob_bear_mitigated": ob_erm,
        "in_bull_ob":        in_bull,
        "in_bear_ob":        in_bear,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 3. Fair Value Gaps ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def fair_value_gaps(
    df: pd.DataFrame,
    fvg_threshold: float = 0.0,
    max_fvgs: int = 5,
    mitigation: str = "Close",
) -> pd.DataFrame:
    """
    Port of dFVG() logic.

    Bull FVG: low[i] > high[i-2] AND close[i-1] > open[i-1]
    Bear FVG: low[i-2] > high[i] AND close[i-1] < open[i-1]

    In-zone check (Pine exact):
        Bull: low <= fvg.top AND close >= fvg.btm
        Bear: high >= fvg.btm AND close <= fvg.top

    Columns:
        fvg_bull_top/btm    float  active bull FVG
        fvg_bear_top/btm    float  active bear FVG
        fvg_bull_avg        float  midpoint
        fvg_bear_avg        float
        fvg_bull_sl         float  SL for bull FVG signal (btm - ATR*0.5)
        fvg_bear_sl         float  SL for bear FVG signal (top + ATR*0.5)
        fvg_bull_new        bool   new bull FVG this bar
        fvg_bear_new        bool   new bear FVG this bar
        in_bull_fvg         bool   price inside bull FVG
        in_bear_fvg         bool   price inside bear FVG
    """
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr   = _atr14(df)
    n = len(df)

    fvg_bt  = np.full(n, np.nan)
    fvg_bb  = np.full(n, np.nan)
    fvg_ba  = np.full(n, np.nan)
    fvg_bsl = np.full(n, np.nan)
    fvg_ert = np.full(n, np.nan)
    fvg_erb = np.full(n, np.nan)
    fvg_bea = np.full(n, np.nan)
    fvg_bsl2 = np.full(n, np.nan)
    fvg_bn  = np.zeros(n, dtype=bool)
    fvg_rn  = np.zeros(n, dtype=bool)
    in_bull = np.zeros(n, dtype=bool)
    in_bear = np.zeros(n, dtype=bool)

    # (top, btm)
    active_bull: list[tuple[float, float]] = []
    active_bear: list[tuple[float, float]] = []

    def _mit_bull(i: int, top: float, btm: float) -> bool:
        avg = (top + btm) / 2
        if mitigation == "Close":   return min(close[i], open_[i]) < btm
        if mitigation == "Wick":    return low[i] < btm
        return low[i] < avg

    def _mit_bear(i: int, top: float, btm: float) -> bool:
        avg = (top + btm) / 2
        if mitigation == "Close":   return max(close[i], open_[i]) > top
        if mitigation == "Wick":    return high[i] > top
        return high[i] > avg

    for i in range(2, n):
        # New bull FVG
        if low[i] > high[i - 2] and close[i - 1] > open_[i - 1]:
            if (low[i] - high[i - 2]) > fvg_threshold:
                active_bull.append((low[i], high[i - 2]))
                if len(active_bull) > max_fvgs:
                    active_bull.pop(0)
                fvg_bn[i] = True

        # New bear FVG
        if low[i - 2] > high[i] and close[i - 1] < open_[i - 1]:
            if (low[i - 2] - high[i]) > fvg_threshold:
                active_bear.append((low[i - 2], high[i]))
                if len(active_bear) > max_fvgs:
                    active_bear.pop(0)
                fvg_rn[i] = True

        # Mitigation
        active_bull = [(t, b) for (t, b) in active_bull if not _mit_bull(i, t, b)]
        active_bear = [(t, b) for (t, b) in active_bear if not _mit_bear(i, t, b)]

        if active_bull:
            top, btm = active_bull[-1]
            fvg_bt[i]  = top
            fvg_bb[i]  = btm
            fvg_ba[i]  = (top + btm) / 2
            fvg_bsl[i] = btm - atr[i] * 0.5
            if low[i] <= top and close[i] >= btm:
                in_bull[i] = True

        if active_bear:
            top, btm = active_bear[-1]
            fvg_ert[i]  = top
            fvg_erb[i]  = btm
            fvg_bea[i]  = (top + btm) / 2
            fvg_bsl2[i] = top + atr[i] * 0.5
            if high[i] >= btm and close[i] <= top:
                in_bear[i] = True

    return pd.DataFrame({
        "fvg_bull_top":  fvg_bt,
        "fvg_bull_btm":  fvg_bb,
        "fvg_bull_avg":  fvg_ba,
        "fvg_bull_sl":   fvg_bsl,
        "fvg_bear_top":  fvg_ert,
        "fvg_bear_btm":  fvg_erb,
        "fvg_bear_avg":  fvg_bea,
        "fvg_bear_sl":   fvg_bsl2,
        "fvg_bull_new":  fvg_bn,
        "fvg_bear_new":  fvg_rn,
        "in_bull_fvg":   in_bull,
        "in_bear_fvg":   in_bear,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 4. Fisher Transform (Ehlers) ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def fisher_transform(
    df: pd.DataFrame,
    period: int = 14,
    extreme: float = 2.5,
) -> pd.DataFrame:
    """Ehlers Fisher Transform — IIR, adjust=False."""
    src  = ((df["high"] + df["low"]) / 2.0).to_numpy(dtype=float)
    high = pd.Series(df["high"]).rolling(period).max().to_numpy(dtype=float)
    low  = pd.Series(df["low"]).rolling(period).min().to_numpy(dtype=float)
    n = len(src)

    fish = np.full(n, np.nan)
    xp = yp = 0.0
    for i in range(period - 1, n):
        rng = max(high[i] - low[i], 1e-10)
        val = 2.0 * ((src[i] - low[i]) / rng) - 1.0
        x   = 0.33 * val + 0.67 * xp
        v   = max(-0.999, min(0.999, x))
        y   = 0.5 * np.log((1.0 + v) / (1.0 - v)) + 0.5 * yp
        fish[i] = y
        xp, yp  = x, y

    fs   = pd.Series(fish, index=df.index)
    sig  = fs.shift(1)
    bull = (fs > sig) & (fs.shift(1) <= sig.shift(1))
    bear = (fs < sig) & (fs.shift(1) >= sig.shift(1))

    return pd.DataFrame({
        "fisher":              fs,
        "fisher_signal":       sig,
        "fisher_bull":         bull.fillna(False),
        "fisher_bear":         bear.fillna(False),
        "fisher_extreme_bull": fs < -extreme,
        "fisher_extreme_bear": fs > extreme,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 5. Frost Engine (Range Filter) ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def frost_engine(df: pd.DataFrame, mode: str = "Normal") -> pd.DataFrame:
    """Range filter-based trend direction engine."""
    params = {
        "Normal":    (27, 1.5, 55, 1.3),
        "Sensitive": (27, 1.5, 55, 3.5),
        "Extreme":   (27, 1.5, 55, 4.2),
    }
    t1, m1, t2, sens = params.get(mode, params["Normal"])
    close = df["close"]

    def _smrng(x: pd.Series, t: int, m: float) -> pd.Series:
        return _ema(_ema(x.diff().abs().fillna(0), t) * m, t * 2 - 1)

    smrng1 = _smrng(close, t1, m1)
    smrng2 = _smrng(close, t2, sens)
    ca = close.to_numpy(dtype=float)
    sa = smrng1.to_numpy(dtype=float)
    n  = len(ca)
    filt = np.full(n, np.nan)
    filt[0] = ca[0]

    for i in range(1, n):
        if np.isnan(sa[i]):
            filt[i] = filt[i - 1]
            continue
        p, c, r = filt[i - 1], ca[i], sa[i]
        filt[i] = p + r if c > p + r else (p - r if c < p - r else p)

    uc = dc = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    up_cnt, dn_cnt = uc[0], uc[1]
    up_cnt = np.zeros(n, dtype=int)
    dn_cnt = np.zeros(n, dtype=int)
    for i in range(1, n):
        if filt[i] > filt[i - 1]:
            up_cnt[i] = up_cnt[i - 1] + 1
        elif filt[i] < filt[i - 1]:
            dn_cnt[i] = dn_cnt[i - 1] + 1
        else:
            up_cnt[i] = up_cnt[i - 1]
            dn_cnt[i] = dn_cnt[i - 1]

    fb = up_cnt > 0
    bb = dn_cnt > 0
    atr14 = _atr14(df)
    s1 = smrng1.to_numpy(dtype=float)
    s2 = smrng2.to_numpy(dtype=float)
    conf = np.zeros(n, dtype=float)

    for i in range(14, n):
        sc = 0.0
        if fb[i]:
            sc += 2.0
            if ca[i] > filt[i] + atr14[i] * 0.5: sc += 1.5
            if up_cnt[i] >= 3:                     sc += 1.5
            if not np.isnan(s1[i]) and not np.isnan(s2[i]) and s1[i] > s2[i]: sc += 1.5
        elif bb[i]:
            sc += 2.0
            if ca[i] < filt[i] - atr14[i] * 0.5: sc += 1.5
            if dn_cnt[i] >= 3:                     sc += 1.5
            if not np.isnan(s1[i]) and not np.isnan(s2[i]) and s1[i] < s2[i]: sc += 1.5
        if not np.isnan(s1[i]) and not np.isnan(s2[i]):
            avg = (s1[i] + s2[i]) / 2.0
            if avg > 0:
                sc += min(abs(s1[i] - s2[i]) / avg * 10.0, 3.5)
        conf[i] = min(sc, 10.0)

    fd = np.zeros(n, dtype=int)
    fd[fb] =  1
    fd[bb] = -1

    return pd.DataFrame({
        "frost_dir":    fd,
        "frost_conf":   conf,
        "frost_filter": pd.Series(filt, index=df.index),
        "frost_bull":   fb,
        "frost_bear":   bb,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 6. Signal Generator ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def signal_generator(
    df: pd.DataFrame,
    ms_df: pd.DataFrame,
    ob_df: pd.DataFrame,
    fvg_df: pd.DataFrame,
    rr_ratio: float = 2.0,
    ms_len: int = 5,
    htf_state: Optional[MTFState] = None,
) -> pd.DataFrame:
    """
    Exact port of Pine signal logic (f_calcConfluence + signal block).

    Signal Types:
        A: CHoCH event + trend aligned + MTF aligned + (in OB OR in FVG)
        B: BOS event + trend aligned + MTF aligned + in OB
        C: Sweep + MTF aligned + (in OB OR in FVG) + candle direction

    Confluence (0-5):
        +1 CTF trend aligned
        +1 MTF trend aligned
        +1 price in OB
        +1 price in FVG
        +1 sweep event (dnsweep for longs, upsweep for shorts)

    SL:
        OB SL takes priority over FVG SL
        Fallback: low/high ± ATR(14)

    Columns:
        sig_long / sig_short   bool
        sig_type               str   "A" | "B" | "C"
        sig_entry              float
        sig_sl                 float
        sig_tp                 float
        sig_confluence         int   0-5
        sig_reason             str
    """
    close  = df["close"].to_numpy(dtype=float)
    high   = df["high"].to_numpy(dtype=float)
    low    = df["low"].to_numpy(dtype=float)
    open_  = df["open"].to_numpy(dtype=float)
    atr    = _atr14(df)
    n = len(df)

    ms_txt  = ms_df["ms_txt"].to_numpy(dtype=object)
    trend   = ms_df["ms_trend"].to_numpy(dtype=int)
    dnsweep = ms_df["dnsweep"].to_numpy(dtype=bool)
    upsweep = ms_df["upsweep"].to_numpy(dtype=bool)

    in_bull_ob  = ob_df["in_bull_ob"].to_numpy(dtype=bool)
    in_bear_ob  = ob_df["in_bear_ob"].to_numpy(dtype=bool)
    ob_bull_sl  = ob_df["ob_bull_sl"].to_numpy(dtype=float)
    ob_bear_sl  = ob_df["ob_bear_sl"].to_numpy(dtype=float)

    in_bull_fvg  = fvg_df["in_bull_fvg"].to_numpy(dtype=bool)
    in_bear_fvg  = fvg_df["in_bear_fvg"].to_numpy(dtype=bool)
    fvg_bull_sl  = fvg_df["fvg_bull_sl"].to_numpy(dtype=float)
    fvg_bear_sl  = fvg_df["fvg_bear_sl"].to_numpy(dtype=float)

    htf_trend = htf_state.trend if htf_state is not None else 0
    # MTF filter: if no HTF data, allow both directions
    def _mtf_bull(i: int) -> bool:
        return htf_trend == 0 or htf_trend == 1

    def _mtf_bear(i: int) -> bool:
        return htf_trend == 0 or htf_trend == -1

    cooldown = ms_len * 2
    last_sig_bar = -cooldown - 1

    sig_long_arr  = np.zeros(n, dtype=bool)
    sig_short_arr = np.zeros(n, dtype=bool)
    sig_type_arr  = np.full(n, "", dtype=object)
    sig_entry_arr = np.full(n, np.nan)
    sig_sl_arr    = np.full(n, np.nan)
    sig_tp_arr    = np.full(n, np.nan)
    sig_conf_arr  = np.zeros(n, dtype=int)
    sig_reason_arr = np.full(n, "", dtype=object)

    for i in range(20, n):
        if (i - last_sig_bar) < cooldown:
            continue

        txt = str(ms_txt[i]) if ms_txt[i] else ""
        tr  = trend[i]
        mtf_bull = _mtf_bull(i)
        mtf_bear = _mtf_bear(i)

        ib_ob  = bool(in_bull_ob[i])
        ib_fvg = bool(in_bull_fvg[i])
        ir_ob  = bool(in_bear_ob[i])
        ir_fvg = bool(in_bear_fvg[i])

        # ── LONG ──────────────────────────────────────────────────────────────
        long_a = (txt == "choch" and tr == 1 and mtf_bull and (ib_ob or ib_fvg))
        long_b = (txt == "bos"   and tr == 1 and mtf_bull and ib_ob)
        long_c = (bool(dnsweep[i]) and mtf_bull and (ib_ob or ib_fvg) and close[i] > open_[i])

        if long_a or long_b or long_c:
            entry = close[i]
            # OB SL priority → FVG SL → fallback
            if not np.isnan(ob_bull_sl[i]):
                sl = ob_bull_sl[i]
            elif not np.isnan(fvg_bull_sl[i]):
                sl = fvg_bull_sl[i]
            else:
                sl = low[i] - atr[i]
            sl = min(sl, entry - atr[i] * 0.1)  # SL must be below entry
            tp = entry + abs(entry - sl) * rr_ratio

            # f_calcConfluence (isBull=True)
            conf = 0
            if tr == 1:        conf += 1
            if htf_trend == 1: conf += 1
            if ib_ob:          conf += 1
            if ib_fvg:         conf += 1
            if dnsweep[i]:     conf += 1

            stype = "A" if long_a else ("B" if long_b else "C")
            reasons = []
            if txt == "choch": reasons.append("CHoCH")
            if txt == "bos":   reasons.append("BOS")
            if dnsweep[i]:     reasons.append("Sweep")
            if ib_ob:          reasons.append("OB")
            if ib_fvg:         reasons.append("FVG")
            if htf_trend == 1: reasons.append("HTF aligned")

            sig_long_arr[i]   = True
            sig_type_arr[i]   = stype
            sig_entry_arr[i]  = entry
            sig_sl_arr[i]     = sl
            sig_tp_arr[i]     = tp
            sig_conf_arr[i]   = min(conf, 5)
            sig_reason_arr[i] = "LONG " + stype + " | " + " + ".join(reasons)
            last_sig_bar = i
            continue

        # ── SHORT ─────────────────────────────────────────────────────────────
        short_a = (txt == "choch" and tr == -1 and mtf_bear and (ir_ob or ir_fvg))
        short_b = (txt == "bos"   and tr == -1 and mtf_bear and ir_ob)
        short_c = (bool(upsweep[i]) and mtf_bear and (ir_ob or ir_fvg) and close[i] < open_[i])

        if short_a or short_b or short_c:
            entry = close[i]
            if not np.isnan(ob_bear_sl[i]):
                sl = ob_bear_sl[i]
            elif not np.isnan(fvg_bear_sl[i]):
                sl = fvg_bear_sl[i]
            else:
                sl = high[i] + atr[i]
            sl = max(sl, entry + atr[i] * 0.1)  # SL must be above entry
            tp = entry - abs(sl - entry) * rr_ratio

            conf = 0
            if tr == -1:        conf += 1
            if htf_trend == -1: conf += 1
            if ir_ob:           conf += 1
            if ir_fvg:          conf += 1
            if upsweep[i]:      conf += 1

            stype = "A" if short_a else ("B" if short_b else "C")
            reasons = []
            if txt == "choch": reasons.append("CHoCH")
            if txt == "bos":   reasons.append("BOS")
            if upsweep[i]:     reasons.append("Sweep")
            if ir_ob:          reasons.append("OB")
            if ir_fvg:         reasons.append("FVG")
            if htf_trend == -1: reasons.append("HTF aligned")

            sig_short_arr[i]  = True
            sig_type_arr[i]   = stype
            sig_entry_arr[i]  = entry
            sig_sl_arr[i]     = sl
            sig_tp_arr[i]     = tp
            sig_conf_arr[i]   = min(conf, 5)
            sig_reason_arr[i] = "SHORT " + stype + " | " + " + ".join(reasons)
            last_sig_bar = i

    return pd.DataFrame({
        "sig_long":      sig_long_arr,
        "sig_short":     sig_short_arr,
        "sig_type":      sig_type_arr,
        "sig_entry":     sig_entry_arr,
        "sig_sl":        sig_sl_arr,
        "sig_tp":        sig_tp_arr,
        "sig_confluence": sig_conf_arr,
        "sig_reason":    sig_reason_arr,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 7. Confluence Score ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def confluence_score(
    df: pd.DataFrame,
    ms_df: pd.DataFrame,
    ob_df: pd.DataFrame,
    fvg_df: pd.DataFrame,
    fisher_df: pd.DataFrame,
    frost_df: pd.DataFrame,
    htf_trend: int = 0,
) -> pd.Series:
    """Dashboard confluence score 0-10."""
    n = len(df)
    score = np.zeros(n, dtype=float)

    bos_bull   = ms_df["bos_bull"].to_numpy(dtype=bool)
    bos_bear   = ms_df["bos_bear"].to_numpy(dtype=bool)
    choch_bull = ms_df["choch_bull"].to_numpy(dtype=bool)
    choch_bear = ms_df["choch_bear"].to_numpy(dtype=bool)
    dnsweep    = ms_df["dnsweep"].to_numpy(dtype=bool)
    upsweep    = ms_df["upsweep"].to_numpy(dtype=bool)
    ms_trend   = ms_df["ms_trend"].to_numpy(dtype=int)

    in_bull_ob  = ob_df["in_bull_ob"].to_numpy(dtype=bool)
    in_bear_ob  = ob_df["in_bear_ob"].to_numpy(dtype=bool)
    in_bull_fvg = fvg_df["in_bull_fvg"].to_numpy(dtype=bool)
    in_bear_fvg = fvg_df["in_bear_fvg"].to_numpy(dtype=bool)

    fish_bull  = fisher_df["fisher_bull"].to_numpy(dtype=bool)
    fish_bear  = fisher_df["fisher_bear"].to_numpy(dtype=bool)
    fish_xbull = fisher_df["fisher_extreme_bull"].to_numpy(dtype=bool)
    fish_xbear = fisher_df["fisher_extreme_bear"].to_numpy(dtype=bool)

    frost_conf = frost_df["frost_conf"].to_numpy(dtype=float)
    frost_bull = frost_df["frost_bull"].to_numpy(dtype=bool)
    frost_bear = frost_df["frost_bear"].to_numpy(dtype=bool)

    for i in range(10, n):
        s = 0.0
        slc = slice(max(0, i - 3), i + 1)
        tr = ms_trend[i]

        if (bos_bull[slc].any() or choch_bull[slc].any() or
                bos_bear[slc].any() or choch_bear[slc].any()):
            s += 1.5
        if dnsweep[slc].any() or upsweep[slc].any():
            s += 0.5

        if tr >= 0 and in_bull_ob[i]:  s += 1.0
        if tr <= 0 and in_bear_ob[i]:  s += 1.0
        if tr >= 0 and in_bull_fvg[i]: s += 1.0
        if tr <= 0 and in_bear_fvg[i]: s += 1.0

        if htf_trend != 0 and tr == htf_trend: s += 1.0
        if frost_conf[i] >= 6.0:               s += 1.5
        if fish_bull[i] or fish_xbull[i] or fish_bear[i] or fish_xbear[i]: s += 1.0
        if (frost_bull[i] and tr > 0) or (frost_bear[i] and tr < 0):       s += 1.0

        score[i] = min(s, 10.0)

    return pd.Series(score, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 8. Full Analysis ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def smc_analysis(
    df: pd.DataFrame,
    ms_len: int = 5,
    ob_len: int = 5,
    ob_mitigation: str = "Close",
    fvg_threshold: float = 0.0,
    fvg_mitigation: str = "Close",
    fisher_period: int = 14,
    fisher_extreme: float = 2.5,
    frost_mode: str = "Normal",
    rr_ratio: float = 2.0,
    htf_state: Optional[MTFState] = None,
) -> pd.DataFrame:
    """
    Full BUDA MONEY CONCEPTS analysis — all components combined.

    Matches Pine Script indicator inputs:
        ms_len         → mslen (default 5)
        ob_len         → len (OB construction length, default 5)
        ob_mitigation  → obmiti ('Close'|'Wick'|'Avg')
        fvg_threshold  → fvgthresh (0 = show all)
        fvg_mitigation → fvg_src ('Close'|'Wick'|'Avg')
        rr_ratio       → riskReward (default 2.0)
    """
    ms_df     = market_structure(df, ms_len=ms_len)
    ob_df     = order_blocks(df, ms_df, ob_len=ob_len, mitigation=ob_mitigation)
    fvg_df    = fair_value_gaps(df, fvg_threshold=fvg_threshold, mitigation=fvg_mitigation)
    fisher_df = fisher_transform(df, period=fisher_period, extreme=fisher_extreme)
    frost_df  = frost_engine(df, mode=frost_mode)
    sig_df    = signal_generator(df, ms_df, ob_df, fvg_df,
                                  rr_ratio=rr_ratio, ms_len=ms_len, htf_state=htf_state)
    conf      = confluence_score(df, ms_df, ob_df, fvg_df, fisher_df, frost_df,
                                  htf_trend=htf_state.trend if htf_state else 0)

    result = pd.concat([ms_df, ob_df, fvg_df, fisher_df, frost_df, sig_df], axis=1)
    result["confluence"] = conf
    return result


def _build_mtf_state(result: pd.DataFrame) -> MTFState:
    last = result.iloc[-1]

    def _f(col: str) -> float:
        v = last.get(col, np.nan)
        return float(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else np.nan

    return MTFState(
        trend=int(last.get("ms_trend", 0)),
        bos_level=_f("last_bos_level"),
        has_bull_ob=not np.isnan(_f("ob_bull_top")),
        has_bear_ob=not np.isnan(_f("ob_bear_top")),
        has_bull_fvg=not np.isnan(_f("fvg_bull_top")),
        has_bear_fvg=not np.isnan(_f("fvg_bear_top")),
    )


def _tf_minutes(tf: str) -> int:
    unit_map = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    for suffix, mult in unit_map.items():
        if tf.lower().strip().endswith(suffix):
            return int(tf.lower().strip()[:-1]) * mult
    return 0


def smc_all_timeframes(
    dfs: dict[str, pd.DataFrame],
    ms_len: int = 5,
    ob_len: int = 5,
    fisher_period: int = 14,
    fisher_extreme: float = 2.5,
    frost_mode: str = "Normal",
    rr_ratio: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """HTF→LTF cascade: each TF's final MTFState feeds the next lower TF."""
    results: dict[str, pd.DataFrame] = {}
    tf_order = sorted(dfs.keys(), key=lambda t: _tf_minutes(t), reverse=True)
    htf_state: Optional[MTFState] = None

    for tf in tf_order:
        df = dfs.get(tf)
        if df is None or df.empty:
            continue
        res = smc_analysis(
            df, ms_len=ms_len, ob_len=ob_len,
            fisher_period=fisher_period, fisher_extreme=fisher_extreme,
            frost_mode=frost_mode, rr_ratio=rr_ratio, htf_state=htf_state,
        )
        results[tf] = res
        htf_state = _build_mtf_state(res)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 9. Summary for Prompt Builder ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def smc_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Last-bar values per TF — used by prompt_builder."""
    summary: dict[str, dict] = {}

    for tf, df in results.items():
        if df is None or df.empty:
            continue
        last = df.iloc[-1]

        def _f(col: str) -> Optional[float]:
            v = last.get(col, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return float(v)

        def _zone(top_col: str, btm_col: str) -> str:
            t, b = _f(top_col), _f(btm_col)
            return f"[{b:,.2f}–{t:,.2f}]" if (t and b) else "—"

        ob_bull_z = _zone("ob_bull_top", "ob_bull_btm")
        ob_bear_z = _zone("ob_bear_top", "ob_bear_btm")
        fvg_bull  = _zone("fvg_bull_top", "fvg_bull_btm")
        fvg_bear  = _zone("fvg_bear_top", "fvg_bear_btm")

        fv = _f("fisher") or 0.0
        fs = f"{fv:+.2f}"
        if last.get("fisher_extreme_bull", False): fs += " (Extreme Bull)"
        elif last.get("fisher_extreme_bear", False): fs += " (Extreme Bear)"
        elif last.get("fisher_bull", False):  fs += " (CrossUp)"
        elif last.get("fisher_bear", False):  fs += " (CrossDn)"

        fd = int(last.get("frost_dir", 0))
        fc = _f("frost_conf") or 0.0
        frost_s = "Bull" if fd > 0 else ("Bear" if fd < 0 else "Neutral")

        sig_long  = bool(last.get("sig_long", False))
        sig_short = bool(last.get("sig_short", False))
        signal_dict: dict = {}
        if sig_long or sig_short:
            signal_dict = {
                "direction":          "LONG" if sig_long else "SHORT",
                "type":               str(last.get("sig_type", "")),
                "entry":              _f("sig_entry"),
                "sl":                 _f("sig_sl"),
                "tp":                 _f("sig_tp"),
                "confluence_factors": int(last.get("sig_confluence", 0)),
                "reason":             str(last.get("sig_reason", "")),
            }

        summary[tf] = {
            "last_event":  str(last.get("last_event", "—")),
            "bos_level":   _f("last_bos_level"),
            "ms_trend":    int(last.get("ms_trend", 0)),
            "ms_txt":      str(last.get("ms_txt", "")),
            "ob_bull":     ob_bull_z,
            "ob_bear":     ob_bear_z,
            "in_bull_ob":  bool(last.get("in_bull_ob", False)),
            "in_bear_ob":  bool(last.get("in_bear_ob", False)),
            "fvg_bull":    fvg_bull,
            "fvg_bear":    fvg_bear,
            "in_bull_fvg": bool(last.get("in_bull_fvg", False)),
            "in_bear_fvg": bool(last.get("in_bear_fvg", False)),
            "dnsweep":     bool(last.get("dnsweep", False)),
            "upsweep":     bool(last.get("upsweep", False)),
            "fisher":      fs,
            "frost":       f"{frost_s} (conf {fc:.1f})",
            "confluence":  round(_f("confluence") or 0.0, 1),
            "signal":      signal_dict,
        }

    return summary
