"""
Ehlers CyberCycle — Python port of cybercyclev3.pine (core engine).

Adaptive alpha methods supported: Manual, Homodyne, Autocorrelation.
(MAMA and Kalman skipped — manual + homodyne cover 95% of use cases.)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ema_recursive(src: np.ndarray, length: int) -> np.ndarray:
    k = 2.0 / (length + 1)
    out = np.zeros_like(src, dtype=float)
    for i in range(len(src)):
        if np.isnan(src[i]):
            out[i] = out[i - 1] if i > 0 else 0.0
        else:
            prev = out[i - 1] if i > 0 else 0.0
            out[i] = k * src[i] + (1 - k) * prev
    return out


# ─── Alpha: Homodyne Discriminator ────────────────────────────────────────────

def _alpha_homodyne(
    src: np.ndarray,
    min_period: float = 3.0,
    max_period: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(src)
    smooth = np.zeros(n)
    det = np.zeros(n)
    Q1 = np.zeros(n)
    jI = np.zeros(n)
    jQ = np.zeros(n)
    I2 = np.zeros(n)
    Q2 = np.zeros(n)
    Re = np.zeros(n)
    Im = np.zeros(n)
    period = np.full(n, 15.0)
    smooth_period = np.full(n, 15.0)
    alpha = np.full(n, 2.0 / (15.0 + 1.0))

    def _nz(arr, i):
        return arr[i] if i >= 0 and not np.isnan(arr[i]) else 0.0

    for i in range(n):
        if i < 7 or np.isnan(src[i]):
            continue
        smooth[i] = (4.0 * src[i] + 3.0 * _nz(src, i - 1) + 2.0 * _nz(src, i - 2) + _nz(src, i - 3)) / 10.0
        adj = 0.075 * period[i - 1] + 0.54

        det[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2] - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6]) * adj
        Q1[i] = (0.0962 * det[i] + 0.5769 * det[i - 2] - 0.5769 * det[i - 4] - 0.0962 * det[i - 6]) * adj
        I1 = det[i - 3]

        jI[i] = (0.0962 * I1 + 0.5769 * det[i - 5] - 0.5769 * (det[i - 7] if i >= 7 else 0.0) - 0.0962 * (det[i - 9] if i >= 9 else 0.0)) * adj
        jQ[i] = (0.0962 * Q1[i] + 0.5769 * Q1[i - 2] - 0.5769 * Q1[i - 4] - 0.0962 * Q1[i - 6]) * adj

        I2[i] = 0.2 * (I1 - jQ[i]) + 0.8 * I2[i - 1]
        Q2[i] = 0.2 * (Q1[i] + jI[i]) + 0.8 * Q2[i - 1]

        Re[i] = 0.2 * (I2[i] * I2[i - 1] + Q2[i] * Q2[i - 1]) + 0.8 * Re[i - 1]
        Im[i] = 0.2 * (I2[i] * Q2[i - 1] - Q2[i] * I2[i - 1]) + 0.8 * Im[i - 1]

        if abs(Im[i]) > 1e-10 and abs(Re[i]) > 1e-10:
            phase_adv = math.atan(Im[i] / Re[i])
        else:
            phase_adv = 0.0

        if phase_adv > 0.001:
            raw_per = 2.0 * math.pi / phase_adv
        else:
            raw_per = period[i - 1]

        raw_per = max(raw_per, 0.67 * period[i - 1])
        raw_per = min(raw_per, 1.5 * period[i - 1])
        raw_per = max(min_period, min(max_period, raw_per))
        period[i] = 0.2 * raw_per + 0.8 * period[i - 1]

        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]
        smooth_period[i] = max(min_period, min(max_period, smooth_period[i]))

        alpha[i] = 2.0 / (smooth_period[i] + 1.0)

    return alpha, smooth_period


# ─── Alpha: Autocorrelation Periodogram ───────────────────────────────────────

def _alpha_autocorr(
    src: np.ndarray,
    min_period: int = 6,
    max_period: int = 48,
    avg_length: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(src)
    a1 = (0.707 * 2.0 * math.pi) / max_period
    alpha_hp = (math.cos(a1) + math.sin(a1) - 1.0) / math.cos(a1)
    a1ss = math.exp(-1.414 * math.pi / min_period)
    b1ss = 2.0 * a1ss * math.cos(1.414 * math.pi / min_period)
    c2 = b1ss
    c3 = -a1ss * a1ss
    c1 = 1.0 - c2 - c3

    hp = np.zeros(n)
    filt = np.zeros(n)
    best_period = np.full(n, 15.0)
    alpha_arr = np.full(n, 2.0 / 16.0)

    for i in range(n):
        if i < 2 or np.isnan(src[i]):
            continue
        hp[i] = (
            (1.0 - alpha_hp / 2.0) ** 2 * (src[i] - 2.0 * src[i - 1] + src[i - 2])
            + 2.0 * (1.0 - alpha_hp) * hp[i - 1]
            - (1.0 - alpha_hp) ** 2 * hp[i - 2]
        )
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2.0 + c2 * filt[i - 1] + c3 * filt[i - 2]

    step = max(1, (max_period - min_period) // 10)

    for i in range(n):
        if i < max_period * avg_length or np.isnan(src[i]):
            continue
        max_corr = 0.0
        best_p = best_period[i - 1]
        for p in range(min_period, max_period + 1, step):
            cnt = avg_length * p
            cnt_use = min(cnt - 1, 199)
            if i - (cnt_use + p) < 0:
                continue
            sx = sy = sxx = syy = sxy = 0.0
            for j in range(cnt_use + 1):
                x = filt[i - j]
                y = filt[i - j - p]
                sx += x; sy += y
                sxx += x * x; syy += y * y
                sxy += x * y
            denom = (cnt * sxx - sx * sx) * (cnt * syy - sy * sy)
            corr = (cnt * sxy - sx * sy) / math.sqrt(denom) if denom > 0 else 0.0
            if corr > max_corr:
                max_corr = corr
                best_p = float(p)
        best_period[i] = 0.25 * best_p + 0.75 * best_period[i - 1]
        best_period[i] = max(min_period, min(max_period, best_period[i]))
        alpha_arr[i] = 2.0 / (best_period[i] + 1.0)

    return alpha_arr, best_period


# ─── CyberCycle Core ──────────────────────────────────────────────────────────

def cybercycle(
    df: pd.DataFrame,
    alpha_method: str = "manual",
    manual_alpha: float = 0.42,
    itrend_alpha: float = 0.09,
    trigger_ema: int = 9,
    ob_level: float = 1.5,
    os_level: float = -1.5,
    fisher_lookback: int = 10,
    hd_min_period: float = 3.0,
    hd_max_period: float = 40.0,
    ac_min_period: int = 6,
    ac_max_period: int = 48,
) -> pd.DataFrame:
    """
    Adaptive Ehlers CyberCycle.

    alpha_method: 'manual' | 'homodyne' | 'autocorr'

    Outputs:
        cycle             raw cyber cycle value (centered around 0)
        trigger           EMA of cycle
        cycle_norm        Fisher-normalized [-1, +1] (clipped)
        cycle_osc         0-100 normalized
        cycle_trend       +1 / -1 / 0
        is_ob, is_os      bool
        bull_cross, bear_cross   crossover/under cycle vs trigger
        itrend            adaptive trend value
        itrend_bull, itrend_bear bool
        dominant_period   adaptive period (only meaningful if non-manual)
        cycle_conf        0-10 confidence score
    """
    src = ((df["high"] + df["low"]) / 2.0).to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(src)

    # ── Alpha ────────────────────────────────────────────────────────────────
    if alpha_method == "homodyne":
        alpha_arr, period_arr = _alpha_homodyne(src, hd_min_period, hd_max_period)
    elif alpha_method == "autocorr":
        alpha_arr, period_arr = _alpha_autocorr(src, ac_min_period, ac_max_period)
    else:  # manual
        alpha_arr = np.full(n, manual_alpha)
        period_arr = np.full(n, (2.0 / manual_alpha) - 1.0)

    # ── Smooth filter ────────────────────────────────────────────────────────
    smooth = np.zeros(n)
    cycle = np.zeros(n)
    for i in range(n):
        if np.isnan(src[i]):
            continue
        if i < 3:
            smooth[i] = src[i]
        else:
            smooth[i] = (src[i] + 2.0 * src[i - 1] + 2.0 * src[i - 2] + src[i - 3]) / 6.0

        if i < 7:
            cycle[i] = (src[i] - 2.0 * src[i - 1] + src[i - 2]) / 4.0 if i >= 2 else 0.0
        else:
            a = alpha_arr[i]
            a1 = (1.0 - 0.5 * a) ** 2
            a2 = 2.0 * (1.0 - a)
            a3 = (1.0 - a) ** 2
            cycle[i] = (
                a1 * (smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2])
                + a2 * cycle[i - 1]
                - a3 * cycle[i - 2]
            )

    # ── Trigger EMA ──────────────────────────────────────────────────────────
    trigger = _ema_recursive(cycle, trigger_ema)

    # ── iTrend recurrence ────────────────────────────────────────────────────
    itrend = np.zeros(n)
    a = itrend_alpha
    for i in range(n):
        if i < 2:
            itrend[i] = close[i] if not np.isnan(close[i]) else 0.0
            continue
        itrend[i] = (
            (a - a * a / 4.0) * close[i]
            + 0.5 * a * a * close[i - 1]
            - (a - 0.75 * a * a) * close[i - 2]
            + 2.0 * (1.0 - a) * itrend[i - 1]
            - (1.0 - a) ** 2 * itrend[i - 2]
        )

    itrend_bull = np.zeros(n, dtype=bool)
    itrend_bear = np.zeros(n, dtype=bool)
    for i in range(2, n):
        itrend_bull[i] = itrend[i] > itrend[i - 2]
        itrend_bear[i] = itrend[i] < itrend[i - 2]

    # ── Fisher normalization (rolling 10) ────────────────────────────────────
    cycle_s = pd.Series(cycle)
    fH = cycle_s.rolling(fisher_lookback).max().to_numpy()
    fL = cycle_s.rolling(fisher_lookback).min().to_numpy()
    rng = fH - fL
    safe = np.where(rng > 1e-10, rng, 1.0)
    cycle_norm = 2.0 * ((cycle - fL) / safe - 0.5)
    cycle_norm = np.clip(cycle_norm, -1.0, 1.0)

    # 0-100 osc
    cycle_osc = (cycle_norm + 1.0) * 50.0

    # ── Trend / OB / OS ──────────────────────────────────────────────────────
    cycle_trend = np.zeros(n, dtype=int)
    bull_mask = (cycle > trigger) & itrend_bull
    bear_mask = (cycle < trigger) & itrend_bear
    cycle_trend[bull_mask] = 1
    cycle_trend[bear_mask] = -1

    is_ob = cycle > ob_level
    is_os = cycle < os_level

    # ── Crosses ──────────────────────────────────────────────────────────────
    bull_cross = np.zeros(n, dtype=bool)
    bear_cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        prev_ge = cycle[i - 1] >= trigger[i - 1]
        curr_ge = cycle[i] >= trigger[i]
        bull_cross[i] = (not prev_ge) and curr_ge
        bear_cross[i] = prev_ge and (not curr_ge)

    # ── Confidence (0-10) ────────────────────────────────────────────────────
    # Based on cycle magnitude relative to its 50-bar stdev
    cycle_std = pd.Series(cycle).rolling(50).std(ddof=0).to_numpy()
    safe_std = np.where(cycle_std > 1e-10, cycle_std, 1.0)
    z = np.abs(cycle) / safe_std
    cycle_conf = np.clip(z * 3.0, 0.0, 10.0)

    return pd.DataFrame({
        "cycle":           cycle,
        "trigger":         trigger,
        "cycle_norm":      cycle_norm,
        "cycle_osc":       cycle_osc,
        "cycle_trend":     cycle_trend,
        "is_ob":           is_ob,
        "is_os":           is_os,
        "bull_cross":      bull_cross,
        "bear_cross":      bear_cross,
        "itrend":          itrend,
        "itrend_bull":     itrend_bull,
        "itrend_bear":     itrend_bear,
        "dominant_period": period_arr,
        "cycle_conf":      cycle_conf,
    }, index=df.index)


# ─── MTF wrappers ─────────────────────────────────────────────────────────────

def cybercycle_all_timeframes(
    dfs: dict[str, pd.DataFrame],
    alpha_method: str = "manual",
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        out[tf] = cybercycle(df, alpha_method=alpha_method)
    return out


def cybercycle_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for tf, df in results.items():
        if df is None or df.empty:
            continue
        valid = df.dropna(subset=["cycle"])
        if valid.empty:
            continue
        last = valid.iloc[-1]
        cyc = float(last["cycle"])
        trig = float(last["trigger"])
        trend = int(last["cycle_trend"])

        if trend > 0:
            tstr = "↑ Bull"
        elif trend < 0:
            tstr = "↓ Bear"
        else:
            tstr = "→ Neutral"

        zone = "Neutral"
        if bool(last["is_ob"]):
            zone = "OverBought"
        elif bool(last["is_os"]):
            zone = "OverSold"

        # signal in last 3 bars
        tail = df.tail(3)
        sigs = []
        if tail["bull_cross"].any():
            sigs.append("BullCross")
        if tail["bear_cross"].any():
            sigs.append("BearCross")

        summary[tf] = {
            "cycle":     round(cyc, 4),
            "trigger":   round(trig, 4),
            "cycle_osc": round(float(last["cycle_osc"]), 1),
            "trend":     tstr,
            "zone":      zone,
            "period":    round(float(last["dominant_period"]), 1),
            "conf":      round(float(last["cycle_conf"]), 1),
            "signal":    ", ".join(sigs) if sigs else "—",
        }
    return summary
