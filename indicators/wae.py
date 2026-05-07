"""
WAE (Waddah Attar Explosion) + Choppiness Index — trend quality filter.

Source: SMCELITE.pine / SMCLITENOcc.pine
WAE detects momentum explosions vs fades.
Choppiness Index distinguishes trending vs ranging markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


# ─── WAE — Waddah Attar Explosion ─────────────────────────────────────────────

def wae(
    df: pd.DataFrame,
    fast_len: int = 20,
    slow_len: int = 40,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    sensitivity: float = 150.0,
    dead_zone_len: int = 100,
) -> pd.DataFrame:
    """
    Waddah Attar Explosion.

    trend_up   = (MACD_now - MACD_prev) * sensitivity   if positive
    trend_dn   = (MACD_now - MACD_prev) * sensitivity   if negative (abs)
    explosion  = BB_upper - BB_lower
    dead_zone  = ATR(dead_zone_len) * 3.7

    Signal:
        bull explosion: trend_up > explosion AND trend_up > dead_zone
        bear explosion: trend_dn > explosion AND trend_dn > dead_zone
    """
    close = df["close"]

    # MACD
    macd_fast = _ema(close, fast_len)
    macd_slow = _ema(close, slow_len)
    macd = macd_fast - macd_slow
    macd_diff = (macd - macd.shift(1)) * sensitivity

    trend_up = macd_diff.where(macd_diff >= 0, other=0.0)
    trend_dn = (-macd_diff).where(macd_diff < 0, other=0.0)

    # Bollinger Bands
    sma = close.rolling(bb_len).mean()
    std = close.rolling(bb_len).std(ddof=0)
    bb_upper = sma + std * bb_mult
    bb_lower = sma - std * bb_mult
    explosion = bb_upper - bb_lower

    # Dead zone (3.7 × ATR per default Waddah Attar)
    dead_zone = _atr(df, dead_zone_len) * 3.7

    bull_explosion = (trend_up > explosion) & (trend_up > dead_zone)
    bear_explosion = (trend_dn > explosion) & (trend_dn > dead_zone)

    return pd.DataFrame({
        "wae_trend_up":   trend_up,
        "wae_trend_dn":   trend_dn,
        "wae_explosion":  explosion,
        "wae_dead_zone":  dead_zone,
        "wae_bull_explo": bull_explosion,
        "wae_bear_explo": bear_explosion,
    }, index=df.index)


# ─── Choppiness Index ─────────────────────────────────────────────────────────

def choppiness(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Choppiness Index = 100 * log10( sum(ATR_1, n) / (max(High,n) - min(Low,n)) ) / log10(n)

    Range 0-100:
        < 38.2  -> trending (strong)
        > 61.8  -> choppy / ranging
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    sum_tr = tr.rolling(length).sum()
    range_n = high.rolling(length).max() - low.rolling(length).min()
    safe_range = range_n.where(range_n > 0, other=np.nan)

    chop = 100.0 * np.log10(sum_tr / safe_range) / np.log10(length)
    return chop


# ─── Combined Trend Quality ───────────────────────────────────────────────────

def trend_quality(
    df: pd.DataFrame,
    wae_kwargs: dict | None = None,
    chop_len: int = 14,
) -> pd.DataFrame:
    wae_kwargs = wae_kwargs or {}
    w = wae(df, **wae_kwargs)
    chop = choppiness(df, length=chop_len)

    is_trending = chop < 38.2
    is_choppy = chop > 61.8

    return pd.DataFrame({
        **{c: w[c] for c in w.columns},
        "chop_idx":     chop,
        "is_trending":  is_trending,
        "is_choppy":    is_choppy,
    }, index=df.index)


# ─── MTF Wrappers ─────────────────────────────────────────────────────────────

def trend_quality_all_timeframes(
    dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            continue
        out[tf] = trend_quality(df)
    return out


def trend_quality_summary(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Per-TF summary: WAE direction + chop status.
    """
    summary: dict[str, dict] = {}
    for tf, df in results.items():
        if df is None or df.empty:
            continue
        valid = df.dropna(subset=["chop_idx"])
        if valid.empty:
            continue
        last = valid.iloc[-1]

        if bool(last["wae_bull_explo"]):
            wae_state = "↑ Bull Explosion"
        elif bool(last["wae_bear_explo"]):
            wae_state = "↓ Bear Explosion"
        elif last["wae_trend_up"] > last["wae_trend_dn"]:
            wae_state = "Bull Fade"
        else:
            wae_state = "Bear Fade"

        chop_val = float(last["chop_idx"])
        if chop_val < 38.2:
            regime = "Trending"
        elif chop_val > 61.8:
            regime = "Choppy"
        else:
            regime = "Mixed"

        summary[tf] = {
            "wae_state": wae_state,
            "chop_idx":  round(chop_val, 1),
            "regime":    regime,
            "trend_up":  round(float(last["wae_trend_up"]), 2),
            "trend_dn":  round(float(last["wae_trend_dn"]), 2),
            "explosion": round(float(last["wae_explosion"]), 2),
        }
    return summary
