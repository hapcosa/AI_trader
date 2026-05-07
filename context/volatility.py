"""
Volatility context — ATR percentile, realized volatility, regime classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def realized_volatility(close: pd.Series, length: int = 30, annualize_factor: float = 365.0) -> pd.Series:
    """Annualized realized vol from log returns."""
    ret = np.log(close / close.shift(1))
    return ret.rolling(length).std(ddof=0) * np.sqrt(annualize_factor)


def atr_percentile(df: pd.DataFrame, atr_len: int = 14, lookback: int = 90) -> pd.Series:
    """ATR rank within its own last `lookback` bars (0-100)."""
    atr = _atr(df, atr_len)
    return atr.rolling(lookback).apply(
        lambda x: (np.sum(x[-1] >= x) - 1) / max(len(x) - 1, 1) * 100.0,
        raw=True,
    )


def volatility_context(
    df: pd.DataFrame,
    atr_len: int = 14,
    rv_len: int = 30,
    rank_lookback: int = 90,
) -> dict:
    """
    Compute volatility regime for the given DF.

    Returns dict with:
        atr           latest ATR
        atr_pct       latest ATR percentile (0-100)
        realized_vol  latest RV
        regime        'expansion' | 'contraction' | 'normal'
    """
    if df is None or df.empty:
        return {"regime": "n/a"}

    atr = _atr(df, atr_len)
    rv = realized_volatility(df["close"], rv_len)
    pct = atr_percentile(df, atr_len, rank_lookback)

    last_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else float("nan")
    last_rv = float(rv.iloc[-1]) if not pd.isna(rv.iloc[-1]) else float("nan")
    last_pct = float(pct.iloc[-1]) if not pd.isna(pct.iloc[-1]) else float("nan")

    if not np.isnan(last_pct):
        if last_pct >= 75:
            regime = "expansion"
        elif last_pct <= 25:
            regime = "contraction"
        else:
            regime = "normal"
    else:
        regime = "n/a"

    return {
        "atr":           round(last_atr, 4) if not np.isnan(last_atr) else None,
        "atr_pct":       round(last_pct, 1) if not np.isnan(last_pct) else None,
        "realized_vol":  round(last_rv, 4) if not np.isnan(last_rv) else None,
        "regime":        regime,
    }


def volatility_all_timeframes(dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
    return {tf: volatility_context(df) for tf, df in dfs.items() if df is not None and not df.empty}
