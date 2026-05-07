"""
Market correlations — DXY, S&P 500, Gold, BTC dominance.

Pulls daily data via yfinance for context macro of the analyzed symbol.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# Symbols always tracked
CORRELATION_SYMBOLS: dict[str, str] = {
    "DXY":     "DX-Y.NYB",     # US Dollar Index
    "SP500":   "^GSPC",        # S&P 500
    "GOLD":    "GC=F",         # Gold Futures
    "BTC_USD": "BTC-USD",      # BTC reference
    "ETH_BTC": "ETH-BTC",      # ETH/BTC dominance proxy
}


def _trend_label(returns: pd.Series, n_bars: int = 3) -> str:
    """Classify last N bars as up / down / flat based on cumulative return."""
    tail = returns.tail(n_bars).dropna()
    if tail.empty:
        return "flat"
    cum = (1.0 + tail).prod() - 1.0
    if cum > 0.005:
        return "up"
    if cum < -0.005:
        return "down"
    return "flat"


def _fetch_daily(symbol: str, days: int = 30) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None

    start = datetime.now(tz=timezone.utc) - timedelta(days=days + 5)
    try:
        t = yf.Ticker(symbol)
        df = t.history(start=start.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_correlations(
    skip_btc_for: str | None = None,
) -> dict[str, dict]:
    """
    Fetch daily snapshots of correlation symbols.
    Returns dict per symbol with: close, change_1d_pct, trend_3d.
    """
    out: dict[str, dict] = {}
    for label, ticker in CORRELATION_SYMBOLS.items():
        # Skip BTC_USD if symbol IS already BTC
        if skip_btc_for and "BTC" in skip_btc_for.upper() and label == "BTC_USD":
            continue
        df = _fetch_daily(ticker, days=10)
        if df is None or df.empty or "close" not in df.columns:
            out[label] = {"close": None, "change_1d_pct": None, "trend_3d": "n/a"}
            continue

        close = df["close"]
        last = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else last
        change_1d = (last / prev - 1.0) * 100.0 if prev > 0 else 0.0
        ret = close.pct_change()
        trend = _trend_label(ret, n_bars=3)

        out[label] = {
            "ticker":        ticker,
            "close":         round(last, 4),
            "change_1d_pct": round(change_1d, 3),
            "trend_3d":      trend,
        }
    return out


def correlation_summary(corrs: dict[str, dict]) -> dict[str, str]:
    """Convert raw correlation data into bull/bear/neutral labels."""
    summary: dict[str, str] = {}
    for k, v in corrs.items():
        trend = v.get("trend_3d", "flat")
        if trend == "up":
            summary[k] = "bullish"
        elif trend == "down":
            summary[k] = "bearish"
        else:
            summary[k] = "neutral"
    return summary
