"""Macro snapshot for the user-facing /indicators Macro tab (W3-2).

Aggregates slow-moving market context: Crypto Fear & Greed (alternative.me),
a set of macro tickers via yfinance (DXY, S&P500, Gold, Nasdaq, VIX, US10Y,
ETH/BTC), and current dominance values (USDT.D/BTC.D/OTHERS.D) from the
dominance SQLite. Every fetch degrades gracefully to None so a single dead
feed never breaks the tab.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from pineforge_ai.context.correlations import _fetch_daily, _trend_label

# Macro tickers (yfinance). Labels are display-friendly; keys are stable.
MARKET_SYMBOLS: dict[str, tuple[str, str]] = {
    # key:        (yfinance ticker, label)
    "DXY":      ("DX-Y.NYB", "Dólar (DXY)"),
    "SP500":    ("^GSPC",    "S&P 500"),
    "NASDAQ":   ("^IXIC",    "Nasdaq"),
    "GOLD":     ("GC=F",     "Oro"),
    "VIX":      ("^VIX",     "VIX"),
    "US10Y":    ("^TNX",     "US 10Y"),
    "ETH_BTC":  ("ETH-BTC",  "ETH/BTC"),
}

FNG_URL = "https://api.alternative.me/fng/?limit=1"
_HEADERS = {"User-Agent": "PineForge-AI/3.0"}

DOMINANCE_KEYS = ("USDT.D", "BTC.D", "OTHERS.D")


def _fetch_fear_greed() -> dict[str, Any] | None:
    """Crypto Fear & Greed index (0-100) from alternative.me."""
    try:
        resp = requests.get(FNG_URL, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        d = resp.json()["data"][0]
        return {
            "value": int(d["value"]),
            "classification": str(d.get("value_classification") or ""),
        }
    except Exception:
        return None


def _fetch_markets() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, (ticker, label) in MARKET_SYMBOLS.items():
        df = _fetch_daily(ticker, days=10)
        if df is None or df.empty or "close" not in df.columns:
            out.append({"key": key, "label": label, "ticker": ticker,
                        "close": None, "change_1d_pct": None, "trend_3d": "n/a"})
            continue
        close = df["close"]
        last = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else last
        change = (last / prev - 1.0) * 100.0 if prev > 0 else 0.0
        out.append({
            "key": key, "label": label, "ticker": ticker,
            "close": round(last, 4),
            "change_1d_pct": round(change, 3),
            "trend_3d": _trend_label(close.pct_change(), n_bars=3),
        })
    return out


def _dominance_db_path() -> Path:
    from pineforge_ai.usdt_dominance import reader

    db = os.environ.get("USDT_DOMINANCE_DB")
    return Path(db) if db else reader.DB_PATH


def _fetch_dominance_now() -> list[dict[str, Any]]:
    from pineforge_ai.usdt_dominance import reader

    db_path = _dominance_db_path()
    out: list[dict[str, Any]] = []
    for sym in DOMINANCE_KEYS:
        try:
            val = reader.get_current_value(db_path=db_path, symbol=sym)
        except Exception:
            val = None
        out.append({"key": sym, "value": round(val, 3) if val is not None else None})
    return out


def build_macro_summary() -> dict[str, Any]:
    """Return ``{fear_greed, markets, dominance}`` for the Macro tab."""
    return {
        "fear_greed": _fetch_fear_greed(),
        "markets": _fetch_markets(),
        "dominance": _fetch_dominance_now(),
    }


__all__ = ["build_macro_summary", "MARKET_SYMBOLS", "DOMINANCE_KEYS"]
