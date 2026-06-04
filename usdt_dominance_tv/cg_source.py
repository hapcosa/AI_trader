"""CoinGecko fallback for dominance percentages.

Used only when TradingView is unavailable. CoinGecko's ``/global`` exposes
``market_cap_percentage`` for the major caps, so it can back-fill USDT.D and
BTC.D as point values. It does NOT expose an OTHERS.D series, so that one has
no CoinGecko fallback.
"""
from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)

COINGECKO_URL = "https://api.coingecko.com/api/v3/global"
HEADERS = {"User-Agent": "PineForge-AI/2.0"}

# Canonical series symbol -> CoinGecko market_cap_percentage key.
CG_KEY_BY_SYMBOL: dict[str, str] = {
    "USDT.D": "usdt",
    "BTC.D": "btc",
}


def fetch_percentages() -> dict[str, float]:
    """Return available dominance percentages keyed by canonical symbol.

    Empty dict on any failure. Only symbols CoinGecko exposes are returned
    (USDT.D, BTC.D); OTHERS.D is absent by design.
    """
    try:
        resp = requests.get(COINGECKO_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        pct = resp.json()["data"]["market_cap_percentage"]
    except Exception as e:
        log.warning("CoinGecko fetch failed: %s", e)
        return {}

    out: dict[str, float] = {}
    for symbol, cg_key in CG_KEY_BY_SYMBOL.items():
        if cg_key in pct:
            try:
                out[symbol] = float(pct[cg_key])
            except (TypeError, ValueError):
                continue
    return out


def fetch_dominance() -> float | None:
    """Back-compat shim: USDT.D point value, or None."""
    return fetch_percentages().get("USDT.D")
