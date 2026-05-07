"""CoinGecko fallback for USDT dominance percentage."""
from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)

COINGECKO_URL = "https://api.coingecko.com/api/v3/global"
HEADERS = {"User-Agent": "PineForge-AI/2.0"}


def fetch_dominance() -> float | None:
    try:
        resp = requests.get(COINGECKO_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data["data"]["market_cap_percentage"]["usdt"])
    except Exception as e:
        log.warning("CoinGecko fetch failed: %s", e)
        return None
