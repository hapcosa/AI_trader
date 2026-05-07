"""Fetches global crypto market cap data from CoinGecko public API."""

import json
import urllib.request
from datetime import datetime, timezone

_ENDPOINT = "https://api.coingecko.com/api/v3/global"


def fetch_market_cap(timeout=10):
    try:
        with urllib.request.urlopen(_ENDPOINT, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode())

        data = raw["data"]
        dominances = data["market_cap_percentage"]

        btc = float(dominances.get("btc", 0.0))
        eth = float(dominances.get("eth", 0.0))
        altcoin = 100.0 - btc - eth

        top5 = dict(
            sorted(dominances.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        top5 = {k.upper(): float(v) for k, v in top5.items()}

        return {
            "btc_dominance": btc,
            "eth_dominance": eth,
            "altcoin_dominance": altcoin,
            "total_market_cap_usd": float(data["total_market_cap"].get("usd", 0.0)),
            "total_volume_24h_usd": float(data["total_volume"].get("usd", 0.0)),
            "market_cap_change_24h_pct": float(
                data.get("market_cap_change_percentage_24h_usd", 0.0)
            ),
            "top_dominances": top5,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception:
        return None


def build_market_cap_summary(timeout=10):
    data = fetch_market_cap(timeout=timeout)

    if data is None:
        return {"available": False}

    btc = data["btc_dominance"]
    if btc > 55:
        btc_zone = "High (>55%) — BTC season, altcoins lagging"
    elif btc >= 45:
        btc_zone = "Mid (45-55%) — transitional"
    else:
        btc_zone = "Low (<45%) — altseason territory"

    change = data["market_cap_change_24h_pct"]
    if change > 2:
        mc_signal = "EXPANDING — risk-on"
    elif change < -2:
        mc_signal = "CONTRACTING — risk-off"
    else:
        mc_signal = "STABLE"

    return {
        "available": True,
        "btc_zone": btc_zone,
        "mc_signal": mc_signal,
        **data,
    }
