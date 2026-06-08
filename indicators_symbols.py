"""Symbol catalog for the /indicators pair picker.

Returns the symbols the candle store has already downloaded (so the UI lists
them first, instant) plus a curated crypto catalog for the rest. The picker
renders two sections: "downloaded" then "altcoins" (catalog minus downloaded).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Curated crypto perp catalog (BASE/USDT). Mirrors AI_trader's symbol list,
# crypto-only. The store's downloaded set is layered on top at request time.
CATALOG: tuple[str, ...] = (
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT", "POL/USDT",
    "UNI/USDT", "ATOM/USDT", "LTC/USDT", "BCH/USDT", "NEAR/USDT", "OP/USDT",
    "ARB/USDT", "INJ/USDT", "TRX/USDT", "APT/USDT", "SUI/USDT", "TON/USDT",
    "WIF/USDT", "PEPE/USDT", "SHIB/USDT", "ETC/USDT", "FIL/USDT", "AAVE/USDT",
    "ENA/USDT", "HBAR/USDT", "HYPE/USDT", "KAS/USDT", "ONDO/USDT", "PENDLE/USDT",
    "PENGU/USDT", "PEOPLE/USDT", "PIPPIN/USDT", "PI/USDT", "PLUME/USDT",
    "PNUT/USDT", "POPCAT/USDT", "POWR/USDT", "PUMP/USDT", "PYTH/USDT",
    "RAY/USDT", "RENDER/USDT", "RUNE/USDT", "TAO/USDT", "TIA/USDT",
    "TRUMP/USDT", "XLM/USDT", "XMR/USDT", "ZEC/USDT", "LPT/USDT", "ORDI/USDT",
    "FARTCOIN/USDT", "MORPHO/USDT", "CHZ/USDT", "CFX/USDT", "CELO/USDT",
)


def build_symbols() -> dict[str, Any]:
    """Return ``{downloaded, catalog}``.

    ``downloaded`` = symbols the store has bars for (instant on the page).
    ``catalog`` = the curated list unioned with downloaded, sorted — the UI
    derives "altcoins" as catalog minus downloaded.
    """
    from pineforge_ai.crypto_ohlcv import reader

    db = os.environ.get("CRYPTO_OHLCV_DB")
    db_path = Path(db) if db else reader.DB_PATH
    downloaded = reader.store_symbols(db_path=db_path)

    catalog = sorted(set(CATALOG) | set(downloaded))
    return {"downloaded": downloaded, "catalog": catalog}


__all__ = ["build_symbols", "CATALOG"]
