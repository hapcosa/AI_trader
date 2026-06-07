"""ccxt OHLCV source — Bitget USDT-M perp 1-minute bars.

Wraps a single ccxt exchange handle (``defaultType=swap``) and exposes
``fetch_recent(symbol, n_bars)`` returning a UTC-indexed OHLCV DataFrame. The
public symbol is the clean spot pair (``BTC/USDT``); under the hood it maps to
the ccxt swap symbol (``BTC/USDT:USDT``) so the store keys off the clean pair.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd


log = logging.getLogger("crypto_ohlcv_ccxt")

# Bitget caps OHLCV pages; 1000 is the safe per-call limit.
MAX_BARS_PER_CALL = 1000


def perp_symbol(symbol: str, exchange_id: str) -> str:
    """Map a clean spot pair to the ccxt swap symbol for Bitget (BASE/QUOTE →
    BASE/QUOTE:QUOTE). Leaves already-suffixed symbols / non-Bitget untouched."""
    s = (symbol or "").strip()
    if exchange_id.lower() == "bitget" and "/" in s and ":" not in s:
        quote = s.split("/", 1)[1]
        return f"{s}:{quote}"
    return s


class CcxtSource:
    """Thin wrapper over a ccxt exchange handle for 1m OHLCV fetches."""

    def __init__(self, exchange_id: str = "bitget") -> None:
        import ccxt  # imported lazily so the module loads without ccxt in tests

        self.exchange_id = exchange_id
        cls = getattr(ccxt, exchange_id)
        self.exchange = cls(
            {"enableRateLimit": True, "options": {"defaultType": "swap"}}
        )
        self.exchange.load_markets()

    def _resolve(self, symbol: str) -> str:
        """Return a market symbol ccxt knows, mapping clean pairs to the perp."""
        mapped = perp_symbol(symbol, self.exchange_id)
        if mapped in self.exchange.markets:
            return mapped
        # Fall back to a loose match (strip separators) before giving up.
        compact = mapped.replace("/", "").replace(":", "").upper()
        for k in self.exchange.markets:
            if k.replace("/", "").replace(":", "").upper() == compact:
                return k
        raise ValueError(f"symbol {symbol!r} ({mapped}) not found on {self.exchange_id}")

    def fetch_recent(self, n_bars: int, symbol: str) -> pd.DataFrame:
        """Fetch the most recent ``n_bars`` 1m candles for ``symbol``.

        Paginates backwards with ``since`` so backfills of many days work.
        Returns a UTC-indexed DataFrame [open, high, low, close, volume].
        """
        market = self._resolve(symbol)
        n_bars = max(1, int(n_bars))
        since = int(
            (datetime.now(tz=timezone.utc) - timedelta(minutes=n_bars + 2)).timestamp()
            * 1000
        )

        rows: list[list] = []
        cursor = since
        while True:
            batch = self.exchange.fetch_ohlcv(
                market, "1m", since=cursor, limit=MAX_BARS_PER_CALL
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < MAX_BARS_PER_CALL:
                break
            cursor = batch[-1][0] + 1
            # Stop once we have enough; the loop is for deep backfills only.
            if len(rows) >= n_bars + MAX_BARS_PER_CALL:
                break
            time.sleep(self.exchange.rateLimit / 1000.0)

        if not rows:
            raise ValueError(f"ccxt returned no data for {market} on {self.exchange_id}")

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        return df.iloc[-n_bars:]
