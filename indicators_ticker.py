"""Live ticker — current price for the /indicators header (fast poll).

A lightweight last-price lookup via ccxt ``fetch_ticker``, separate from the
candle store (60s granularity) so the UI can poll every few seconds. The ccxt
exchange handle is cached process-wide so ``load_markets`` runs once, not on
every poll.
"""
from __future__ import annotations

from typing import Any

from pineforge_ai.indicators_summary import DEFAULT_EXCHANGE, _ccxt_symbol

# Process-wide ccxt handle cache (load_markets is expensive — do it once).
_EX_CACHE: dict[str, Any] = {}


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def _get_exchange(exchange_id: str):
    ex = _EX_CACHE.get(exchange_id)
    if ex is None:
        import ccxt

        ex = getattr(ccxt, exchange_id)(
            {"enableRateLimit": True, "options": {"defaultType": "swap"}}
        )
        ex.load_markets()
        _EX_CACHE[exchange_id] = ex
    return ex


def build_ticker(*, symbol: str, exchange: str = DEFAULT_EXCHANGE) -> dict[str, Any]:
    """Return ``{symbol, last, bid, ask, change_pct, time}`` for ``symbol``.

    ``symbol`` is the clean pair (BTC/USDT); the Bitget perp is queried under
    the hood. Raises ValueError on a bad symbol, RuntimeError when the ticker
    is unavailable.
    """
    symbol = (symbol or "").strip()
    if not symbol:
        raise ValueError("symbol is required")

    ex = _get_exchange(exchange)
    market = _ccxt_symbol(symbol, exchange)
    if market not in ex.markets:
        compact = market.replace("/", "").replace(":", "").upper()
        match = next(
            (k for k in ex.markets if k.replace("/", "").replace(":", "").upper() == compact),
            None,
        )
        if match is None:
            raise ValueError(f"symbol {symbol!r} not found on {exchange}")
        market = match

    try:
        t = ex.fetch_ticker(market)
    except Exception as e:  # network / exchange error
        raise RuntimeError(f"ticker unavailable for {symbol}: {e}") from e

    return {
        "symbol": symbol,
        "last": _num(t.get("last")),
        "bid": _num(t.get("bid")),
        "ask": _num(t.get("ask")),
        "change_pct": _num(t.get("percentage")),
        "time": int(t["timestamp"]) if t.get("timestamp") else None,
    }


__all__ = ["build_ticker"]
