"""Per-timeframe price range — the current candle's high-low for the header.

For the selected symbol, returns the latest (in-progress) candle's high/low/last
and range % for each requested timeframe, so the UI can show how wide price is
swinging on 15m/1h/4h/1d at a glance. Uses the same store/ccxt routing as the
series/price builders (store for intraday, live for deep TFs).
"""
from __future__ import annotations

import math
from typing import Any

from pineforge_ai.indicators_summary import (
    DEFAULT_EXCHANGE,
    DOMINANCE_SYMBOLS,
    _crypto_store_dfs,
    _dominance_dfs,
    _market_symbol,
)

DEFAULT_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_FETCH_CANDLES = 3  # only the latest bucket is needed


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _routed_df(symbol: str, tf: str, source: str, exchange: str):
    if symbol.upper() in DOMINANCE_SYMBOLS:
        dfs = _dominance_dfs(symbol.upper(), [tf], _FETCH_CANDLES)
    else:
        from pineforge_ai.data.fetcher import detect_source, fetch_multi_timeframe

        src = source if source != "auto" else detect_source(symbol)
        dfs = _crypto_store_dfs(symbol, [tf], _FETCH_CANDLES) if src == "ccxt" else {}
        if tf not in dfs:
            fetch_symbol = _market_symbol(symbol, tf, exchange) if src == "ccxt" else symbol
            dfs = fetch_multi_timeframe(
                symbol=fetch_symbol, timeframes=[tf], candles=_FETCH_CANDLES,
                source=src, exchange=exchange,
            )
    return dfs.get(tf) if dfs else None


def build_ranges(
    *,
    symbol: str,
    timeframes: list[str] | None = None,
    source: str = "auto",
    exchange: str = DEFAULT_EXCHANGE,
) -> dict[str, Any]:
    """Return ``{symbol, ranges:{tf:{high,low,last,range_pct}}}``.

    ``range_pct`` is ``(high-low)/low*100`` of the latest candle (None when the
    TF has no data).
    """
    symbol = (symbol or "").strip()
    if not symbol:
        raise ValueError("symbol is required")
    tfs = [t.strip() for t in (timeframes or DEFAULT_TIMEFRAMES) if t.strip()]
    if not tfs:
        raise ValueError("at least one timeframe is required")

    ranges: dict[str, Any] = {}
    for tf in tfs:
        df = _routed_df(symbol, tf, source, exchange)
        if df is None or df.empty:
            ranges[tf] = None
            continue
        row = df.iloc[-1]
        hi, lo, last = _num(row["high"]), _num(row["low"]), _num(row["close"])
        rng_pct = (hi - lo) / lo * 100.0 if (hi is not None and lo) else None
        ranges[tf] = {"high": hi, "low": lo, "last": last, "range_pct": rng_pct}

    return {"symbol": symbol, "ranges": ranges}


__all__ = ["build_ranges", "DEFAULT_TIMEFRAMES"]
