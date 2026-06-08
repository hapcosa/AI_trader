"""Price + EMA overlay — feeds the /indicators EMA panel (E3b).

EMAs are not oscillators (0-100/centered); they overlay price. This builder
returns OHLC candles plus N EMA lines for one (symbol, timeframe), using the
same store/ccxt routing as ``build_indicator_series`` so it reads from the local
candle store for intraday TFs and falls back to ccxt live for deep TFs.
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

DEFAULT_CANDLES = 300
DEFAULT_EMAS = (20, 50, 200)


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def build_price_overlay(
    *,
    symbol: str,
    timeframe: str,
    emas: list[int] | None = None,
    source: str = "auto",
    exchange: str = DEFAULT_EXCHANGE,
    candles: int = DEFAULT_CANDLES,
) -> dict[str, Any]:
    """Return ``{symbol, timeframe, candles, emas}`` for the EMA price panel.

    ``candles`` is ``[{time, open, high, low, close}]`` (oldest-first); ``emas``
    is ``[{length, points:[{time, value}]}]``.
    """
    symbol = (symbol or "").strip()
    if not symbol:
        raise ValueError("symbol is required")
    timeframe = (timeframe or "").strip() or "1h"
    if source not in {"auto", "yfinance", "ccxt"}:
        raise ValueError("source must be auto, yfinance, or ccxt")
    if candles < 1:
        raise ValueError("candles must be greater than 0")
    lengths = [int(x) for x in (emas if emas is not None else DEFAULT_EMAS) if int(x) > 0]
    if not lengths:
        raise ValueError("at least one positive EMA length is required")

    if symbol.upper() in DOMINANCE_SYMBOLS:
        dfs = _dominance_dfs(symbol.upper(), [timeframe], candles)
    else:
        from pineforge_ai.data.fetcher import detect_source, fetch_multi_timeframe

        src = source if source != "auto" else detect_source(symbol)
        dfs = _crypto_store_dfs(symbol, [timeframe], candles) if src == "ccxt" else {}
        if timeframe not in dfs:
            fetch_symbol = _market_symbol(symbol, timeframe, exchange) if src == "ccxt" else symbol
            dfs = fetch_multi_timeframe(
                symbol=fetch_symbol, timeframes=[timeframe], candles=candles,
                source=src, exchange=exchange,
            )
    df = dfs.get(timeframe) if dfs else None
    if df is None or df.empty:
        raise RuntimeError("empty dfs — no OHLCV returned")

    close = df["close"].astype(float)
    ema_lines: list[dict[str, Any]] = []
    for length in lengths:
        e = close.ewm(span=length, adjust=False).mean()
        points = [
            {"time": int(ts.timestamp()), "value": v}
            for ts, raw in zip(e.index, e)
            if (v := _num(raw)) is not None
        ]
        ema_lines.append({"length": length, "points": points[-candles:]})

    out_candles: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        o, h, lo, c = _num(row["open"]), _num(row["high"]), _num(row["low"]), _num(row["close"])
        if None in (o, h, lo, c):
            continue
        out_candles.append({"time": int(ts.timestamp()), "open": o, "high": h, "low": lo, "close": c})
    out_candles = out_candles[-candles:]

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": out_candles,
        "emas": ema_lines,
    }


__all__ = ["build_price_overlay", "DEFAULT_EMAS", "DEFAULT_CANDLES"]
