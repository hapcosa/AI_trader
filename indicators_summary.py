"""Indicator summaries as JSON — the data source for the configurable notifier.

The dashboard-side NotifierDigestScheduler (signalsTrading W1-5) consumes this:
given a symbol, timeframes and a list of indicators, it fetches OHLCV, computes
the requested indicator summaries (the same last-bar-per-TF state used to build
prompts) and returns plain JSON. Keeping the indicator math here — where the
ports already live — lets the notifier stay free of pandas/DB (it only formats
and dispatches what the scheduler posts to it).
"""
from __future__ import annotations

import math
from typing import Any

from pineforge_ai.config import ALL_INDICATORS
from pineforge_ai.runner import (
    _build_indicator_summaries,
    parse_indicators,
    parse_timeframes,
)

# Public indicator name -> internal summary key used by _build_indicator_summaries.
_NAME_TO_KEY: dict[str, str] = {
    "wavetrend": "wt",
    "luxalgo": "lux",
    "smc": "smc",
    "wae": "tq",
    "itrend": "it",
    "ict": "ict",
    "trendlines": "tl",
    "pulse": "pulse",
    "abyss": "abyss",
    "tide": "tide",
    "athenea": "athenea",
}

# Default candles per timeframe — above WARMUP_BARS (150) so indicators settle,
# but bounded so the fetch stays cheap for a per-cycle digest.
DEFAULT_CANDLES = 300


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars / NaN / nested containers to JSON-native types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # numpy scalars expose .item(); plain floats need the NaN guard.
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_indicators_summary(
    *,
    symbol: str,
    timeframes: Any = None,
    indicators: Any = None,
    source: str = "auto",
    exchange: str = "binance",
    candles: int = DEFAULT_CANDLES,
) -> dict[str, Any]:
    """Compute requested indicator summaries for ``symbol`` across ``timeframes``.

    Returns ``{symbol, source, exchange, timeframes, indicators, summaries}``
    where ``summaries`` is keyed by the public indicator name, each value a
    ``{tf: reading}`` dict (None when that indicator failed to compute).
    """
    symbol = (symbol or "").strip()
    if not symbol:
        raise ValueError("symbol is required")
    if source not in {"auto", "yfinance", "ccxt"}:
        raise ValueError("source must be auto, yfinance, or ccxt")
    if candles < 1:
        raise ValueError("candles must be greater than 0")

    tf_list = parse_timeframes(timeframes)
    ind_list = parse_indicators(indicators)  # validates against ALL_INDICATORS

    from pineforge_ai.data.fetcher import detect_source, fetch_multi_timeframe

    actual_source = source if source != "auto" else detect_source(symbol)
    dfs = fetch_multi_timeframe(
        symbol=symbol,
        timeframes=tf_list,
        candles=candles,
        source=actual_source,
        exchange=exchange,
    )
    if not dfs:
        raise RuntimeError("empty dfs — no OHLCV returned")

    raw = _build_indicator_summaries(dfs, ind_list, emit=None)

    summaries: dict[str, Any] = {}
    for name in ind_list:
        key = _NAME_TO_KEY[name]
        summaries[name] = _json_safe(raw.get(key))

    return {
        "symbol": symbol,
        "source": actual_source,
        "exchange": exchange,
        "timeframes": tf_list,
        "indicators": ind_list,
        "summaries": summaries,
    }


__all__ = ["build_indicators_summary", "ALL_INDICATORS", "DEFAULT_CANDLES"]
