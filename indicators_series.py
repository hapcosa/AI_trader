"""Indicator oscillator series — feeds the user-facing /indicators live charts.

`/api/indicators/summary` (W1-2/W2-2) returns the last-bar state per TF. The live
oscillator needs the full series (osc + trig over time) so the dashboard can draw
the line + trigger + OB/OS zones and push updates over WebSocket (W3).

v1 covers the BudAI family + WaveTrend (all ported, with known OB/OS levels).
Dominance symbols (USDT.D/BTC.D/OTHERS.D) resolve against the dominance SQLite,
like the summary endpoint.
"""
from __future__ import annotations

import math
from typing import Any

from pineforge_ai.indicators.budai_abyss import budai_abyss
from pineforge_ai.indicators.budai_athenea import budai_athenea
from pineforge_ai.indicators.budai_moneyflow_tide import budai_moneyflow_tide
from pineforge_ai.indicators.budai_pulse import budai_pulse
from pineforge_ai.indicators.classics import macd, rsi, stochastic
from pineforge_ai.indicators.wavetrend import wavetrend
from pineforge_ai.indicators_summary import (
    DEFAULT_EXCHANGE,
    DOMINANCE_SYMBOLS,
    _ccxt_symbol,
    _crypto_store_dfs,
    _dominance_dfs,
)

# indicator -> compute fn + which columns are the oscillator/trigger, the OB/OS
# guide levels, and the scale ("0-100" or "centered" around 0). OB/OS match each
# indicator's own thresholds. "hist" (optional) names a histogram column emitted
# alongside osc/trig (MACD). ob/os may be None when the indicator has no bounded
# extremes (MACD), in which case the UI skips OB/OS lines + the reading chip.
_SERIES: dict[str, dict[str, Any]] = {
    "pulse":     {"fn": budai_pulse,            "osc": "osc",      "trig": "trig",       "ob": 80.0,  "os": 20.0,   "scale": "0-100"},
    "abyss":     {"fn": budai_abyss,            "osc": "wt1",      "trig": "wt2",        "ob": 53.0,  "os": -53.0,  "scale": "centered"},
    "tide":      {"fn": budai_moneyflow_tide,   "osc": "fast",     "trig": "slow",       "ob": 60.0,  "os": -60.0,  "scale": "centered"},
    "athenea":   {"fn": budai_athenea,          "osc": "osc",      "trig": "trig",       "ob": 80.0,  "os": 20.0,   "scale": "0-100"},
    "wavetrend": {"fn": wavetrend,              "osc": "osc_norm", "trig": "trig_norm",  "ob": 75.0,  "os": 25.0,   "scale": "0-100"},
    "rsi":       {"fn": rsi,                    "osc": "rsi",      "trig": "rsi_signal", "ob": 70.0,  "os": 30.0,   "scale": "0-100"},
    "stochastic": {"fn": stochastic,           "osc": "k",        "trig": "d",          "ob": 80.0,  "os": 20.0,   "scale": "0-100"},
    "macd":      {"fn": macd,                   "osc": "macd",     "trig": "signal",     "ob": None,  "os": None,   "scale": "centered", "hist": "hist"},
}

SERIES_INDICATORS = tuple(_SERIES)

DEFAULT_CANDLES = 300


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def build_indicator_series(
    *,
    symbol: str,
    timeframe: str,
    indicator: str,
    source: str = "auto",
    exchange: str = DEFAULT_EXCHANGE,
    candles: int = DEFAULT_CANDLES,
) -> dict[str, Any]:
    """Compute one indicator's oscillator series for one (symbol, timeframe).

    Returns ``{symbol, timeframe, indicator, scale, ob, os, points}`` where
    ``points`` is ``[{time, osc, trig}]`` (oldest-first, NaN→None).
    """
    symbol = (symbol or "").strip()
    if not symbol:
        raise ValueError("symbol is required")
    indicator = (indicator or "").strip().lower()
    if indicator not in _SERIES:
        raise ValueError(
            f"unknown indicator '{indicator}'. valid: {', '.join(SERIES_INDICATORS)}"
        )
    timeframe = (timeframe or "").strip() or "1h"
    if source not in {"auto", "yfinance", "ccxt"}:
        raise ValueError("source must be auto, yfinance, or ccxt")
    if candles < 1:
        raise ValueError("candles must be greater than 0")

    spec = _SERIES[indicator]

    if symbol.upper() in DOMINANCE_SYMBOLS:
        dfs = _dominance_dfs(symbol.upper(), [timeframe], candles)
    else:
        from pineforge_ai.data.fetcher import detect_source, fetch_multi_timeframe

        src = source if source != "auto" else detect_source(symbol)
        # Hybrid: read intraday TFs from the local candle store (instant); fall
        # back to ccxt live for deep TFs (1d/1w) or symbols not in the store.
        dfs = _crypto_store_dfs(symbol, [timeframe], candles) if src == "ccxt" else {}
        if timeframe not in dfs:
            fetch_symbol = _ccxt_symbol(symbol, exchange) if src == "ccxt" else symbol
            dfs = fetch_multi_timeframe(
                symbol=fetch_symbol, timeframes=[timeframe], candles=candles,
                source=src, exchange=exchange,
            )
    df = dfs.get(timeframe) if dfs else None
    if df is None or df.empty:
        raise RuntimeError("empty dfs — no OHLCV returned")

    res = spec["fn"](df)
    osc_col, trig_col = spec["osc"], spec["trig"]
    hist_col = spec.get("hist")

    points: list[dict[str, Any]] = []
    for idx, o, t, h in zip(
        res.index,
        res[osc_col],
        res[trig_col],
        res[hist_col] if hist_col else [None] * len(res),
    ):
        osc_v, trig_v = _num(o), _num(t)
        if osc_v is None and trig_v is None:
            continue
        ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        point: dict[str, Any] = {"time": int(ts.timestamp()), "osc": osc_v, "trig": trig_v}
        if hist_col:
            point["hist"] = _num(h)
        points.append(point)

    points = points[-candles:]
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "indicator": indicator,
        "scale": spec["scale"],
        "ob": spec["ob"],
        "os": spec["os"],
        "points": points,
    }


__all__ = ["build_indicator_series", "SERIES_INDICATORS", "DEFAULT_CANDLES"]
