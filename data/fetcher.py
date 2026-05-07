"""OHLCV data fetcher — auto-detects yfinance vs ccxt by symbol format."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from pineforge_ai.config import (
    CANDLES_PER_DAY,
    RESAMPLE_RULES,
    WARMUP_BARS,
    YFINANCE_INTERVAL_MAP,
    YFINANCE_RESAMPLE,
)


# ─── Source Detection ────────────────────────────────────────────────────────

def detect_source(symbol: str) -> str:
    """
    Auto-detect data source from symbol format.
    BTC/USDT, ETH/BTC → ccxt
    AAPL, ^FTSE, EURUSD=X → yfinance
    """
    if "/" in symbol and not symbol.endswith("=X"):
        return "ccxt"
    return "yfinance"


# ─── Candle Count ─────────────────────────────────────────────────────────────

def candles_needed(days: int, timeframe: str) -> int:
    cpd = CANDLES_PER_DAY.get(timeframe, 24.0)
    return math.ceil(days * cpd) + WARMUP_BARS


def candles_needed_for_count(candles: int) -> int:
    return candles + WARMUP_BARS


def _validate_history_window(days: int | None, candles: int | None) -> None:
    if days is not None and candles is not None:
        raise ValueError("Use days or candles, not both")
    if days is None and candles is None:
        raise ValueError("Either days or candles is required")
    if days is not None and days < 1:
        raise ValueError("days must be greater than 0")
    if candles is not None and candles < 1:
        raise ValueError("candles must be greater than 0")


def _requested_candles(days: int | None, candles: int | None, timeframe: str) -> int:
    _validate_history_window(days, candles)
    if candles is not None:
        return candles_needed_for_count(candles)
    return candles_needed(days or 1, timeframe)


# ─── OHLCV Normalization ─────────────────────────────────────────────────────

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard columns and UTC-aware DatetimeIndex."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df[["open", "high", "low", "close", "volume"]].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_index()
    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to a coarser timeframe."""
    resampled = df.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
    return resampled


# ─── yfinance Fetcher ────────────────────────────────────────────────────────

def _fetch_yfinance(
    symbol: str,
    timeframe: str,
    days: int | None = None,
    candles: int | None = None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed. Run: pip install yfinance") from e

    n = _requested_candles(days, candles, timeframe)
    needs_resample = timeframe in YFINANCE_RESAMPLE

    if needs_resample:
        # Descarga en 1h y resamplea
        base_tf = "1h"
        base_days = math.ceil(n / CANDLES_PER_DAY["1h"]) + 5
    else:
        base_tf = YFINANCE_INTERVAL_MAP.get(timeframe, "1d")
        if candles is not None:
            cpd = CANDLES_PER_DAY.get(timeframe, 24.0)
            base_days = math.ceil(n / cpd) + 10
        else:
            base_days = (days or 1) + 10

    # yfinance limita histórico por intervalo
    max_days = {
        "1m": 7, "3m": 60, "5m": 60, "15m": 60, "30m": 60,
        "1h": 730, "2h": 730,
    }
    if base_tf in max_days:
        base_days = min(base_days, max_days[base_tf])

    start = datetime.now(tz=timezone.utc) - timedelta(days=base_days)
    ticker = yf.Ticker(symbol)
    raw = ticker.history(interval=base_tf, start=start.strftime("%Y-%m-%d"), auto_adjust=True)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for {symbol} [{base_tf}]")

    df = _normalize_df(raw)

    if needs_resample:
        rule = YFINANCE_RESAMPLE[timeframe]
        df = _resample_ohlcv(df, rule)

    return df.iloc[-n:]


# ─── ccxt Fetcher ─────────────────────────────────────────────────────────────

def _fetch_ccxt(
    symbol: str,
    timeframe: str,
    days: int | None = None,
    exchange_id: str = "binance",
    candles: int | None = None,
) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError as e:
        raise ImportError("ccxt not installed. Run: pip install ccxt") from e

    n = _requested_candles(days, candles, timeframe)

    # ccxt timeframe mapping (algunos exchanges no soportan todos)
    ccxt_tf_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h",
        "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w",
    }
    ccxt_tf = ccxt_tf_map.get(timeframe)
    if ccxt_tf is None:
        raise ValueError(f"Unsupported timeframe for ccxt: {timeframe}")

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    # Algunos exchanges no soportan ciertos TF → fallback a resample desde TF menor
    markets = exchange.load_markets()
    if symbol not in markets:
        # Intentar variaciones del símbolo
        alt = symbol.replace("/", "")
        matches = [k for k in markets if k.replace("/", "") == alt.upper()]
        if matches:
            symbol = matches[0]
        else:
            raise ValueError(f"Symbol {symbol} not found on {exchange_id}")

    # Calculate actual days needed based on candle count + warmup
    cpd = CANDLES_PER_DAY.get(timeframe, 24.0)
    days_needed = math.ceil(n / cpd) + 5
    since_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=days_needed)).timestamp() * 1000)

    all_candles = []
    limit = 1000
    fetch_since = since_ms

    while True:
        candles = exchange.fetch_ohlcv(symbol, ccxt_tf, since=fetch_since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        if len(candles) < limit:
            break
        fetch_since = candles[-1][0] + 1

    if not all_candles:
        raise ValueError(f"ccxt returned no data for {symbol} [{ccxt_tf}] on {exchange_id}")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = _normalize_df(df)

    return df.iloc[-n:]


# ─── Public API ───────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    days: int | None = None,
    source: str = "auto",
    exchange: str = "binance",
    candles: int | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol.

    Args:
        symbol:    Ticker (BTC/USDT, AAPL, ^FTSE, EURUSD=X)
        timeframe: Candle size (1h, 4h, 1d, ...)
        days:      History in days from now
        candles:   Final candle count requested, plus warmup internally
        source:    'auto' | 'yfinance' | 'ccxt'
        exchange:  ccxt exchange id (default: binance)

    Returns:
        pd.DataFrame with columns [open, high, low, close, volume], UTC index
    """
    if source == "auto":
        source = detect_source(symbol)

    if source == "ccxt":
        return _fetch_ccxt(symbol, timeframe, days=days, exchange_id=exchange, candles=candles)
    elif source == "yfinance":
        return _fetch_yfinance(symbol, timeframe, days=days, candles=candles)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'auto', 'yfinance' or 'ccxt'")


def fetch_multi_timeframe(
    symbol: str,
    timeframes: list[str],
    days: int | None = None,
    source: str = "auto",
    exchange: str = "binance",
    candles: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple timeframes.

    Returns:
        dict[timeframe, DataFrame]
    """
    result: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        try:
            result[tf] = fetch_ohlcv(
                symbol, tf, days=days, source=source, exchange=exchange, candles=candles
            )
            if candles is not None:
                visible = min(candles, len(result[tf]))
                warmup = max(0, len(result[tf]) - visible)
                print(f"  [{tf}] {visible} velas descargadas (+{warmup} warmup)")
            else:
                print(f"  [{tf}] {len(result[tf])} velas descargadas")
        except Exception as e:
            print(f"  [{tf}] ERROR: {e}")
    return result
