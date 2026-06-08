"""Crypto OHLCV reader — reads the SQLite store written by the ccxt daemon
(``crypto_ohlcv_ccxt``) and returns OHLCV DataFrames resampled to any timeframe.

Mirror of ``usdt_dominance/reader.py``. The store holds 1-minute bars keyed by
``(symbol, ts)`` where ``symbol`` is the clean spot pair (``BTC/USDT``). The
indicators series endpoint reads here for intraday timeframes (store coverage)
instead of fetching live, falling back to ccxt for deep timeframes the 1m store
can't reach (1d/1w).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "crypto_ohlcv.db"

RESAMPLE_RULES: dict[str, str] = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
    "1w":  "1W",
}

# Timeframes the 1m store serves from local SQLite. Capped at 1h: Bitget serves
# only ~200-bar pages for far-past 1m history, so backfilling enough 1m to fill
# 300×4h candles (~83 days) per symbol is impractical. 4h+ fall back to ccxt
# live (few bars, one fast page), keeping the store shallow and quick to backfill.
STORE_TIMEFRAMES: frozenset[str] = frozenset({"1m", "5m", "15m", "1h"})


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _load_bars(db_path: Path, days: int, symbol: str) -> pd.DataFrame:
    """Return DataFrame [open, high, low, close, volume] indexed by UTC datetime
    for ``symbol``, limited to the last ``days``. Empty on any miss."""
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not db_path.exists():
        return empty

    cutoff_ts = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp())
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        if not _table_exists(conn, "bars_1m"):
            conn.close()
            return empty
        df = pd.read_sql(
            "SELECT ts, open, high, low, close, volume FROM bars_1m "
            "WHERE symbol = ? AND ts >= ? ORDER BY ts ASC",
            conn,
            params=(symbol, cutoff_ts),
        )
        conn.close()
    except Exception:
        return empty

    if df.empty:
        return empty

    df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.drop(columns=["ts"])


def has_symbol(symbol: str, db_path: Path = DB_PATH) -> bool:
    """True if the store currently holds any bars for ``symbol``."""
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        if not _table_exists(conn, "bars_1m"):
            conn.close()
            return False
        row = conn.execute(
            "SELECT 1 FROM bars_1m WHERE symbol = ? LIMIT 1", (symbol,)
        ).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def get_ohlcv(
    timeframe: str = "1h",
    days: int = 30,
    db_path: Path = DB_PATH,
    symbol: str = "BTC/USDT",
) -> pd.DataFrame:
    """Load 1-minute bars for ``symbol`` and resample to ``timeframe``.

    Returns DataFrame [open, high, low, close, volume] with a UTC DatetimeIndex.
    Empty DataFrame if no data is available.
    """
    rule = RESAMPLE_RULES.get(timeframe, "1h")
    bars = _load_bars(db_path, days=days, symbol=symbol)
    if bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    ohlcv = bars.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return ohlcv
