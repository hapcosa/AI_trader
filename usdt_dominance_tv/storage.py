"""SQLite storage for USDT.D 1-minute OHLCV bars."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_1m (
    ts     INTEGER PRIMARY KEY,
    open   REAL NOT NULL,
    high   REAL NOT NULL,
    low    REAL NOT NULL,
    close  REAL NOT NULL,
    volume REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bars_1m_ts ON bars_1m(ts);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def minute_ts(dt: datetime) -> int:
    return int(dt.replace(second=0, microsecond=0).timestamp())


def get_last_ts(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT MAX(ts) FROM bars_1m").fetchone()
    return int(row[0]) if row and row[0] is not None else None


def get_first_ts(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT MIN(ts) FROM bars_1m").fetchone()
    return int(row[0]) if row and row[0] is not None else None


def count_bars(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM bars_1m").fetchone()
    return int(row[0]) if row else 0


def upsert_bars(conn: sqlite3.Connection, df: pd.DataFrame, source: str) -> int:
    """
    Upsert OHLCV rows from a DataFrame.
    df must have a UTC DatetimeIndex and columns [open, high, low, close, volume].
    Timestamps are truncated to minute boundary. Returns number of rows written.
    """
    if df is None or df.empty:
        return 0
    rows: list[tuple] = []
    for idx, row in df.iterrows():
        ts_dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        ts = minute_ts(ts_dt)
        try:
            o = float(row["open"])
            h = float(row["high"])
            lo = float(row["low"])
            c = float(row["close"])
            v = float(row.get("volume", 0.0) or 0.0)
        except (KeyError, ValueError, TypeError):
            continue
        rows.append((ts, o, h, lo, c, v, source))
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO bars_1m "
        "(ts, open, high, low, close, volume, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_single(
    conn: sqlite3.Connection,
    ts: int,
    value: float,
    source: str,
    volume: float = 0.0,
) -> None:
    """Insert a single tick where o=h=l=c=value (used for CoinGecko fallback)."""
    conn.execute(
        "INSERT OR REPLACE INTO bars_1m "
        "(ts, open, high, low, close, volume, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, value, value, value, value, volume, source),
    )
    conn.commit()


def stats_by_source(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT source, COUNT(*) AS n, MIN(ts) AS min_ts, MAX(ts) AS max_ts "
        "FROM bars_1m GROUP BY source"
    )
    return [
        {"source": r[0], "count": r[1], "min_ts": r[2], "max_ts": r[3]}
        for r in cur.fetchall()
    ]
