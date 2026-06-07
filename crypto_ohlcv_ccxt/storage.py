"""SQLite storage for crypto 1-minute OHLCV bars (multi-symbol).

Each row is keyed by ``(symbol, ts)`` so several pairs (BTC/USDT, ETH/USDT, …)
share one table. ``symbol`` is the clean spot pair (``BTC/USDT``) even though
the daemon fetches the perp under the hood — the UI and the reader key off the
clean pair. Mirrors ``usdt_dominance_tv.storage`` minus the legacy single-series
migration (this is a fresh schema), plus a retention prune.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_1m (
    symbol TEXT    NOT NULL,
    ts     INTEGER NOT NULL,
    open   REAL    NOT NULL,
    high   REAL    NOT NULL,
    low    REAL    NOT NULL,
    close  REAL    NOT NULL,
    volume REAL    NOT NULL DEFAULT 0,
    source TEXT    NOT NULL,
    PRIMARY KEY (symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_1m_symbol_ts ON bars_1m(symbol, ts);
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


def get_last_ts(conn: sqlite3.Connection, symbol: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(ts) FROM bars_1m WHERE symbol = ?", (symbol,)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def get_first_ts(conn: sqlite3.Connection, symbol: str) -> int | None:
    row = conn.execute(
        "SELECT MIN(ts) FROM bars_1m WHERE symbol = ?", (symbol,)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def count_bars(conn: sqlite3.Connection, symbol: str | None = None) -> int:
    if symbol is None:
        row = conn.execute("SELECT COUNT(*) FROM bars_1m").fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE symbol = ?", (symbol,)
        ).fetchone()
    return int(row[0]) if row else 0


def upsert_bars(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    symbol: str,
    source: str = "bitget",
) -> int:
    """Upsert OHLCV rows from a DataFrame for ``symbol``.

    df must have a UTC DatetimeIndex and columns [open, high, low, close,
    volume]. Timestamps are truncated to minute boundary. Returns rows written.
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
        rows.append((symbol, ts, o, h, lo, c, v, source))
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO bars_1m "
        "(symbol, ts, open, high, low, close, volume, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def prune_old(conn: sqlite3.Connection, retention_days: int) -> int:
    """Delete bars older than ``retention_days`` across all symbols.

    Returns the number of rows removed. A non-positive retention is a no-op.
    """
    if retention_days <= 0:
        return 0
    cutoff = minute_ts(datetime.now(tz=timezone.utc)) - retention_days * 24 * 60 * 60
    cur = conn.execute("DELETE FROM bars_1m WHERE ts < ?", (cutoff,))
    conn.commit()
    return cur.rowcount or 0


def stats_by_symbol(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT symbol, source, COUNT(*) AS n, MIN(ts) AS min_ts, MAX(ts) AS max_ts "
        "FROM bars_1m GROUP BY symbol, source"
    )
    return [
        {"symbol": r[0], "source": r[1], "count": r[2], "min_ts": r[3], "max_ts": r[4]}
        for r in cur.fetchall()
    ]
