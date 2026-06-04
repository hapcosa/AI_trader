"""SQLite storage for dominance 1-minute OHLCV bars (multi-series).

Each row is keyed by ``(symbol, ts)`` so several dominance series (USDT.D,
BTC.D, OTHERS.D, …) can share one table. Legacy single-series databases
(``bars_1m`` with PK on ``ts`` only and no ``symbol`` column) are migrated
in place on open: existing rows are tagged ``symbol='USDT.D'``.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# Canonical symbol of the legacy single-series schema.
LEGACY_SYMBOL = "USDT.D"

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
    _migrate_schema(conn)
    conn.commit()
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Create the multi-series schema, migrating a legacy single-series DB.

    Idempotent: a no-op once ``bars_1m`` already has a ``symbol`` column.
    """
    if not _table_exists(conn, "bars_1m"):
        conn.executescript(SCHEMA)
        return

    if "symbol" in _columns(conn, "bars_1m"):
        # Already multi-series — just make sure the index exists.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bars_1m_symbol_ts ON bars_1m(symbol, ts);"
        )
        return

    # Legacy single-series schema (PK on ts, no symbol). Rebuild with the new
    # PK and tag every existing row as USDT.D.
    conn.executescript(
        """
        BEGIN;
        ALTER TABLE bars_1m RENAME TO bars_1m_legacy;
        CREATE TABLE bars_1m (
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
        INSERT INTO bars_1m (symbol, ts, open, high, low, close, volume, source)
            SELECT '%s', ts, open, high, low, close, volume, source
            FROM bars_1m_legacy;
        DROP TABLE bars_1m_legacy;
        COMMIT;
        CREATE INDEX IF NOT EXISTS idx_bars_1m_symbol_ts ON bars_1m(symbol, ts);
        """
        % LEGACY_SYMBOL
    )


def minute_ts(dt: datetime) -> int:
    return int(dt.replace(second=0, microsecond=0).timestamp())


def get_last_ts(conn: sqlite3.Connection, symbol: str = LEGACY_SYMBOL) -> int | None:
    row = conn.execute(
        "SELECT MAX(ts) FROM bars_1m WHERE symbol = ?", (symbol,)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def get_first_ts(conn: sqlite3.Connection, symbol: str = LEGACY_SYMBOL) -> int | None:
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
    source: str,
) -> int:
    """
    Upsert OHLCV rows from a DataFrame for ``symbol``.
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


def upsert_single(
    conn: sqlite3.Connection,
    symbol: str,
    ts: int,
    value: float,
    source: str,
    volume: float = 0.0,
) -> None:
    """Insert a single tick where o=h=l=c=value (used for CoinGecko fallback)."""
    conn.execute(
        "INSERT OR REPLACE INTO bars_1m "
        "(symbol, ts, open, high, low, close, volume, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (symbol, ts, value, value, value, value, volume, source),
    )
    conn.commit()


def stats_by_source(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT symbol, source, COUNT(*) AS n, MIN(ts) AS min_ts, MAX(ts) AS max_ts "
        "FROM bars_1m GROUP BY symbol, source"
    )
    return [
        {"symbol": r[0], "source": r[1], "count": r[2], "min_ts": r[3], "max_ts": r[4]}
        for r in cur.fetchall()
    ]
