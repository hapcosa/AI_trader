"""W0-2A — multi-series dominance storage + migration + reader filtering.

Covers:
  - fresh open_db builds the (symbol, ts) schema
  - a legacy single-series bars_1m is migrated in place, rows tagged USDT.D
  - several series share one table without ts collisions
  - CoinGecko fallback parses the percentages it can supply
  - the USDT.D reader filters by symbol (and still reads legacy DBs)
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from usdt_dominance_tv import cg_source, storage
from usdt_dominance import reader


def _bars(start_min: int, closes: list[float]) -> pd.DataFrame:
    idx = pd.to_datetime(
        [(start_min + i) * 60 for i in range(len(closes))], unit="s", utc=True
    )
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [0.0] * len(closes),
        },
        index=idx,
    )


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


# ─── schema / migration ─────────────────────────────────────────────────────

def test_open_db_fresh_creates_multiseries_schema(tmp_path):
    conn = storage.open_db(tmp_path / "dom.db")
    cols = _columns(conn, "bars_1m")
    assert "symbol" in cols
    # symbol + ts compose the primary key
    pk = [r[1] for r in conn.execute("PRAGMA table_info(bars_1m)") if r[5] > 0]
    assert set(pk) == {"symbol", "ts"}
    conn.close()


def test_migrate_legacy_singleseries(tmp_path):
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(str(db))
    raw.executescript(
        """
        CREATE TABLE bars_1m (
            ts INTEGER PRIMARY KEY, open REAL, high REAL, low REAL,
            close REAL, volume REAL DEFAULT 0, source TEXT
        );
        INSERT INTO bars_1m VALUES (60, 5.0, 5.1, 4.9, 5.0, 0, 'tv');
        INSERT INTO bars_1m VALUES (120, 5.2, 5.3, 5.1, 5.2, 0, 'tv');
        """
    )
    raw.commit()
    raw.close()

    conn = storage.open_db(db)
    assert "symbol" in _columns(conn, "bars_1m")
    assert storage.count_bars(conn) == 2
    assert storage.count_bars(conn, "USDT.D") == 2
    rows = conn.execute(
        "SELECT symbol, ts, close FROM bars_1m ORDER BY ts"
    ).fetchall()
    assert rows == [("USDT.D", 60, 5.0), ("USDT.D", 120, 5.2)]
    # Legacy table must be gone.
    assert not storage._table_exists(conn, "bars_1m_legacy")
    conn.close()


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "idem.db"
    conn = storage.open_db(db)
    storage.upsert_bars(conn, _bars(1, [5.0, 5.1]), symbol="USDT.D", source="tv")
    conn.close()
    # Re-open: must not wipe data or raise.
    conn2 = storage.open_db(db)
    assert storage.count_bars(conn2, "USDT.D") == 2
    conn2.close()


# ─── multi-series writes ────────────────────────────────────────────────────

def test_series_share_table_without_collision(tmp_path):
    conn = storage.open_db(tmp_path / "multi.db")
    # Same timestamps for two series — must not collide on PK.
    storage.upsert_bars(conn, _bars(1, [5.0, 5.1, 5.2]), symbol="USDT.D", source="tv")
    storage.upsert_bars(conn, _bars(1, [55.0, 55.4, 55.8]), symbol="BTC.D", source="tv")

    assert storage.count_bars(conn, "USDT.D") == 3
    assert storage.count_bars(conn, "BTC.D") == 3
    assert storage.count_bars(conn) == 6
    assert storage.get_last_ts(conn, "USDT.D") == 3 * 60
    assert storage.get_last_ts(conn, "BTC.D") == 3 * 60
    assert storage.get_last_ts(conn, "OTHERS.D") is None
    conn.close()


def test_upsert_single_per_symbol(tmp_path):
    conn = storage.open_db(tmp_path / "single.db")
    storage.upsert_single(conn, "BTC.D", 60, 55.5, source="coingecko")
    row = conn.execute(
        "SELECT open, high, low, close, source FROM bars_1m WHERE symbol='BTC.D'"
    ).fetchone()
    assert row == (55.5, 55.5, 55.5, 55.5, "coingecko")
    conn.close()


# ─── CoinGecko fallback ─────────────────────────────────────────────────────

def test_cg_fetch_percentages(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"market_cap_percentage": {"btc": 54.3, "usdt": 5.1, "eth": 12.0}}}

    monkeypatch.setattr(cg_source.requests, "get", lambda *a, **k: _Resp())
    pct = cg_source.fetch_percentages()
    assert pct == {"USDT.D": 5.1, "BTC.D": 54.3}  # OTHERS.D not available from CG
    assert cg_source.fetch_dominance() == 5.1


def test_cg_fetch_percentages_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(cg_source.requests, "get", _boom)
    assert cg_source.fetch_percentages() == {}
    assert cg_source.fetch_dominance() is None


# ─── reader filtering ───────────────────────────────────────────────────────

def test_reader_filters_by_symbol(tmp_path):
    # get_current_value() filters to the last day, so use recent timestamps.
    now_min = int(datetime.now(tz=timezone.utc).timestamp()) // 60
    conn = storage.open_db(tmp_path / "reader.db")
    db = tmp_path / "reader.db"
    storage.upsert_bars(conn, _bars(now_min - 1, [5.0, 5.5]), symbol="USDT.D", source="tv")
    storage.upsert_bars(conn, _bars(now_min - 1, [55.0, 56.0]), symbol="BTC.D", source="tv")
    conn.close()

    assert reader.get_current_value(db_path=db) == 5.5  # default USDT.D
    assert reader.get_current_value(db_path=db, symbol="BTC.D") == 56.0
    assert reader.get_current_value(db_path=db, symbol="OTHERS.D") is None


def test_reader_reads_legacy_db_without_symbol(tmp_path):
    now_min = int(datetime.now(tz=timezone.utc).timestamp()) // 60
    db = tmp_path / "legacy_reader.db"
    raw = sqlite3.connect(str(db))
    raw.execute(
        """
        CREATE TABLE bars_1m (
            ts INTEGER PRIMARY KEY, open REAL, high REAL, low REAL,
            close REAL, volume REAL DEFAULT 0, source TEXT
        )
        """
    )
    raw.execute(
        "INSERT INTO bars_1m VALUES (?, 5.0, 5.1, 4.9, 5.0, 0, 'tv')",
        (now_min * 60,),
    )
    raw.commit()
    raw.close()
    # Reader must not choke on the pre-migration schema.
    assert reader.get_current_value(db_path=db) == 5.0
