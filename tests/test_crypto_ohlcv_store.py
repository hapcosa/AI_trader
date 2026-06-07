"""E1.1 — crypto OHLCV candle store (ccxt → SQLite).

Covers:
  - fresh open_db builds the (symbol, ts) schema
  - several pairs share one table without ts collisions
  - upsert dedupes on (symbol, ts) (INSERT OR REPLACE)
  - retention prune drops only bars older than the cutoff
  - perp_symbol maps clean spot pairs to the Bitget swap symbol
  - daemon symbol parsing dedupes preserving order
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from crypto_ohlcv_ccxt import daemon, storage
from crypto_ohlcv_ccxt.source import perp_symbol


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
            "volume": [1.0] * len(closes),
        },
        index=idx,
    )


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


# ─── schema ──────────────────────────────────────────────────────────────────

def test_open_db_creates_multisymbol_schema(tmp_path):
    conn = storage.open_db(tmp_path / "c.db")
    cols = _columns(conn, "bars_1m")
    assert {"symbol", "ts", "open", "high", "low", "close", "volume", "source"} <= set(cols)
    pk = [r[1] for r in conn.execute("PRAGMA table_info(bars_1m)") if r[5] > 0]
    assert set(pk) == {"symbol", "ts"}
    conn.close()


def test_open_db_is_idempotent(tmp_path):
    db = tmp_path / "c.db"
    storage.open_db(db).close()
    conn = storage.open_db(db)  # second open must not raise
    assert storage.count_bars(conn) == 0
    conn.close()


# ─── upsert / multi-symbol ───────────────────────────────────────────────────

def test_symbols_share_table_without_collision(tmp_path):
    conn = storage.open_db(tmp_path / "c.db")
    storage.upsert_bars(conn, _bars(0, [100, 101, 102]), symbol="BTC/USDT")
    storage.upsert_bars(conn, _bars(0, [10, 11, 12]), symbol="ETH/USDT")
    assert storage.count_bars(conn, "BTC/USDT") == 3
    assert storage.count_bars(conn, "ETH/USDT") == 3
    assert storage.count_bars(conn) == 6
    conn.close()


def test_upsert_dedupes_on_symbol_ts(tmp_path):
    conn = storage.open_db(tmp_path / "c.db")
    storage.upsert_bars(conn, _bars(0, [100, 101, 102]), symbol="BTC/USDT")
    # Re-write the same minutes with new closes — count stays, values replace.
    storage.upsert_bars(conn, _bars(0, [200, 201, 202]), symbol="BTC/USDT")
    assert storage.count_bars(conn, "BTC/USDT") == 3
    last = conn.execute(
        "SELECT close FROM bars_1m WHERE symbol='BTC/USDT' ORDER BY ts DESC LIMIT 1"
    ).fetchone()[0]
    assert last == 202
    conn.close()


def test_upsert_empty_is_noop(tmp_path):
    conn = storage.open_db(tmp_path / "c.db")
    assert storage.upsert_bars(conn, pd.DataFrame(), symbol="BTC/USDT") == 0
    assert storage.upsert_bars(conn, None, symbol="BTC/USDT") == 0
    conn.close()


# ─── retention prune ─────────────────────────────────────────────────────────

def test_prune_old_drops_only_stale_bars(tmp_path):
    conn = storage.open_db(tmp_path / "c.db")
    now = datetime.now(tz=timezone.utc)
    old_min = int((now - timedelta(days=120)).timestamp()) // 60
    fresh_min = int((now - timedelta(days=1)).timestamp()) // 60
    storage.upsert_bars(conn, _bars(old_min, [1, 2, 3]), symbol="BTC/USDT")
    storage.upsert_bars(conn, _bars(fresh_min, [4, 5, 6]), symbol="BTC/USDT")

    removed = storage.prune_old(conn, retention_days=90)
    assert removed == 3
    assert storage.count_bars(conn, "BTC/USDT") == 3  # only the fresh ones remain
    conn.close()


def test_prune_zero_retention_is_noop(tmp_path):
    conn = storage.open_db(tmp_path / "c.db")
    storage.upsert_bars(conn, _bars(0, [1, 2, 3]), symbol="BTC/USDT")
    assert storage.prune_old(conn, retention_days=0) == 0
    assert storage.count_bars(conn) == 3
    conn.close()


# ─── symbol mapping / parsing ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "symbol,exchange,expected",
    [
        ("BTC/USDT", "bitget", "BTC/USDT:USDT"),
        ("ETH/USDT", "bitget", "ETH/USDT:USDT"),
        ("BTC/USDT:USDT", "bitget", "BTC/USDT:USDT"),  # already perp
        ("BTC/USDT", "binance", "BTC/USDT"),           # non-bitget untouched
        ("USDT.D", "bitget", "USDT.D"),                # dominance untouched
    ],
)
def test_perp_symbol_mapping(symbol, exchange, expected):
    assert perp_symbol(symbol, exchange) == expected


def test_parse_symbols_dedupes_preserving_order(monkeypatch):
    monkeypatch.setenv("CRYPTO_OHLCV_SYMBOLS", "BTC/USDT, ETH/USDT ,BTC/USDT,SOL/USDT")
    assert daemon._parse_symbols() == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_parse_symbols_defaults(monkeypatch):
    monkeypatch.delenv("CRYPTO_OHLCV_SYMBOLS", raising=False)
    syms = daemon._parse_symbols()
    assert "BTC/USDT" in syms and len(syms) >= 5
