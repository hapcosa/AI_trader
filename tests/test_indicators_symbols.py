"""Symbol catalog for the /indicators pair picker."""
from __future__ import annotations

import indicators_symbols as isym
from crypto_ohlcv import reader as creader
from crypto_ohlcv_ccxt import storage


def _seed(path, symbols):
    conn = storage.open_db(path)
    import pandas as pd
    idx = pd.to_datetime([0, 60], unit="s", utc=True)
    df = pd.DataFrame({"open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1], "volume": [0, 0]}, index=idx)
    for s in symbols:
        storage.upsert_bars(conn, df, symbol=s)
    conn.close()


def test_store_symbols(tmp_path):
    db = tmp_path / "c.db"
    _seed(db, ["ETH/USDT", "BTC/USDT"])
    assert creader.store_symbols(db_path=db) == ["BTC/USDT", "ETH/USDT"]  # sorted


def test_store_symbols_missing_db(tmp_path):
    assert creader.store_symbols(db_path=tmp_path / "nope.db") == []


def test_build_symbols_sections(tmp_path, monkeypatch):
    db = tmp_path / "c.db"
    _seed(db, ["BTC/USDT", "Z-CUSTOM/USDT"])  # one in catalog, one not
    monkeypatch.setenv("CRYPTO_OHLCV_DB", str(db))
    out = isym.build_symbols()
    assert out["downloaded"] == ["BTC/USDT", "Z-CUSTOM/USDT"]
    # catalog is the curated set unioned with downloaded, sorted + deduped.
    assert "BTC/USDT" in out["catalog"]
    assert "ETH/USDT" in out["catalog"]          # from CATALOG
    assert "Z-CUSTOM/USDT" in out["catalog"]      # downloaded but not in CATALOG
    assert out["catalog"] == sorted(set(out["catalog"]))
    # commodities are their own section (perp metals/energy), not in catalog.
    assert "XAU/USDT" in out["commodities"]
    assert "NATGAS/USDT" in out["commodities"]
    assert "XAU/USDT" not in out["catalog"]
