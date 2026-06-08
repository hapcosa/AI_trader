"""E1.2 — crypto candle-store reader + hybrid routing.

Covers:
  - get_ohlcv resamples 1m store bars to the requested timeframe
  - has_symbol reflects what the store holds
  - missing db / unknown symbol / deep TF return empty (caller falls back live)
  - build_indicator_series reads the store for intraday TFs (no ccxt) and falls
    back to ccxt for 1d/1w or untracked symbols
  - build_indicators_summary serves store TFs from the store, live TFs from ccxt
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import indicators_series as iser
import indicators_summary as isum
from crypto_ohlcv import reader as creader
from crypto_ohlcv_ccxt import storage


def _ohlcv_1m(n: int, start_min: int | None = None) -> pd.DataFrame:
    # Default to bars ending at the current minute so the reader's day-cutoff
    # filter keeps them (the store holds recent history).
    now_min = int(pd.Timestamp.now(tz="UTC").timestamp()) // 60
    base = start_min if start_min is not None else now_min - (n - 1)
    idx = pd.to_datetime([(base + i) * 60 for i in range(n)], unit="s", utc=True)
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
         "volume": rng.uniform(1, 5, n)},
        index=idx,
    )


def _seed_store(path, symbol="BTC/USDT", minutes=600):
    conn = storage.open_db(path)
    storage.upsert_bars(conn, _ohlcv_1m(minutes), symbol=symbol)
    conn.close()


# ─── reader ──────────────────────────────────────────────────────────────────

def test_get_ohlcv_resamples_to_15m(tmp_path):
    db = tmp_path / "c.db"
    _seed_store(db, minutes=600)  # 10h of 1m
    df = creader.get_ohlcv("15m", days=2, db_path=db, symbol="BTC/USDT")
    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # 600 one-minute bars → ~40 fifteen-minute candles (40 or 41 depending on
    # where the current-minute boundary falls relative to the 15m grid).
    assert len(df) in (40, 41)


def test_has_symbol(tmp_path):
    db = tmp_path / "c.db"
    _seed_store(db, symbol="ETH/USDT")
    assert creader.has_symbol("ETH/USDT", db_path=db) is True
    assert creader.has_symbol("DOGE/USDT", db_path=db) is False


def test_get_ohlcv_missing_db_is_empty(tmp_path):
    df = creader.get_ohlcv("1h", days=2, db_path=tmp_path / "nope.db", symbol="BTC/USDT")
    assert df.empty


def test_store_timeframes_excludes_deep(tmp_path):
    assert "1h" in creader.STORE_TIMEFRAMES
    # 4h+ fall back to ccxt live (deep 1m backfill on Bitget is impractical).
    assert "4h" not in creader.STORE_TIMEFRAMES
    assert "1d" not in creader.STORE_TIMEFRAMES
    assert "1w" not in creader.STORE_TIMEFRAMES


# ─── routing: series ─────────────────────────────────────────────────────────

def test_series_reads_store_for_intraday(monkeypatch, tmp_path):
    db = tmp_path / "c.db"
    _seed_store(db, symbol="BTC/USDT", minutes=900)
    monkeypatch.setenv("CRYPTO_OHLCV_DB", str(db))
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    # ccxt must NOT be touched when the store serves the TF.
    monkeypatch.setattr(
        "pineforge_ai.data.fetcher.fetch_multi_timeframe",
        lambda **kw: (_ for _ in ()).throw(AssertionError("ccxt should not be used")),
    )
    res = iser.build_indicator_series(symbol="BTC/USDT", timeframe="15m", indicator="pulse")
    assert res["indicator"] == "pulse"
    assert len(res["points"]) > 0


def test_series_falls_back_to_ccxt_for_deep_tf(monkeypatch, tmp_path):
    db = tmp_path / "c.db"
    _seed_store(db, symbol="BTC/USDT", minutes=900)
    monkeypatch.setenv("CRYPTO_OHLCV_DB", str(db))
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    called = {}

    def _fetch(**kw):
        called["tf"] = kw["timeframes"]
        return {kw["timeframes"][0]: _ohlcv_1m(300)}

    monkeypatch.setattr("pineforge_ai.data.fetcher.fetch_multi_timeframe", _fetch)
    res = iser.build_indicator_series(symbol="BTC/USDT", timeframe="1d", indicator="pulse")
    assert called["tf"] == ["1d"]  # deep TF went live
    assert len(res["points"]) > 0


def test_series_falls_back_for_untracked_symbol(monkeypatch, tmp_path):
    db = tmp_path / "c.db"
    _seed_store(db, symbol="BTC/USDT", minutes=900)
    monkeypatch.setenv("CRYPTO_OHLCV_DB", str(db))
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    called = {}

    def _fetch(**kw):
        called["used"] = True
        return {kw["timeframes"][0]: _ohlcv_1m(300)}

    monkeypatch.setattr("pineforge_ai.data.fetcher.fetch_multi_timeframe", _fetch)
    iser.build_indicator_series(symbol="DOGE/USDT", timeframe="15m", indicator="pulse")
    assert called.get("used") is True  # not in store → live


# ─── routing: summary ────────────────────────────────────────────────────────

def test_summary_splits_store_and_live_tfs(monkeypatch, tmp_path):
    db = tmp_path / "c.db"
    _seed_store(db, symbol="BTC/USDT", minutes=2000)
    monkeypatch.setenv("CRYPTO_OHLCV_DB", str(db))
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    live_calls = {}

    def _fetch(**kw):
        live_calls["tfs"] = kw["timeframes"]
        return {tf: _ohlcv_1m(300) for tf in kw["timeframes"]}

    monkeypatch.setattr("pineforge_ai.data.fetcher.fetch_multi_timeframe", _fetch)
    res = isum.build_indicators_summary(
        symbol="BTC/USDT", timeframes="1h,1d", indicators="pulse"
    )
    # 1h served by store, 1d goes live.
    assert live_calls["tfs"] == ["1d"]
    assert set(res["timeframes"]) == {"1h", "1d"}
