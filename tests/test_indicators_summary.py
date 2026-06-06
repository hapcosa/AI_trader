"""W1-2 — indicator summaries JSON helper (data source for the notifier digest).

Mocks the OHLCV fetch and the heavy indicator computation so the test stays
fast and offline; asserts name->key mapping, JSON sanitization and validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import indicators_summary as isum


def test_json_safe_handles_numpy_and_nan():
    out = isum._json_safe(
        {
            "f": np.float64(1.25),
            "b": np.bool_(True),
            "i": np.int64(7),
            "nan": float("nan"),
            "nested": [np.float64(2.0), {"x": np.bool_(False)}],
            "s": "ok",
        }
    )
    assert out["f"] == 1.25 and isinstance(out["f"], float)
    assert out["b"] is True and isinstance(out["b"], bool)
    assert out["i"] == 7
    assert out["nan"] is None
    assert out["nested"] == [2.0, {"x": False}]
    assert out["s"] == "ok"


def _patch_pipeline(monkeypatch, raw_summaries):
    """Stub fetch + computation so build_indicators_summary runs offline."""
    monkeypatch.setattr(
        "pineforge_ai.data.fetcher.detect_source", lambda symbol: "ccxt"
    )
    monkeypatch.setattr(
        "pineforge_ai.data.fetcher.fetch_multi_timeframe",
        lambda **kw: {tf: pd.DataFrame({"close": [1.0]}) for tf in kw["timeframes"]},
    )
    monkeypatch.setattr(isum, "_build_indicator_summaries", lambda dfs, ind_list, emit: raw_summaries)


def test_build_maps_names_and_sanitizes(monkeypatch):
    raw = {
        "pulse": {"1h": {"osc": np.float64(62.5), "is_bull": np.bool_(True), "trig": float("nan")}},
        "smc": {"1h": {"bos": np.bool_(False)}},
        "wt": None,  # wavetrend failed
    }
    _patch_pipeline(monkeypatch, raw)

    res = isum.build_indicators_summary(
        symbol="BTC/USDT", timeframes="1h", indicators="pulse,smc,wavetrend"
    )

    assert res["symbol"] == "BTC/USDT"
    assert res["source"] == "ccxt"
    assert res["timeframes"] == ["1h"]
    assert res["indicators"] == ["pulse", "smc", "wavetrend"]
    # Keyed by public indicator name (wavetrend, not 'wt').
    assert set(res["summaries"]) == {"pulse", "smc", "wavetrend"}
    assert res["summaries"]["pulse"]["1h"]["osc"] == 62.5
    assert res["summaries"]["pulse"]["1h"]["is_bull"] is True
    assert res["summaries"]["pulse"]["1h"]["trig"] is None  # NaN sanitized
    assert res["summaries"]["smc"]["1h"]["bos"] is False
    assert res["summaries"]["wavetrend"] is None  # failed computation passes through


def test_build_rejects_unknown_indicator(monkeypatch):
    _patch_pipeline(monkeypatch, {})
    with pytest.raises(ValueError):
        isum.build_indicators_summary(symbol="BTC/USDT", indicators="bogus")


def test_build_requires_symbol():
    with pytest.raises(ValueError):
        isum.build_indicators_summary(symbol="   ", indicators="pulse")


def test_build_rejects_bad_source():
    with pytest.raises(ValueError):
        isum.build_indicators_summary(symbol="BTC/USDT", source="nasdaq")


def test_build_rejects_bad_candles():
    with pytest.raises(ValueError):
        isum.build_indicators_summary(symbol="BTC/USDT", candles=0)


# ─── dominance branch (reads the dominance SQLite, not ccxt) ────────

def test_build_dominance_uses_reader(monkeypatch):
    # ccxt path must NOT be touched for dominance symbols.
    def _boom(**kw):
        raise AssertionError("fetch_multi_timeframe should not be called for dominance")

    monkeypatch.setattr("pineforge_ai.data.fetcher.fetch_multi_timeframe", _boom)
    monkeypatch.setattr(
        "pineforge_ai.usdt_dominance.reader.get_ohlcv",
        lambda **kw: pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0]}),
    )
    monkeypatch.setattr(
        isum, "_build_indicator_summaries",
        lambda dfs, ind_list, emit: {"pulse": {"1h": {"trend": "↑ Bull", "zone": "OverSold"}}},
    )

    res = isum.build_indicators_summary(symbol="USDT.D", timeframes="1h", indicators="pulse")
    assert res["source"] == "dominance"
    assert res["symbol"] == "USDT.D"
    assert res["summaries"]["pulse"]["1h"]["trend"] == "↑ Bull"


def test_build_dominance_others_d_routes_to_reader(monkeypatch):
    seen = {}

    def _reader(**kw):
        seen["symbol"] = kw.get("symbol")
        return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0]})

    monkeypatch.setattr("pineforge_ai.usdt_dominance.reader.get_ohlcv", _reader)
    monkeypatch.setattr(isum, "_build_indicator_summaries", lambda dfs, ind_list, emit: {"pulse": {"1h": {}}})

    res = isum.build_indicators_summary(symbol="OTHERS.D", timeframes="1h", indicators="pulse")
    assert res["source"] == "dominance"
    assert seen["symbol"] == "OTHERS.D"


def test_build_dominance_empty_raises(monkeypatch):
    monkeypatch.setattr(
        "pineforge_ai.usdt_dominance.reader.get_ohlcv",
        lambda **kw: pd.DataFrame(),  # no data
    )
    with pytest.raises(RuntimeError):
        isum.build_indicators_summary(symbol="BTC.D", timeframes="1h", indicators="pulse")


# ─── Bitget-perp default (the platform trades Bitget USDT-M perps) ──

def test_ccxt_symbol_bitget_perp():
    assert isum._ccxt_symbol("BTC/USDT", "bitget") == "BTC/USDT:USDT"
    assert isum._ccxt_symbol("ETH/USDT", "bitget") == "ETH/USDT:USDT"
    # idempotent / already swap
    assert isum._ccxt_symbol("BTC/USDT:USDT", "bitget") == "BTC/USDT:USDT"
    # other exchanges untouched
    assert isum._ccxt_symbol("BTC/USDT", "binance") == "BTC/USDT"
    # dominance (no '/') untouched
    assert isum._ccxt_symbol("USDT.D", "bitget") == "USDT.D"


def test_default_exchange_is_bitget():
    assert isum.DEFAULT_EXCHANGE == "bitget"


def test_build_default_fetches_bitget_perp(monkeypatch):
    captured = {}

    def _fetch(**kw):
        captured["symbol"] = kw["symbol"]
        captured["exchange"] = kw["exchange"]
        return {kw["timeframes"][0]: pd.DataFrame({"close": [1.0]})}

    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    monkeypatch.setattr("pineforge_ai.data.fetcher.fetch_multi_timeframe", _fetch)
    monkeypatch.setattr(isum, "_build_indicator_summaries", lambda dfs, ind_list, emit: {"pulse": {"1h": {}}})

    # No exchange passed → defaults to bitget; crypto pair charted on the perp.
    res = isum.build_indicators_summary(symbol="BTC/USDT", timeframes="1h", indicators="pulse")
    assert captured["exchange"] == "bitget"
    assert captured["symbol"] == "BTC/USDT:USDT"
    assert res["symbol"] == "BTC/USDT"  # response keeps the clean symbol


# ─── HTTP endpoint (thin wrapper) ───────────────────────────────────

def test_endpoint_returns_summary(monkeypatch):
    monkeypatch.setenv("DOMINANCE_DIGEST_ENABLED", "false")  # don't start scheduler
    from fastapi.testclient import TestClient

    captured = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return {"symbol": kwargs["symbol"], "summaries": {"pulse": {"1h": {"osc": 50}}}}

    # The handler imports the function lazily from pineforge_ai.indicators_summary.
    monkeypatch.setattr(
        "pineforge_ai.indicators_summary.build_indicators_summary", _fake_build
    )

    from web.app import app

    with TestClient(app) as client:
        r = client.get(
            "/api/indicators/summary",
            params={"symbol": "BTC/USDT", "tf": "1h", "inds": "pulse"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summaries"]["pulse"]["1h"]["osc"] == 50
    assert captured["symbol"] == "BTC/USDT"
    assert captured["timeframes"] == "1h"
    assert captured["indicators"] == "pulse"


def test_endpoint_400_on_bad_indicator(monkeypatch):
    monkeypatch.setenv("DOMINANCE_DIGEST_ENABLED", "false")
    from fastapi.testclient import TestClient

    def _raise(**kwargs):
        raise ValueError("Invalid indicators: ['bogus']")

    monkeypatch.setattr(
        "pineforge_ai.indicators_summary.build_indicators_summary", _raise
    )

    from web.app import app

    with TestClient(app) as client:
        r = client.get("/api/indicators/summary", params={"symbol": "BTC/USDT", "inds": "bogus"})
    assert r.status_code == 400
