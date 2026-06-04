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
