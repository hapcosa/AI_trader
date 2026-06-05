"""W3-2 — macro snapshot (Fear & Greed + tickers + dominance)."""
from __future__ import annotations

import pandas as pd
import pytest

import indicators_macro as im


def test_fetch_fear_greed_parses(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"value": "61", "value_classification": "Greed"}]}

    monkeypatch.setattr(im.requests, "get", lambda *a, **k: _Resp())
    fng = im._fetch_fear_greed()
    assert fng == {"value": 61, "classification": "Greed"}


def test_fetch_fear_greed_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(im.requests, "get", _boom)
    assert im._fetch_fear_greed() is None


def test_build_macro_summary_shape(monkeypatch):
    # Stub each feed so it runs offline.
    monkeypatch.setattr(im, "_fetch_fear_greed", lambda: {"value": 50, "classification": "Neutral"})

    def _daily(ticker, days=10):
        return pd.DataFrame({"close": [100.0, 101.0, 102.0]})

    monkeypatch.setattr(im, "_fetch_daily", _daily)
    monkeypatch.setattr(
        "pineforge_ai.usdt_dominance.reader.get_current_value",
        lambda **kw: 5.0 if kw.get("symbol") == "USDT.D" else 55.0,
    )

    res = im.build_macro_summary()
    assert res["fear_greed"]["value"] == 50
    keys = {m["key"] for m in res["markets"]}
    assert {"DXY", "SP500", "NASDAQ", "GOLD", "VIX", "US10Y", "ETH_BTC"} <= keys
    sp = next(m for m in res["markets"] if m["key"] == "SP500")
    assert sp["close"] == 102.0 and sp["change_1d_pct"] is not None
    dom = {d["key"]: d["value"] for d in res["dominance"]}
    assert dom["USDT.D"] == 5.0 and dom["BTC.D"] == 55.0


def test_build_macro_summary_tolerates_dead_feeds(monkeypatch):
    monkeypatch.setattr(im, "_fetch_fear_greed", lambda: None)
    monkeypatch.setattr(im, "_fetch_daily", lambda ticker, days=10: None)
    monkeypatch.setattr(
        "pineforge_ai.usdt_dominance.reader.get_current_value",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("no db")),
    )
    res = im.build_macro_summary()
    assert res["fear_greed"] is None
    assert all(m["close"] is None for m in res["markets"])
    assert all(d["value"] is None for d in res["dominance"])


def test_endpoint_returns_macro(monkeypatch):
    monkeypatch.setenv("DOMINANCE_DIGEST_ENABLED", "false")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        "pineforge_ai.indicators_macro.build_macro_summary",
        lambda: {"fear_greed": {"value": 42, "classification": "Fear"}, "markets": [], "dominance": []},
    )
    from web.app import app

    with TestClient(app) as client:
        r = client.get("/api/indicators/macro")
    assert r.status_code == 200, r.text
    assert r.json()["fear_greed"]["value"] == 42
