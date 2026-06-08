"""Live ticker for the /indicators header."""
from __future__ import annotations

import pytest

import indicators_ticker as it


class _FakeEx:
    def __init__(self):
        self.markets = {"BTC/USDT:USDT": {}}
        self.calls = 0

    def fetch_ticker(self, market):
        self.calls += 1
        assert market == "BTC/USDT:USDT"  # clean pair → perp
        return {"last": 63000.5, "bid": 63000.0, "ask": 63001.0,
                "percentage": 1.23, "timestamp": 1780945200000}


def test_build_ticker_maps_perp_and_shape(monkeypatch):
    fake = _FakeEx()
    monkeypatch.setattr(it, "_get_exchange", lambda ex: fake)
    out = it.build_ticker(symbol="BTC/USDT")
    assert out == {
        "symbol": "BTC/USDT", "last": 63000.5, "bid": 63000.0,
        "ask": 63001.0, "change_pct": 1.23, "time": 1780945200000,
    }


def test_build_ticker_rejects_empty_symbol():
    with pytest.raises(ValueError):
        it.build_ticker(symbol="  ")


def test_build_ticker_unknown_symbol(monkeypatch):
    monkeypatch.setattr(it, "_get_exchange", lambda ex: _FakeEx())
    with pytest.raises(ValueError):
        it.build_ticker(symbol="NOPE/USDT")


def test_build_ticker_exchange_error_is_runtime(monkeypatch):
    class _Boom(_FakeEx):
        def fetch_ticker(self, market):
            raise Exception("network down")

    monkeypatch.setattr(it, "_get_exchange", lambda ex: _Boom())
    with pytest.raises(RuntimeError):
        it.build_ticker(symbol="BTC/USDT")
