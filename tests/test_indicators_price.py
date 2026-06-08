"""E3b — price + EMA overlay builder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import indicators_price as ip


def _ohlcv(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    idx = pd.to_datetime([i * 3600 for i in range(n)], unit="s", utc=True)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": rng.uniform(1, 10, n)},
        index=idx,
    )


def _patch_fetch(monkeypatch, df):
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    monkeypatch.setattr(ip, "_crypto_store_dfs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pineforge_ai.data.fetcher.fetch_multi_timeframe",
        lambda **kw: {kw["timeframes"][0]: df},
    )


def test_price_overlay_shape(monkeypatch):
    _patch_fetch(monkeypatch, _ohlcv())
    res = ip.build_price_overlay(symbol="BTC/USDT", timeframe="1h", emas=[20, 50])
    assert res["symbol"] == "BTC/USDT" and res["timeframe"] == "1h"
    assert len(res["candles"]) > 100
    c = res["candles"][-1]
    assert set(c) == {"time", "open", "high", "low", "close"}
    assert isinstance(c["time"], int)
    # One EMA line per requested length, tagged + with points.
    assert [e["length"] for e in res["emas"]] == [20, 50]
    assert all(len(e["points"]) > 0 for e in res["emas"])
    p = res["emas"][0]["points"][-1]
    assert set(p) == {"time", "value"}


def test_price_overlay_default_emas(monkeypatch):
    _patch_fetch(monkeypatch, _ohlcv())
    res = ip.build_price_overlay(symbol="BTC/USDT", timeframe="1h")
    assert [e["length"] for e in res["emas"]] == [20, 50, 200]


def test_price_overlay_candles_cap(monkeypatch):
    _patch_fetch(monkeypatch, _ohlcv(300))
    res = ip.build_price_overlay(symbol="BTC/USDT", timeframe="1h", emas=[20], candles=50)
    assert len(res["candles"]) <= 50
    assert len(res["emas"][0]["points"]) <= 50


def test_price_overlay_rejects_empty_symbol():
    with pytest.raises(ValueError):
        ip.build_price_overlay(symbol="  ", timeframe="1h")


def test_price_overlay_rejects_no_valid_ema():
    with pytest.raises(ValueError):
        ip.build_price_overlay(symbol="BTC/USDT", timeframe="1h", emas=[0, -5])


def test_price_overlay_empty_raises(monkeypatch):
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    monkeypatch.setattr(ip, "_crypto_store_dfs", lambda *a, **k: {})
    monkeypatch.setattr("pineforge_ai.data.fetcher.fetch_multi_timeframe", lambda **kw: {})
    with pytest.raises(RuntimeError):
        ip.build_price_overlay(symbol="BTC/USDT", timeframe="1h")
