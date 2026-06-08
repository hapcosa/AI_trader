"""Per-timeframe price range for the /indicators header."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import indicators_ranges as ir


def _ohlcv(n: int = 5) -> pd.DataFrame:
    idx = pd.to_datetime([i * 3600 for i in range(n)], unit="s", utc=True)
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {"open": close, "high": close + 2, "low": close - 1, "close": close, "volume": [1.0] * n},
        index=idx,
    )


def _patch(monkeypatch, df):
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    monkeypatch.setattr(ir, "_crypto_store_dfs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pineforge_ai.data.fetcher.fetch_multi_timeframe",
        lambda **kw: {kw["timeframes"][0]: df},
    )


def test_ranges_shape_and_pct(monkeypatch):
    _patch(monkeypatch, _ohlcv())
    res = ir.build_ranges(symbol="BTC/USDT", timeframes=["15m", "1h"])
    assert res["symbol"] == "BTC/USDT"
    assert set(res["ranges"]) == {"15m", "1h"}
    r = res["ranges"]["1h"]
    assert set(r) == {"high", "low", "last", "range_pct"}
    # last candle: high=close+2, low=close-1 → range = 3/low*100
    assert r["range_pct"] == pytest.approx((r["high"] - r["low"]) / r["low"] * 100.0)


def test_ranges_default_timeframes(monkeypatch):
    _patch(monkeypatch, _ohlcv())
    res = ir.build_ranges(symbol="BTC/USDT")
    assert list(res["ranges"]) == ["15m", "1h", "4h", "1d"]


def test_ranges_missing_tf_is_none(monkeypatch):
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    monkeypatch.setattr(ir, "_crypto_store_dfs", lambda *a, **k: {})
    monkeypatch.setattr("pineforge_ai.data.fetcher.fetch_multi_timeframe", lambda **kw: {})
    res = ir.build_ranges(symbol="BTC/USDT", timeframes=["1h"])
    assert res["ranges"]["1h"] is None


def test_ranges_rejects_empty_symbol():
    with pytest.raises(ValueError):
        ir.build_ranges(symbol="  ")
