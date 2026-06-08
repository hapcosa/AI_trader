"""E3a — classic oscillators (RSI, Stochastic, MACD) + series payload extensions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import indicators_series as iser
from indicators.classics import macd, rsi, stochastic


def _ohlcv(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(11)
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
    monkeypatch.setattr(iser, "_crypto_store_dfs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pineforge_ai.data.fetcher.fetch_multi_timeframe",
        lambda **kw: {kw["timeframes"][0]: df},
    )


# ─── pure compute ────────────────────────────────────────────────────────────

def test_rsi_shape_and_bounds():
    out = rsi(_ohlcv())
    assert list(out.columns) == ["rsi", "rsi_signal"]
    v = out["rsi"].dropna()
    assert ((v >= 0) & (v <= 100)).all()


def test_stochastic_shape_and_bounds():
    out = stochastic(_ohlcv())
    assert list(out.columns) == ["k", "d"]
    v = out["k"].dropna()
    assert ((v >= 0) & (v <= 100)).all()


def test_macd_has_hist_equal_to_line_minus_signal():
    out = macd(_ohlcv())
    assert list(out.columns) == ["macd", "signal", "hist"]
    tail = out.dropna().iloc[-1]
    assert abs(tail["hist"] - (tail["macd"] - tail["signal"])) < 1e-9


# ─── series payload ──────────────────────────────────────────────────────────

def test_series_rsi(monkeypatch):
    _patch_fetch(monkeypatch, _ohlcv())
    res = iser.build_indicator_series(symbol="BTC/USDT", timeframe="1h", indicator="rsi")
    assert res["scale"] == "0-100"
    assert res["ob"] == 70.0 and res["os"] == 30.0
    p = res["points"][-1]
    assert set(p) == {"time", "osc", "trig"}  # no hist for rsi
    assert 0.0 <= p["osc"] <= 100.0


def test_series_stochastic(monkeypatch):
    _patch_fetch(monkeypatch, _ohlcv())
    res = iser.build_indicator_series(symbol="BTC/USDT", timeframe="1h", indicator="stochastic")
    assert res["scale"] == "0-100"
    assert res["ob"] == 80.0 and res["os"] == 20.0


def test_series_macd_has_hist_and_null_obos(monkeypatch):
    _patch_fetch(monkeypatch, _ohlcv())
    res = iser.build_indicator_series(symbol="BTC/USDT", timeframe="1h", indicator="macd")
    assert res["scale"] == "centered"
    assert res["ob"] is None and res["os"] is None
    p = res["points"][-1]
    assert "hist" in p
    assert p["hist"] is not None


def test_classics_registered():
    for name in ("rsi", "stochastic", "macd"):
        assert name in iser.SERIES_INDICATORS
