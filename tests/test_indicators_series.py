"""W3-1 — indicator oscillator series for the live charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import indicators_series as iser


def _ohlcv(n: int = 250) -> pd.DataFrame:
    """Synthetic random-walk OHLCV with a UTC minute index."""
    rng = np.random.default_rng(7)
    idx = pd.to_datetime(
        [i * 3600 for i in range(n)], unit="s", utc=True
    )
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": rng.uniform(1, 10, n)},
        index=idx,
    )


def _patch_fetch(monkeypatch, df):
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    # Isolate from any real local candle store (a running daemon may have
    # written AI_trader/data/crypto_ohlcv.db) so these exercise the live path.
    monkeypatch.setattr(iser, "_crypto_store_dfs", lambda *a, **k: {})
    monkeypatch.setattr(
        "pineforge_ai.data.fetcher.fetch_multi_timeframe",
        lambda **kw: {kw["timeframes"][0]: df},
    )


def test_series_pulse_shape(monkeypatch):
    _patch_fetch(monkeypatch, _ohlcv())
    res = iser.build_indicator_series(symbol="BTC/USDT", timeframe="1h", indicator="pulse")
    assert res["indicator"] == "pulse"
    assert res["scale"] == "0-100"
    assert res["ob"] == 80.0 and res["os"] == 20.0
    assert len(res["points"]) > 100
    p = res["points"][-1]
    assert set(p) == {"time", "osc", "trig"}
    assert isinstance(p["time"], int)
    # oscillator settled (not None) on the last bar
    assert p["osc"] is not None and 0.0 <= p["osc"] <= 100.0


def test_series_abyss_is_centered(monkeypatch):
    _patch_fetch(monkeypatch, _ohlcv())
    res = iser.build_indicator_series(symbol="BTC/USDT", timeframe="1h", indicator="abyss")
    assert res["scale"] == "centered"
    assert res["ob"] == 53.0 and res["os"] == -53.0


def test_series_rejects_unknown_indicator():
    with pytest.raises(ValueError):
        iser.build_indicator_series(symbol="BTC/USDT", timeframe="1h", indicator="bogus")


def test_series_requires_symbol():
    with pytest.raises(ValueError):
        iser.build_indicator_series(symbol="  ", timeframe="1h", indicator="pulse")


def test_series_empty_raises(monkeypatch):
    monkeypatch.setattr("pineforge_ai.data.fetcher.detect_source", lambda s: "ccxt")
    monkeypatch.setattr(iser, "_crypto_store_dfs", lambda *a, **k: {})
    monkeypatch.setattr("pineforge_ai.data.fetcher.fetch_multi_timeframe", lambda **kw: {})
    with pytest.raises(RuntimeError):
        iser.build_indicator_series(symbol="BTC/USDT", timeframe="1h", indicator="pulse")


def test_series_dominance_uses_reader(monkeypatch):
    seen = {}

    def _dom(symbol, tfs, candles):
        seen["symbol"] = symbol
        return {tfs[0]: _ohlcv()}

    # ccxt must not be touched for dominance
    monkeypatch.setattr(
        "pineforge_ai.data.fetcher.fetch_multi_timeframe",
        lambda **kw: (_ for _ in ()).throw(AssertionError("ccxt should not be used")),
    )
    monkeypatch.setattr(iser, "_dominance_dfs", _dom)
    res = iser.build_indicator_series(symbol="USDT.D", timeframe="1h", indicator="pulse")
    assert seen["symbol"] == "USDT.D"
    assert len(res["points"]) > 0


def test_series_candles_cap(monkeypatch):
    _patch_fetch(monkeypatch, _ohlcv(300))
    res = iser.build_indicator_series(symbol="BTC/USDT", timeframe="1h", indicator="wavetrend", candles=50)
    assert len(res["points"]) <= 50
