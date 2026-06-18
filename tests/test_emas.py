"""Tests for the EMAs indicator."""
import numpy as np
import pandas as pd

from indicators.emas import emas, emas_all_timeframes, emas_summary


def _synthetic_uptrend(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Strong uptrend with minor noise so EMA20 > EMA50 > EMA200 at the end."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # Strong linear trend + a cyclical component + small noise.
    base = 100.0 + 0.5 * t + 5.0 * np.sin(t / 20.0)
    close = base + rng.normal(0, 0.8, n)
    high = np.maximum(close, base) + rng.uniform(0.2, 1.5, n)
    low = np.minimum(close, base) - rng.uniform(0.2, 1.5, n)
    open_ = close + rng.normal(0, 0.4, n)
    volume = rng.uniform(50, 200, n)
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="1h")
    return df


def test_emas_outputs_columns_and_bull_stack():
    df = _synthetic_uptrend()
    out = emas(df)

    assert list(out.columns) == ["close", "ema20", "ema50", "ema200"]
    assert len(out) == len(df)

    last = out.iloc[-1]
    assert last["ema20"] > last["ema50"] > last["ema200"]
    assert last["close"] > last["ema20"]


def test_emas_all_timeframes_shape():
    df = _synthetic_uptrend()
    res = emas_all_timeframes({"1h": df, "4h": df})
    assert set(res.keys()) == {"1h", "4h"}
    for tf, out in res.items():
        assert list(out.columns) == ["close", "ema20", "ema50", "ema200"]
        assert len(out) == len(df)


def test_emas_summary_keys_and_price_vs():
    df = _synthetic_uptrend()
    res = emas_all_timeframes({"1h": df})
    summ = emas_summary(res)

    assert set(summ.keys()) == {"1h"}
    s = summ["1h"]
    expected_keys = {
        "close", "ema20", "ema50", "ema200",
        "price_vs_ema20", "price_vs_ema50", "price_vs_ema200",
        "stack", "slope20",
    }
    assert set(s.keys()) == expected_keys

    assert s["stack"] == "bull"
    assert s["price_vs_ema20"] == "above"
    assert s["price_vs_ema50"] == "above"
    assert s["price_vs_ema200"] == "above"
    assert s["slope20"] in ("up", "flat")

    # Values are rounded floats (or "—" for NaN).
    assert isinstance(s["close"], (int, float))
    assert isinstance(s["ema20"], (int, float))


def test_emas_summary_empty_df():
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    res = emas_all_timeframes({"1h": empty})
    summ = emas_summary(res)
    assert summ == {}
