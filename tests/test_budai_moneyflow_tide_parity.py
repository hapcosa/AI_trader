"""
Parity test for indicators.budai_moneyflow_tide vs an independent transcription
of budai_moneyflow_tide.pine (incl. a from-scratch ta.mfi).
"""
import numpy as np
import pandas as pd

from indicators.budai_moneyflow_tide import (
    budai_moneyflow_tide,
    budai_moneyflow_tide_all_timeframes,
    budai_moneyflow_tide_summary,
)


def _synthetic(n=450, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 100.0 + 9.0 * np.sin(t / 10.0) + 5.0 * np.sin(t / 2.9) + np.cumsum(rng.normal(0, 0.2, n))
    close = base + rng.normal(0, 0.4, n)
    high = np.maximum(close, base) + rng.uniform(0.1, 0.9, n)
    low = np.minimum(close, base) - rng.uniform(0.1, 0.9, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(50, 200, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _mfi_ref(src, volume, length):
    s = src.to_numpy(float)
    v = volume.to_numpy(float)
    n = len(s)
    change = np.zeros(n)
    change[1:] = s[1:] - s[:-1]
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - length + 1)
        up = sum(v[j] * s[j] for j in range(lo, i + 1) if (j > 0 and change[j] > 0.0))
        dn = sum(v[j] * s[j] for j in range(lo, i + 1) if (j > 0 and change[j] < 0.0))
        if dn == 0.0:
            out[i] = 100.0 if up > 0.0 else 50.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + up / dn)
    return pd.Series(out, index=src.index)


def _reference(df, p):
    src = (df["high"] + df["low"] + df["close"]) / 3.0
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    fast = _ema(((_mfi_ref(src, volume, p["fast_len"]) - 50.0) * 2.0).clip(-100, 100), p["smooth"])
    slow = _ema(((_mfi_ref(src, volume, p["slow_len"]) - 50.0) * 2.0).clip(-100, 100), p["smooth"])

    hl = high - low
    mfm = (((close - low) - (high - close)) / hl).where(hl != 0.0, 0.0)
    vol = volume.where(volume != 0.0, 1.0)
    cmf = ((mfm * vol).rolling(p["cmf_len"], min_periods=1).sum()
           / vol.rolling(p["cmf_len"], min_periods=1).sum())

    bull = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    bear = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    return fast, slow, cmf, bull, bear


def test_tide_arrays_match_reference():
    df = _synthetic()
    p = dict(fast_len=14, slow_len=28, cmf_len=20, smooth=2)
    fast_r, slow_r, cmf_r, bull_r, bear_r = _reference(df, p)
    out = budai_moneyflow_tide(df)

    pd.testing.assert_series_equal(out["fast"], fast_r, check_names=False, rtol=1e-9, atol=1e-9)
    pd.testing.assert_series_equal(out["slow"], slow_r, check_names=False, rtol=1e-9, atol=1e-9)
    pd.testing.assert_series_equal(out["cmf"], cmf_r, check_names=False, rtol=1e-9, atol=1e-9)
    assert out["bull_cross"].equals(bull_r.rename("bull_cross"))
    assert out["bear_cross"].equals(bear_r.rename("bear_cross"))
    assert (bull_r.sum() + bear_r.sum()) >= 5


def test_tide_strong_signals_in_zone():
    df = _synthetic()
    out = budai_moneyflow_tide(df, os_lvl=-10.0, ob_lvl=10.0)  # relaxed for signals
    n_strong = 0
    for i in range(len(df)):
        if bool(out["strong_up"].iloc[i]):
            assert out["fast"].iloc[i] <= -10.0 and out["cmf"].iloc[i] > 0.0
            n_strong += 1
        if bool(out["strong_dn"].iloc[i]):
            assert out["fast"].iloc[i] >= 10.0 and out["cmf"].iloc[i] < 0.0
            n_strong += 1
    assert n_strong >= 1


def test_tide_summary_shape():
    df = _synthetic()
    res = budai_moneyflow_tide_all_timeframes({"1h": df, "4h": df})
    summ = budai_moneyflow_tide_summary(res)
    assert set(summ.keys()) == {"1h", "4h"}
    for s in summ.values():
        assert set(s.keys()) == {"fast", "slow", "zone", "trend", "flow", "cmf", "signal"}
        assert s["zone"] in ("Distribution-OB", "Accumulation-OS", "Neutral")
