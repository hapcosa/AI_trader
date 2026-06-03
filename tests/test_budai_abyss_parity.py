"""
Parity test for indicators.budai_abyss vs an independent transcription of
budai_abyss.pine (same Pine source of truth as the KryptoLab port).
"""
import numpy as np
import pandas as pd

from indicators.budai_abyss import (
    budai_abyss,
    budai_abyss_all_timeframes,
    budai_abyss_summary,
)


def _synthetic(n=400, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 100.0 + 9.0 * np.sin(t / 9.0) + 5.0 * np.sin(t / 2.7) + np.cumsum(rng.normal(0, 0.2, n))
    close = base + rng.normal(0, 0.4, n)
    high = np.maximum(close, base) + rng.uniform(0.1, 0.9, n)
    low = np.minimum(close, base) - rng.uniform(0.1, 0.9, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(50, 200, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _reference(df, p):
    src = (df["high"] + df["low"] + df["close"]) / 3.0   # hlc3
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    esa = _ema(src, p["chan_len"])
    de = _ema((src - esa).abs(), p["chan_len"])
    ci = ((src - esa) / (0.015 * de)).where(de != 0.0, 0.0)
    wt1 = _ema(ci, p["avg_len"])
    wt2 = wt1.rolling(p["trig_len"]).mean()

    hl = high - low
    mfm = (((close - low) - (high - close)) / hl).where(hl != 0.0, 0.0)
    vol = volume.where(volume != 0.0, 1.0)
    cmf = ((mfm * vol).rolling(p["mf_len"], min_periods=1).sum()
           / vol.rolling(p["mf_len"], min_periods=1).sum())

    bull = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    bear = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    strong_up = bull & (wt1 <= p["os_x"]) & (cmf > 0.0)
    strong_dn = bear & (wt1 >= p["ob_x"]) & (cmf < 0.0)
    return wt1, wt2, cmf, bull, bear, strong_up, strong_dn


def test_abyss_arrays_match_reference():
    df = _synthetic()
    p = dict(chan_len=9, avg_len=12, trig_len=3, mf_len=14, ob_x=53.0, os_x=-53.0)
    wt1_r, wt2_r, cmf_r, bull_r, bear_r, su_r, sd_r = _reference(df, p)
    out = budai_abyss(df)

    pd.testing.assert_series_equal(out["wt1"], wt1_r, check_names=False, rtol=1e-9, atol=1e-9)
    pd.testing.assert_series_equal(out["wt2"], wt2_r, check_names=False, rtol=1e-9, atol=1e-9)
    pd.testing.assert_series_equal(out["cmf"], cmf_r, check_names=False, rtol=1e-9, atol=1e-9)
    assert out["bull_cross"].equals(bull_r.rename("bull_cross"))
    assert out["bear_cross"].equals(bear_r.rename("bear_cross"))
    assert out["strong_up"].equals(su_r.rename("strong_up"))
    assert out["strong_dn"].equals(sd_r.rename("strong_dn"))
    assert (bull_r.sum() + bear_r.sum()) >= 5


def test_abyss_strong_signals_in_extreme_zone():
    df = _synthetic()
    out = budai_abyss(df)
    for i in range(len(df)):
        if bool(out["strong_up"].iloc[i]):
            assert out["wt1"].iloc[i] <= -53.0 and out["cmf"].iloc[i] > 0.0
        if bool(out["strong_dn"].iloc[i]):
            assert out["wt1"].iloc[i] >= 53.0 and out["cmf"].iloc[i] < 0.0


def test_abyss_summary_shape():
    df = _synthetic()
    res = budai_abyss_all_timeframes({"1h": df, "4h": df})
    summ = budai_abyss_summary(res)
    assert set(summ.keys()) == {"1h", "4h"}
    for s in summ.values():
        assert set(s.keys()) == {"wt1", "wt2", "zone", "trend", "flow", "cmf", "signal"}
        assert s["zone"] in ("OverBought-Extreme", "OverSold-Extreme", "Neutral")
