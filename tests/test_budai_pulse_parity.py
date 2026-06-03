"""
Parity test for indicators.budai_pulse vs an independent line-by-line
transcription of budai_pulse.pine (same Pine source of truth as the KryptoLab
port). The reference re-derives osc/trig/cmf with explicit pandas ops, so a
transcription bug surfaces as a mismatch.
"""
import numpy as np
import pandas as pd

from indicators.budai_pulse import (
    budai_pulse,
    budai_pulse_all_timeframes,
    budai_pulse_summary,
)


def _synthetic(n=400, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 100.0 + 8.0 * np.sin(t / 11.0) + 4.0 * np.sin(t / 3.3) + np.cumsum(rng.normal(0, 0.15, n))
    close = base + rng.normal(0, 0.4, n)
    high = np.maximum(close, base) + rng.uniform(0.1, 0.8, n)
    low = np.minimum(close, base) - rng.uniform(0.1, 0.8, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(50, 200, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _fnorm(s, length):
    lo = s.rolling(length, min_periods=1).min()
    hi = s.rolling(length, min_periods=1).max()
    rng = hi - lo
    return (100.0 * (s - lo) / rng).where(rng != 0.0, 50.0)


def _reference(df, p):
    src = (df["high"] + df["low"] + df["close"]) / 3.0   # hlc3
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    n = len(df)

    esa = _ema(src, p["chan_len"])
    de = _ema((src - esa).abs(), p["chan_len"])
    ci = ((src - esa) / (0.015 * de)).where(de != 0.0, 0.0)
    wt1 = _ema(ci, p["avg_len"])
    wt2 = wt1.rolling(p["trig_len"]).mean()

    s = src.to_numpy(float)
    cog = np.zeros(n)
    for t in range(n):
        num = den = 0.0
        for i in range(p["cog_len"]):
            pv = s[t - i] if t - i >= 0 else 0.0
            num += (1 + i) * pv
            den += pv
        cog[t] = (-num / den + (p["cog_len"] + 1) / 2.0) if den != 0.0 else 0.0
    cog = pd.Series(cog, index=df.index)

    mom = close - close.shift(p["mom_len"])

    hl = high - low
    mfm = (((close - low) - (high - close)) / hl).where(hl != 0.0, 0.0)
    vol = volume.where(volume != 0.0, 1.0)
    cmf = ((mfm * vol).rolling(p["mf_len"], min_periods=1).sum()
           / vol.rolling(p["mf_len"], min_periods=1).sum())

    nl = p["norm_len"]
    osc = _fnorm(wt1, nl) * 0.6 + _fnorm(cog, nl) * 0.25 + _fnorm(mom, nl) * 0.15
    trig = _fnorm(wt2, nl)
    bull = (osc > trig) & (osc.shift(1) <= trig.shift(1))
    bear = (osc < trig) & (osc.shift(1) >= trig.shift(1))
    return osc, trig, cmf, bull, bear


def test_pulse_arrays_match_reference():
    df = _synthetic()
    p = dict(chan_len=9, avg_len=12, trig_len=3, cog_len=9, mom_len=10,
             mf_len=14, norm_len=100)
    osc_r, trig_r, cmf_r, bull_r, bear_r = _reference(df, p)
    out = budai_pulse(df)

    pd.testing.assert_series_equal(out["osc"], osc_r, check_names=False, rtol=1e-9, atol=1e-9)
    pd.testing.assert_series_equal(out["trig"], trig_r, check_names=False, rtol=1e-9, atol=1e-9)
    pd.testing.assert_series_equal(out["cmf"], cmf_r, check_names=False, rtol=1e-9, atol=1e-9)
    assert out["bull_cross"].equals(bull_r.rename("bull_cross"))
    assert out["bear_cross"].equals(bear_r.rename("bear_cross"))
    assert (bull_r.sum() + bear_r.sum()) >= 5


def test_pulse_osc_source_changes_output():
    df = _synthetic()
    a = budai_pulse(df, osc_source="hlc3")["osc"]
    b = budai_pulse(df, osc_source="close")["osc"]
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_pulse_summary_shape():
    df = _synthetic()
    res = budai_pulse_all_timeframes({"1h": df, "4h": df})
    summ = budai_pulse_summary(res)
    assert set(summ.keys()) == {"1h", "4h"}
    for tf, s in summ.items():
        assert set(s.keys()) == {"osc", "trig", "zone", "trend", "flow", "cmf", "signal"}
        assert s["zone"] in ("OverBought", "OverSold", "Neutral")
        assert s["trend"] in ("↑ Bull", "↓ Bear")
        assert s["flow"] in ("Accumulation", "Distribution")
