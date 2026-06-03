"""
Parity test for indicators.budai_athenea vs an independent transcription of
budai_athenea_oscillator.pine. The slope is cross-checked: the module computes
the OLS slope; the reference computes linreg(0)-linreg(1), so a match confirms
that identity.
"""
import numpy as np
import pandas as pd

from indicators.budai_athenea import (
    budai_athenea,
    budai_athenea_all_timeframes,
    budai_athenea_summary,
)


def _synthetic(n=500, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 100.0 + 10.0 * np.sin(t / 13.0) + 6.0 * np.sin(t / 3.1) + np.cumsum(rng.normal(0, 0.25, n))
    close = base + rng.normal(0, 0.5, n)
    high = np.maximum(close, base) + rng.uniform(0.1, 1.0, n)
    low = np.minimum(close, base) - rng.uniform(0.1, 1.0, n)
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


def _linreg(s, length, offset):
    """ta.linreg(s, length, offset) via rolling OLS."""
    def _val(y):
        x = np.arange(len(y), dtype=float)
        m, b = np.polyfit(x, y, 1)
        return m * (len(y) - 1 - offset) + b
    return s.rolling(length).apply(_val, raw=True)


def _tr(high, low, close):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def _reference(df, p):
    src = (df["high"] + df["low"] + df["close"]) / 3.0
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    n = len(df)

    esa = _ema(src, p["chan_len"])
    de = _ema((src - esa).abs(), p["chan_len"])
    ci = ((src - esa) / (0.015 * de)).where(de != 0.0, 0.0)
    wt1 = _ema(ci, p["avg_len"])
    wt2 = wt1.rolling(p["trig_len"]).mean()

    slope = _linreg(src, p["slope_len"], 0) - _linreg(src, p["slope_len"], 1)

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

    nl = p["norm_len"]
    osc = _fnorm(wt1, nl) * 0.5 + _fnorm(slope, nl) * 0.3 + _fnorm(cog, nl) * 0.2
    trig = _fnorm(wt2, nl)

    hl = high - low
    mfm = (((close - low) - (high - close)) / hl).where(hl != 0.0, 0.0)
    vol = volume.where(volume != 0.0, 1.0)
    cmf = ((mfm * vol).rolling(p["mf_len"], min_periods=1).sum()
           / vol.rolling(p["mf_len"], min_periods=1).sum())

    basis = src.rolling(p["bb_len"]).mean()
    dev = p["bb_mult"] * src.rolling(p["bb_len"]).std(ddof=0)
    bb_up, bb_lo = basis + dev, basis - dev
    kc_ma = src.rolling(p["kc_len"]).mean()
    kc_rng = _tr(high, low, close)
    kc_rng_ma = kc_rng.rolling(p["kc_len"]).mean()
    kc_up, kc_lo = kc_ma + kc_rng_ma * p["kc_mult"], kc_ma - kc_rng_ma * p["kc_mult"]
    sqz_on = (bb_lo > kc_lo) & (bb_up < kc_up)
    released = sqz_on.shift(1, fill_value=False) & (~sqz_on)

    hhv = src.rolling(p["vix_len"], min_periods=1).max()
    vix_raw = ((hhv - low) / hhv * 100.0).where(hhv != 0.0, 0.0)
    vix_fix = _ema(vix_raw, 2)
    b_up_v = vix_fix.rolling(p["bb_len_v"]).mean() + p["bb_mult_v"] * vix_fix.rolling(p["bb_len_v"]).std(ddof=0)
    range_hi = vix_fix.rolling(p["ph_len"], min_periods=1).max() * p["ph_up"]
    panic = (vix_fix >= b_up_v) | (vix_fix >= range_hi)

    cu = (osc > trig) & (osc.shift(1) <= trig.shift(1))
    cd = (osc < trig) & (osc.shift(1) >= trig.shift(1))
    entry_buy = cu & ((osc < p["os_lvl"]) | panic) & (cmf > 0.0)
    entry_sell = cd & (osc > p["ob_lvl"]) & (cmf < 0.0)
    sqz_buy = released & (osc > 50.0) & (cmf > 0.0)
    sqz_sell = released & (osc < 50.0) & (cmf < 0.0)
    return osc, trig, cmf, entry_buy | sqz_buy, entry_sell | sqz_sell


_PARAMS = dict(chan_len=9, avg_len=12, trig_len=3, slope_len=20, cog_len=9,
               norm_len=100, mf_len=14, ob_lvl=80.0, os_lvl=20.0,
               bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5,
               vix_len=22, bb_len_v=20, bb_mult_v=2.0, ph_len=50, ph_up=0.85)


def test_athenea_arrays_match_reference():
    df = _synthetic()
    osc_r, trig_r, cmf_r, buy_r, sell_r = _reference(df, _PARAMS)
    out = budai_athenea(df)

    pd.testing.assert_series_equal(out["osc"], osc_r, check_names=False, rtol=1e-7, atol=1e-7)
    pd.testing.assert_series_equal(out["trig"], trig_r, check_names=False, rtol=1e-9, atol=1e-9)
    pd.testing.assert_series_equal(out["cmf"], cmf_r, check_names=False, rtol=1e-9, atol=1e-9)
    assert out["signal_buy"].equals(buy_r.rename("signal_buy"))
    assert out["signal_sell"].equals(sell_r.rename("signal_sell"))
    assert (buy_r.sum() + sell_r.sum()) >= 3


def test_athenea_summary_shape():
    df = _synthetic()
    res = budai_athenea_all_timeframes({"1h": df, "4h": df})
    summ = budai_athenea_summary(res)
    assert set(summ.keys()) == {"1h", "4h"}
    for s in summ.values():
        assert set(s.keys()) == {"osc", "trig", "zone", "trend", "flow", "panic", "squeeze_release", "signal"}
        assert s["zone"] in ("OverBought", "OverSold", "Neutral")
