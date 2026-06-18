"""Tests for the per-session summary builder (dashboard Sessions page)."""
from datetime import datetime, time as dtime, timezone

import numpy as np
import pandas as pd

from pineforge_ai import session_summary as ss


def _synthetic_uptrend(end: datetime, periods: int = 6 * 96) -> pd.DataFrame:
    """15m UTC OHLCV with a steady uptrend so move_pct > 0."""
    idx = pd.date_range(end=end, periods=periods, freq="15min", tz="UTC")
    close = np.linspace(100.0, 200.0, periods)
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(periods, 1000.0),
        },
        index=idx,
    )
    return df


def test_last_completed_window_overnight():
    # Sydney 21:00 -> 06:00 (overnight). At 08:00 UTC the last completed window
    # opened the previous day 21:00 and closed today 06:00.
    now = datetime(2026, 6, 10, 8, 0, tzinfo=timezone.utc)
    open_dt, close_dt = ss._last_completed_window(
        now, dtime(21, 0, tzinfo=timezone.utc), dtime(6, 0, tzinfo=timezone.utc)
    )
    assert open_dt == datetime(2026, 6, 9, 21, 0, tzinfo=timezone.utc)
    assert close_dt == datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc)
    assert close_dt <= now


def test_last_completed_window_intraday_not_yet_closed():
    # London+NY 13:30 -> 17:00. At 14:00 today the window is still open, so the
    # last COMPLETED occurrence is yesterday's.
    now = datetime(2026, 6, 10, 14, 0, tzinfo=timezone.utc)
    open_dt, close_dt = ss._last_completed_window(
        now, dtime(13, 30, tzinfo=timezone.utc), dtime(17, 0, tzinfo=timezone.utc)
    )
    assert open_dt.date() == datetime(2026, 6, 9).date()
    assert close_dt <= now


def test_build_summary_structured(monkeypatch):
    end = datetime(2026, 6, 10, 20, 0, tzinfo=timezone.utc)
    df = _synthetic_uptrend(end)
    monkeypatch.setattr(ss, "fetch_ohlcv", lambda *a, **k: df)

    out = ss.build_session_summary("BTC/USDT", "13:30", "17:00", exchange="bitget")

    assert out["available"] is True
    assert out["symbol"] == "BTC/USDT"
    # Uptrend → bullish session move, price above EMA50.
    assert out["range"]["move_pct"] > 0
    assert out["trend"]["direction"] == "bull"
    assert out["trend"]["price_vs_ema50"] == "above"
    assert out["window"]["candles"] > 0
    # SMC block present (values may be None on synthetic data, keys must exist).
    for key in ("last_event", "sweep", "fvg_bull", "fvg_bear"):
        assert key in out["smc"]


def test_build_summary_no_data(monkeypatch):
    monkeypatch.setattr(ss, "fetch_ohlcv", lambda *a, **k: pd.DataFrame())
    out = ss.build_session_summary("BTC/USDT", "13:30", "17:00")
    assert out["available"] is False
    assert out["reason"] == "no_data"
