"""
Per-session market summary for the dashboard "Sessions" page (on-demand).

Given a symbol and a daily UTC window [start,end] (the dashboard resolves the
window from its built-in session/overlap table and passes it here), computes a
compact, structured read of the MOST RECENT COMPLETED occurrence of that window:
range / % move, trend vs EMAs (20/50), relative volume, and SMC structure
(BOS/CHoCH, liquidity sweeps, FVGs) at the session close.

Returns STRUCTURED fields only — the frontend composes the human verdict with
i18n so it stays consistent with the EN/ES locale. Reuses the same data fetcher
and `smc_buda` engine as the prompt builder, plus `classics._ema` for EMAs
(intentionally NOT the optional `emas` indicator module so this works
independently of that change).
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta, timezone

import numpy as np

from pineforge_ai.data.fetcher import fetch_ohlcv
from pineforge_ai.indicators.classics import _ema
from pineforge_ai.indicators.smc_buda import smc_all_timeframes, smc_summary

# Intraday detail TF + how much history to pull. 6 days of 15m comfortably
# covers the last completed instance of every daily session window plus enough
# warmup for EMA50 / SMC structure.
DEFAULT_TF = "15m"
DEFAULT_DAYS = 6


def _parse_hhmm(value: str) -> dtime:
    h, m = value.strip().split(":")
    return dtime(int(h), int(m), tzinfo=timezone.utc)


def _last_completed_window(
    now: datetime, start: dtime, end: dtime
) -> tuple[datetime, datetime]:
    """Most recent [open,close] occurrence of the daily window that has already
    CLOSED at `now`. Handles overnight windows (close <= start)."""
    today = now.date()
    open_dt = datetime.combine(today, start.replace(tzinfo=None), tzinfo=timezone.utc)
    close_dt = datetime.combine(today, end.replace(tzinfo=None), tzinfo=timezone.utc)
    if close_dt <= open_dt:
        close_dt += timedelta(days=1)  # overnight
    # Walk back until the window is fully in the past.
    while close_dt > now:
        open_dt -= timedelta(days=1)
        close_dt -= timedelta(days=1)
    return open_dt, close_dt


def _round(value: float | None, decimals: int = 2) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), decimals)


def build_session_summary(
    symbol: str,
    start_hhmm: str,
    end_hhmm: str,
    *,
    exchange: str = "bitget",
    source: str = "auto",
    tf: str = DEFAULT_TF,
) -> dict:
    """Compute the structured summary. Raises ValueError on bad input."""
    start = _parse_hhmm(start_hhmm)
    end = _parse_hhmm(end_hhmm)

    df = fetch_ohlcv(symbol, tf, days=DEFAULT_DAYS, source=source, exchange=exchange)
    if df is None or df.empty:
        return {"available": False, "reason": "no_data", "symbol": symbol, "tf": tf}

    now = df.index[-1].to_pydatetime()
    open_dt, close_dt = _last_completed_window(now, start, end)
    seg = df[(df.index >= open_dt) & (df.index < close_dt)]
    if seg.empty:
        return {"available": False, "reason": "no_candles", "symbol": symbol, "tf": tf}

    o = float(seg["open"].iloc[0])
    h = float(seg["high"].max())
    low = float(seg["low"].min())
    c = float(seg["close"].iloc[-1])
    move_pct = (c - o) / o * 100.0 if o else 0.0
    range_pct = (h - low) / o * 100.0 if o else 0.0

    # EMAs computed on the full series UP TO the session close (no lookahead).
    upto = df[df.index < close_dt]["close"]
    ema20 = _ema(upto, 20).iloc[-1] if len(upto) else np.nan
    ema50 = _ema(upto, 50).iloc[-1] if len(upto) else np.nan
    price_vs_ema50 = (
        None if np.isnan(ema50) else ("above" if c >= ema50 else "below")
    )
    if not np.isnan(ema20) and not np.isnan(ema50):
        ema_stack = "bull" if ema20 > ema50 else ("bear" if ema20 < ema50 else "flat")
    else:
        ema_stack = None

    direction = "bull" if move_pct > 0.1 else ("bear" if move_pct < -0.1 else "flat")

    # Relative volume: this session's mean candle volume vs the overall mean.
    overall_vol = float(df["volume"].mean()) or 0.0
    rel_volume = (
        _round(float(seg["volume"].mean()) / overall_vol) if overall_vol else None
    )

    # SMC structure as of the session close (slice up to close → read last bar).
    smc: dict = {}
    try:
        sliced = df[df.index < close_dt]
        smc_res = smc_all_timeframes({tf: sliced})
        smc = smc_summary(smc_res).get(tf, {}) or {}
    except Exception:
        smc = {}

    sweep = "down" if smc.get("dnsweep") else ("up" if smc.get("upsweep") else None)

    return {
        "available": True,
        "symbol": symbol,
        "tf": tf,
        "window": {
            "start_utc": start_hhmm,
            "end_utc": end_hhmm,
            "open_iso": open_dt.isoformat(),
            "close_iso": close_dt.isoformat(),
            "candles": int(len(seg)),
        },
        "range": {
            "open": _round(o),
            "high": _round(h),
            "low": _round(low),
            "close": _round(c),
            "move_pct": _round(move_pct),
            "range_pct": _round(range_pct),
        },
        "trend": {
            "direction": direction,
            "price_vs_ema50": price_vs_ema50,
            "ema_stack": ema_stack,
            "ema20": _round(None if np.isnan(ema20) else float(ema20)),
            "ema50": _round(None if np.isnan(ema50) else float(ema50)),
        },
        "volatility": {
            "range_pct": _round(range_pct),
            "rel_volume": rel_volume,
        },
        "smc": {
            "last_event": smc.get("last_event") or None,
            "sweep": sweep,
            "fvg_bull": smc.get("fvg_bull") if smc.get("fvg_bull") != "—" else None,
            "fvg_bear": smc.get("fvg_bear") if smc.get("fvg_bear") != "—" else None,
            "confluence": smc.get("confluence"),
        },
    }
