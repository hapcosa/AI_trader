"""
USDT Dominance reader — reads SQLite ticks from daemon.py and returns OHLCV DataFrames.

The daemon stores 1-minute ticks; this module aggregates them to any timeframe on read.
Returns gracefully empty/unavailable if the daemon has not run yet.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "usdt_dominance.db"

RESAMPLE_RULES: dict[str, str] = {
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}

# Threshold (percentage points) to call bull/bear trend
_TREND_THRESHOLD = 0.05


# ─── Internal loader ──────────────────────────────────────────────────────────

def _load_ticks(db_path: Path, days: int = 30) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=["usdt_pct"])

    cutoff_ts = int(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp()
    )
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        df = pd.read_sql(
            "SELECT ts, usdt_pct FROM ticks WHERE ts >= ? ORDER BY ts ASC",
            conn,
            params=(cutoff_ts,),
        )
        conn.close()
    except Exception:
        return pd.DataFrame(columns=["usdt_pct"])

    if df.empty:
        return df

    df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop(columns=["ts"])
    return df


# ─── Public API ───────────────────────────────────────────────────────────────

def get_ohlcv(
    timeframe: str = "1d",
    days: int = 30,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Load minute ticks and resample to OHLCV candles.

    Returns DataFrame with columns [open, high, low, close, volume]
    and UTC DatetimeIndex. Empty DataFrame if no data available.
    """
    rule = RESAMPLE_RULES.get(timeframe, "1D")
    ticks = _load_ticks(db_path, days=days)
    if ticks.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    ohlcv = ticks["usdt_pct"].resample(rule).agg(
        open="first", high="max", low="min", close="last", volume="count"
    ).dropna(subset=["close"])
    return ohlcv


def get_current_value(db_path: Path = DB_PATH) -> float | None:
    """Return most recent usdt_pct tick, or None if no data."""
    ticks = _load_ticks(db_path, days=1)
    if ticks.empty:
        return None
    return float(ticks["usdt_pct"].iloc[-1])


def get_trend(
    timeframe: str = "1h",
    lookback_periods: int = 3,
    db_path: Path = DB_PATH,
) -> str:
    """Classify trend as 'bull' | 'bear' | 'neutral' over last N candles."""
    ohlcv = get_ohlcv(timeframe=timeframe, days=7, db_path=db_path)
    if len(ohlcv) < lookback_periods + 1:
        return "neutral"
    recent = ohlcv["close"].iloc[-1]
    past = ohlcv["close"].iloc[-(lookback_periods + 1)]
    delta = recent - past
    if delta > _TREND_THRESHOLD:
        return "bull"
    if delta < -_TREND_THRESHOLD:
        return "bear"
    return "neutral"


def get_zone(value: float) -> str:
    """Classify USDT dominance zone."""
    if value > 5.0:
        return "High (>5%) — risk-off"
    if value > 3.0:
        return "Mid (3-5%) — neutral"
    return "Low (<3%) — risk-on"


def build_usdt_summary(
    db_path: Path = DB_PATH,
    ohlcv_days: int = 14,
) -> dict:
    """
    Aggregator used by prompt_builder.

    Returns:
        {
            current:   float | None,
            zone:      str,
            trend_1d:  str,
            trend_4h:  str,
            trend_1h:  str,
            ohlcv_1d:  pd.DataFrame,  # last ohlcv_days of 1d candles
            available: bool,
        }
    """
    current = get_current_value(db_path=db_path)
    available = current is not None

    if not available:
        return {
            "current": None,
            "zone": "—",
            "trend_1d": "neutral",
            "trend_4h": "neutral",
            "trend_1h": "neutral",
            "ohlcv_1d": pd.DataFrame(),
            "available": False,
        }

    zone = get_zone(current)
    trend_1d = get_trend("1d", lookback_periods=3, db_path=db_path)
    trend_4h = get_trend("4h", lookback_periods=3, db_path=db_path)
    trend_1h = get_trend("1h", lookback_periods=3, db_path=db_path)
    ohlcv_1d = get_ohlcv("1d", days=ohlcv_days, db_path=db_path)

    return {
        "current": current,
        "zone": zone,
        "trend_1d": trend_1d,
        "trend_4h": trend_4h,
        "trend_1h": trend_1h,
        "ohlcv_1d": ohlcv_1d,
        "available": True,
    }
