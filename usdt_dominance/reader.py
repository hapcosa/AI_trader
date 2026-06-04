"""
USDT Dominance reader — reads SQLite bars from the TradingView daemon
(`usdt_dominance_tv`) and returns OHLCV DataFrames resampled to any timeframe.

Backwards compatible with the old `ticks(ts, usdt_pct)` schema: if the new
`bars_1m` table is absent, falls back to the legacy table where o=h=l=c=usdt_pct.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "usdt_dominance.db"

RESAMPLE_RULES: dict[str, str] = {
    "1m":  "1min",
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
    "1w":  "1W",
}

# Threshold (percentage points) to call bull/bear trend
_TREND_THRESHOLD = 0.05


# ─── Internal loader ──────────────────────────────────────────────────────────

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def _load_bars(
    db_path: Path,
    days: int = 30,
    symbol: str = "USDT.D",
) -> pd.DataFrame:
    """Return DataFrame [open, high, low, close, volume] indexed by UTC datetime.

    ``symbol`` selects the dominance series in the multi-series ``bars_1m``
    schema. Legacy single-series DBs (no ``symbol`` column) are read as-is.
    """
    if not db_path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    cutoff_ts = int(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp()
    )
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        if _table_exists(conn, "bars_1m"):
            if _has_column(conn, "bars_1m", "symbol"):
                df = pd.read_sql(
                    "SELECT ts, open, high, low, close, volume FROM bars_1m "
                    "WHERE symbol = ? AND ts >= ? ORDER BY ts ASC",
                    conn,
                    params=(symbol, cutoff_ts),
                )
            else:
                df = pd.read_sql(
                    "SELECT ts, open, high, low, close, volume FROM bars_1m "
                    "WHERE ts >= ? ORDER BY ts ASC",
                    conn,
                    params=(cutoff_ts,),
                )
        elif _table_exists(conn, "ticks"):
            legacy = pd.read_sql(
                "SELECT ts, usdt_pct FROM ticks WHERE ts >= ? ORDER BY ts ASC",
                conn,
                params=(cutoff_ts,),
            )
            if legacy.empty:
                conn.close()
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )
            df = pd.DataFrame({
                "ts":     legacy["ts"],
                "open":   legacy["usdt_pct"],
                "high":   legacy["usdt_pct"],
                "low":    legacy["usdt_pct"],
                "close":  legacy["usdt_pct"],
                "volume": 0.0,
            })
        else:
            conn.close()
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        conn.close()
    except Exception:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop(columns=["ts"])
    return df


# ─── Public API ───────────────────────────────────────────────────────────────

def get_ohlcv(
    timeframe: str = "1d",
    days: int = 30,
    db_path: Path = DB_PATH,
    symbol: str = "USDT.D",
) -> pd.DataFrame:
    """
    Load 1-minute bars and resample to OHLCV at the given timeframe.

    Returns DataFrame with columns [open, high, low, close, volume]
    and UTC DatetimeIndex. Empty DataFrame if no data available.
    """
    rule = RESAMPLE_RULES.get(timeframe, "1D")
    bars = _load_bars(db_path, days=days, symbol=symbol)
    if bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    ohlcv = bars.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return ohlcv


def get_current_value(db_path: Path = DB_PATH, symbol: str = "USDT.D") -> float | None:
    """Return most recent close value, or None if no data."""
    bars = _load_bars(db_path, days=1, symbol=symbol)
    if bars.empty:
        return None
    return float(bars["close"].iloc[-1])


def get_trend(
    timeframe: str = "1h",
    lookback_periods: int = 3,
    db_path: Path = DB_PATH,
    symbol: str = "USDT.D",
) -> str:
    """Classify trend as 'bull' | 'bear' | 'neutral' over last N candles."""
    ohlcv = get_ohlcv(timeframe=timeframe, days=7, db_path=db_path, symbol=symbol)
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
