"""SQLite cache of OHLCV per (symbol, timeframe) — multi-source, on-demand refresh.

Source is encoded as the prefix of the cached symbol key:
    "CRYPTOCAP:USDT.D"   → TradingView (default for any unknown prefix)
    "BINANCE:BTCUSDT"    → TradingView via BINANCE exchange feed
    "OANDA:XAUUSD"       → TradingView via OANDA
    "BITGET:BTCUSDT"     → ccxt Bitget futures (USDT-perp)

Bare keys without a colon are migrated to ``CRYPTOCAP:`` on first DB open
(legacy "USDT.D" data preserved).
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "usdt_dominance_tfs.db"

# TFs to cache + target bar count (warmup-sufficient for WT/AMO/SMC)
TARGET_BARS: dict[str, int] = {
    "1h": 500,
    "4h": 400,
    "1d": 300,
    "1w": 200,
    "1M": 100,  # TV cap: CRYPTOCAP:USDT.D starts 2018, ~100 monthly bars max.
}

# Seconds per TF — used for staleness check.
# 1M approximated to 30d.
TF_SECONDS: dict[str, int] = {
    "1h":  3600,
    "4h":  14400,
    "1d":  86400,
    "1w":  604800,
    "1M":  2592000,
}

# tvDatafeed Interval mapping — resolved lazily so the module imports even when
# the library is missing (e.g. host context outside the daemon container).
_INTERVAL_NAME: dict[str, str] = {
    "1h": "in_1_hour",
    "4h": "in_4_hour",
    "1d": "in_daily",
    "1w": "in_weekly",
    "1M": "in_monthly",
}

# ccxt Bitget timeframe codes (no remap needed for these labels).
_CCXT_TF: dict[str, str] = {
    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w", "1M": "1M",
}


def _normalize_symbol(symbol: str) -> str:
    """Map bare legacy tickers to FQ EXCHANGE:SYMBOL keys."""
    return symbol if ":" in symbol else f"CRYPTOCAP:{symbol}"


def _split_key(symbol_key: str) -> tuple[str, str]:
    """Return (exchange_or_source, ticker)."""
    if ":" in symbol_key:
        a, b = symbol_key.split(":", 1)
        return a.upper(), b
    return "CRYPTOCAP", symbol_key


_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_sym_tf_ts ON bars(symbol, timeframe, ts DESC);
"""


def open_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(_SCHEMA)
    # One-time migration: bare tickers → CRYPTOCAP-prefixed FQ keys.
    try:
        conn.execute(
            "UPDATE bars SET symbol = 'CRYPTOCAP:' || symbol "
            "WHERE instr(symbol, ':') = 0"
        )
        conn.commit()
    except Exception:
        pass
    return conn


def get_last_ts(conn: sqlite3.Connection, symbol: str, tf: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(ts) FROM bars WHERE symbol=? AND timeframe=?", (symbol, tf)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def count_bars(conn: sqlite3.Connection, symbol: str, tf: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM bars WHERE symbol=? AND timeframe=?", (symbol, tf)
    ).fetchone()
    return int(row[0]) if row else 0


def upsert_df(
    conn: sqlite3.Connection,
    symbol: str,
    tf: str,
    df: pd.DataFrame,
) -> int:
    if df is None or df.empty:
        return 0
    rows: list[tuple] = []
    for idx, r in df.iterrows():
        ts_dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        ts = int(ts_dt.timestamp())
        try:
            rows.append((
                symbol, tf, ts,
                float(r["open"]), float(r["high"]),
                float(r["low"]),  float(r["close"]),
                float(r.get("volume", 0.0) or 0.0),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO bars (symbol, timeframe, ts, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET
             open=excluded.open, high=excluded.high, low=excluded.low,
             close=excluded.close, volume=excluded.volume""",
        rows,
    )
    conn.commit()
    return len(rows)


def get_ohlcv(
    symbol: str,
    tf: str,
    limit: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    """Return OHLCV DataFrame for (symbol, tf) ordered oldest→newest, UTC index."""
    symbol = _normalize_symbol(symbol)
    own = conn is None
    if own:
        conn = open_db()
    try:
        n = limit or TARGET_BARS.get(tf, 500)
        cur = conn.execute(
            """SELECT ts, open, high, low, close, volume
                 FROM bars
                WHERE symbol=? AND timeframe=?
                ORDER BY ts DESC
                LIMIT ?""",
            (symbol, tf, n),
        )
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.drop(columns=["ts"]).sort_index()
        return df
    finally:
        if own:
            conn.close()


def _fetch_tv(exchange: str, symbol: str, tf: str, n_bars: int) -> pd.DataFrame:
    """Pull n_bars from TradingView for (exchange:symbol, tf)."""
    import os
    try:
        from tvDatafeed import TvDatafeed, Interval
    except Exception as e:
        log.error("tvdatafeed not importable: %s", e)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    interval = getattr(Interval, _INTERVAL_NAME[tf])
    u, p = os.environ.get("TV_USERNAME"), os.environ.get("TV_PASSWORD")
    try:
        tv = TvDatafeed(username=u, password=p) if (u and p) else TvDatafeed()
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
    except Exception as e:
        log.warning("TV fetch failed ex=%s sym=%s tf=%s err=%s", exchange, symbol, tf, e)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df.sort_index()


def _fetch_bitget(symbol: str, tf: str, n_bars: int) -> pd.DataFrame:
    """Pull n_bars from Bitget USDT-perp via ccxt. `symbol` like 'BTCUSDT'."""
    try:
        import ccxt
    except Exception as e:
        log.error("ccxt not importable: %s", e)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    tf_code = _CCXT_TF.get(tf)
    if not tf_code:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Map plain ticker → unified ccxt symbol for USDT-margined perp swap.
    base = symbol.replace("USDT", "").replace("/", "")
    unified = f"{base}/USDT:USDT"

    try:
        ex = ccxt.bitget({"options": {"defaultType": "swap"}})
        ex.options["defaultType"] = "swap"
        candles = ex.fetch_ohlcv(unified, timeframe=tf_code, limit=n_bars)
    except Exception as e:
        log.warning("Bitget fetch failed sym=%s tf=%s err=%s", symbol, tf, e)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(candles, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"]).sort_index()
    return df


def _fetch(symbol_key: str, tf: str, n_bars: int) -> pd.DataFrame:
    src, ticker = _split_key(symbol_key)
    if src == "BITGET":
        return _fetch_bitget(ticker, tf, n_bars)
    return _fetch_tv(src, ticker, tf, n_bars)


def refresh(
    symbol: str,
    tf: str,
    conn: sqlite3.Connection | None = None,
    force_full: bool = False,
) -> int:
    """Pull missing bars from the right source and upsert into cache."""
    symbol = _normalize_symbol(symbol)
    own = conn is None
    if own:
        conn = open_db()
    try:
        last_ts = get_last_ts(conn, symbol, tf)
        now_ts = int(datetime.now(tz=timezone.utc).timestamp())
        target = TARGET_BARS.get(tf, 500)
        tf_sec = TF_SECONDS.get(tf, 3600)

        if force_full or last_ts is None:
            n_bars = target
        else:
            elapsed = max(0, (now_ts - last_ts) // tf_sec)
            if elapsed == 0:
                n_bars = 2
            else:
                n_bars = min(target, max(5, elapsed + 5))

        df = _fetch(symbol, tf, n_bars)
        wrote = upsert_df(conn, symbol, tf, df)
        log.info("cache refresh sym=%s tf=%s wrote=%d last_ts=%s", symbol, tf, wrote, last_ts)
        return wrote
    finally:
        if own:
            conn.close()


def refresh_all(
    symbol: str = "USDT.D",
    tfs: list[str] | None = None,
) -> dict[str, int]:
    """Refresh every cached TF for `symbol` (FQ EXCHANGE:SYMBOL or legacy bare)."""
    tfs = tfs or list(TARGET_BARS.keys())
    conn = open_db()
    out: dict[str, int] = {}
    try:
        for tf in tfs:
            out[tf] = refresh(symbol, tf, conn=conn)
    finally:
        conn.close()
    return out


def get_dfs(
    symbol: str = "USDT.D",
    tfs: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return {tf: DataFrame} ready to feed wavetrend/luxalgo/smc."""
    tfs = tfs or list(TARGET_BARS.keys())
    symbol = _normalize_symbol(symbol)
    conn = open_db()
    out: dict[str, pd.DataFrame] = {}
    try:
        for tf in tfs:
            out[tf] = get_ohlcv(symbol, tf, conn=conn)
    finally:
        conn.close()
    return out
