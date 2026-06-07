"""Crypto OHLCV daemon — Bitget USDT-M perp 1-minute bars → SQLite.

Polls a configurable set of pairs every minute and stores 1-minute OHLCV bars
in SQLite, one row per ``(symbol, ts)``. On startup, backfills missing history
per symbol up to ``HISTORY_DAYS_INITIAL``. Prunes bars older than
``RETENTION_DAYS`` each cycle. Mirrors ``usdt_dominance_tv.daemon`` but sources
from ccxt (Bitget swap) instead of TradingView, with no CoinGecko fallback.

Run: python -m crypto_ohlcv_ccxt.daemon
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import storage
from .source import MAX_BARS_PER_CALL, CcxtSource
from .storage import minute_ts


log = logging.getLogger("crypto_ohlcv_ccxt")

DEFAULT_SYMBOLS = (
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,"
    "ADA/USDT,DOGE/USDT,AVAX/USDT,LINK/USDT,SUI/USDT"
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


def _parse_symbols() -> list[str]:
    raw = _env_str("CRYPTO_OHLCV_SYMBOLS", DEFAULT_SYMBOLS)
    seen: set[str] = set()
    out: list[str] = []
    for it in raw.split(","):
        s = it.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _setup_logging() -> None:
    level = _env_str("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _sleep_until_next_minute(stop: threading.Event, offset_seconds: int = 8) -> None:
    """Sleep until (next minute + offset), polling stop each second.

    The offset gives the exchange time to close the 1m candle before we fetch.
    """
    now = datetime.now(tz=timezone.utc)
    next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    target = next_min.timestamp() + offset_seconds
    while not stop.is_set():
        remaining = target - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 1.0))


def _initial_backfill(
    conn, src: CcxtSource, symbols: list[str], history_days: int
) -> None:
    """Backfill HISTORY_DAYS of 1m bars for empty/sparse symbols."""
    now_ts = minute_ts(datetime.now(tz=timezone.utc))
    target_minutes = history_days * 24 * 60

    for symbol in symbols:
        last_ts = storage.get_last_ts(conn, symbol)
        if last_ts is None:
            n_bars = target_minutes
            log.info("[%s] Empty — backfilling %d minutes (~%d days)",
                     symbol, n_bars, history_days)
        else:
            gap = max(0, (now_ts - last_ts) // 60)
            if gap == 0:
                continue
            n_bars = min(gap + 5, target_minutes)
            log.info("[%s] Gap of %d minutes at startup — fetching %d bars",
                     symbol, gap, n_bars)
        try:
            df = src.fetch_recent(n_bars=n_bars, symbol=symbol)
            wrote = storage.upsert_bars(conn, df, symbol=symbol)
            log.info("[%s] Backfilled %d bars", symbol, wrote)
        except Exception as e:
            log.error("[%s] Backfill failed: %s", symbol, e)


def _tick(conn, src: CcxtSource, symbols: list[str]) -> None:
    now = datetime.now(tz=timezone.utc)
    current_ts = minute_ts(now)
    for symbol in symbols:
        last_ts = storage.get_last_ts(conn, symbol)
        gap_minutes = 1 if last_ts is None else max(1, (current_ts - last_ts) // 60)
        n_bars = max(3, min(gap_minutes + 2, MAX_BARS_PER_CALL))
        try:
            df = src.fetch_recent(n_bars=n_bars, symbol=symbol)
            wrote = storage.upsert_bars(conn, df, symbol=symbol)
            last_close = float(df["close"].iloc[-1]) if len(df) else float("nan")
            log.info("[%s] ok — wrote=%d gap=%d last=%.6g",
                     symbol, wrote, gap_minutes, last_close)
        except Exception as e:
            log.warning("[%s] fetch failed: %s", symbol, e)


def run() -> None:
    _setup_logging()

    db_path = Path(_env_str("DB_PATH", "/app/data/crypto_ohlcv.db"))
    poll_interval = _env_int("POLL_INTERVAL", 60)
    history_days = _env_int("HISTORY_DAYS_INITIAL", 30)
    retention_days = _env_int("RETENTION_DAYS", 90)
    exchange_id = _env_str("CRYPTO_OHLCV_EXCHANGE", "bitget")
    symbols = _parse_symbols()

    log.info("Starting daemon — db=%s exchange=%s symbols=%s poll=%ds "
             "history_days=%d retention_days=%d",
             db_path, exchange_id, ",".join(symbols), poll_interval,
             history_days, retention_days)

    stop = threading.Event()

    def _on_signal(sig, frame):
        log.info("Shutdown signal %s received — closing", sig)
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    conn = storage.open_db(db_path)

    try:
        src = CcxtSource(exchange_id=exchange_id)
    except Exception as e:
        log.error("Cannot initialize ccxt source: %s — exiting", e)
        conn.close()
        raise

    try:
        _initial_backfill(conn, src, symbols, history_days)
    except Exception as e:
        log.error("Initial backfill error: %s", e)

    try:
        while not stop.is_set():
            try:
                _tick(conn, src, symbols)
                removed = storage.prune_old(conn, retention_days)
                if removed:
                    log.info("Pruned %d bars older than %d days", removed, retention_days)
            except Exception as e:
                log.exception("Tick error: %s", e)

            if poll_interval >= 60:
                _sleep_until_next_minute(stop)
            else:
                deadline = time.monotonic() + poll_interval
                while not stop.is_set() and time.monotonic() < deadline:
                    time.sleep(0.5)
    finally:
        conn.close()
        log.info("Daemon stopped.")


if __name__ == "__main__":
    run()
