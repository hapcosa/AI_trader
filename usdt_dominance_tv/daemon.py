"""Dominance daemon — TradingView primary, CoinGecko fallback (multi-series).

Polls several CRYPTOCAP dominance series (USDT.D, BTC.D, OTHERS.D by default)
every minute and stores 1-minute OHLCV bars in SQLite, one row per
``(symbol, ts)``. On startup, backfills missing history per series. After
consecutive TV failures, falls back to CoinGecko for the series it can supply
(USDT.D, BTC.D) until TV recovers.

Run: python -m usdt_dominance_tv.daemon
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import cg_source, storage
from .storage import minute_ts
from .tv_source import TV_MAX_BARS_PER_CALL, TVSource


log = logging.getLogger("usdt_dominance_tv")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


def _parse_symbols() -> list[str]:
    """Resolve the series to track.

    ``TV_SYMBOLS`` (CSV) wins. Otherwise fall back to the legacy single
    ``TV_SYMBOL`` (default USDT.D) plus the new BTC.D/OTHERS.D series, deduped
    preserving order.
    """
    raw = os.environ.get("TV_SYMBOLS")
    if raw:
        items = raw.split(",")
    else:
        legacy = _env_str("TV_SYMBOL", "USDT.D")
        items = [legacy, "BTC.D", "OTHERS.D"]
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
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


def _sleep_until_next_minute(stop: threading.Event, offset_seconds: int = 5) -> None:
    """Sleep until (next minute + offset_seconds), polling stop event each second."""
    now = datetime.now(tz=timezone.utc)
    next_min = now.replace(second=0, microsecond=0) + _one_minute()
    target = next_min.timestamp() + offset_seconds
    while not stop.is_set():
        remaining = target - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 1.0))


def _one_minute():
    from datetime import timedelta
    return timedelta(minutes=1)


def _initial_backfill(
    conn,
    tv: TVSource,
    symbols: list[str],
    exchange: str,
    history_days: int,
) -> None:
    """If a series' history is empty or sparse, backfill HISTORY_DAYS from TV."""
    now_ts = minute_ts(datetime.now(tz=timezone.utc))
    target_minutes = history_days * 24 * 60

    for symbol in symbols:
        last_ts = storage.get_last_ts(conn, symbol)
        if last_ts is None:
            n_bars = min(target_minutes, TV_MAX_BARS_PER_CALL)
            log.info("[%s] Empty — backfilling %d minutes (~%d days) from TV",
                     symbol, n_bars, n_bars // 1440)
        else:
            gap = max(0, (now_ts - last_ts) // 60)
            if gap == 0:
                continue
            n_bars = min(gap + 5, TV_MAX_BARS_PER_CALL)
            log.info("[%s] Gap of %d minutes at startup — fetching %d bars",
                     symbol, gap, n_bars)
        try:
            df = tv.fetch_recent(n_bars=n_bars, symbol=symbol, exchange=exchange)
            wrote = storage.upsert_bars(conn, df, symbol=symbol, source="tv")
            log.info("[%s] Backfilled %d bars from TV", symbol, wrote)
        except Exception as e:
            log.error("[%s] Backfill failed: %s", symbol, e)


def _tick(
    conn,
    tv: TVSource,
    symbols: list[str],
    exchange: str,
    state: dict,
    fail_threshold: int,
) -> None:
    now = datetime.now(tz=timezone.utc)
    current_ts = minute_ts(now)

    any_tv_ok = False
    failed_symbols: list[str] = []
    for symbol in symbols:
        last_ts = storage.get_last_ts(conn, symbol)
        if last_ts is None:
            gap_minutes = 1
        else:
            gap_minutes = max(1, (current_ts - last_ts) // 60)
        n_bars = max(3, min(gap_minutes + 2, TV_MAX_BARS_PER_CALL))
        try:
            df = tv.fetch_recent(n_bars=n_bars, symbol=symbol, exchange=exchange)
            wrote = storage.upsert_bars(conn, df, symbol=symbol, source="tv")
            last_close = float(df["close"].iloc[-1]) if len(df) else float("nan")
            log.info("[%s] TV ok — wrote=%d gap=%d last=%.6f",
                     symbol, wrote, gap_minutes, last_close)
            any_tv_ok = True
        except Exception as e:
            failed_symbols.append(symbol)
            log.warning("[%s] TV fetch failed: %s", symbol, e)

    if any_tv_ok and not failed_symbols:
        state["tv_fails"] = 0
        return
    if failed_symbols:
        state["tv_fails"] = state.get("tv_fails", 0) + 1
        log.warning("TV partial/total failure (%d/%d) for %s",
                    state["tv_fails"], fail_threshold, ",".join(failed_symbols))

    # Fallback to CoinGecko after threshold, for the series it can supply.
    if state["tv_fails"] >= fail_threshold:
        pct = cg_source.fetch_percentages()
        if not pct:
            log.warning("CoinGecko fallback also failed — skipping minute")
            return
        for symbol in failed_symbols:
            value = pct.get(symbol)
            if value is None:
                log.warning("[%s] No CoinGecko fallback available — skipping", symbol)
                continue
            storage.upsert_single(conn, symbol, current_ts, value, source="coingecko")
            log.info("[%s] CoinGecko fallback wrote tick %s = %.4f%%",
                     symbol, now.strftime("%Y-%m-%d %H:%M UTC"), value)


def run() -> None:
    _setup_logging()

    db_path = Path(_env_str("DB_PATH", "/app/data/usdt_dominance.db"))
    poll_interval = _env_int("POLL_INTERVAL", 60)
    history_days = _env_int("HISTORY_DAYS_INITIAL", 30)
    fail_threshold = _env_int("TV_FAIL_THRESHOLD", 3)
    symbols = _parse_symbols()
    tv_exchange = _env_str("TV_EXCHANGE", "CRYPTOCAP")
    tv_user = os.environ.get("TV_USERNAME") or None
    tv_pass = os.environ.get("TV_PASSWORD") or None

    log.info("Starting daemon — db=%s exchange=%s symbols=%s poll=%ds history_days=%d auth=%s",
             db_path, tv_exchange, ",".join(symbols), poll_interval, history_days,
             "yes" if tv_user else "no")

    stop = threading.Event()

    def _on_signal(sig, frame):
        log.info("Shutdown signal %s received — closing", sig)
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    conn = storage.open_db(db_path)

    try:
        tv = TVSource(
            symbol=symbols[0], exchange=tv_exchange,
            username=tv_user, password=tv_pass,
        )
    except Exception as e:
        log.error("Cannot initialize TV source: %s — running in CoinGecko-only mode", e)
        tv = None

    state: dict = {"tv_fails": fail_threshold}  # start "failed" so CG used if TV None

    if tv is not None:
        try:
            _initial_backfill(conn, tv, symbols, tv_exchange, history_days)
        except Exception as e:
            log.error("Initial backfill error: %s", e)

    try:
        while not stop.is_set():
            try:
                if tv is not None:
                    _tick(conn, tv, symbols, tv_exchange, state, fail_threshold)
                else:
                    # Pure CoinGecko mode — only the series CG can supply.
                    now = datetime.now(tz=timezone.utc)
                    pct = cg_source.fetch_percentages()
                    for symbol in symbols:
                        value = pct.get(symbol)
                        if value is not None:
                            storage.upsert_single(
                                conn, symbol, minute_ts(now), value, source="coingecko"
                            )
                            log.info("[%s] CG-only tick %.4f%%", symbol, value)
            except Exception as e:
                log.exception("Tick error: %s", e)

            if poll_interval >= 60:
                _sleep_until_next_minute(stop, offset_seconds=5)
            else:
                # short polling: simple sleep for tests
                deadline = time.monotonic() + poll_interval
                while not stop.is_set() and time.monotonic() < deadline:
                    time.sleep(0.5)
    finally:
        conn.close()
        log.info("Daemon stopped.")


if __name__ == "__main__":
    run()
