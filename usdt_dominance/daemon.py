"""
USDT Dominance daemon — polls CoinGecko /global every 60s, stores ticks in SQLite.

Usage:
    python -m pineforge_ai.usdt_dominance.daemon

SQLite DB: pineforge_ai/data/usdt_dominance.db
Schema: ticks(ts INTEGER PRIMARY KEY, usdt_pct REAL)
  ts = Unix seconds UTC, truncated to the current minute.
"""

from __future__ import annotations

import signal
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

COINGECKO_URL = "https://api.coingecko.com/api/v3/global"
POLL_INTERVAL_SECONDS = 60
DB_PATH = Path(__file__).parent.parent / "data" / "usdt_dominance.db"
_HEADERS = {"User-Agent": "PineForge-AI/2.0"}


# ─── DB helpers ───────────────────────────────────────────────────────────────

def _open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ticks ("
        "ts      INTEGER PRIMARY KEY, "
        "usdt_pct REAL NOT NULL"
        ");"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(ts);")
    conn.commit()
    return conn


def _store_tick(conn: sqlite3.Connection, ts: int, value: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ticks (ts, usdt_pct) VALUES (?, ?)", (ts, value)
    )
    conn.commit()


def _minute_ts(dt: datetime) -> int:
    """Truncate to current minute (Unix seconds UTC)."""
    return int(dt.replace(second=0, microsecond=0).timestamp())


# ─── CoinGecko fetch ──────────────────────────────────────────────────────────

def _fetch_usdt_dominance() -> float | None:
    try:
        resp = requests.get(COINGECKO_URL, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data["data"]["market_cap_percentage"]["usdt"])
    except Exception as e:
        print(f"[USDT.D daemon] fetch error: {e}", flush=True)
        return None


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_daemon(
    db_path: Path = DB_PATH,
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> None:
    stop_event = threading.Event()

    def _handle_signal(sig, frame):
        print("\n[USDT.D daemon] Shutdown signal received. Closing...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"[USDT.D daemon] Starting — DB: {db_path}", flush=True)
    conn = _open_db(db_path)

    try:
        while not stop_event.is_set():
            now = datetime.now(tz=timezone.utc)
            ts = _minute_ts(now)
            value = _fetch_usdt_dominance()

            if value is not None:
                _store_tick(conn, ts, value)
                print(
                    f"[USDT.D daemon] {now.strftime('%Y-%m-%d %H:%M UTC')} "
                    f"→ {value:.4f}%",
                    flush=True,
                )

            # Sleep poll_interval seconds, checking stop_event every 1s
            deadline = time.monotonic() + poll_interval
            while not stop_event.is_set() and time.monotonic() < deadline:
                time.sleep(1)
    finally:
        conn.close()
        print("[USDT.D daemon] Stopped.", flush=True)


if __name__ == "__main__":
    run_daemon()
