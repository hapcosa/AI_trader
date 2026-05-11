"""Reader for indicatorsForge alerts SQLite DB.

The notifier service (indicatorsForge) persists every dispatched alert into
`alerts.db` via write-ahead pattern. This module exposes that data to
prompt_builder so the IA receives discrete TradingView events (cross, sweep,
momentum shift) alongside the OHLCV bars summary.

DB path: AI_trader/data/alerts.db  (bind-mounted from notifier container
`/app/ai_data/alerts.db`). Schema defined in
`indicatorsForge/notifier/persistence/ai_trader_store.py`.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "alerts.db"


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(
                float(raw) / (1000 if raw > 1e12 else 1), tz=timezone.utc
            )
        except Exception:
            return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
    return None


def get_recent_alerts(
    source_type: str = "usdt_dominance",
    hours: int = 24,
    limit: int = 50,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """Return alerts of `source_type` from the last `hours`, newest first.

    Each row is a dict with keys: ts (datetime UTC), symbol, timeframe,
    signal, direction, price, raw (parsed JSON dict).
    Empty list if DB missing or no rows.
    """
    if not db_path.exists():
        return []
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        cur = conn.execute(
            """SELECT ts, source_type, symbol, timeframe, signal, direction, price, raw_json
                 FROM alerts
                WHERE source_type = ? AND ts >= ?
                ORDER BY ts DESC
                LIMIT ?""",
            (source_type, cutoff, limit),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return []

    out: list[dict] = []
    for r in rows:
        try:
            raw = json.loads(r["raw_json"]) if r["raw_json"] else {}
        except Exception:
            raw = {}
        out.append({
            "ts": _parse_ts(r["ts"]) or datetime.now(tz=timezone.utc),
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "signal": r["signal"],
            "direction": r["direction"],
            "price": r["price"],
            "raw": raw,
        })
    return out


def build_usdt_alerts_summary(
    hours: int = 24,
    db_path: Path = DB_PATH,
) -> dict:
    """Aggregate USDT.D alerts for prompt injection.

    Returns:
        {
            available: bool,
            count: int,
            window_hours: int,
            last_signal: str | None,
            last_direction: str | None,
            last_ts: datetime | None,
            alerts: list[dict]  # most recent first, up to 10
        }
    """
    alerts = get_recent_alerts("usdt_dominance", hours=hours, db_path=db_path)
    if not alerts:
        return {
            "available": False,
            "count": 0,
            "window_hours": hours,
            "last_signal": None,
            "last_direction": None,
            "last_ts": None,
            "alerts": [],
        }
    last = alerts[0]
    return {
        "available": True,
        "count": len(alerts),
        "window_hours": hours,
        "last_signal": last["signal"],
        "last_direction": last["direction"],
        "last_ts": last["ts"],
        "alerts": alerts[:10],
    }
