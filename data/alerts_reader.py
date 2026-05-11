"""Sync reader for alerts.db. Used by prompt_builder to enrich AI prompts.

The DB is written by indicatorsForge (notifier service) via write-ahead.
Reader uses stdlib sqlite3 — no extra deps. Read-only access.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_DEFAULT_DB = Path(__file__).resolve().parent / "alerts.db"


def _db_path() -> Path:
    raw = os.getenv("AI_TRADER_DB_PATH")
    return Path(raw) if raw else _DEFAULT_DB


def _connect() -> sqlite3.Connection:
    path = _db_path()
    if not path.exists():
        # Empty DB so reader doesn't crash before notifier writes anything.
        path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(path).close()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    raw = out.pop("raw_json", None)
    if raw:
        try:
            out["payload"] = json.loads(raw)
        except json.JSONDecodeError:
            out["payload"] = None
    return out


def get_recent_alerts(
    symbol: str | None = None,
    source_type: str | None = None,
    since_hours: int = 24,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return alerts from the last N hours, newest first."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    sql = "SELECT * FROM alerts WHERE ts >= ?"
    params: list[Any] = [cutoff]
    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol)
    if source_type:
        sql += " AND source_type = ?"
        params.append(source_type)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)

    try:
        with _connect() as conn:
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    return [_row_to_dict(r) for r in rows]


def get_dominance_trend(hours: int = 24) -> dict[str, Any]:
    """Return latest USDT dominance value + delta over the window."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    sql = """SELECT ts, price, signal, direction, raw_json
             FROM alerts
             WHERE source_type = 'usdt_dominance' AND ts >= ?
             ORDER BY ts ASC"""
    try:
        with _connect() as conn:
            rows = conn.execute(sql, [cutoff]).fetchall()
    except sqlite3.OperationalError:
        rows = []

    if not rows:
        return {"available": False, "hours": hours}

    first = rows[0]
    last = rows[-1]
    first_v = first["price"]
    last_v = last["price"]
    delta = None
    if first_v is not None and last_v is not None:
        try:
            delta = float(last_v) - float(first_v)
        except (TypeError, ValueError):
            delta = None

    return {
        "available": True,
        "hours": hours,
        "first_ts": first["ts"],
        "last_ts": last["ts"],
        "first_value": first_v,
        "last_value": last_v,
        "delta": delta,
        "direction": last["direction"],
        "last_signal": last["signal"],
        "samples": len(rows),
    }


def format_alerts_block(alerts: list[dict[str, Any]], max_items: int = 20) -> str:
    """Render alerts as a readable text block for prompt embedding."""
    if not alerts:
        return "Sin alertas recientes."
    lines: list[str] = []
    for a in alerts[:max_items]:
        ts = a.get("ts", "?")
        st = a.get("source_type", "?")
        sym = a.get("symbol") or "—"
        tf = a.get("timeframe") or "—"
        sig = a.get("signal") or "—"
        direction = a.get("direction") or "—"
        price = a.get("price")
        price_s = f"{float(price):.6g}" if isinstance(price, (int, float)) else "—"
        lines.append(f"  [{ts}] {st} · {sym} {tf} · {sig} · dir={direction} · px={price_s}")
    if len(alerts) > max_items:
        lines.append(f"  … (+{len(alerts) - max_items} más)")
    return "\n".join(lines)


def format_dominance_block(trend: dict[str, Any]) -> str:
    if not trend.get("available"):
        return f"USDT.D: sin datos en últimas {trend.get('hours', '?')}h."
    delta = trend.get("delta")
    delta_s = f"{delta:+.4f}%" if isinstance(delta, (int, float)) else "—"
    return (
        f"USDT.D actual: {trend.get('last_value')}% "
        f"(Δ {delta_s} en {trend.get('hours')}h, {trend.get('samples')} muestras) "
        f"· dir={trend.get('direction')} · last_signal={trend.get('last_signal')}"
    )
