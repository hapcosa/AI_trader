"""Global market session analysis — all times in UTC."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone

import pandas as pd


@dataclass
class Session:
    name: str
    open_utc: time      # UTC open time
    close_utc: time     # UTC close time
    crosses_midnight: bool = False
    exchange: str = ""
    description: str = ""

    def is_active(self, dt_utc: datetime) -> bool:
        t = dt_utc.time().replace(second=0, microsecond=0)
        if self.crosses_midnight:
            return t >= self.open_utc or t < self.close_utc
        return self.open_utc <= t < self.close_utc


# ─── Session Definitions ─────────────────────────────────────────────────────

SESSIONS: list[Session] = [
    Session(
        name="Sydney",
        open_utc=time(21, 0),
        close_utc=time(6, 0),
        crosses_midnight=True,
        exchange="ASX",
        description="Australia — apertura semanal, liquidez baja",
    ),
    Session(
        name="Tokyo",
        open_utc=time(0, 0),
        close_utc=time(9, 0),
        exchange="TSE / JPX",
        description="Japón — Asia principal, JPY activo",
    ),
    Session(
        name="Shanghai",
        open_utc=time(1, 30),
        close_utc=time(8, 0),
        exchange="SSE / SZSE",
        description="China — mayor economía asiática, CNY activo",
    ),
    Session(
        name="India",
        open_utc=time(3, 45),
        close_utc=time(10, 0),
        exchange="BSE / NSE",
        description="India — mercado emergente de alta liquidez",
    ),
    Session(
        name="Frankfurt",
        open_utc=time(7, 0),
        close_utc=time(16, 0),
        exchange="XETRA / Deutsche Börse",
        description="Europa continental — EUR activo",
    ),
    Session(
        name="London",
        open_utc=time(8, 0),
        close_utc=time(17, 0),
        exchange="LSE",
        description="Sesión europea principal — alta liquidez, GBP/EUR activos",
    ),
    Session(
        name="New York",
        open_utc=time(13, 30),
        close_utc=time(22, 0),
        exchange="NYSE / NASDAQ",
        description="Sesión americana — máxima liquidez global, USD dominante",
    ),
    Session(
        name="Wall Street Core",
        open_utc=time(14, 30),
        close_utc=time(21, 0),
        exchange="NYSE core hours",
        description="Horas pico NYSE — mayor volumen institucional",
    ),
]

# Solapamientos clave (máxima liquidez)
OVERLAPS: list[dict] = [
    {
        "name": "London + New York",
        "open_utc": time(13, 30),
        "close_utc": time(17, 0),
        "description": "MÁXIMA LIQUIDEZ GLOBAL — movimientos institucionales más fuertes",
        "liquidity": "máxima",
    },
    {
        "name": "Tokyo + London",
        "open_utc": time(8, 0),
        "close_utc": time(9, 0),
        "description": "Transición Asia-Europa — posibles reversals de sesión",
        "liquidity": "media-alta",
    },
    {
        "name": "Sydney + Tokyo",
        "open_utc": time(0, 0),
        "close_utc": time(6, 0),
        "description": "Asia/Pacífico — liquidez moderada, AUD/JPY activos",
        "liquidity": "media",
    },
]


# ─── Public API ───────────────────────────────────────────────────────────────

def get_active_sessions(dt_utc: datetime | None = None) -> list[Session]:
    """Return sessions active at the given UTC datetime (default: now)."""
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)
    return [s for s in SESSIONS if s.is_active(dt_utc)]


def get_active_overlaps(dt_utc: datetime | None = None) -> list[dict]:
    """Return active overlap windows at the given UTC datetime."""
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)
    t = dt_utc.time().replace(second=0, microsecond=0)
    active = []
    for ov in OVERLAPS:
        if ov["open_utc"] <= t < ov["close_utc"]:
            active.append(ov)
    return active


def get_session_status(dt_utc: datetime | None = None) -> dict:
    """
    Full session status at a given UTC datetime.

    Returns:
        {
          'active': [Session, ...],
          'overlaps': [dict, ...],
          'closed': [Session, ...],
          'liquidity': 'máxima' | 'alta' | 'media' | 'baja',
          'summary': str,
        }
    """
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)

    active = get_active_sessions(dt_utc)
    overlaps = get_active_overlaps(dt_utc)
    closed = [s for s in SESSIONS if s not in active]

    # Liquidity level
    if any(ov["liquidity"] == "máxima" for ov in overlaps):
        liquidity = "máxima"
    elif len(active) >= 3 or any(s.name in ("London", "New York") for s in active):
        liquidity = "alta"
    elif len(active) >= 2:
        liquidity = "media"
    else:
        liquidity = "baja"

    active_names = [s.name for s in active]
    overlap_names = [ov["name"] for ov in overlaps]

    if overlap_names:
        summary = f"Overlap activo: {', '.join(overlap_names)} | {liquidity.capitalize()} liquidez"
    elif active_names:
        summary = f"Activas: {', '.join(active_names)} | Liquidez {liquidity}"
    else:
        summary = "Sin sesión activa (mercados cerrados)"

    return {
        "active": active,
        "overlaps": overlaps,
        "closed": closed,
        "liquidity": liquidity,
        "summary": summary,
        "datetime_utc": dt_utc,
    }


def classify_candles_by_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'session' column to a DataFrame with UTC DatetimeIndex.
    Each candle is tagged with the primary active session (or 'Asian Transition', 'Off-Hours').
    """
    # Priority order for labeling (highest liquidity first)
    priority = ["New York", "London", "Tokyo", "Shanghai", "India", "Frankfurt", "Sydney"]

    def _label(dt):
        active = [s.name for s in SESSIONS if s.is_active(dt) and s.name in priority]
        if not active:
            return "Off-Hours"
        # Return highest priority
        for p in priority:
            if p in active:
                return p
        return active[0]

    df = df.copy()
    df["session"] = df.index.map(_label)
    return df


def session_volume_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average volume per session for context analysis.
    Requires 'session' column (run classify_candles_by_session first).
    """
    if "session" not in df.columns:
        df = classify_candles_by_session(df)
    stats = df.groupby("session")["volume"].agg(
        avg_volume="mean",
        total_volume="sum",
        candle_count="count",
    ).reset_index()
    stats = stats.sort_values("avg_volume", ascending=False)
    return stats


def format_session_block(dt_utc: datetime | None = None) -> str:
    """Format session status as prompt block text."""
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)

    status = get_session_status(dt_utc)
    active_names = [s.name for s in status["active"]]
    overlap_names = [ov["name"] for ov in status["overlaps"]]
    closed_names = [s.name for s in status["closed"]]

    lines = []
    if overlap_names:
        lines.append(f"Overlap : {' + '.join(overlap_names)} — {status['overlaps'][0]['description']}")
    if active_names:
        lines.append(f"Activas : {', '.join(active_names)}")
    if closed_names:
        lines.append(f"Cerradas: {', '.join(closed_names)}")
    lines.append(f"Liquidez: {status['liquidity'].upper()}")

    return "\n".join(lines)
