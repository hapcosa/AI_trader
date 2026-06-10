"""Global market session analysis — all times in UTC."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone

import pandas as pd

# Stock-exchange holiday calendars drive the "important session" windows below.
# Crypto trades 24/7, but London/NY liquidity follows the LSE/NYSE calendar, so
# weekends and exchange holidays mean no real session. `holidays` is optional —
# if it's missing we degrade to weekend-only detection rather than crashing.
try:  # pragma: no cover - import guard
    import holidays as _holidays

    _HOLIDAY_CALS: dict[str, object] = {
        "US": _holidays.financial_holidays("NYSE"),
        "UK": _holidays.country_holidays("UK"),
    }
except Exception:  # pragma: no cover - lib absent / API drift
    _HOLIDAY_CALS = {}


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


# ─── Any-time mindset: important sessions + next-session lookahead ────────────

# Sessions that drive real liquidity for the mindset analysis. Asia is kept as
# context (manipulation/accumulation before Europe), but the actionable windows
# are London, New York and their overlap.
IMPORTANT_SESSION_NAMES: tuple[str, ...] = ("London", "New York")
CONTEXT_SESSION_NAMES: tuple[str, ...] = ("Tokyo", "Shanghai", "Sydney")

# Which holiday calendar gates each important session.
_SESSION_MARKET: dict[str, str] = {"London": "UK", "New York": "US"}

_IMPORTANT_SESSIONS: list[Session] = [s for s in SESSIONS if s.name in IMPORTANT_SESSION_NAMES]


def _market_open_on(d, market: str) -> bool:
    """Is the given stock market trading on date `d`? False on weekends and on
    that market's holidays (NYSE for US, LSE/UK bank holidays for UK)."""
    if d.weekday() >= 5:  # Saturday / Sunday
        return False
    cal = _HOLIDAY_CALS.get(market)
    if cal is not None and d in cal:
        return False
    return True


def is_weekend(dt_utc: datetime | None = None) -> bool:
    """True on Saturday/Sunday UTC (stock sessions closed; crypto still 24/7)."""
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)
    return dt_utc.weekday() >= 5


def active_important_sessions(dt_utc: datetime | None = None) -> list[Session]:
    """Important sessions (London/NY) active *and* whose market is open today."""
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)
    return [
        s
        for s in _IMPORTANT_SESSIONS
        if s.is_active(dt_utc) and _market_open_on(dt_utc.date(), _SESSION_MARKET[s.name])
    ]


def session_closing_soon(
    dt_utc: datetime | None = None, minutes: int = 60
) -> Session | None:
    """If an active important session closes within `minutes`, return it. Used to
    advise on both the current and the *next* session when we're at the tail."""
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)
    for s in active_important_sessions(dt_utc):
        close_dt = datetime.combine(dt_utc.date(), s.close_utc, tzinfo=timezone.utc)
        if timedelta(0) <= (close_dt - dt_utc) <= timedelta(minutes=minutes):
            return s
    return None


def next_important_session(
    dt_utc: datetime | None = None, horizon_days: int = 9
) -> tuple[datetime | None, Session | None]:
    """Earliest important session that *opens* strictly after `dt_utc`, skipping
    weekends and exchange holidays. Returns (open_datetime_utc, session)."""
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)
    candidates: list[tuple[datetime, Session]] = []
    for s in _IMPORTANT_SESSIONS:
        market = _SESSION_MARKET[s.name]
        for day_offset in range(horizon_days):
            day = (dt_utc + timedelta(days=day_offset)).date()
            open_dt = datetime.combine(day, s.open_utc, tzinfo=timezone.utc)
            if open_dt <= dt_utc or not _market_open_on(day, market):
                continue
            candidates.append((open_dt, s))
            break
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def _humanize_delta(delta: timedelta) -> str:
    """'en 3h 20m' / 'en 2d 4h' for a future timedelta."""
    total_min = int(delta.total_seconds() // 60)
    if total_min < 0:
        return "ya"
    days, rem = divmod(total_min, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"en {days}d {hours}h"
    if hours:
        return f"en {hours}h {mins}m"
    return f"en {mins}m"


_WEEKDAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def get_session_context(dt_utc: datetime | None = None) -> dict:
    """Dynamic, time-agnostic session context for the any-time mindset.

    Captures: which important sessions are live now (and whether one is closing
    soon), the next important session and how far off it is, plus the weekend/
    holiday state of the US/UK exchanges. Crypto is 24/7 — these windows only
    describe where stock-driven liquidity concentrates."""
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)

    active = active_important_sessions(dt_utc)
    closing = session_closing_soon(dt_utc)
    overlaps = [ov for ov in get_active_overlaps(dt_utc) if ov["name"] == "London + New York"]
    next_open, next_sess = next_important_session(dt_utc)

    return {
        "datetime_utc": dt_utc,
        "weekday": _WEEKDAYS_ES[dt_utc.weekday()],
        "is_weekend": is_weekend(dt_utc),
        "us_open": _market_open_on(dt_utc.date(), "US"),
        "uk_open": _market_open_on(dt_utc.date(), "UK"),
        "active_important": active,
        "closing_soon": closing,
        "overlap_active": bool(overlaps),
        "next_session": next_sess,
        "next_session_open": next_open,
    }


def format_session_context_block(dt_utc: datetime | None = None) -> str:
    """Render `get_session_context` as a prompt block for the mindset mode."""
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)
    ctx = get_session_context(dt_utc)

    lines: list[str] = []
    lines.append(
        f"Hora UTC: {dt_utc.strftime('%Y-%m-%d %H:%M')} ({ctx['weekday']})"
    )

    if ctx["is_weekend"]:
        lines.append(
            "Fin de semana: las bolsas (LSE/NYSE) están cerradas. "
            "Crypto opera 24/7 pero sin liquidez institucional de sesión."
        )
    else:
        closed_markets = []
        if not ctx["us_open"]:
            closed_markets.append("NYSE (feriado US)")
        if not ctx["uk_open"]:
            closed_markets.append("LSE (feriado UK)")
        if closed_markets:
            lines.append(f"Feriado de bolsa: {', '.join(closed_markets)} cerrada(s) hoy.")

    active = ctx["active_important"]
    if active:
        parts = []
        for s in active:
            close_dt = datetime.combine(dt_utc.date(), s.close_utc, tzinfo=timezone.utc)
            parts.append(f"{s.name} (cierra {s.close_utc.strftime('%H:%M')} UTC, {_humanize_delta(close_dt - dt_utc)})")
        lines.append("Sesión importante activa: " + ", ".join(parts))
        if ctx["overlap_active"]:
            lines.append("Overlap London + New York — MÁXIMA LIQUIDEZ GLOBAL")
    else:
        lines.append("Sin sesión importante activa (London/NY cerradas ahora).")

    if ctx["closing_soon"] is not None:
        s = ctx["closing_soon"]
        lines.append(
            f"⚠ {s.name} está cerrando pronto — asesora la sesión actual Y la siguiente."
        )

    if ctx["next_session"] is not None and ctx["next_session_open"] is not None:
        nxt = ctx["next_session"]
        when = ctx["next_session_open"]
        lines.append(
            f"Próxima sesión importante: {nxt.name} — abre "
            f"{when.strftime('%Y-%m-%d %H:%M')} UTC ({_humanize_delta(when - dt_utc)})"
        )

    lines.append(
        "Nota: crypto es 24/7; estas ventanas marcan dónde se concentra la "
        "liquidez de bolsa. Fines de semana y feriados (NYSE/LSE) no hay sesión."
    )
    return "\n".join(lines)
