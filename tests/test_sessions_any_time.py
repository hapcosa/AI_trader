"""Any-time mindset session logic: important sessions, next-session lookahead,
weekend/holiday awareness, and the rendered context block."""

from datetime import datetime, timezone

import pytest

from pineforge_ai.sessions import (
    active_important_sessions,
    format_session_context_block,
    get_session_context,
    is_weekend,
    next_important_session,
    session_closing_soon,
)


def _utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# 2026-06-08 is a Monday; 2026-06-13 a Saturday; 2026-06-14 a Sunday.
MON = _utc(2026, 6, 8, 15, 0)   # 15:00 UTC → London + NY both active (overlap)
SAT = _utc(2026, 6, 13, 15, 0)


def test_active_important_during_london_ny_overlap():
    active = {s.name for s in active_important_sessions(MON)}
    assert active == {"London", "New York"}


def test_no_important_session_overnight():
    # 03:00 UTC Monday — Asia time, London/NY closed.
    assert active_important_sessions(_utc(2026, 6, 8, 3, 0)) == []


def test_weekend_has_no_important_sessions():
    assert is_weekend(SAT) is True
    assert active_important_sessions(SAT) == []


def test_next_session_skips_the_weekend():
    # Friday 21:00 UTC, after London close and inside NY tail. The next London
    # open must be the following Monday, not Saturday.
    fri = _utc(2026, 6, 12, 21, 0)
    when, sess = next_important_session(fri)
    assert sess is not None
    assert when.weekday() < 5  # not Sat/Sun
    assert when > fri


def test_next_session_skips_us_holiday():
    # 2026-07-03 is the observed NYSE holiday for July 4 (Saturday). Late on
    # 2026-07-02 (Thu) after NY opens, the next NY open should skip 07-03.
    when, sess = next_important_session(_utc(2026, 7, 2, 23, 0))
    # Whatever the next important session is, if it's New York it must not land
    # on the 3rd (NYSE closed).
    nyse_holiday = when.date() == datetime(2026, 7, 3).date() and sess.name == "New York"
    assert not nyse_holiday


def test_closing_soon_flags_tail_of_session():
    # London closes 17:00 UTC; at 16:30 it's within the 60-min tail.
    s = session_closing_soon(_utc(2026, 6, 8, 16, 30))
    assert s is not None and s.name == "London"
    # At 14:00 it's mid-session, nothing closing soon.
    assert session_closing_soon(_utc(2026, 6, 8, 14, 0)) is None


def test_context_block_mentions_weekend_and_next_session():
    block = format_session_context_block(SAT)
    assert "Fin de semana" in block
    assert "Próxima sesión importante" in block


def test_context_dict_shape():
    ctx = get_session_context(MON)
    assert ctx["overlap_active"] is True
    assert ctx["is_weekend"] is False
    assert ctx["next_session"] is not None
    assert {s.name for s in ctx["active_important"]} == {"London", "New York"}
