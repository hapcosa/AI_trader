"""TVSource: authenticated login failure must fall back to anonymous TV
(not to CoinGecko), so dominance keeps the real CRYPTOCAP series incl OTHERS.D.
"""
from __future__ import annotations

import usdt_dominance_tv.tv_source as tv


def _fake_tvdf(record):
    class _FakeTVDF:
        def __init__(self, username=None, password=None):
            if username or password:
                record.append("auth_attempt")
                raise ValueError("Login failed - check your credentials")
            record.append("anon")
    return _FakeTVDF


def test_falls_back_to_anonymous_on_auth_failure(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(tv, "_HAS_TVDF", True)
    monkeypatch.setattr(tv, "TvDatafeed", _fake_tvdf(calls))

    src = tv.TVSource(symbol="USDT.D", exchange="CRYPTOCAP", username="u", password="p")
    client = src._client_or_init()

    assert calls == ["auth_attempt", "anon"]  # tried auth, then anonymous
    assert client is not None


def test_anonymous_when_no_creds(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(tv, "_HAS_TVDF", True)

    class _FakeTVDF:
        def __init__(self, username=None, password=None):
            seen["auth"] = bool(username and password)

    monkeypatch.setattr(tv, "TvDatafeed", _FakeTVDF)
    src = tv.TVSource()  # no creds
    src._client_or_init()
    assert seen["auth"] is False


def test_authenticated_when_login_ok(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(tv, "_HAS_TVDF", True)

    class _FakeTVDF:
        def __init__(self, username=None, password=None):
            seen["auth"] = bool(username and password)  # login succeeds

    monkeypatch.setattr(tv, "TvDatafeed", _FakeTVDF)
    src = tv.TVSource(username="u", password="p")
    src._client_or_init()
    assert seen["auth"] is True  # used authenticated client (no fallback)
