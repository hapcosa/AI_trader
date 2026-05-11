"""Async scheduler that posts dominance digests to the indicatorsForge notifier.

Two cadences:
- 4H tick at every 04/08/12/16/20/00 UTC: send 1h+4h indicators digest.
- DAILY tick at 00:00 UTC: send 1d indicators digest (alongside the 4h one).

Posting is HTTP POST to `${NOTIFIER_URL}/ingest/indicator_dominance/{TOKEN}`.
Failures are logged but don't crash the loop.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import aiohttp

from .builder import build_digest_body, DOMINANCE_SYMBOLS

log = logging.getLogger(__name__)

# Hours of day when the 4h tick fires (UTC).
TICK_HOURS_4H: tuple[int, ...] = (0, 4, 8, 12, 16, 20)

# Slight offset so candles have actually closed on TV.
TICK_OFFSET_SECONDS: int = 90


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _next_4h_tick(now: datetime | None = None) -> datetime:
    n = now or _now_utc()
    base = n.replace(minute=0, second=0, microsecond=0) + timedelta(seconds=TICK_OFFSET_SECONDS)
    candidates = [
        base.replace(hour=h) for h in TICK_HOURS_4H
    ]
    future = [c for c in candidates if c > n]
    if future:
        return min(future)
    # Wrap to tomorrow's first slot
    tomorrow = (n + timedelta(days=1)).replace(hour=TICK_HOURS_4H[0], minute=0,
                                                second=0, microsecond=0)
    return tomorrow + timedelta(seconds=TICK_OFFSET_SECONDS)


async def _post_digest(
    session: aiohttp.ClientSession,
    notifier_url: str,
    token: str,
    digest_kind: str,
    body: str,
) -> bool:
    url = f"{notifier_url.rstrip('/')}/ingest/indicator_dominance/{token}"
    payload = {
        "source_type": "indicator_dominance",
        "symbol": "DOMINANCE",
        "timeframe": "DIGEST",
        "digest_kind": digest_kind,
        "body": body,
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=20)) as r:
            ok = r.status in (200, 202)
            if not ok:
                detail = (await r.text())[:300]
                log.warning("digest_post non_ok status=%s detail=%s", r.status, detail)
            return ok
    except Exception as e:
        log.error("digest_post failed: %s", e)
        return False


class DominanceDigestScheduler:
    """Background asyncio scheduler. Start/stop via FastAPI lifespan."""

    def __init__(
        self,
        notifier_url: str | None = None,
        token: str | None = None,
        symbols: tuple[str, ...] = DOMINANCE_SYMBOLS,
        tfs_4h: tuple[str, ...] = ("1h", "4h"),
        tfs_daily: tuple[str, ...] = ("1d",),
    ) -> None:
        self.notifier_url = notifier_url or _env("NOTIFIER_URL", "http://notifier:8090")
        self.token = token or _env("INTERNAL_INGEST_TOKEN")
        self.symbols = symbols
        self.tfs_4h = list(tfs_4h)
        self.tfs_daily = list(tfs_daily)
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if not self.token:
            log.warning("dominance_digest disabled — INTERNAL_INGEST_TOKEN empty")
            return
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._loop(), name="dominance_digest")
        log.info(
            "dominance_digest_started url=%s symbols=%s tfs_4h=%s tfs_daily=%s",
            self.notifier_url, list(self.symbols), self.tfs_4h, self.tfs_daily,
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        log.info("dominance_digest_stopped")

    async def _loop(self) -> None:
        assert self._session is not None
        while True:
            try:
                target = _next_4h_tick()
                wait = max(1.0, (target - _now_utc()).total_seconds())
                log.info("dominance_digest_sleep_until=%s wait=%.1fs",
                         target.isoformat(), wait)
                await asyncio.sleep(wait)
                await self._fire(target)
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("dominance_digest_loop_error: %s", e)
                await asyncio.sleep(60)

    async def _fire(self, target: datetime) -> None:
        assert self._session is not None

        # Always send the 4h digest at every tick.
        try:
            body = await asyncio.to_thread(
                build_digest_body,
                tfs=self.tfs_4h,
                symbols=self.symbols,
                refresh=True,
            )
            await _post_digest(self._session, self.notifier_url, self.token,
                               "4H", body)
            log.info("dominance_digest_4h_dispatched at=%s", target.isoformat())
        except Exception as e:
            log.error("dominance_digest_4h_failed: %s", e)

        # Additionally fire the daily digest only at the 00:00 UTC tick.
        if target.hour == 0:
            try:
                body_d = await asyncio.to_thread(
                    build_digest_body,
                    tfs=self.tfs_daily,
                    symbols=self.symbols,
                    refresh=True,
                )
                await _post_digest(self._session, self.notifier_url, self.token,
                                   "DAILY", body_d)
                log.info("dominance_digest_daily_dispatched at=%s", target.isoformat())
            except Exception as e:
                log.error("dominance_digest_daily_failed: %s", e)

    async def fire_once(self, kind: str = "4H") -> bool:
        """Manual trigger — useful for tests."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        tfs = self.tfs_4h if kind.upper() == "4H" else self.tfs_daily
        body = await asyncio.to_thread(
            build_digest_body, tfs=tfs, symbols=self.symbols, refresh=True,
        )
        return await _post_digest(self._session, self.notifier_url, self.token,
                                  kind.upper(), body)
