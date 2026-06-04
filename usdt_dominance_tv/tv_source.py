"""TradingView OHLCV fetcher for CRYPTOCAP:USDT.D via tvdatafeed-enhanced."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)

try:
    from tvDatafeed import Interval, TvDatafeed
    _HAS_TVDF = True
except Exception as e:  # pragma: no cover
    log.error("tvdatafeed not importable: %s", e)
    Interval = None  # type: ignore
    TvDatafeed = None  # type: ignore
    _HAS_TVDF = False


# Per-call cap of the TV WS protocol — practical safe limit.
TV_MAX_BARS_PER_CALL = 5000


class TVSource:
    def __init__(
        self,
        symbol: str = "USDT.D",
        exchange: str = "CRYPTOCAP",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        if not _HAS_TVDF:
            raise RuntimeError("tvdatafeed library not available")
        self.symbol = symbol
        self.exchange = exchange
        self._username = username or None
        self._password = password or None
        self._client: TvDatafeed | None = None

    def _client_or_init(self) -> TvDatafeed:
        if self._client is None:
            if self._username and self._password:
                self._client = TvDatafeed(
                    username=self._username, password=self._password
                )
            else:
                self._client = TvDatafeed()
        return self._client

    def fetch_recent(
        self,
        n_bars: int,
        symbol: str | None = None,
        exchange: str | None = None,
    ) -> pd.DataFrame:
        """Fetch the last n_bars 1-minute bars. Returns DataFrame with UTC index.

        ``symbol``/``exchange`` override the instance defaults so a single
        client can serve several dominance series (USDT.D, BTC.D, OTHERS.D).
        """
        n = max(1, min(int(n_bars), TV_MAX_BARS_PER_CALL))
        client = self._client_or_init()
        df = client.get_hist(
            symbol=symbol or self.symbol,
            exchange=exchange or self.exchange,
            interval=Interval.in_1_minute,
            n_bars=n,
        )
        return self._normalize(df)

    def fetch_paged(self, total_bars: int, symbol: str | None = None) -> pd.DataFrame:
        """
        Fetch up to total_bars by repeated calls. tvdatafeed-enhanced returns the
        most recent n_bars each call; for very large backfills we accept a single
        biggest-possible window (TV_MAX_BARS_PER_CALL) since the lib does not
        expose `before_ts` paging on every fork. Caller pages by date if needed.
        """
        return self.fetch_recent(total_bars, symbol=symbol)

    @staticmethod
    def _normalize(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = df.copy()
        # tvdatafeed returns columns: symbol, open, high, low, close, volume
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        # Index: ensure UTC tz-aware DatetimeIndex
        idx = pd.to_datetime(df.index)
        if idx.tz is None:
            # tvdatafeed returns naive timestamps assumed UTC
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        df.index = idx
        return df.sort_index()


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)
