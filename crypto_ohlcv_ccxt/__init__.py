"""Crypto OHLCV candle store (ccxt → SQLite).

A daemon that mirrors ``usdt_dominance_tv`` but sources 1-minute OHLCV from
**Bitget USDT-M perp** via ccxt (``defaultType=swap``) for a configurable set
of symbols, accumulating them in ``crypto_ohlcv.db``. The indicators series
endpoint reads from this store (resampled per timeframe) instead of fetching
live, killing the per-request fetch lag.
"""
