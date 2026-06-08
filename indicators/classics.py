"""Classic oscillators — RSI, Stochastic, MACD.

Standard textbook TA (not Pine ports), shaped to the indicators-series contract:
each returns a DataFrame whose columns feed ``build_indicator_series`` as
osc/trig (+ optional histogram for MACD). Helpers match the budai family
(``ewm(adjust=False)`` = Pine ``ta.ema``; Wilder RMA for RSI).
"""
from __future__ import annotations

import pandas as pd


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's moving average (used by RSI)."""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def rsi(df: pd.DataFrame, length: int = 14, signal: int = 14) -> pd.DataFrame:
    """Wilder RSI (0-100) plus an SMA signal line for the osc/trig crossover.

    Columns: ``rsi`` (oscillator), ``rsi_signal`` (SMA of RSI).
    """
    close = df["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there are no losses RSI is 100; when no gains, 0.
    out = out.where(avg_loss != 0.0, 100.0).where(avg_gain != 0.0, 0.0)
    return pd.DataFrame({"rsi": out, "rsi_signal": _sma(out, signal)})


def stochastic(
    df: pd.DataFrame, k: int = 14, d: int = 3, smooth_k: int = 3
) -> pd.DataFrame:
    """Slow Stochastic (0-100). Columns: ``k`` (%K), ``d`` (%D)."""
    low_min = df["low"].astype(float).rolling(k).min()
    high_max = df["high"].astype(float).rolling(k).max()
    rng = high_max - low_min
    raw_k = 100.0 * (df["close"].astype(float) - low_min) / rng
    raw_k = raw_k.where(rng != 0.0, 50.0)
    k_line = _sma(raw_k, smooth_k)  # slow %K
    d_line = _sma(k_line, d)        # %D
    return pd.DataFrame({"k": k_line, "d": d_line})


def macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD (centered on 0). Columns: ``macd`` (line), ``signal``, ``hist``."""
    close = df["close"].astype(float)
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}
    )
