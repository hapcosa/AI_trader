"""Default configuration and constants."""

from __future__ import annotations

# Velas por día por timeframe (24h continuo — crypto/forex)
# Para mercados con horario (stocks) yfinance maneja la limitación internamente
CANDLES_PER_DAY: dict[str, float] = {
    "1m":  1440.0,
    "3m":  480.0,
    "5m":  288.0,
    "15m": 96.0,
    "30m": 48.0,
    "1h":  24.0,
    "2h":  12.0,
    "4h":  6.0,
    "6h":  4.0,
    "8h":  3.0,
    "12h": 2.0,
    "1d":  1.0,
    "3d":  0.333,
    "1w":  0.143,
}

# Barras adicionales para warmup de indicadores (ventanas largas, normalización)
WARMUP_BARS = 150

DEFAULT_TIMEFRAMES: list[str] = ["1h", "4h", "1d"]
ALL_INDICATORS: list[str] = [
    "wavetrend", "luxalgo", "smc", "wae", "itrend", "ict", "trendlines",
    # BudAI pineforge oscillators (ports mirror KryptoLab strategies)
    "pulse", "abyss", "tide", "athenea",
]
DEFAULT_INDICATORS: list[str] = ALL_INDICATORS
DEFAULT_DAYS: int = 60
DEFAULT_EXCHANGE: str = "binance"
DEFAULT_OUTPUT_DIR: str = "pineforge_ai/output"

# Timeframes válidos soportados por el sistema
VALID_TIMEFRAMES: set[str] = set(CANDLES_PER_DAY.keys())

# Mapeo de timeframes para yfinance (intervalo -> period string)
YFINANCE_INTERVAL_MAP: dict[str, str] = {
    "1m":  "1m",
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",    # yfinance no soporta 4h nativamente, se resampleará desde 1h
    "6h":  "1h",    # se resamplea desde 1h
    "8h":  "1h",    # se resamplea desde 1h
    "12h": "1h",    # se resamplea desde 1h
    "1d":  "1d",
    "1w":  "1wk",
}

# Timeframes que requieren resample en yfinance (no nativos)
YFINANCE_RESAMPLE: dict[str, str] = {
    "4h":  "4h",
    "6h":  "6h",
    "8h":  "8h",
    "12h": "12h",
    "3d":  "3D",
}

# Mapeo timeframe → regla pandas resample
RESAMPLE_RULES: dict[str, str] = {
    "1m":  "1min",
    "3m":  "3min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "8h":  "8h",
    "12h": "12h",
    "1d":  "1D",
    "3d":  "3D",
    "1w":  "1W",
}
