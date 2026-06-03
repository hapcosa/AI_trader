# PineForge AI — Python Market Analysis System

> **Submódulo de signalsTrading**. Este repo (`hapcosa/AI_trader`) existe exclusivamente para servir al ecosistema signalsTrading. El parent (`hapcosa/signalsTrading`) lo vendora como submódulo en `AI_trader/` y mantiene un puntero a un commit exacto de aquí.

## Pipeline (rama → tests → PR → bump pointer)

Doc canónico del pipeline integrado: [hapcosa/signalsTrading → docs/operacion/pipeline-git-docker.md](https://github.com/hapcosa/signalsTrading/blob/master/docs/operacion/pipeline-git-docker.md). Para cambios dentro de este repo:

1. Rama `feat/<scope>-<resumen>` desde `master` (acá, en `AI_trader/`).
2. Commits, push a `hapcosa/AI_trader`, PR contra `master` del submódulo, review, merge.
3. En `hapcosa/signalsTrading`: rama propia + `git add AI_trader && git commit -m "chore: update AI_trader submodule"` + PR + merge.
4. La validación end-to-end corre contra el repo principal: `scripts/test-compose.sh` con el SHA actualizado del submódulo. Por eso un cambio acá que rompa el wiring de `pineforge-api` con `dashboard-api` se detecta en el CI del padre, no en este repo aislado.

Para cambios coordinados padre + submódulo: 2 PRs enlazados, mergear primero el del submódulo, después actualizar puntero y mergear el del padre (Caso C del doc canónico).

Los `docker-compose.yml` dentro de este repo son para dev aislado del submódulo. Para signalsTrading manda el compose raíz del padre.

---

## Rol del Desarrollador
Ingeniero experto en trading algorítmico. Portar Pine Script v6 a Python con fidelidad matemática exacta. Dominio de SMC, indicadores Ehlers, análisis multi-timeframe.

## Objetivo
CLI que descarga OHLCV → calcula indicadores de elite → analiza sesiones → genera prompt estructurado para IA. Parche v2: envío Anthropic API → JSON con entradas.

---

## Stack
- Python 3.11+
- pandas ≥ 2.0, numpy ≥ 1.24
- yfinance (stocks, ETFs, forex, índices: AAPL, ^FTSE, EURUSD=X)
- ccxt (crypto: BTC/USDT en Binance/Bybit/etc.)
- scipy (linreg rolling)
- click (CLI)

---

## Estructura del Proyecto

```
pineforge_ai/
├── CLAUDE.md           ← este archivo
├── __init__.py
├── cli.py              ← entry point: python -m pineforge_ai.cli
├── config.py           ← defaults y constantes
├── sessions.py         ← sesiones de mercado globales (UTC)
├── prompt_builder.py   ← ensambla y guarda el prompt
├── data/
│   ├── __init__.py
│   └── fetcher.py      ← descarga OHLCV (auto-detect yfinance vs ccxt)
├── indicators/
│   ├── __init__.py
│   ├── wavetrend.py    ← port de oscilador_v26.pine (WaveTrend + normalización)
│   ├── luxalgo_amo.py  ← port de luxalgooscilator.pine (AMO + AMA + divergencias)
│   └── smc_elite.py    ← port de SMCELITE.pine (estructura + OBs + FVGs + Fisher + Frost)
└── output/             ← prompts generados como {SYMBOL}_{DATETIME}.txt
```

---

## Convenciones Críticas

### DataFrame de entrada (contrato de todos los indicadores)
```python
# Index: DatetimeIndex, UTC timezone-aware
# Columnas obligatorias:
df.columns = ['open', 'high', 'low', 'close', 'volume']
```

### EMA (equivalente a ta.ema de Pine Script)
```python
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()
```

### SMA
```python
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()
```

### linreg rolling (equivalente a ta.linreg(src, length, 0) en Pine)
Regresión lineal de las últimas `length` barras → valor en la barra actual (offset=0):
```python
def linreg_rolling(series: pd.Series, length: int) -> pd.Series:
    def _lr(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(y), dtype=float)
        m, b = np.polyfit(x, y, 1)
        return m * (len(y) - 1) + b  # valor en último punto
    return series.rolling(length).apply(_lr, raw=True)
```

### Pivots (equivalente a ta.pivothigh / ta.pivotlow)
Pivot en barra `i` se confirma SOLO cuando han pasado `right` barras después → sin lookahead:
```python
def pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    result = pd.Series(np.nan, index=series.index)
    arr = series.to_numpy()
    for i in range(left, len(arr) - right):
        window_l = arr[i - left:i]
        window_r = arr[i + 1:i + right + 1]
        if arr[i] > np.nanmax(window_l) and arr[i] > np.nanmax(window_r):
            result.iloc[i] = arr[i]
    return result

def pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    result = pd.Series(np.nan, index=series.index)
    arr = series.to_numpy()
    for i in range(left, len(arr) - right):
        window_l = arr[i - left:i]
        window_r = arr[i + 1:i + right + 1]
        if arr[i] < np.nanmin(window_l) and arr[i] < np.nanmin(window_r):
            result.iloc[i] = arr[i]
    return result
```

### Anti-lookahead
- `rolling(N).max()/.min()` incluye barra actual → correcto, equivalente a Pine
- Pivots: señal en barra `i`, se detecta en barra `i + right` — nunca adelantar
- Fisher IIR: acumulación iterativa (loop), no vectorizable sin estado previo

---

## Indicadores — Resumen de Outputs

### wavetrend.py
Fuente: `oscilador_v26.pine`

```python
result = wavetrend(df, hyperwave_len=6, trigger_len=2, kernel_factor1=0.8, kernel_factor2=0.3)
# result es pd.DataFrame con columnas:
# osc_norm    float 0-100  Oscilador principal normalizado
# trig_norm   float 0-100  Línea trigger (SMA del WT1)
# hyper_norm  float 0-100  HyperWave (EMA doble suavizada)
# sma_osc     float        SMA(21) del oscilador
# is_bull_mom bool         osc > trigger
# mom_cross_up   bool      crossover(osc, trigger)
# mom_cross_dn   bool      crossunder(osc, trigger)
# rvol        float        Relative volume
# pressure_pct float       Buy pressure %

def wavetrend_mtf_trend(df) -> float:
    # retorna 1.0 (bull), -1.0 (bear), 0.0 (neutral)
    # equivalente a f_mtfTrend() en Pine
```

### luxalgo_amo.py
Fuente: `luxalgooscilator.pine`

```python
result = adaptive_momentum(df, length=14, smoothing=9, divergence_length=4)
# result es pd.DataFrame con columnas:
# amo         float  Adaptive Momentum Oscillator (sin suavizar)
# amo_smooth  float  linreg(amo, smoothing) — línea principal
# ama         float  Adaptive Moving Average de amo_smooth
# bull_div    bool   Divergencia alcista detectada
# bear_div    bool   Divergencia bajista detectada
```

### smc_elite.py
Fuente: `SMCELITE.pine`

```python
result = smc_analysis(df, ms_len=5, ob_len=10, fisher_period=14, fisher_extreme=2.5)
# result es pd.DataFrame con columnas:
# swing_high      float   Pivot high (NaN si no hay)
# swing_low       float   Pivot low (NaN si no hay)
# bos_bull        bool    Break of Structure alcista
# bos_bear        bool    Break of Structure bajista
# choch_bull      bool    Change of Character alcista
# choch_bear      bool    Change of Character bajista
# ob_bull_top/btm float   Order Block alcista activo (top/bottom)
# ob_bear_top/btm float   Order Block bajista activo
# fvg_bull_top/btm float  Fair Value Gap alcista activo
# fvg_bear_top/btm float  Fair Value Gap bajista activo
# fisher          float   Fisher Transform value
# fisher_signal   float   Fisher trigger line
# fisher_bull     bool    Fisher crossover (compra)
# fisher_bear     bool    Fisher crossunder (venta)
# frost_dir       int     Frost Engine dirección: 1/-1/0
# frost_conf      float   Frost confidence score 0-10
# confluence      float   Confluence Score 0-10
```

### Familia BudAI pineforge (osciladores)

Ports pandas 1:1 de los Pine de `pineforge/Osciladores/*.pine`, espejo de las
estrategias KryptoLab (paridad verificada con tests de transcripción Pine
independiente). Cada módulo expone `budai_<x>(df, …)` → DataFrame,
`budai_<x>_all_timeframes(dfs)` y `budai_<x>_summary(results)` (estado del último
bar por TF para el prompt). Todos exponen `osc_source` (default `hlc3`, el del
Pine; `close` en señales no aplicables).

| Módulo | Fuente Pine | Núcleo | Señal |
|--------|-------------|--------|-------|
| `budai_pulse.py` | budai_pulse.pine | WaveTrend + COG + Momentum norm. 0-100 | cruce osc/trig + Money Flow |
| `budai_abyss.py` | budai_abyss.pine | WaveTrend crudo (±100) | cruce wt1/wt2 en zona extrema + MF |
| `budai_moneyflow_tide.py` | budai_moneyflow_tide.pine | doble MFI centrado ±100 | cruce fast/slow en zona OB/OS + MF |
| `budai_athenea.py` | budai_athenea_oscillator.pine | WaveTrend + LinReg slope + COG + squeeze + Vix-Fix | cruce cíclico (o pánico) / squeeze release |

Wiring: registrados en `config.ALL_INDICATORS` (`pulse`/`abyss`/`tide`/`athenea`),
computados en `runner._compute_summaries` y renderizados por
`prompt_builder._build_budai_*`. Tests en `tests/` (correr con un venv con
pytest+pandas; `tests/conftest.py` aliasa `pineforge_ai`→repo root como en el
runtime Docker `COPY . ./pineforge_ai`).

---

## Lógica de Descarga de Velas

```python
CANDLES_PER_DAY = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
    "30m": 48,  "1h": 24,  "2h": 12,  "4h": 6,
    "6h": 4,    "8h": 3,   "12h": 2,  "1d": 1,
    "3d": 0.33, "1w": 0.143
}
WARMUP_BARS = 150  # para indicadores con ventanas largas

n_candles = math.ceil(days * CANDLES_PER_DAY[tf]) + WARMUP_BARS
```

---

## Sesiones de Mercado (UTC)

| Sesión | Apertura UTC | Cierre UTC | Notas |
|--------|-------------|------------|-------|
| Sydney | 21:00 | 06:00 | Cruza medianoche |
| Tokyo | 00:00 | 09:00 | |
| Shanghai | 01:30 | 08:00 | |
| India BSE/NSE | 03:45 | 10:00 | |
| Frankfurt XETRA | 07:00 | 16:00 | |
| London LSE | 08:00 | 17:00 | |
| New York NYSE/NASDAQ | 13:30 | 22:00 | |
| Wall Street Core | 14:30 | 21:00 | |

Solapamientos clave:
- **London + NY**: 13:30–17:00 UTC → máxima liquidez
- **Tokyo + London**: 08:00–09:00 UTC → transición Asia-Europa

---

## CLI — Uso Completo

```bash
# Instalación
pip install -e /home/obrero/programacion/PineForge

# Crypto (ccxt/Binance)
python -m pineforge_ai.cli \
  --symbol BTC/USDT \
  --indicators all \
  --timeframes 5m,15m,1h,4h,1d \
  --days 30 \
  --source ccxt \
  --exchange binance \
  --output ./pineforge_ai/output/

# Stocks/índices (yfinance)
python -m pineforge_ai.cli \
  --symbol AAPL \
  --indicators wavetrend,smc \
  --timeframes 1h,4h,1d \
  --days 60 \
  --source yfinance

# Solo un indicador para test rápido
python -m pineforge_ai.cli --symbol ETH/USDT --indicators wavetrend --timeframes 1h --days 5
```

### Flags
| Flag | Default | Descripción |
|------|---------|-------------|
| `--symbol` | requerido | BTCUSDT, BTC/USDT, AAPL, ^FTSE, EURUSD=X |
| `--indicators` | all | wavetrend, luxalgo, smc — separados por coma |
| `--timeframes` | 1h,4h,1d | Separados por coma |
| `--days` | 60 | Días hacia atrás desde ahora |
| `--source` | auto | yfinance \| ccxt (auto-detect por símbolo) |
| `--exchange` | binance | Exchange para ccxt |
| `--output` | ./pineforge_ai/output | Directorio de salida |

---

## Formato del Prompt de Salida

Archivo: `{SYMBOL}_{YYYY-MM-DD_HHMM}UTC.txt`

```
═══════════════════════════════════════════════════════════════
  PINEFORGE AI — MARKET ANALYSIS REQUEST
  Symbol   : BTC/USDT  |  Source: Binance (ccxt)
  Generated: 2026-04-30 18:45 UTC
  Data     : 1H×820 velas | 4H×205 velas | 1D×90 velas
═══════════════════════════════════════════════════════════════

[SESSION STATUS]
Active  : London (cerrando), New York (abierta)
Overlap : London-NY — ALTA LIQUIDEZ
Closed  : Sydney, Tokyo, Shanghai, India, Frankfurt

[PRICE SNAPSHOT]
TF   | Open      | High      | Low       | Close     | Volume
1H   | 94,200.00 | 94,850.00 | 93,900.00 | 94,700.00 | 1.23B
4H   | 93,100.00 | 95,200.00 | 92,800.00 | 94,700.00 | 4.87B
1D   | 91,500.00 | 95,500.00 | 91,200.00 | 94,700.00 | 18.4B

[WAVETREND OSCILLATOR — MTF]
TF   | Osc   | Trigger | Hyper | Trend  | Signal
5M   |  62.1 |   58.3  |  60.5 | ↑ Bull | —
15M  |  71.4 |   65.0  |  68.2 | ↑ Bull | MomCrossUp
1H   |  55.2 |   52.1  |  53.8 | ↑ Bull | —
4H   |  48.9 |   51.3  |  50.1 | ↓ Bear | —
1D   |  61.0 |   55.0  |  57.3 | ↑ Bull | —
MTF Alignment: 4/5 Bull (Strong)

[LUXALGO ADAPTIVE MOMENTUM — MTF]
TF   | AMO    | AMA    | Dir    | Divergence
1H   |  +1.23 |  +0.87 | ↑ Bull | —
4H   |  -0.42 |  -0.18 | ↓ Bear | Bull Div (forming)
1D   |  +2.10 |  +1.95 | ↑ Bull | —

[SMC ELITE — MARKET STRUCTURE]
TF   | Last Event | Level     | OB Zone               | FVG Zone
1H   | BOS Bull   | 93,800.00 | Bull OB [93.2k-93.5k] | Bull FVG [93.5k-93.8k]
4H   | CHoCH Bear | 95,200.00 | Bear OB [94.9k-95.2k] | Bear FVG [94.8k-95.1k]
1D   | BOS Bull   | 91,000.00 | Bull OB [90.3k-90.8k] | —

[SMC CONFLUENCE SCORES]
TF   | Score | Fisher        | Frost     | Key Signal
1H   | 7/10  | +2.1 (Extreme Bull) | Bull (conf 8) | Fisher CrossUp + OB touch
4H   | 3/10  | -0.8 (Neutral)     | Bear (conf 5) | CHoCH sin confirmar
1D   | 6/10  | +1.4 (Bull)        | Bull (conf 7) | BOS Bull confirmado

[ANALYSIS REQUEST]
Eres un trader institucional experto en Smart Money Concepts y análisis técnico de elite.
Con los datos anteriores (indicadores WaveTrend, LuxAlgo AMO y SMC Elite en múltiples
timeframes), realiza el siguiente análisis:

1. TENDENCIA DOMINANTE: Determina la dirección macro (HTF) y micro (LTF) del mercado.
2. ZONAS CLAVE: Identifica OBs, FVGs, niveles BOS/CHoCH de mayor relevancia.
3. ENTRADAS POTENCIALES: Para cada setup válido, especifica:
   - Dirección (long/short)
   - Zona de entrada (precio o rango)
   - Stop Loss (nivel y razón)
   - Take Profit 1 y 2 (niveles)
   - Risk/Reward ratio
   - Timeframe de ejecución
4. SESIÓN ÓPTIMA: Cuándo ejecutar considerando liquidez y solapamientos.
5. RIESGOS: Confluencias en contra, divergencias sin resolver, eventos macro.
6. CONTEXTO GLOBAL: Correlación con DXY, índices principales, mercados relacionados.

Responde ÚNICAMENTE en JSON con esta estructura exacta:
{
  "trend_analysis": {
    "dominant": "bullish|bearish|neutral",
    "htf_bias": "string",
    "ltf_bias": "string",
    "by_tf": {"1H": "bull|bear|neutral", "4H": "...", "1D": "..."}
  },
  "key_zones": [
    {"type": "OB|FVG|BOS|CHoCH", "direction": "bull|bear", "tf": "1H", "level_top": 0.0, "level_bottom": 0.0, "strength": 8}
  ],
  "entries": [
    {
      "id": 1,
      "direction": "long|short",
      "entry_zone": [93500.0, 93800.0],
      "stop_loss": 92400.0,
      "take_profit_1": 96000.0,
      "take_profit_2": 98500.0,
      "risk_reward": 1.8,
      "execution_tf": "1H",
      "trigger": "string — qué debe pasar para activar la entrada",
      "confidence": 7
    }
  ],
  "optimal_session": "London-NY overlap (13:30-17:00 UTC)",
  "risks": ["string", "string"],
  "global_context": {
    "dxy": "bullish|bearish|neutral",
    "sp500": "bullish|bearish|neutral",
    "btc_dominance": "string",
    "notes": "string"
  }
}
═══════════════════════════════════════════════════════════════
```

---

## Parche v2 — Envío a API (stub ya incluido)

```python
# En prompt_builder.py
def send_to_ai(prompt: str, api_key: str, model: str = "claude-opus-4-7") -> dict:
    """
    Envía el prompt a la API de Anthropic y retorna el JSON de análisis.
    Implementar en v2 usando anthropic SDK con prompt caching.
    """
    raise NotImplementedError("Implementar en v2")
```

---

## Errores Comunes — Python

### EWM con adjust=False vs adjust=True
```python
# adjust=False = comportamiento idéntico a Pine Script EMA
series.ewm(span=length, adjust=False).mean()  # ✅
series.ewm(span=length, adjust=True).mean()   # ❌ difiere en primeras barras
```

### NaN en rolling con min_periods
```python
# Siempre especificar min_periods=1 solo cuando se acepta resultado parcial
# Para indicadores: dejar min_periods=length (default) → NaN hasta tener suficientes datos
```

### Timezone en yfinance
```python
# yfinance retorna index sin tz en algunos casos → siempre normalizar
df.index = pd.to_datetime(df.index, utc=True)
```
