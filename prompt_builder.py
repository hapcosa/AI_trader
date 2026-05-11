"""
Master Prompt Builder v3 — assembles structured market analysis prompt.
Modes: 'signal' (JSON setups) | 'mindset' (Pre-NY Protocol checklist).

Includes WaveTrend, LuxAlgo AMO, SMC Buda, WAE+Chop, Ehlers iTrend, ICT, Trendlines,
USDT Dominance, 14D OHLCV history, correlations and volatility context.
Implements send_to_ai() against Anthropic API with prompt caching.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pineforge_ai.sessions import format_session_block


# ─── System prompt: SIGNAL mode (cacheable) ───────────────────────────────────

SYSTEM_PROMPT = """Eres un trader institucional senior con 15+ años de experiencia.
Especialidades: Smart Money Concepts (SMC), Inner Circle Trader (ICT), indicadores Ehlers,
análisis multi-timeframe, gestión de riesgo institucional y correlación de mercados globales.

Tu tarea es analizar datos de indicadores técnicos en múltiples timeframes y producir un
análisis JSON riguroso con setups de entrada accionables.

REGLAS ESTRICTAS:
1. Responde SIEMPRE y ÚNICAMENTE en JSON válido, sin texto extra antes o después.
2. NO inventes niveles. Solo cita precios que aparecen en los datos provistos.
3. Cada zona y setup debe citar el TF de origen.
4. RR < 1.5 → descartar setup. Confianza < 6 → descartar.
5. Si los indicadores muestran señales mixtas, prefiere "neutral" antes que forzar dirección.
6. SL siempre tiene justificación estructural (OB invalidado, swing roto, etc).
7. Si no hay setups válidos, devuelve "entries": [] — no rellenes con setups débiles.

Estructura JSON exacta requerida:
{
  "market_state": {"phase": "accumulation|distribution|markup|markdown|ranging", "confidence": 0-10},
  "trend_analysis": {
    "dominant": "bullish|bearish|neutral",
    "htf_bias": "string",
    "ltf_bias": "string",
    "mtf_alignment": "confluencia|divergencia|mixto",
    "by_tf": {"<TF>": "bull|bear|neutral"}
  },
  "key_zones": [
    {"type":"OB|FVG|BOS|CHoCH|BSL|SSL", "direction":"bull|bear", "tf":"<TF>",
     "level_top":0.0, "level_bottom":0.0, "strength":0-10, "description":"string"}
  ],
  "entries": [
    {"id":1, "direction":"long|short",
     "entry_zone":[low, high], "stop_loss":0.0, "sl_reason":"string",
     "take_profit_1":0.0, "take_profit_2":0.0, "risk_reward":0.0,
     "execution_tf":"<TF>", "trigger":"string", "invalidation":"string",
     "confluence_factors":["string"], "confidence":0-10, "session":"string"}
  ],
  "optimal_session": "string",
  "risks": ["string"],
  "global_context": {"dxy":"...", "sp500":"...", "gold":"...", "btc_dominance":"...", "notes":"string"},
  "cycle_phase": "peak|trough|rising|falling|unknown",
  "summary": "string max 3 frases"
}"""


# ─── System prompt: MINDSET mode (Pre-NY Protocol) ────────────────────────────

SYSTEM_PROMPT_MINDSET = """Eres un trader institucional SMC con protocolo Pre-NY activado.
Tu tarea: analizar sesión de Asia y London antes de apertura de Nueva York,
evaluar USDT.D, BTC.D y estructura de BTC, producir un plan de trading legible.

REGLAS ESTRICTAS:
1. Responde en TEXTO ESTRUCTURADO con secciones, NO en JSON.
2. Usa los datos OHLCV reales para identificar HIGH/LOW de Asia y London.
3. Asia session = 00:00–09:00 UTC. London = 08:00–13:30 UTC.
4. Si faltan datos escribe "N/D" — no inventes niveles.
5. USDT.D subiendo → sesgo SHORT crypto. Bajando → sesgo LONG crypto.
6. BTC.D alto (>55%) → BTC season. BTC.D bajo (<45%) → altseason.
7. Liquidity taken = Caso A (NY continúa). Intacta = Caso B (NY manipula primero).

FORMATO DE RESPUESTA OBLIGATORIO:

═══ SESIÓN ASIA ═══
High: [precio exacto]
Low: [precio exacto]
Tipo: [Rango / Tendencia]
Liquidez: [Arriba / Abajo / Ambos / Ninguno]

═══ SESIÓN LONDON ═══
Rompió High de Asia: [Sí / No]
Rompió Low de Asia: [Sí / No]
Tipo de movimiento: [Manipulación / Intención / Poco claro]
Dirección del sweep: [Arriba / Abajo / Ninguno]
FVG u OB post-sweep: [descripción o N/D]

═══ DOMINANCIAS ═══
USDT.D: [valor%] | Tendencia: [Alcista/Bajista/Neutral] | Zona: [High/Mid/Low] | Sesgo: [Long/Short/Neutral]
BTC.D:  [valor%] | Zona: [BTC season / Transitional / Altseason] | Impacto: [descripción breve]

═══ ESTRUCTURA BTC ═══
Ubicación: [PDH / PDL / Mid-range]
Evento reciente: [BOS Alcista / BOS Bajista / CHoCH / Ninguno]
Sesgo: [Alcista / Bajista / Neutral]

═══ ESTADO DE MERCADO ═══
Liquidez tomada: [Sí / No]
Caso: [A — NY continúa tendencia / B — NY manipula primero]
Conclusión: [una o dos frases]

═══ ESCENARIOS ═══

▶ LONG
  Condición: [qué debe ocurrir para activar]
  Entrada: [precio o rango]
  Stop Loss: [precio]
  TP1: [precio] | TP2: [precio]
  Invalidación: [qué cancela el setup]

▶ SHORT
  Condición: [qué debe ocurrir para activar]
  Entrada: [precio o rango]
  Stop Loss: [precio]
  TP1: [precio] | TP2: [precio]
  Invalidación: [qué cancela el setup]

═══ VEREDICTO ═══
¿Operar ahora?: [Sí / No / Esperar confirmación]
Razón: [explicación directa]
Sesión óptima: [nombre y horario UTC]
Checklist:
  [✅/❌] Sweep visto
  [✅/❌] Confirmación post-sweep
  [✅/❌] USDT.D alineado
  [✅/❌] Estructura clara"""


_SUMMARY_INSTRUCTION = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[INSTRUCCIÓN ADICIONAL — RESUMEN PARA WEB]

Después de tu análisis completo, agrega una sección final con este formato exacto.
No analices desde cero. Solo resume e interpreta lo que ya dijiste.

REGLAS:
- Máximo 120 palabras en el resumen total.
- Lenguaje simple, emojis, apto para alguien nuevo en trading.
- No prometas ganancias. No digas "100% seguro".
- Si no hay señal clara: escribe "⚠️ Esperar confirmación".
- Si faltan datos para entrada, TP o SL, escribe "N/D" — no los inventes.

━━━ RESUMEN WEB ━━━
🟠 {SYMBOL}

📍 Estado: [Resumen corto del mercado en 1 frase]
🧭 Sesgo: [🟢 Alcista / 🔴 Bajista / 🟡 Mixto]
🎯 Plan: [LONG / SHORT / ⚠️ Esperar confirmación]

📌 Niveles clave:
   Entrada: [precio o N/D]
   SL: [precio o N/D]
   TP: [precio(s) o N/D]

🧠 Por qué: [1–2 frases simples]
⚠️ Cuidado: [qué debe evitar el usuario]
🏁 Veredicto: [decisión final en 1 frase]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


# ─── Formatting Helpers ───────────────────────────────────────────────────────

def _fmt_price(val, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    try:
        return f"{float(val):,.{decimals}f}"
    except Exception:
        return "—"


def _fmt_vol(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if val >= 1e9: return f"{val/1e9:.2f}B"
    if val >= 1e6: return f"{val/1e6:.2f}M"
    if val >= 1e3: return f"{val/1e3:.2f}K"
    return f"{val:.2f}"


def _separator(char: str = "═", width: int = 70) -> str:
    return char * width


def _section(title: str) -> str:
    return f"\n[{title}]"


def _row(*cols, widths: list[int]) -> str:
    parts = []
    for i, (col, w) in enumerate(zip(cols, widths)):
        s = str(col)
        parts.append(s.ljust(w) if i == 0 else s.center(w))
    return "  ".join(parts)


# ─── Block Builders ───────────────────────────────────────────────────────────

def _build_header(symbol, source, exchange, timeframes, candle_counts, dt_utc):
    sep = _separator()
    candles_str = "  |  ".join(f"{tf.upper()}×{candle_counts.get(tf, '?')}" for tf in timeframes)
    return (
        f"{sep}\n"
        f"  AI TRADER v3 — MARKET ANALYSIS REQUEST\n"
        f"  Symbol   : {symbol}  |  Source: {source.upper()} ({exchange})\n"
        f"  Generated: {dt_utc.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"  Data     : {candles_str}\n"
        f"{sep}"
    )


def _build_session(dt_utc):
    return _section("SESSION STATUS") + "\n" + format_session_block(dt_utc)


def _build_price(dfs, timeframes):
    lines = [_section("PRICE SNAPSHOT")]
    lines.append(_row("TF", "Open", "High", "Low", "Close", "Volume",
                      widths=[5, 14, 14, 14, 14, 12]))
    lines.append("-" * 80)
    for tf in timeframes:
        df = dfs.get(tf)
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        lines.append(_row(
            tf.upper(),
            _fmt_price(last["open"]),
            _fmt_price(last["high"]),
            _fmt_price(last["low"]),
            _fmt_price(last["close"]),
            _fmt_vol(last["volume"]),
            widths=[5, 14, 14, 14, 14, 12],
        ))
    return "\n".join(lines)


def _build_wavetrend(wt, timeframes):
    lines = [_section("WAVETREND OSCILLATOR — MTF")]
    lines.append(_row("TF", "Osc", "Trig", "Hyper", "EOT", "Trend", "Vol", "Signal",
                      widths=[5, 7, 7, 7, 7, 11, 8, 22]))
    lines.append("-" * 90)
    bull = bear = total = 0
    for tf in timeframes:
        d = wt.get(tf)
        if d is None: continue
        total += 1
        if "Bull" in d["trend"]: bull += 1
        if "Bear" in d["trend"]: bear += 1
        lines.append(_row(
            tf.upper(),
            d.get("osc", "—"),
            d.get("trigger", "—"),
            d.get("hyper", "—"),
            d.get("eot", "—"),
            d["trend"],
            d.get("vol_regime", "—"),
            d["signal"],
            widths=[5, 7, 7, 7, 7, 11, 8, 22],
        ))
    if total:
        align = (
            f"STRONG BULL ({bull}/{total})" if bull == total else
            f"STRONG BEAR ({bear}/{total})" if bear == total else
            f"Mixed: {bull}B/{bear}Bear/{total-bull-bear}N"
        )
        lines.append(f"\nMTF Alignment: {align}")
    return "\n".join(lines)


def _build_luxalgo(lux, timeframes):
    lines = [_section("LUXALGO ADAPTIVE MOMENTUM — MTF")]
    lines.append(_row("TF", "AMO", "AMA", "Dir", "Divergence",
                      widths=[5, 11, 11, 12, 22]))
    lines.append("-" * 65)
    for tf in timeframes:
        d = lux.get(tf)
        if d is None: continue
        amo_v = d.get("amo", 0.0)
        ama_v = d.get("ama", 0.0)
        lines.append(_row(
            tf.upper(),
            f"{amo_v:+.4f}" if isinstance(amo_v, (int, float)) else "—",
            f"{ama_v:+.4f}" if isinstance(ama_v, (int, float)) else "—",
            d.get("direction", "—"),
            d.get("divergence", "—"),
            widths=[5, 11, 11, 12, 22],
        ))
    return "\n".join(lines)


def _build_trend_quality(tq, timeframes):
    lines = [_section("TREND QUALITY — WAE + CHOPPINESS")]
    lines.append(_row("TF", "WAE State", "TrendUp", "TrendDn", "Chop", "Regime",
                      widths=[5, 22, 9, 9, 7, 11]))
    lines.append("-" * 72)
    for tf in timeframes:
        d = tq.get(tf)
        if d is None: continue
        lines.append(_row(
            tf.upper(),
            d.get("wae_state", "—"),
            f"{d.get('trend_up', 0):.2f}",
            f"{d.get('trend_dn', 0):.2f}",
            f"{d.get('chop_idx', 0):.1f}",
            d.get("regime", "—"),
            widths=[5, 22, 9, 9, 7, 11],
        ))
    return "\n".join(lines)


def _build_itrend(it, timeframes):
    lines = [_section("EHLERS iTREND — ADAPTIVE FILTER")]
    lines.append(_row("TF", "Value", "Slope%", "Trend", "Signal",
                      widths=[5, 14, 10, 12, 28]))
    lines.append("-" * 75)
    for tf in timeframes:
        d = it.get(tf)
        if d is None: continue
        lines.append(_row(
            tf.upper(),
            _fmt_price(d.get("value"), 4),
            f"{d.get('slope', 0):+.4f}",
            d.get("trend", "—"),
            d.get("signal", "—"),
            widths=[5, 14, 10, 12, 28],
        ))
    return "\n".join(lines)


def _build_smc(smc, timeframes):
    lines = [_section("SMC BUDA — MARKET STRUCTURE")]
    lines.append(_row("TF", "Last Event", "BOS Level", "Bull OB", "Bear OB",
                      widths=[5, 14, 14, 25, 25]))
    lines.append("-" * 90)
    for tf in timeframes:
        d = smc.get(tf)
        if d is None: continue
        ob_b = d.get("ob_bull", "—")
        if d.get("in_bull_ob"): ob_b += " ◄ IN"
        ob_s = d.get("ob_bear", "—")
        if d.get("in_bear_ob"): ob_s += " ◄ IN"
        lines.append(_row(
            tf.upper(),
            d.get("last_event", "—"),
            _fmt_price(d.get("bos_level")),
            ob_b, ob_s,
            widths=[5, 14, 14, 25, 25],
        ))
    lines.append("")
    lines.append(_row("TF", "Bull FVG", "Bear FVG", "Sweep", "Fisher", "Score",
                      widths=[5, 22, 22, 8, 22, 8]))
    lines.append("-" * 92)
    for tf in timeframes:
        d = smc.get(tf)
        if d is None: continue
        fb = d.get("fvg_bull", "—")
        if d.get("in_bull_fvg"): fb += " ◄"
        fs = d.get("fvg_bear", "—")
        if d.get("in_bear_fvg"): fs += " ◄"
        sw = "↓ Dn" if d.get("dnsweep") else ("↑ Up" if d.get("upsweep") else "—")
        lines.append(_row(
            tf.upper(), fb, fs, sw,
            d.get("fisher", "—"),
            f"{d.get('confluence', 0)}/10",
            widths=[5, 22, 22, 8, 22, 8],
        ))

    sigs = []
    for tf in timeframes:
        d = smc.get(tf)
        if d is None: continue
        s = d.get("signal")
        if s:
            sigs.append(
                f"  {tf.upper()} {s.get('direction','')} Type-{s.get('type','')} "
                f"({s.get('confluence_factors',0)}/5) | "
                f"Entry {_fmt_price(s.get('entry'))} | SL {_fmt_price(s.get('sl'))} | "
                f"TP {_fmt_price(s.get('tp'))} | {s.get('reason','')}"
            )
    if sigs:
        lines.append("\n[SMC SIGNALS]")
        lines.extend(sigs)
    return "\n".join(lines)


def _build_ict(ict, timeframes):
    lines = [_section("ICT CONCEPTS — STRUCTURE & LIQUIDITY")]
    lines.append(_row("TF", "Last Event", "BSL", "SSL", "Bull FVG", "Bear FVG", "Recent",
                      widths=[5, 12, 12, 12, 18, 18, 24]))
    lines.append("-" * 105)
    for tf in timeframes:
        d = ict.get(tf)
        if d is None: continue
        bull_fvg = (
            f"[{_fmt_price(d['fvg_bull_btm'])}-{_fmt_price(d['fvg_bull_top'])}]"
            if d.get("fvg_bull_top") else "—"
        )
        bear_fvg = (
            f"[{_fmt_price(d['fvg_bear_btm'])}-{_fmt_price(d['fvg_bear_top'])}]"
            if d.get("fvg_bear_top") else "—"
        )
        lines.append(_row(
            tf.upper(),
            d.get("last_event", "—"),
            _fmt_price(d.get("bsl_level")),
            _fmt_price(d.get("ssl_level")),
            bull_fvg, bear_fvg,
            d.get("signals", "—"),
            widths=[5, 12, 12, 12, 18, 18, 24],
        ))
    return "\n".join(lines)


def _build_trendlines(tl, timeframes):
    lines = [_section("TRENDLINES — DIAGONAL STRUCTURE")]
    lines.append(_row("TF", "Upper TL", "Lower TL", "Slope↑", "Slope↓", "Signal",
                      widths=[5, 14, 14, 12, 12, 18]))
    lines.append("-" * 80)
    for tf in timeframes:
        d = tl.get(tf)
        if d is None: continue
        lines.append(_row(
            tf.upper(),
            _fmt_price(d.get("upper_tl")),
            _fmt_price(d.get("lower_tl")),
            f"{d.get('slope_up', 0):.4f}",
            f"{d.get('slope_dn', 0):.4f}",
            d.get("signals", "—"),
            widths=[5, 14, 14, 12, 12, 18],
        ))
    return "\n".join(lines)


def _build_correlations(corr):
    if not corr:
        return _section("MARKET CORRELATIONS") + "\n  (no data)"
    lines = [_section("MARKET CORRELATIONS")]
    lines.append(_row("Asset", "Close", "Δ 1D %", "Trend 3D",
                      widths=[12, 14, 10, 12]))
    lines.append("-" * 55)
    for k, v in corr.items():
        lines.append(_row(
            k,
            _fmt_price(v.get("close"), 4),
            f"{v.get('change_1d_pct', 0):+.2f}%" if v.get("change_1d_pct") is not None else "—",
            v.get("trend_3d", "—"),
            widths=[12, 14, 10, 12],
        ))
    return "\n".join(lines)


def _build_volatility(vol, timeframes):
    if not vol:
        return _section("VOLATILITY REGIME") + "\n  (no data)"
    lines = [_section("VOLATILITY REGIME")]
    lines.append(_row("TF", "ATR", "ATR %ile", "Realized Vol", "Regime",
                      widths=[5, 12, 10, 14, 14]))
    lines.append("-" * 65)
    for tf in timeframes:
        d = vol.get(tf)
        if d is None: continue
        lines.append(_row(
            tf.upper(),
            _fmt_price(d.get("atr"), 4),
            f"{d.get('atr_pct')}" if d.get("atr_pct") is not None else "—",
            f"{d.get('realized_vol')}" if d.get("realized_vol") is not None else "—",
            d.get("regime", "—"),
            widths=[5, 12, 10, 14, 14],
        ))
    return "\n".join(lines)


def _build_market_cap(mc_data: dict | None) -> str:
    lines = [_section("CRYPTO MARKET — BTC & ALTCOIN DOMINANCE")]
    if not mc_data or not mc_data.get("available"):
        lines.append("  (CoinGecko unavailable)")
        return "\n".join(lines)

    btc = mc_data.get("btc_dominance", 0.0)
    eth = mc_data.get("eth_dominance", 0.0)
    alt = mc_data.get("altcoin_dominance", 0.0)
    total_mc = mc_data.get("total_market_cap_usd", 0.0)
    vol_24h = mc_data.get("total_volume_24h_usd", 0.0)
    mc_change = mc_data.get("market_cap_change_24h_pct", 0.0)
    btc_zone = mc_data.get("btc_zone", "—")
    mc_signal = mc_data.get("mc_signal", "—")
    top5 = mc_data.get("top_dominances", {})
    fetched_at = mc_data.get("fetched_at", "—")

    def _fmt_mc(val):
        if val >= 1e12: return f"${val/1e12:.3f}T"
        if val >= 1e9:  return f"${val/1e9:.2f}B"
        if val >= 1e6:  return f"${val/1e6:.1f}M"
        return f"${val:.0f}"

    lines.append(f"  BTC.D : {btc:.2f}%  |  ETH.D: {eth:.2f}%  |  Altcoins: {alt:.1f}%")
    lines.append(f"  Zone  : {btc_zone}")
    lines.append(f"  Total : {_fmt_mc(total_mc)}  |  Vol 24h: {_fmt_mc(vol_24h)}  |  Δ24h: {mc_change:+.2f}%  →  {mc_signal}")
    if top5:
        top5_str = "  |  ".join(f"{k.upper()}={v:.1f}%" for k, v in top5.items())
        lines.append(f"  Top5  : {top5_str}")
    lines.append(f"  Source: CoinGecko  |  As of: {fetched_at}")
    return "\n".join(lines)


def _build_usdt_dominance(usdt_data: dict | None) -> str:
    lines = [_section("USDT DOMINANCE")]
    if not usdt_data or not usdt_data.get("available"):
        lines.append("  (daemon not running — no USDT.D data)")
        lines.append("  Start with: python -m pineforge_ai.usdt_dominance.daemon")
        return "\n".join(lines)

    cur = usdt_data.get("current")
    zone = usdt_data.get("zone", "—")
    t1d = usdt_data.get("trend_1d", "—")
    t4h = usdt_data.get("trend_4h", "—")
    t1h = usdt_data.get("trend_1h", "—")

    trend_arrow = {"bull": "↑", "bear": "↓", "neutral": "→"}
    signal = (
        "🔴 SHORT bias (risk-off)" if t1h == "bear" and t4h == "bear"
        else "🟢 LONG bias (risk-on)" if t1h == "bull" and t4h == "bull"
        else "⚪ Neutral"
    )

    lines.append(f"  Current : {cur:.4f}%  |  Zone: {zone}")
    lines.append(
        f"  Trend   : 1D={trend_arrow.get(t1d,'?')}{t1d}  "
        f"4H={trend_arrow.get(t4h,'?')}{t4h}  "
        f"1H={trend_arrow.get(t1h,'?')}{t1h}"
    )
    lines.append(f"  Signal  : {signal}")

    df14 = usdt_data.get("ohlcv_1d")
    if df14 is not None and not df14.empty:
        lines.append("")
        lines.append("  [14D DAILY OHLCV]")
        lines.append("  Date          Open      High      Low       Close")
        lines.append("  " + "-" * 52)
        for ts, row in df14.tail(14).iterrows():
            date_str = ts.strftime("%Y-%m-%d")
            lines.append(
                f"  {date_str}   "
                f"{row['open']:.4f}    {row['high']:.4f}    "
                f"{row['low']:.4f}    {row['close']:.4f}"
            )
    return "\n".join(lines)


def _build_ohlcv_history(
    dfs: dict[str, pd.DataFrame],
    timeframes: list[str],
    days: int = 14,
) -> str:
    lines = [_section("OHLCV — 14D HISTORY")]
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)

    for tf in timeframes:
        df = dfs.get(tf)
        if df is None or df.empty:
            continue
        sub = df[df.index >= cutoff].copy()
        if sub.empty:
            continue
        lines.append(f"\n  [{tf.upper()}]")
        lines.append(
            "  datetime(UTC)        open           high           low            close          volume"
        )
        lines.append("  " + "-" * 100)
        for ts, row in sub.iterrows():
            dt_str = ts.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  {dt_str}   "
                f"{_fmt_price(row['open']):>14}  "
                f"{_fmt_price(row['high']):>14}  "
                f"{_fmt_price(row['low']):>14}  "
                f"{_fmt_price(row['close']):>14}  "
                f"{_fmt_vol(row['volume']):>10}"
            )
    return "\n".join(lines)


def _build_session_history(dfs: dict, n_sessions: int = 5) -> str:
    preferred = ["1h", "30m", "15m", "2h", "4h"]
    df = None
    used_tf = None
    for tf in preferred:
        candidate = dfs.get(tf)
        if candidate is not None and not candidate.empty:
            df = candidate.copy()
            used_tf = tf
            break

    lines = [_section(f"SESSION HISTORY — ÚLTIMAS {n_sessions} SESIONES [{used_tf or '?'}]")]

    if df is None:
        lines.append("  (sin datos horarios disponibles)")
        return "\n".join(lines)

    # Sessions defined in UTC hours [start, end)
    SESSIONS = [
        ("Asia",   0,  9),
        ("London", 8,  14),
        ("NY",    13,  22),
    ]

    df["_date"] = df.index.normalize()
    dates = sorted(df["_date"].unique())[-n_sessions:]

    lines.append(_row("Fecha", "Sesion", "Open", "High", "Low", "Close", widths=[12, 8, 14, 14, 14, 14]))
    lines.append("-" * 80)

    for d in dates:
        day_df = df[df["_date"] == d]
        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        first_in_day = True
        for sess_name, h_start, h_end in SESSIONS:
            sess_df = day_df[(day_df.index.hour >= h_start) & (day_df.index.hour < h_end)]
            if sess_df.empty:
                continue
            lines.append(_row(
                date_str if first_in_day else "",
                sess_name,
                _fmt_price(sess_df["open"].iloc[0]),
                _fmt_price(sess_df["high"].max()),
                _fmt_price(sess_df["low"].min()),
                _fmt_price(sess_df["close"].iloc[-1]),
                widths=[12, 8, 14, 14, 14, 14],
            ))
            first_in_day = False
        lines.append("")

    df.drop(columns=["_date"], inplace=True)
    return "\n".join(lines)


def _build_recent_alerts(symbol: str | None, hours: int = 24) -> str:
    """Pull recent alerts from indicatorsForge SQLite DB and format for the prompt."""
    try:
        from AI_trader.data.alerts_reader import (
            format_alerts_block,
            format_dominance_block,
            get_dominance_trend,
            get_recent_alerts,
        )
    except ImportError:
        try:
            from .data.alerts_reader import (  # type: ignore
                format_alerts_block,
                format_dominance_block,
                get_dominance_trend,
                get_recent_alerts,
            )
        except ImportError:
            return ""

    smc_alerts = get_recent_alerts(symbol=symbol, source_type="indicator_smc", since_hours=hours, limit=15)
    tv_alerts = get_recent_alerts(symbol=symbol, source_type="tradingview_alert", since_hours=hours, limit=10)
    ai_alerts = get_recent_alerts(symbol=symbol, source_type="ai_signal", since_hours=hours, limit=5)
    dom_trend = get_dominance_trend(hours=hours)

    if not smc_alerts and not tv_alerts and not ai_alerts and not dom_trend.get("available"):
        return ""

    parts = [_section(f"RECENT ALERTS — last {hours}h")]
    parts.append("[USDT DOMINANCE CONTEXT]")
    parts.append(format_dominance_block(dom_trend))
    if smc_alerts:
        parts.append("\n[SMC ALERTS]")
        parts.append(format_alerts_block(smc_alerts, max_items=15))
    if tv_alerts:
        parts.append("\n[TRADINGVIEW ALERTS]")
        parts.append(format_alerts_block(tv_alerts, max_items=10))
    if ai_alerts:
        parts.append("\n[PRIOR AI SIGNALS]")
        parts.append(format_alerts_block(ai_alerts, max_items=5))
    return "\n".join(parts)


def _build_pretrain_context(pretrain_summary):
    if not pretrain_summary:
        return ""
    lines = [_section("PRETRAIN CONTEXT — PRIOR ITERATIONS")]
    for entry in pretrain_summary:
        lines.append(f"  • {entry}")
    return "\n".join(lines)


def _build_request(symbol, timeframes, dt_utc, ai_summary: bool = False):
    tf_str = ", ".join(t.upper() for t in timeframes)
    summary_block = _SUMMARY_INSTRUCTION.replace("{SYMBOL}", symbol) if ai_summary else ""
    return f"""{_section("ROLE & ANALYSIS RULES")}
{SYSTEM_PROMPT}

{_section("ANALYSIS REQUEST")}
Símbolo: {symbol}
Timeframes: {tf_str}
Hora UTC: {dt_utc.strftime('%Y-%m-%d %H:%M')}

Aplica el rol y reglas definidos arriba sobre los datos de indicadores provistos.
Devuelve ÚNICAMENTE el JSON con la estructura especificada. Sin texto adicional.
{summary_block}
{_separator()}"""


def _build_request_mindset(symbol, timeframes, dt_utc, ai_summary: bool = False):
    tf_str = ", ".join(t.upper() for t in timeframes)
    summary_block = _SUMMARY_INSTRUCTION.replace("{SYMBOL}", symbol) if ai_summary else ""
    return f"""{_section("ROLE & ANALYSIS RULES")}
{SYSTEM_PROMPT_MINDSET}

{_section("PRE-NY PROTOCOL — MINDSET MODE")}

[ASIA SESSION — 00:00–09:00 UTC]
1. ¿Asia fue rango o tendencia? (analiza OHLCV dentro de esa ventana horaria)
2. ¿Dónde está el HIGH de Asia? (precio exacto del máximo dentro de 00:00-09:00 UTC)
3. ¿Dónde está el LOW de Asia? (precio exacto del mínimo dentro de 00:00-09:00 UTC)
4. ¿Se tomaron esos niveles durante London, o siguen intactos?
5. ¿Dónde está la liquidez clara? (stops acumulados por encima o debajo)

[LONDON SESSION — 08:00–13:30 UTC]
6. ¿London rompió el HIGH de Asia hacia arriba (upsweep de liquidez)?
7. ¿London rompió el LOW de Asia hacia abajo (downsweep)?
8. ¿Qué tipo de movimiento fue: manipulación (reversión post-sweep) o intención (continuación)?
9. ¿Hay CHoCH o BOS visible en 15m/1h después del sweep?
10. ¿Hay Fair Value Gap (FVG) o Order Block relevante post-sweep?

[USDT.D + BTC.D STATUS]
11. ¿USDT.D está subiendo o bajando? ¿En qué zona? (High >5% / Mid 3-5% / Low <3%)
12. ¿Qué sesgo implica USDT.D para crypto? (risk-on → LONG / risk-off → SHORT / neutral)
13. ¿BTC.D está en zona alta (>55%), media o baja (<45%)? ¿Favorece BTC o altcoins?

[BTC STRUCTURE]
14. ¿BTC está en PDH (Previous Day High), PDL (Previous Day Low) o mid-range?
15. ¿Hay BOS o CHoCH reciente en BTC que confirme o invalide el setup?

[ANALYSIS REQUEST]
Símbolo: {symbol}
Timeframes: {tf_str}
Hora UTC: {dt_utc.strftime('%Y-%m-%d %H:%M')}

Con los datos OHLCV (sección OHLCV — 14D HISTORY), indicadores, dominancias y sesiones:
1. Responde el checklist Pre-NY con datos reales
2. Define: ¿mercado ya tomó liquidez (Caso A) o aún no (Caso B)?
3. Plantea 2 escenarios (LONG y SHORT) con condiciones y niveles exactos
4. Da veredicto claro: operar ahora o esperar — con checklist final

Usa el FORMATO DE RESPUESTA definido arriba (secciones con ═══, NO JSON).
{summary_block}
{_separator()}"""


# ─── Main Builder ─────────────────────────────────────────────────────────────

def build_prompt(
    symbol: str,
    dfs: dict[str, pd.DataFrame],
    timeframes: list[str],
    wt_summary: dict | None = None,
    lux_summary: dict | None = None,
    smc_sum: dict | None = None,
    tq_summary: dict | None = None,
    cc_summary: dict | None = None,
    ict_sum: dict | None = None,
    tl_summary: dict | None = None,
    it_summary: dict | None = None,
    correlations: dict | None = None,
    volatility: dict | None = None,
    pretrain_summary: list[str] | None = None,
    usdt_data: dict | None = None,
    market_cap_data: dict | None = None,
    source: str = "auto",
    exchange: str = "binance",
    candle_counts: dict[str, int] | None = None,
    dt_utc: datetime | None = None,
    mode: str = "signal",
    ohlcv_history_days: int = 14,
    ai_summary: bool = False,
) -> str:
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)
    if candle_counts is None:
        candle_counts = {tf: len(df) for tf, df in dfs.items() if df is not None}

    blocks = [
        _build_header(symbol, source, exchange, timeframes, candle_counts, dt_utc),
        _build_session(dt_utc),
        _build_price(dfs, timeframes),
    ]

    if wt_summary:   blocks.append(_build_wavetrend(wt_summary, timeframes))
    if lux_summary:  blocks.append(_build_luxalgo(lux_summary, timeframes))
    if tq_summary:   blocks.append(_build_trend_quality(tq_summary, timeframes))
    if it_summary:   blocks.append(_build_itrend(it_summary, timeframes))
    if smc_sum:      blocks.append(_build_smc(smc_sum, timeframes))
    if ict_sum:      blocks.append(_build_ict(ict_sum, timeframes))
    if tl_summary:   blocks.append(_build_trendlines(tl_summary, timeframes))

    if correlations: blocks.append(_build_correlations(correlations))
    if volatility:   blocks.append(_build_volatility(volatility, timeframes))

    blocks.append(_build_market_cap(market_cap_data))
    blocks.append(_build_usdt_dominance(usdt_data))

    # Recent alerts from indicatorsForge (TradingView/SMC/USDT.D/AI signals)
    alerts_block = _build_recent_alerts(symbol=symbol, hours=24)
    if alerts_block:
        blocks.append(alerts_block)

    # Raw OHLCV history (14 days)
    blocks.append(_build_ohlcv_history(dfs, timeframes, days=ohlcv_history_days))

    if pretrain_summary:
        blocks.append(_build_pretrain_context(pretrain_summary))

    if mode == "mindset":
        blocks.append(_build_session_history(dfs, n_sessions=5))
        blocks.append(_build_request_mindset(symbol, timeframes, dt_utc, ai_summary=ai_summary))
    else:
        blocks.append(_build_request(symbol, timeframes, dt_utc, ai_summary=ai_summary))

    return "\n\n".join(blocks)


def save_prompt(prompt_text: str, symbol: str, output_dir: str = "pineforge_ai/output",
                 dt_utc: datetime | None = None) -> str:
    if dt_utc is None:
        dt_utc = datetime.now(tz=timezone.utc)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "-").replace("^", "").replace("=", "")
    fp = os.path.join(output_dir, f"{safe}_{dt_utc.strftime('%Y-%m-%d_%H%M')}UTC.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    return os.path.abspath(fp)


# ─── Anthropic API call ───────────────────────────────────────────────────────

def send_to_ai(
    prompt: str,
    api_key: str | None = None,
    model: str = "claude-opus-4-7",
    max_tokens: int = 4096,
    system_prompt: str | None = None,
    mode: str = "signal",
) -> dict:
    """
    Send prompt to Anthropic API. System prompt is cached (cache_control=ephemeral).

    Returns parsed JSON dict from Claude's response.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Install with: pip install anthropic") from e

    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY (env var or api_key arg).")

    if system_prompt:
        sys_text = system_prompt
    elif mode == "mindset":
        sys_text = SYSTEM_PROMPT_MINDSET
    else:
        sys_text = SYSTEM_PROMPT

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    full = "\n".join(text_parts).strip()

    if full.startswith("```"):
        first_nl = full.find("\n")
        last_fence = full.rfind("```")
        full = full[first_nl + 1: last_fence].strip()

    try:
        data = json.loads(full)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude response not valid JSON: {e}\nRaw: {full[:500]}") from e

    usage = getattr(resp, "usage", None)
    if usage is not None:
        data["_usage"] = {
            "input_tokens":                getattr(usage, "input_tokens", None),
            "output_tokens":               getattr(usage, "output_tokens", None),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens":     getattr(usage, "cache_read_input_tokens", None),
        }
    return data


def call_claude_raw(
    prompt: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 8096,
    mode: str = "mindset",
) -> dict:
    """Call Claude API and return raw text response (no JSON parsing)."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Install with: pip install anthropic") from e

    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment or .env")

    sys_text = SYSTEM_PROMPT_MINDSET if mode == "mindset" else SYSTEM_PROMPT

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    response_text = "\n".join(text_parts).strip()

    usage = getattr(resp, "usage", None)
    usage_dict = {}
    if usage is not None:
        usage_dict = {
            "input_tokens":                getattr(usage, "input_tokens", 0),
            "output_tokens":               getattr(usage, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens":     getattr(usage, "cache_read_input_tokens", 0),
        }

    return {"response": response_text, "model": model, "usage": usage_dict}
