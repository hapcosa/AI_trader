"""FastAPI app for generating AI Trader prompt files from a browser."""

from __future__ import annotations

import os
from html import escape
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pineforge_ai.config import ALL_INDICATORS, DEFAULT_EXCHANGE
from pineforge_ai.runner import generate_prompt, generate_prompt_file


TIMEFRAME_ORDER = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
]
DEFAULT_WEB_TIMEFRAMES = {"1h", "4h", "1d", "1w"}


app = FastAPI(title="AI Trader Web Runner")


INDEX_HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Trader</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #65717a;
      --line: #d9e0e5;
      --accent: #087f8c;
      --accent-strong: #05606a;
      --warn: #b85c00;
      --bad: #a33434;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.1;
      font-weight: 780;
      letter-spacing: 0;
    }
    .status {
      min-height: 22px;
      color: var(--muted);
      font-size: 14px;
      text-align: right;
    }
    form {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 8px 28px rgba(23, 32, 38, 0.06);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 14px;
    }
    .field {
      grid-column: span 4;
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }
    .field.wide { grid-column: span 12; }
    .field.mid { grid-column: span 6; }
    label,
    legend {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    input[type="text"],
    input[type="number"],
    select {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 11px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }
    fieldset {
      margin: 0;
      padding: 0;
      border: 0;
    }
    .segmented,
    .checks {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .choice,
    .check {
      position: relative;
      display: inline-flex;
      align-items: center;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      overflow: hidden;
    }
    .choice input,
    .check input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .choice span,
    .check span {
      padding: 9px 12px;
      font-size: 14px;
      line-height: 1;
      white-space: nowrap;
    }
    .choice input:checked + span,
    .check input:checked + span {
      background: #e7f5f6;
      color: var(--accent-strong);
      box-shadow: inset 0 0 0 1px var(--accent);
    }
    .toggle {
      grid-column: span 12;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--ink);
      font-size: 14px;
      font-weight: 650;
      text-transform: none;
      width: fit-content;
    }
    .toggle input {
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }
    .actions {
      display: flex;
      justify-content: flex-end;
      gap: 12px;
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 16px;
    }
    button {
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      padding: 0 18px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 760;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled {
      cursor: wait;
      opacity: 0.72;
    }
    .error { color: var(--bad); }
    .done { color: var(--accent-strong); }
    .btn-secondary {
      background: #fff;
      color: var(--accent);
      border: 1.5px solid var(--accent);
    }
    .btn-secondary:hover { background: #e7f5f6; }
    .prompt-panel {
      margin-top: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 8px 28px rgba(23, 32, 38, 0.06);
    }
    .prompt-panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .prompt-panel-header span {
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
    }
    .prompt-textarea {
      width: 100%;
      height: 420px;
      font-family: "Fira Mono", "Cascadia Code", "Courier New", monospace;
      font-size: 12.5px;
      line-height: 1.55;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      resize: vertical;
      color: var(--ink);
      background: #fafbfc;
      box-sizing: border-box;
    }
    .copy-btn {
      min-height: 34px;
      padding: 0 14px;
      font-size: 13px;
    }
    .tf-rows { display: flex; flex-wrap: wrap; gap: 8px; }
    .tf-row {
      display: inline-flex;
      align-items: center;
      gap: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }
    .tf-row .tf-chk {
      position: relative;
      display: inline-flex;
      align-items: center;
      min-height: 38px;
      cursor: pointer;
    }
    .tf-row .tf-chk input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .tf-row .tf-chk span {
      padding: 9px 10px;
      font-size: 14px;
      line-height: 1;
      white-space: nowrap;
    }
    .tf-row .tf-chk input:checked + span {
      background: #e7f5f6;
      color: var(--accent-strong);
    }
    .tf-row .tf-sep {
      width: 1px;
      background: var(--line);
      align-self: stretch;
    }
    .tf-row input[type="number"] {
      width: 64px;
      min-height: 38px;
      border: none;
      border-radius: 0;
      padding: 0 8px;
      font-size: 13px;
      color: var(--muted);
      background: transparent;
    }
    .tf-row input[type="number"]:focus { outline: none; color: var(--ink); }
    .ai-response-panel {
      margin-top: 18px;
      background: #0f1923;
      border: 1px solid #1e3040;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 8px 28px rgba(0,0,0,0.18);
    }
    .ai-response-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .ai-response-header span {
      font-size: 13px;
      font-weight: 700;
      color: #4ecdc4;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .ai-response-meta {
      font-size: 11px;
      color: #5a7a8a;
    }
    .ai-response-body {
      color: #d4e8f0;
      font-family: "Fira Mono", "Cascadia Code", "Courier New", monospace;
      font-size: 13px;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .btn-ai {
      background: linear-gradient(135deg, #087f8c, #05606a);
      font-weight: 760;
    }
    .btn-ai:hover { background: linear-gradient(135deg, #05606a, #03484f); }
    .btn-ai:disabled { opacity: 0.5; cursor: not-allowed; }
    .spinner {
      display: inline-block;
      width: 14px; height: 14px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 760px) {
      main { width: min(100vw - 20px, 1120px); padding-top: 18px; }
      header { align-items: start; flex-direction: column; }
      .status { text-align: left; }
      .field,
      .field.mid { grid-column: span 12; }
      .actions { justify-content: stretch; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>AI Trader</h1>
      <div id="status" class="status"></div>
    </header>

    <form id="runner-form">
      <div class="grid">
        <div class="field">
          <label for="symbol">Simbolo</label>
          <select id="symbol" name="symbol">
            <option value="BTC/USDT" selected>BTC/USDT</option>
            <option value="ETH/USDT">ETH/USDT</option>
            <option value="SOL/USDT">SOL/USDT</option>
            <option value="BNB/USDT">BNB/USDT</option>
            <option value="XRP/USDT">XRP/USDT</option>
            <option value="DOGE/USDT">DOGE/USDT</option>
            <option value="ADA/USDT">ADA/USDT</option>
            <option value="AVAX/USDT">AVAX/USDT</option>
            <option value="LINK/USDT">LINK/USDT</option>
            <option value="DOT/USDT">DOT/USDT</option>
            <option value="MATIC/USDT">MATIC/USDT</option>
            <option value="UNI/USDT">UNI/USDT</option>
            <option value="ATOM/USDT">ATOM/USDT</option>
            <option value="LTC/USDT">LTC/USDT</option>
            <option value="BCH/USDT">BCH/USDT</option>
            <option value="NEAR/USDT">NEAR/USDT</option>
            <option value="OP/USDT">OP/USDT</option>
            <option value="ARB/USDT">ARB/USDT</option>
            <option value="INJ/USDT">INJ/USDT</option>
            <option value="TRX/USDT">TRX/USDT</option>
            <option value="APT/USDT">APT/USDT</option>
            <option value="SUI/USDT">SUI/USDT</option>
            <option value="TON/USDT">TON/USDT</option>
            <option value="WIF/USDT">WIF/USDT</option>
            <option value="PEPE/USDT">PEPE/USDT</option>
            <option value="SHIB/USDT">SHIB/USDT</option>
            <option value="ETC/USDT">ETC/USDT</option>
            <option value="FIL/USDT">FIL/USDT</option>
          </select>
        </div>

        <div class="field">
          <label for="mode">Modo</label>
          <select id="mode" name="mode">
            <option value="mindset" selected>mindset</option>
            <option value="signal">signal</option>
          </select>
        </div>

        <div class="field">
          <label for="source">Fuente</label>
          <select id="source" name="source">
            <option value="auto" selected>auto</option>
            <option value="ccxt">ccxt</option>
            <option value="yfinance">yfinance</option>
          </select>
        </div>

        <div class="field">
          <label for="exchange">Exchange</label>
          <select id="exchange" name="exchange">
            <option value="binance" selected>Binance</option>
            <option value="bybit">Bybit</option>
            <option value="bitget">Bitget</option>
            <option value="okx">OKX</option>
            <option value="kraken">Kraken</option>
            <option value="coinbase">Coinbase</option>
            <option value="kucoin">KuCoin</option>
            <option value="gateio">Gate.io</option>
          </select>
        </div>

        <fieldset class="field wide">
          <legend>Timeframes — velas por TF (default 200)</legend>
          <div class="tf-rows">__TIMEFRAME_ROWS__</div>
        </fieldset>

        <fieldset class="field wide">
          <legend>Indicadores</legend>
          <div class="checks">__INDICATOR_OPTIONS__</div>
        </fieldset>

        <div class="field mid">
          <label for="model">Modelo Claude</label>
          <select id="model" name="model">
            <option value="claude-sonnet-4-6" selected>Sonnet 4.6 — Recomendado</option>
            <option value="claude-opus-4-7">Opus 4.7 — Mejor análisis</option>
            <option value="claude-haiku-4-5-20251001">Haiku 4.5 — Más rápido</option>
          </select>
        </div>

        <label class="toggle">
          <input type="checkbox" name="context" checked>
          Contexto de mercado
        </label>
        <label class="toggle">
          <input type="checkbox" name="ai_summary">
          Resumen IA (agrega resumen breve con emojis al final del prompt)
        </label>
      </div>

      <div class="actions">
        <button id="btn-show" type="button" class="btn-secondary">Mostrar prompt</button>
        <button id="btn-claude" type="button" class="btn-ai">Analizar con Claude</button>
        <button id="submit" type="submit">Descargar</button>
      </div>
    </form>

    <div id="prompt-panel" class="prompt-panel" style="display:none">
      <div class="prompt-panel-header">
        <span>Prompt generado</span>
        <button id="copy-btn" class="copy-btn" type="button">Copiar</button>
      </div>
      <textarea id="prompt-textarea" class="prompt-textarea" readonly></textarea>
    </div>

    <div id="ai-panel" class="ai-response-panel" style="display:none">
      <div class="ai-response-header">
        <span>Respuesta Claude</span>
        <span id="ai-meta" class="ai-response-meta"></span>
      </div>
      <div id="ai-body" class="ai-response-body"></div>
    </div>
  </main>

  <script>
    const form = document.querySelector("#runner-form");
    const statusEl = document.querySelector("#status");
    const submit = document.querySelector("#submit");
    const btnShow = document.querySelector("#btn-show");
    const btnClaude = document.querySelector("#btn-claude");
    const promptPanel = document.querySelector("#prompt-panel");
    const promptTextarea = document.querySelector("#prompt-textarea");
    const copyBtn = document.querySelector("#copy-btn");
    const aiPanel = document.querySelector("#ai-panel");
    const aiBody = document.querySelector("#ai-body");
    const aiMeta = document.querySelector("#ai-meta");

    let lastPrompt = null;
    let lastMode = null;

    function checkedValues(name) {
      return [...form.querySelectorAll(`[name="${name}"]:checked`)].map((el) => el.value);
    }

    function fileNameFromHeader(header) {
      const value = header || "";
      const match = /filename\\*?=(?:UTF-8''|")?([^";]+)/i.exec(value);
      if (!match) return "ai_trader_prompt.txt";
      return decodeURIComponent(match[1].replace(/"/g, ""));
    }

    function buildPayload() {
      const checkedTfs = checkedValues("timeframes");
      const candles_per_tf = {};
      checkedTfs.forEach((tf) => {
        const inp = form.querySelector(`input[name="candles_${tf}"]`);
        candles_per_tf[tf] = inp ? Math.max(1, parseInt(inp.value) || 200) : 200;
      });
      return {
        symbol: form.symbol.value,
        mode: form.mode.value,
        source: form.source.value,
        exchange: form.exchange.value,
        timeframes: checkedTfs,
        candles_per_tf,
        indicators: checkedValues("indicators"),
        context: form.context.checked,
        ai_summary: form.ai_summary.checked
      };
    }

    function setLoading(flag) {
      submit.disabled = flag;
      btnShow.disabled = flag;
      btnClaude.disabled = flag;
    }

    function fmtTokens(usage) {
      if (!usage) return "";
      const cached = (usage.cache_read_input_tokens || 0);
      const inp = usage.input_tokens || 0;
      const out = usage.output_tokens || 0;
      return `in: ${inp} | out: ${out} | cache hit: ${cached}`;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      statusEl.className = "status";
      statusEl.textContent = "Generando...";
      setLoading(true);
      try {
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(buildPayload())
        });
        if (!response.ok) {
          let detail = "No se pudo generar el archivo.";
          try { const d = await response.json(); detail = d.detail || detail; } catch (_) {}
          throw new Error(detail);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = fileNameFromHeader(response.headers.get("Content-Disposition"));
        document.body.appendChild(link); link.click(); link.remove();
        URL.revokeObjectURL(url);
        statusEl.className = "status done";
        statusEl.textContent = "Descarga iniciada.";
      } catch (error) {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      } finally { setLoading(false); }
    });

    btnShow.addEventListener("click", async () => {
      statusEl.className = "status";
      statusEl.textContent = "Generando prompt...";
      setLoading(true);
      promptPanel.style.display = "none";
      try {
        const payload = buildPayload();
        const response = await fetch("/api/prompt-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          let detail = "No se pudo generar el prompt.";
          try { const d = await response.json(); detail = d.detail || detail; } catch (_) {}
          throw new Error(detail);
        }
        const data = await response.json();
        lastPrompt = data.prompt || "";
        lastMode = payload.mode;
        promptTextarea.value = lastPrompt;
        promptPanel.style.display = "block";
        promptPanel.scrollIntoView({ behavior: "smooth", block: "start" });
        statusEl.className = "status done";
        statusEl.textContent = "Prompt listo. Puedes enviarlo a Claude.";
      } catch (error) {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      } finally { setLoading(false); }
    });

    btnClaude.addEventListener("click", async () => {
      const prompt = lastPrompt || promptTextarea.value;
      if (!prompt) {
        statusEl.className = "status error";
        statusEl.textContent = "Genera el prompt primero (Mostrar prompt).";
        return;
      }
      statusEl.className = "status";
      statusEl.textContent = "";
      statusEl.innerHTML = '<span class="spinner"></span>Enviando a Claude...';
      setLoading(true);
      aiPanel.style.display = "none";

      try {
        const model = form.model.value;
        const mode = lastMode || form.mode.value;
        const response = await fetch("/api/send-to-ai", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, model, mode })
        });
        if (!response.ok) {
          let detail = "Error al llamar a Claude.";
          try { const d = await response.json(); detail = d.detail || detail; } catch (_) {}
          throw new Error(detail);
        }
        const data = await response.json();
        aiBody.textContent = data.response || "";
        aiMeta.textContent = fmtTokens(data.usage) + (data.model ? "  |  " + data.model : "");
        aiPanel.style.display = "block";
        aiPanel.scrollIntoView({ behavior: "smooth", block: "start" });
        statusEl.className = "status done";
        statusEl.textContent = "Análisis completo.";
      } catch (error) {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      } finally { setLoading(false); }
    });

    copyBtn.addEventListener("click", async () => {
      const text = promptTextarea.value;
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = "Copiado!";
        setTimeout(() => { copyBtn.textContent = "Copiar"; }, 1800);
      } catch (_) {
        promptTextarea.select();
        document.execCommand("copy");
        copyBtn.textContent = "Copiado!";
        setTimeout(() => { copyBtn.textContent = "Copiar"; }, 1800);
      }
    });
  </script>
</body>
</html>
"""


def _checkboxes(name: str, values: list[str], checked: set[str]) -> str:
    items = []
    for value in values:
        safe = escape(value)
        attr = " checked" if value in checked else ""
        items.append(
            f'<label class="check"><input type="checkbox" name="{name}" '
            f'value="{safe}"{attr}><span>{safe}</span></label>'
        )
    return "".join(items)


def _tf_rows(timeframes: list[str], checked: set[str], default_candles: int = 200) -> str:
    items = []
    for tf in timeframes:
        safe = escape(tf)
        attr = " checked" if tf in checked else ""
        items.append(
            f'<div class="tf-row">'
            f'<label class="tf-chk">'
            f'<input type="checkbox" name="timeframes" value="{safe}"{attr}>'
            f'<span>{safe}</span>'
            f'</label>'
            f'<div class="tf-sep"></div>'
            f'<input type="number" name="candles_{safe}" value="{default_candles}" min="10" step="1">'
            f'</div>'
        )
    return "".join(items)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} must be an integer") from e


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _history_from_payload(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    if "days" in payload or "candles" in payload:
        return _optional_int(payload.get("days"), "days"), _optional_int(
            payload.get("candles"), "candles"
        )

    history_mode = str(payload.get("history_mode", "candles")).strip().lower()
    amount = _optional_int(payload.get("amount"), "amount")
    if amount is None:
        raise ValueError("amount is required")
    if history_mode == "days":
        return amount, None
    if history_mode == "candles":
        return None, amount
    raise ValueError("history_mode must be days or candles")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (
        INDEX_HTML.replace(
            "__TIMEFRAME_ROWS__",
            _tf_rows(TIMEFRAME_ORDER, DEFAULT_WEB_TIMEFRAMES),
        )
        .replace(
            "__INDICATOR_OPTIONS__",
            _checkboxes("indicators", list(ALL_INDICATORS), set(ALL_INDICATORS)),
        )
    )
    return HTMLResponse(html)


@app.get("/api/options")
def options() -> JSONResponse:
    return JSONResponse(
        {
            "timeframes": TIMEFRAME_ORDER,
            "default_timeframes": sorted(DEFAULT_WEB_TIMEFRAMES),
            "indicators": ALL_INDICATORS,
            "default_exchange": DEFAULT_EXCHANGE,
        }
    )


@app.post("/api/generate")
async def generate(request: Request) -> FileResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON object body is required")

        context_enabled = _bool_value(payload.get("context"), default=True)
        if "no_context" in payload:
            no_context = _bool_value(payload.get("no_context"))
        else:
            no_context = not context_enabled
        ai_summary = _bool_value(payload.get("ai_summary"), default=False)
        candles_per_tf = payload.get("candles_per_tf") or None
        days, candles = (None, None) if candles_per_tf else _history_from_payload(payload)

        result = generate_prompt_file(
            symbol=str(payload.get("symbol", "")).strip(),
            indicators=payload.get("indicators", "all"),
            timeframes=payload.get("timeframes"),
            days=days,
            candles=candles,
            candles_per_tf=candles_per_tf,
            source=str(payload.get("source", "auto")).strip() or "auto",
            exchange=str(payload.get("exchange", DEFAULT_EXCHANGE)).strip() or DEFAULT_EXCHANGE,
            no_context=no_context,
            mode=str(payload.get("mode", "mindset")).strip() or "mindset",
            ai_summary=ai_summary,
            send_to_ai=False,
            emit=None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not result.file_path:
        raise HTTPException(status_code=500, detail="Prompt file was not created")

    return FileResponse(
        result.file_path,
        media_type="text/plain",
        filename=os.path.basename(result.file_path),
    )


@app.post("/api/prompt-text")
async def prompt_text(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON object body is required")

        context_enabled = _bool_value(payload.get("context"), default=True)
        no_context = not context_enabled if "no_context" not in payload else _bool_value(payload.get("no_context"))
        ai_summary = _bool_value(payload.get("ai_summary"), default=False)
        candles_per_tf = payload.get("candles_per_tf") or None
        days, candles = (None, None) if candles_per_tf else _history_from_payload(payload)

        result = generate_prompt(
            symbol=str(payload.get("symbol", "")).strip(),
            indicators=payload.get("indicators", "all"),
            timeframes=payload.get("timeframes"),
            days=days,
            candles=candles,
            candles_per_tf=candles_per_tf,
            source=str(payload.get("source", "auto")).strip() or "auto",
            exchange=str(payload.get("exchange", DEFAULT_EXCHANGE)).strip() or DEFAULT_EXCHANGE,
            no_context=no_context,
            mode=str(payload.get("mode", "mindset")).strip() or "mindset",
            ai_summary=ai_summary,
            send_to_ai=False,
            save=False,
            emit=None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse({"prompt": result.prompt})


@app.post("/api/send-to-ai")
async def send_to_ai_endpoint(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON object body is required")

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")

        model = str(payload.get("model", "claude-sonnet-4-6")).strip() or "claude-sonnet-4-6"
        mode = str(payload.get("mode", "mindset")).strip() or "mindset"

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or api_key.startswith("sk-ant-..."):
            raise ValueError(
                "ANTHROPIC_API_KEY no configurado. "
                "Agrega tu key en AI_trader/.env → ANTHROPIC_API_KEY=sk-ant-..."
            )

        from pineforge_ai.prompt_builder import call_claude_raw
        result = call_claude_raw(prompt=prompt, api_key=api_key, model=model, mode=mode)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(result)
