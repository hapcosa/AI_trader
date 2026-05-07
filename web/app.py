"""FastAPI app for generating PineForge prompt files from a browser."""

from __future__ import annotations

import os
from html import escape
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pineforge_ai.config import ALL_INDICATORS, DEFAULT_EXCHANGE
from pineforge_ai.runner import generate_prompt_file


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


app = FastAPI(title="PineForge Web Runner")


INDEX_HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PineForge</title>
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
      <h1>PineForge</h1>
      <div id="status" class="status"></div>
    </header>

    <form id="runner-form">
      <div class="grid">
        <div class="field">
          <label for="symbol">Simbolo</label>
          <input id="symbol" name="symbol" type="text" value="BTC/USDT" autocomplete="off" required>
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

        <fieldset class="field mid">
          <legend>Historial</legend>
          <div class="segmented">
            <label class="choice"><input type="radio" name="history_mode" value="candles" checked><span>Velas</span></label>
            <label class="choice"><input type="radio" name="history_mode" value="days"><span>Dias</span></label>
          </div>
        </fieldset>

        <div class="field mid">
          <label for="amount">Cantidad</label>
          <input id="amount" name="amount" type="number" value="300" min="1" step="1" required>
        </div>

        <div class="field">
          <label for="exchange">Exchange</label>
          <input id="exchange" name="exchange" type="text" value="__DEFAULT_EXCHANGE__" autocomplete="off">
        </div>

        <fieldset class="field wide">
          <legend>Timeframes</legend>
          <div class="checks">__TIMEFRAME_OPTIONS__</div>
        </fieldset>

        <fieldset class="field wide">
          <legend>Indicadores</legend>
          <div class="checks">__INDICATOR_OPTIONS__</div>
        </fieldset>

        <label class="toggle">
          <input type="checkbox" name="context" checked>
          Contexto de mercado
        </label>
      </div>

      <div class="actions">
        <button id="submit" type="submit">Ejecutar y descargar</button>
      </div>
    </form>
  </main>

  <script>
    const form = document.querySelector("#runner-form");
    const statusEl = document.querySelector("#status");
    const submit = document.querySelector("#submit");

    function checkedValues(name) {
      return [...form.querySelectorAll(`[name="${name}"]:checked`)].map((item) => item.value);
    }

    function fileNameFromHeader(header) {
      const value = header || "";
      const match = /filename\\*?=(?:UTF-8''|")?([^";]+)/i.exec(value);
      if (!match) return "pineforge_prompt.txt";
      return decodeURIComponent(match[1].replace(/"/g, ""));
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      statusEl.className = "status";
      statusEl.textContent = "Generando...";
      submit.disabled = true;

      const payload = {
        symbol: form.symbol.value.trim(),
        mode: form.mode.value,
        source: form.source.value,
        exchange: form.exchange.value.trim(),
        history_mode: form.history_mode.value,
        amount: Number(form.amount.value),
        timeframes: checkedValues("timeframes"),
        indicators: checkedValues("indicators"),
        context: form.context.checked
      };

      try {
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (!response.ok) {
          let detail = "No se pudo generar el archivo.";
          try {
            const data = await response.json();
            detail = data.detail || detail;
          } catch (_) {}
          throw new Error(detail);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = fileNameFromHeader(response.headers.get("Content-Disposition"));
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);

        statusEl.className = "status done";
        statusEl.textContent = "Descarga iniciada.";
      } catch (error) {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      } finally {
        submit.disabled = false;
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
            "__TIMEFRAME_OPTIONS__",
            _checkboxes("timeframes", TIMEFRAME_ORDER, DEFAULT_WEB_TIMEFRAMES),
        )
        .replace(
            "__INDICATOR_OPTIONS__",
            _checkboxes("indicators", list(ALL_INDICATORS), set(ALL_INDICATORS)),
        )
        .replace("__DEFAULT_EXCHANGE__", escape(DEFAULT_EXCHANGE))
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

        days, candles = _history_from_payload(payload)
        context_enabled = _bool_value(payload.get("context"), default=True)
        if "no_context" in payload:
            no_context = _bool_value(payload.get("no_context"))
        else:
            no_context = not context_enabled

        result = generate_prompt_file(
            symbol=str(payload.get("symbol", "")).strip(),
            indicators=payload.get("indicators", "all"),
            timeframes=payload.get("timeframes"),
            days=days,
            candles=candles,
            source=str(payload.get("source", "auto")).strip() or "auto",
            exchange=str(payload.get("exchange", DEFAULT_EXCHANGE)).strip() or DEFAULT_EXCHANGE,
            no_context=no_context,
            mode=str(payload.get("mode", "mindset")).strip() or "mindset",
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
