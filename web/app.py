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
    .ai-launch {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px dashed var(--line);
      flex-wrap: wrap;
    }
    .ai-launch-label {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      margin-right: 4px;
    }
    .ai-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 38px;
      padding: 0 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.08s ease, box-shadow 0.12s ease;
    }
    .ai-btn:hover {
      box-shadow: 0 4px 12px rgba(23, 32, 38, 0.10);
      transform: translateY(-1px);
    }
    .ai-btn svg {
      width: 18px;
      height: 18px;
      flex-shrink: 0;
    }
    .ai-btn.chatgpt:hover { border-color: #10a37f; }
    .ai-btn.claude:hover  { border-color: #cc785c; }
    .ai-btn.gemini:hover  { border-color: #4285f4; }
    .ai-btn.deepseek:hover{ border-color: #4d6bfe; }
    .combo { position: relative; }
    .combo-list {
      position: absolute;
      top: calc(100% + 4px);
      left: 0;
      right: 0;
      z-index: 50;
      max-height: 280px;
      overflow-y: auto;
      margin: 0;
      padding: 4px 0;
      list-style: none;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: 0 8px 24px rgba(23, 32, 38, 0.12);
    }
    .combo-list li {
      padding: 8px 12px;
      cursor: pointer;
      font-size: 14px;
      color: var(--ink);
    }
    .combo-list li:hover,
    .combo-list li.active {
      background: #e7f5f6;
      color: var(--accent-strong);
    }
    .combo-list li.empty {
      color: var(--muted);
      cursor: default;
      font-style: italic;
    }
    .combo-list li.empty:hover { background: transparent; color: var(--muted); }
    .toast {
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%) translateY(-20px);
      z-index: 1000;
      min-width: 320px;
      max-width: 90vw;
      padding: 14px 22px;
      background: #087f8c;
      color: #fff;
      border-radius: 8px;
      box-shadow: 0 12px 36px rgba(0,0,0,0.22);
      font-size: 15px;
      font-weight: 600;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.22s ease, transform 0.22s ease;
      text-align: center;
    }
    .toast.show {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
      pointer-events: auto;
    }
    .toast.error { background: #a33434; }
    .toast kbd {
      display: inline-block;
      padding: 2px 7px;
      margin: 0 2px;
      background: rgba(255,255,255,0.22);
      border-radius: 4px;
      font-family: "Fira Mono", "Cascadia Code", monospace;
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
  <div id="toast" class="toast" role="status" aria-live="polite"></div>

  <main>
    <header>
      <h1>AI Trader</h1>
      <div id="status" class="status"></div>
    </header>

    <form id="runner-form">
      <div class="grid">
        <div class="field combo-field">
          <label for="symbol-search">Simbolo</label>
          <div class="combo">
            <input type="text" id="symbol-search" autocomplete="off" placeholder="Buscar par..." value="BTC/USDT">
            <input type="hidden" name="symbol" id="symbol" value="BTC/USDT">
            <ul id="symbol-list" class="combo-list" hidden></ul>
          </div>
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

        <div class="field mid">
          <label for="api_key">Anthropic API Key</label>
          <input type="password" id="api_key" name="api_key" placeholder="sk-ant-..." autocomplete="off">
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
      <div class="ai-launch">
        <span class="ai-launch-label">Abrir en IA externa:</span>
        <button class="ai-btn chatgpt" data-ai="chatgpt" type="button" title="ChatGPT — prefill soportado">
          <svg viewBox="0 0 24 24" fill="#10a37f" xmlns="http://www.w3.org/2000/svg"><path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91 6.046 6.046 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a5.985 5.985 0 0 0-3.998 2.9 6.046 6.046 0 0 0 .743 7.097 5.98 5.98 0 0 0 .51 4.911 6.051 6.051 0 0 0 6.515 2.9A5.985 5.985 0 0 0 13.26 24a6.056 6.056 0 0 0 5.772-4.206 5.99 5.99 0 0 0 3.997-2.9 6.056 6.056 0 0 0-.747-7.073zM13.26 22.43a4.476 4.476 0 0 1-2.876-1.04l.141-.081 4.779-2.758a.795.795 0 0 0 .392-.681v-6.737l2.02 1.168a.071.071 0 0 1 .038.052v5.583a4.504 4.504 0 0 1-4.494 4.494zM3.6 18.304a4.47 4.47 0 0 1-.535-3.014l.142.085 4.783 2.759a.771.771 0 0 0 .78 0l5.843-3.369v2.332a.08.08 0 0 1-.033.062L9.74 19.95a4.5 4.5 0 0 1-6.14-1.646zM2.34 7.896a4.485 4.485 0 0 1 2.366-1.973V11.6a.766.766 0 0 0 .388.676l5.815 3.355-2.02 1.168a.076.076 0 0 1-.071 0l-4.83-2.786A4.504 4.504 0 0 1 2.34 7.872zm16.597 3.855l-5.833-3.387L15.119 7.2a.076.076 0 0 1 .071 0l4.83 2.791a4.494 4.494 0 0 1-.676 8.105v-5.678a.79.79 0 0 0-.407-.667zm2.01-3.023l-.141-.085-4.774-2.782a.776.776 0 0 0-.785 0L9.409 9.23V6.897a.066.066 0 0 1 .028-.061l4.83-2.787a4.5 4.5 0 0 1 6.68 4.66zm-12.64 4.135l-2.02-1.164a.08.08 0 0 1-.038-.057V6.075a4.5 4.5 0 0 1 7.375-3.453l-.142.08L8.704 5.46a.795.795 0 0 0-.393.681zm1.097-2.365l2.602-1.5 2.607 1.5v2.999l-2.597 1.5-2.607-1.5z"/></svg>
          <span>ChatGPT</span>
        </button>
        <button class="ai-btn claude" data-ai="claude" type="button" title="Claude — prefill soportado">
          <svg viewBox="0 0 24 24" fill="#cc785c" xmlns="http://www.w3.org/2000/svg"><path d="M4.709 15.955l4.72-2.647.08-.23-.08-.128h-.23l-.79-.048-2.698-.073-2.34-.097-2.265-.122-.571-.121L0 11.784l.055-.352.48-.321.686.06 1.52.103 2.278.158 1.652.097 2.449.255h.389l.055-.157-.134-.098-.103-.097-2.358-1.596-2.552-1.688-1.336-.972-.724-.491-.364-.462-.158-1.008.656-.722.881.06.225.061.892.686 1.908 1.476 2.491 1.833.365.304.146-.103.018-.072-.164-.274-1.355-2.446-1.446-2.49-.644-1.032-.17-.619a2.97 2.97 0 01-.104-.729L6.283.134 6.696 0l.996.134.42.364.62 1.414 1.002 2.229 1.555 3.03.456.898.243.832.091.255h.158V8.91l.128-1.706.237-2.095.23-2.695.08-.76.376-.91.747-.492.584.28.48.685-.067.444-.286 1.851-.559 2.903-.364 1.942h.212l.243-.242.985-1.306 1.652-2.064.73-.82.85-.904.547-.431h1.033l.76 1.129-.34 1.166-1.064 1.347-.881 1.142-1.264 1.7-.79 1.36.073.11.188-.02 2.856-.606 1.543-.28 1.841-.315.833.388.091.395-.328.807-1.969.486-2.309.462-3.438.813-.042.03.049.061 1.549.146.662.036h1.622l3.02.225.79.522.473.638-.079.485-1.215.62-1.64-.389-3.829-.91-1.312-.329h-.182v.11l1.093 1.068 2.006 1.81 2.509 2.33.127.578-.322.455-.34-.049-2.205-1.657-.851-.747-1.926-1.62h-.128v.17l.444.649 2.345 3.521.122 1.08-.17.353-.608.213-.668-.122-1.374-1.925-1.415-2.167-1.143-1.943-.14.08-.674 7.254-.316.37-.729.28-.607-.461-.322-.747.322-1.476.389-1.924.315-1.53.286-1.9.17-.632-.012-.042-.14.018-1.434 1.967-2.18 2.945-1.726 1.845-.414.164-.717-.37.066-.662.401-.589 2.388-3.036 1.44-1.882.93-1.086-.006-.158h-.055L4.132 18.56l-1.13.146-.487-.456.061-.746.231-.243 1.908-1.312-.006.006z"/></svg>
          <span>Claude</span>
        </button>
        <button class="ai-btn gemini" data-ai="gemini" type="button" title="Gemini — sin prefill, prompt en portapapeles">
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="gemG" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#4285f4"/><stop offset="50%" stop-color="#9b72cb"/><stop offset="100%" stop-color="#d96570"/></linearGradient></defs><path fill="url(#gemG)" d="M12 24A14.304 14.304 0 0 0 0 12 14.304 14.304 0 0 0 12 0a14.305 14.305 0 0 0 12 12 14.305 14.305 0 0 0-12 12"/></svg>
          <span>Gemini</span>
        </button>
        <button class="ai-btn deepseek" data-ai="deepseek" type="button" title="DeepSeek — sin prefill, prompt en portapapeles">
          <svg viewBox="0 0 24 24" fill="#4d6bfe" xmlns="http://www.w3.org/2000/svg"><path d="M23.748 4.482c-.254-.124-.364.113-.512.234-.051.039-.094.09-.137.136-.372.397-.806.657-1.373.626-.829-.046-1.537.214-2.163.848-.133-.782-.575-1.248-1.247-1.548-.352-.156-.708-.311-.955-.65-.172-.241-.219-.51-.305-.774-.055-.16-.11-.323-.293-.35-.2-.031-.278.136-.356.276-.313.572-.434 1.202-.422 1.84.027 1.436.633 2.58 1.838 3.393.137.093.172.187.129.323-.082.28-.18.552-.266.833-.055.179-.137.217-.329.14a5.526 5.526 0 0 1-1.736-1.18c-.857-.828-1.631-1.742-2.597-2.458a11.365 11.365 0 0 0-.689-.471c-.985-.957.13-1.743.388-1.836.27-.098.093-.432-.779-.428-.872.004-1.67.295-2.687.684a3.055 3.055 0 0 1-.465.137 9.597 9.597 0 0 0-2.883-.102c-1.885.21-3.39 1.102-4.497 2.623C.082 8.606-.231 10.684.152 12.85c.403 2.284 1.569 4.175 3.36 5.653 1.858 1.533 3.997 2.284 6.438 2.14 1.482-.085 3.133-.284 4.994-1.86.47.234.962.327 1.78.397.63.059 1.236-.03 1.705-.128.735-.156.684-.837.419-.961-2.155-1.004-1.682-.595-2.113-.926 1.096-1.296 2.746-2.642 3.392-7.003.05-.347.007-.565 0-.845-.004-.17.035-.237.23-.256a4.173 4.173 0 0 0 1.545-.475c1.396-.763 1.96-2.015 2.093-3.517.02-.23-.004-.467-.247-.588zM11.581 18c-2.089-1.642-3.102-2.183-3.52-2.16-.392.024-.321.471-.235.763.09.288.207.486.371.739.114.167.192.416-.113.603-.673.416-1.842-.14-1.897-.167-1.361-.802-2.5-1.86-3.301-3.307-.774-1.393-1.224-2.887-1.298-4.482-.02-.386.093-.522.477-.592a4.696 4.696 0 0 1 1.529-.039c2.132.312 3.946 1.265 5.468 2.774.868.86 1.525 1.887 2.202 2.891.72 1.066 1.494 2.082 2.48 2.914.348.292.625.514.891.677-.802.09-2.14.11-3.054-.614zm1-6.44a.306.306 0 0 1 .415-.287.302.302 0 0 1 .2.288.306.306 0 0 1-.31.307.303.303 0 0 1-.304-.308zm3.11 1.596c-.2.081-.399.151-.59.16a1.245 1.245 0 0 1-.798-.254c-.274-.23-.47-.358-.552-.758a1.73 1.73 0 0 1 .016-.588c.07-.327-.008-.537-.239-.727-.187-.156-.426-.199-.688-.199a.559.559 0 0 1-.254-.078c-.11-.054-.2-.19-.114-.358.028-.054.16-.186.192-.21.356-.202.767-.136 1.146.016.352.144.618.408 1.001.782.391.452.462.577.685.916.176.265.336.537.445.848.067.195-.019.354-.25.451z"/></svg>
          <span>DeepSeek</span>
        </button>
      </div>
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

    const SYMBOLS = [
      "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","DOGE/USDT","ADA/USDT","AVAX/USDT",
      "LINK/USDT","DOT/USDT","MATIC/USDT","UNI/USDT","ATOM/USDT","LTC/USDT","BCH/USDT","NEAR/USDT",
      "OP/USDT","ARB/USDT","INJ/USDT","TRX/USDT","APT/USDT","SUI/USDT","TON/USDT","WIF/USDT",
      "PEPE/USDT","SHIB/USDT","ETC/USDT","FIL/USDT",
      "AAVE/USDT","APR/USDT","ASTER/USDT","BIO/USDT","BIRB/USDT","CELO/USDT","CETUS/USDT","CFX/USDT",
      "CHZ/USDT","ENA/USDT","ESPORTS/USDT","FARTCOIN/USDT","HBAR/USDT","HUMA/USDT","HYPE/USDT",
      "IP/USDT","KAS/USDT","KMNO/USDT","M/USDT","MORPHO/USDT","MOVR/USDT","MU/USDT","NAORIS/USDT",
      "NEIROCTO/USDT","ONDO/USDT","PARTI/USDT","PENDLE/USDT","PENGU/USDT","PEOPLE/USDT","PHA/USDT",
      "PIEVERSE/USDT","PIPPIN/USDT","PI/USDT","PIXEL/USDT","PLUME/USDT","PNUT/USDT","POL/USDT",
      "POLYX/USDT","POPCAT/USDT","POWER/USDT","POWR/USDT","PROVE/USDT","PUMP/USDT","PUNDIX/USDT",
      "PYTH/USDT","RAVE/USDT","RAY/USDT","RENDER/USDT","RIVER/USDT","RUNE/USDT","SAHARA/USDT",
      "SIREN/USDT","STO/USDT","TAO/USDT","TIA/USDT","TRUMP/USDT","XLM/USDT","XMR/USDT","ZEC/USDT"
    ].sort();

    const symbolInput  = document.querySelector("#symbol-search");
    const symbolHidden = document.querySelector("#symbol");
    const symbolList   = document.querySelector("#symbol-list");
    let activeIdx = -1;
    let filtered = SYMBOLS.slice();

    function renderSymbolList(items) {
      symbolList.innerHTML = "";
      if (!items.length) {
        const li = document.createElement("li");
        li.className = "empty";
        li.textContent = "Sin resultados";
        symbolList.appendChild(li);
        return;
      }
      items.forEach((sym, i) => {
        const li = document.createElement("li");
        li.textContent = sym;
        li.dataset.value = sym;
        if (i === activeIdx) li.classList.add("active");
        li.addEventListener("mousedown", (e) => {
          e.preventDefault();
          selectSymbol(sym);
        });
        symbolList.appendChild(li);
      });
    }

    function filterSymbols(q) {
      const term = q.trim().toUpperCase();
      filtered = term
        ? SYMBOLS.filter((s) => s.toUpperCase().includes(term))
        : SYMBOLS.slice();
      activeIdx = filtered.length ? 0 : -1;
      renderSymbolList(filtered);
    }

    function selectSymbol(sym) {
      symbolInput.value = sym;
      symbolHidden.value = sym;
      symbolList.hidden = true;
    }

    symbolInput.addEventListener("focus", () => {
      filterSymbols(symbolInput.value);
      symbolList.hidden = false;
    });
    symbolInput.addEventListener("input", () => {
      filterSymbols(symbolInput.value);
      symbolList.hidden = false;
      symbolHidden.value = symbolInput.value.toUpperCase();
    });
    symbolInput.addEventListener("blur", () => {
      setTimeout(() => { symbolList.hidden = true; }, 120);
    });
    symbolInput.addEventListener("keydown", (e) => {
      if (symbolList.hidden) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        activeIdx = Math.min(activeIdx + 1, filtered.length - 1);
        renderSymbolList(filtered);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        activeIdx = Math.max(activeIdx - 1, 0);
        renderSymbolList(filtered);
      } else if (e.key === "Enter") {
        if (filtered[activeIdx]) {
          e.preventDefault();
          selectSymbol(filtered[activeIdx]);
        }
      } else if (e.key === "Escape") {
        symbolList.hidden = true;
      }
    });

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
      const apiKey = apiKeyInput.value.trim();
      if (!apiKey) {
        showToast("Ingresa tu <b>Anthropic API Key</b> arriba para analizar con Claude.", "error", 6000);
        apiKeyInput.focus();
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
          body: JSON.stringify({ prompt, model, mode, api_key: apiKey })
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

    const AI_URLS = {
      chatgpt:  { base: "https://chatgpt.com/",          label: "ChatGPT"  },
      claude:   { base: "https://claude.ai/new",         label: "Claude"   },
      gemini:   { base: "https://gemini.google.com/app", label: "Gemini"   },
      deepseek: { base: "https://chat.deepseek.com/",    label: "DeepSeek" }
    };

    const toastEl = document.querySelector("#toast");
    let toastTimer = null;
    function showToast(html, kind = "ok", ms = 4500) {
      toastEl.innerHTML = html;
      toastEl.className = "toast show" + (kind === "error" ? " error" : "");
      if (toastTimer) clearTimeout(toastTimer);
      toastTimer = setTimeout(() => { toastEl.classList.remove("show"); }, ms);
    }

    async function copyToClipboard(text) {
      try { await navigator.clipboard.writeText(text); return true; }
      catch (_) {
        try {
          promptTextarea.select();
          document.execCommand("copy");
          window.getSelection().removeAllRanges();
          return true;
        } catch (__) { return false; }
      }
    }

    document.querySelectorAll(".ai-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const prompt = promptTextarea.value;
        if (!prompt) {
          showToast("Genera el prompt primero (botón <b>Mostrar prompt</b>).", "error");
          return;
        }
        const ai = btn.dataset.ai;
        const cfg = AI_URLS[ai];
        if (!cfg) return;
        const copied = await copyToClipboard(prompt);
        if (!copied) {
          showToast("No se pudo copiar al portapapeles. Copia manualmente del cuadro y pega en " + cfg.label + ".", "error", 7000);
          window.open(cfg.base, "_blank", "noopener,noreferrer");
          return;
        }
        window.open(cfg.base, "_blank", "noopener,noreferrer");
        showToast(
          "✓ Prompt copiado al portapapeles. En " + cfg.label + " pega con <kbd>Ctrl</kbd>+<kbd>V</kbd> y pulsa <kbd>Enter</kbd>.",
          "ok",
          6000
        );
      });
    });

    // ─── API Key persistente en localStorage ─────────────────────────────────
    const apiKeyInput = document.querySelector("#api_key");
    const STORAGE_KEY = "ai_trader_anthropic_key";
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) apiKeyInput.value = saved;
    } catch (_) {}
    apiKeyInput.addEventListener("change", () => {
      try {
        const v = apiKeyInput.value.trim();
        if (v) localStorage.setItem(STORAGE_KEY, v);
        else localStorage.removeItem(STORAGE_KEY);
      } catch (_) {}
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

        api_key = str(payload.get("api_key", "")).strip()
        if not api_key:
            raise ValueError(
                "API Key requerida. Ingresa tu Anthropic API Key en el formulario."
            )
        if not api_key.startswith("sk-ant-"):
            raise ValueError(
                "Formato de API Key inválido. Debe comenzar con 'sk-ant-'."
            )

        from pineforge_ai.prompt_builder import call_claude_raw
        result = call_claude_raw(prompt=prompt, api_key=api_key, model=model, mode=mode)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(result)
