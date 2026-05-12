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
from fastapi.staticfiles import StaticFiles

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


from contextlib import asynccontextmanager
import logging

_log = logging.getLogger("pineforge_ai.web")
_digest_scheduler = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _digest_scheduler
    if os.environ.get("DOMINANCE_DIGEST_ENABLED", "true").lower() == "true":
        try:
            from pineforge_ai.dominance_digest.scheduler import DominanceDigestScheduler
            _digest_scheduler = DominanceDigestScheduler()
            await _digest_scheduler.start()
        except Exception as exc:
            _log.error("dominance_digest_start_failed: %s", exc, exc_info=True)
            _digest_scheduler = None
    try:
        yield
    finally:
        if _digest_scheduler is not None:
            await _digest_scheduler.stop()


app = FastAPI(title="AI Trader Web Runner", lifespan=_lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.post("/api/dominance-digest/fire")
async def _fire_digest(request: Request):
    """Manual trigger for tests. Body: {"kind": "4H" | "DAILY"}"""
    if _digest_scheduler is None:
        raise HTTPException(status_code=503, detail="scheduler not running")
    try:
        body = await request.json()
        kind = str((body or {}).get("kind", "4H")).upper()
    except Exception:
        kind = "4H"
    ok = await _digest_scheduler.fire_once(kind=kind)
    return JSONResponse({"ok": ok, "kind": kind})


INDEX_HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Trader — Terminal de Inteligencia de Mercados</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg-0: #050912;
      --bg-1: #0a1424;
      --bg-2: #0f1c33;
      --panel: rgba(11, 22, 41, 0.78);
      --panel-edge: rgba(0, 229, 255, 0.22);
      --ink: #d9f4ff;
      --muted: #5d7a92;
      --muted-2: #8aa9c0;
      --accent: #00e5ff;
      --accent-strong: #29f7ff;
      --accent-2: #ff3df0;
      --accent-3: #b266ff;
      --accent-soft: rgba(0, 229, 255, 0.12);
      --ok: #00ffa3;
      --bad: #ff4d6d;
      --warn: #ffc857;
      --line: rgba(0, 229, 255, 0.22);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Rajdhani", "Inter", ui-sans-serif, system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1100px 600px at 18% 8%, rgba(0, 229, 255, 0.10), transparent 65%),
        radial-gradient(900px 500px at 92% 92%, rgba(255, 61, 240, 0.08), transparent 60%),
        linear-gradient(180deg, #04070f 0%, #060c1a 60%, #03060f 100%);
      background-attachment: fixed;
      position: relative;
      overflow-x: hidden;
    }
    body::before {
      content: "";
      position: fixed; inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(to right, rgba(0,229,255,0.05) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(0,229,255,0.05) 1px, transparent 1px);
      background-size: 40px 40px;
      mask-image: radial-gradient(ellipse at center, #000 35%, transparent 80%);
      z-index: 0;
    }
    body::after {
      content: "";
      position: fixed; inset: 0;
      pointer-events: none;
      background: repeating-linear-gradient(
        180deg,
        rgba(0, 229, 255, 0.02) 0 1px,
        transparent 1px 3px
      );
      z-index: 0;
      mix-blend-mode: screen;
    }
    .hud-shell {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: 240px 1fr;
      gap: 22px;
      width: min(1360px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0 36px;
      align-items: start;
    }
    /* ---------- SIDEBAR ---------- */
    .sidebar {
      position: sticky;
      top: 22px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 18px 14px 16px;
      background: linear-gradient(180deg, rgba(11,22,41,0.92), rgba(7,14,28,0.85));
      border: 1px solid var(--panel-edge);
      box-shadow:
        0 0 0 1px rgba(0, 229, 255, 0.04) inset,
        0 18px 50px rgba(0, 0, 0, 0.45),
        0 0 30px rgba(0, 229, 255, 0.06);
      clip-path: polygon(
        0 12px, 12px 0,
        100% 0, 100% calc(100% - 12px),
        calc(100% - 12px) 100%, 0 100%
      );
    }
    .ai-core-label {
      writing-mode: vertical-rl;
      transform: rotate(180deg);
      position: absolute;
      left: -2px; top: 14px;
      font-family: "Orbitron", monospace;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.35em;
      color: var(--muted-2);
      opacity: 0.7;
    }
    .avatar-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      display: flex;
      align-items: center;
      justify-content: center;
      background:
        radial-gradient(circle at 50% 70%, rgba(0,229,255,0.14), transparent 58%),
        radial-gradient(circle at 50% 30%, rgba(178,102,255,0.10), transparent 65%),
        #01030c;
      border: 1px solid rgba(0,229,255,0.18);
      border-radius: 4px;
      overflow: hidden;
    }
    .avatar-wrap::before {
      content: "";
      position: absolute; inset: 6px;
      border: 1px dashed rgba(0,229,255,0.18);
      border-radius: 3px;
      pointer-events: none;
    }
    .avatar-wrap .corner {
      position: absolute;
      width: 14px; height: 14px;
      border: 1.5px solid var(--accent);
      filter: drop-shadow(0 0 4px rgba(0,229,255,0.6));
    }
    .avatar-wrap .corner.tl { top: 4px; left: 4px; border-right: 0; border-bottom: 0; }
    .avatar-wrap .corner.tr { top: 4px; right: 4px; border-left: 0; border-bottom: 0; }
    .avatar-wrap .corner.bl { bottom: 4px; left: 4px; border-right: 0; border-top: 0; }
    .avatar-wrap .corner.br { bottom: 4px; right: 4px; border-left: 0; border-top: 0; }
    .avatar-gif {
      position: absolute;
      inset: -5%;
      z-index: 1;
      width: 110%;
      height: 110%;
      max-width: none;
      object-fit: cover;
      object-position: center center;
      filter:
        drop-shadow(0 0 16px rgba(0,229,255,0.55))
        drop-shadow(0 0 18px rgba(255,61,240,0.18));
    }
    @keyframes pulse-halo {
      0%, 100% { transform: scale(1); opacity: 0.55; }
      50%      { transform: scale(1.06); opacity: 0.85; }
    }
    .nucleo {
      display: flex; flex-direction: column; gap: 4px;
      padding: 8px 10px;
      background: rgba(4,8,16,0.55);
      border: 1px solid rgba(0,229,255,0.15);
      border-radius: 3px;
    }
    .nucleo-label {
      font-family: "Orbitron", monospace;
      font-size: 9px;
      letter-spacing: 0.28em;
      color: var(--muted-2);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .nucleo-label .dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: var(--accent-2);
      box-shadow: 0 0 8px var(--accent-2);
      animation: pulse-dot 1.4s ease-in-out infinite;
    }
    .mini-chart { width: 100%; height: 34px; display: block; }
    .mini-chart .line {
      fill: none;
      stroke: var(--accent-2);
      stroke-width: 1.4;
      filter: drop-shadow(0 0 4px var(--accent-2));
    }
    .mini-chart .line2 {
      fill: none;
      stroke: var(--accent);
      stroke-width: 1.1;
      opacity: 0.7;
      filter: drop-shadow(0 0 3px var(--accent));
    }
    @keyframes dash-flow {
      to { stroke-dashoffset: -200; }
    }
    .mini-chart .line, .mini-chart .line2 {
      stroke-dasharray: 4 3;
      animation: dash-flow 8s linear infinite;
    }
    .music-console {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 10px;
      background: rgba(4,8,16,0.55);
      border: 1px solid rgba(0,229,255,0.15);
      border-radius: 3px;
    }
    .music-toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      width: 100%;
      min-height: 38px;
      padding: 0 10px;
      border: 1px solid rgba(0,229,255,0.30);
      border-radius: 3px;
      background: rgba(0,229,255,0.06);
      color: var(--muted-2);
      font-family: "Share Tech Mono", monospace;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      cursor: pointer;
      transition: border-color 0.15s ease, color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }
    .music-toggle:hover,
    .music-toggle.is-on {
      border-color: var(--accent);
      color: var(--accent-strong);
      background: rgba(0,229,255,0.12);
      box-shadow: 0 0 14px rgba(0,229,255,0.25), inset 0 0 14px rgba(0,229,255,0.12);
    }
    .music-toggle .music-led {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--muted);
      box-shadow: none;
      flex: 0 0 8px;
    }
    .music-toggle.is-on .music-led {
      background: var(--ok);
      box-shadow: 0 0 10px var(--ok);
      animation: pulse-dot 1.4s ease-in-out infinite;
    }
    .music-meter {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 3px;
      height: 18px;
      align-items: end;
    }
    .music-meter i {
      display: block;
      height: 25%;
      background: linear-gradient(180deg, var(--accent), var(--accent-2));
      opacity: 0.35;
      box-shadow: 0 0 8px rgba(0,229,255,0.35);
      animation: meter-idle 1.4s ease-in-out infinite;
      animation-play-state: paused;
    }
    .music-console.is-on .music-meter i { animation-play-state: running; }
    .music-meter i:nth-child(2n) { animation-duration: 1.1s; }
    .music-meter i:nth-child(3n) { animation-duration: 1.7s; }
    @keyframes meter-idle {
      0%, 100% { height: 25%; opacity: 0.3; }
      50% { height: 100%; opacity: 0.9; }
    }
    .stat-grid {
      display: flex; flex-direction: column;
      border-top: 1px solid rgba(0,229,255,0.14);
      padding-top: 10px;
    }
    .stat-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 5px 0;
      border-bottom: 1px dashed rgba(0,229,255,0.10);
      font-family: "Share Tech Mono", monospace;
      font-size: 11px;
      letter-spacing: 0.08em;
    }
    .stat-row:last-child { border-bottom: 0; }
    .stat-row label {
      color: var(--muted-2);
      text-transform: uppercase;
      font-weight: 600;
    }
    .stat-row .val { color: var(--accent); font-weight: 700; }
    .stat-row .val.ok { color: var(--ok); }
    .quote {
      margin-top: auto;
      padding: 12px 10px;
      border-top: 1px solid rgba(0,229,255,0.12);
      color: var(--muted-2);
      font-family: "Share Tech Mono", monospace;
      font-size: 10.5px;
      line-height: 1.55;
      letter-spacing: 0.05em;
      text-align: center;
      opacity: 0.85;
    }
    /* ---------- MAIN PANEL ---------- */
    .main-panel {
      position: relative;
      padding: 22px 26px 24px;
      background: linear-gradient(180deg, rgba(11,22,41,0.88), rgba(8,16,30,0.85));
      border: 1px solid var(--panel-edge);
      box-shadow:
        0 0 0 1px rgba(0, 229, 255, 0.04) inset,
        0 24px 60px rgba(0, 0, 0, 0.55),
        0 0 50px rgba(0, 229, 255, 0.05);
      clip-path: polygon(
        0 16px, 16px 0,
        calc(100% - 16px) 0, 100% 16px,
        100% calc(100% - 16px), calc(100% - 16px) 100%,
        16px 100%, 0 calc(100% - 16px)
      );
    }
    .hud-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid rgba(0,229,255,0.12);
      position: relative;
    }
    .hud-header::after {
      content: ""; position: absolute; left: 0; right: 0; bottom: -1px;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--accent) 30%, var(--accent-2) 70%, transparent);
      opacity: 0.55;
      filter: blur(0.3px);
    }
    .title-block .title-row {
      display: flex; align-items: center; gap: 14px;
    }
    .title-block h1 {
      margin: 0;
      font-family: "Orbitron", sans-serif;
      font-size: 30px;
      font-weight: 900;
      letter-spacing: 0.14em;
      color: var(--ink);
      text-transform: uppercase;
      text-shadow: 0 0 18px rgba(0,229,255,0.45), 0 0 2px rgba(255,255,255,0.4);
    }
    .title-block h1 .a2 { color: var(--accent); }
    .title-dots {
      display: inline-flex; gap: 4px;
      padding: 4px 8px;
      border: 1px solid rgba(0,229,255,0.25);
      border-radius: 2px;
      align-items: center;
    }
    .title-dots i {
      width: 5px; height: 5px; border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 6px var(--accent);
      opacity: 0.4;
    }
    .title-dots i:nth-child(1) { animation: blink 1.2s infinite; }
    .title-dots i:nth-child(2) { animation: blink 1.2s 0.3s infinite; }
    .title-dots i:nth-child(3) { animation: blink 1.2s 0.6s infinite; }
    .title-dots i:nth-child(4) { animation: blink 1.2s 0.9s infinite; }
    @keyframes blink { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
    .subtitle {
      margin: 4px 0 0;
      font-family: "Share Tech Mono", monospace;
      font-size: 11px;
      letter-spacing: 0.35em;
      color: var(--muted-2);
      text-transform: uppercase;
    }
    .status-pills {
      display: flex; flex-direction: column; gap: 6px;
      align-items: flex-end;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 5px 10px;
      font-family: "Share Tech Mono", monospace;
      font-size: 10.5px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--ok);
      background: rgba(0, 255, 163, 0.06);
      border: 1px solid rgba(0, 255, 163, 0.35);
      border-radius: 2px;
    }
    .pill .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--ok);
      box-shadow: 0 0 8px var(--ok);
      animation: pulse-dot 1.6s ease-in-out infinite;
    }
    @keyframes pulse-dot {
      0%, 100% { opacity: 1; transform: scale(1); }
      50%      { opacity: 0.55; transform: scale(0.85); }
    }
    .status {
      min-height: 20px;
      color: var(--muted-2);
      font-family: "Share Tech Mono", monospace;
      font-size: 11px;
      letter-spacing: 0.08em;
      margin-top: 6px;
      text-align: right;
    }
    .status.done { color: var(--ok); }
    .status.error { color: var(--bad); }
    /* ---------- FORM ---------- */
    form {
      background: transparent;
      border: 0;
      padding: 0;
      box-shadow: none;
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
      position: relative;
    }
    .field.wide { grid-column: span 12; }
    .field.mid { grid-column: span 6; }
    label,
    legend {
      color: var(--muted-2);
      font-family: "Share Tech Mono", monospace;
      font-size: 10.5px;
      font-weight: 600;
      letter-spacing: 0.22em;
      text-transform: uppercase;
    }
    .input-shell {
      position: relative;
      display: flex;
      align-items: center;
    }
    .input-shell .icon {
      position: absolute;
      left: 12px;
      width: 16px; height: 16px;
      color: var(--accent);
      pointer-events: none;
      filter: drop-shadow(0 0 4px rgba(0,229,255,0.6));
    }
    .input-shell.has-icon input,
    .input-shell.has-icon select {
      padding-left: 38px;
    }
    .asset-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      flex: 0 0 18px;
      border-radius: 50%;
      font-family: "Orbitron", monospace;
      font-size: 9px;
      font-weight: 900;
      letter-spacing: 0;
      color: #04101a;
      background: linear-gradient(135deg, #00e5ff, #29f7ff);
      box-shadow: 0 0 10px rgba(0,229,255,0.55);
      text-transform: uppercase;
    }
    .input-shell .asset-icon {
      position: absolute;
      left: 12px;
      z-index: 2;
      pointer-events: none;
    }
    .asset-icon.btc { background: linear-gradient(135deg, #f7931a, #ffd166); }
    .asset-icon.eth { background: linear-gradient(135deg, #8aa4ff, #e5ecff); }
    .asset-icon.sol { background: linear-gradient(135deg, #14f195, #9945ff); color: #061024; }
    .asset-icon.bnb { background: linear-gradient(135deg, #f3ba2f, #ffe28a); }
    .asset-icon.xrp { background: linear-gradient(135deg, #d9f4ff, #6f8ea6); }
    .asset-icon.doge { background: linear-gradient(135deg, #c2a633, #fff2a6); }
    .asset-icon.ada { background: linear-gradient(135deg, #1f6fff, #7bdcff); }
    .asset-icon.avax { background: linear-gradient(135deg, #e84142, #ff9a9a); color: #fff; }
    .asset-icon.link { background: linear-gradient(135deg, #2a5ada, #8fb1ff); color: #fff; }
    .asset-icon.dot { background: linear-gradient(135deg, #e6007a, #ff8bd0); color: #fff; }
    .asset-icon.matic,
    .asset-icon.pol { background: linear-gradient(135deg, #8247e5, #c7a7ff); color: #fff; }
    .asset-icon.uni { background: linear-gradient(135deg, #ff007a, #ff9bd0); color: #fff; }
    .asset-icon.ltc { background: linear-gradient(135deg, #345d9d, #b8c8e8); color: #fff; }
    .asset-icon.sui { background: linear-gradient(135deg, #6fbcf0, #d9f4ff); }
    .asset-icon.ton { background: linear-gradient(135deg, #0098ea, #8bdcff); }
    .asset-icon.xau,
    .asset-icon.xaut,
    .asset-icon.paxg { background: linear-gradient(135deg, #d4af37, #fff2a6); }
    .asset-icon.xag,
    .asset-icon.xpd,
    .asset-icon.xpt { background: linear-gradient(135deg, #b8c2cc, #f4fbff); }
    .asset-icon.copper { background: linear-gradient(135deg, #b87333, #ffbf75); }
    .asset-icon.cl,
    .asset-icon.bz,
    .asset-icon.natgas,
    .asset-icon.xom,
    .asset-icon.oxy { background: linear-gradient(135deg, #1fd06f, #f4d35e); }
    .asset-icon.spy,
    .asset-icon.qqq,
    .asset-icon.tqqq,
    .asset-icon.sqqq,
    .asset-icon.soxl,
    .asset-icon.soxs { background: linear-gradient(135deg, #00e5ff, #b266ff); color: #fff; }
    .asset-icon.nvda,
    .asset-icon.tsla,
    .asset-icon.aapl,
    .asset-icon.googl,
    .asset-icon.amzn,
    .asset-icon.meta,
    .asset-icon.msft,
    .asset-icon.amd { background: linear-gradient(135deg, #16213e, #00e5ff); color: #fff; }
    input[type="text"],
    input[type="number"],
    input[type="password"],
    select {
      width: 100%;
      min-height: 44px;
      border: 1px solid var(--line);
      border-radius: 3px;
      padding: 0 14px;
      color: var(--ink);
      background: rgba(4, 10, 22, 0.7);
      font: inherit;
      font-family: "Rajdhani", sans-serif;
      font-size: 15px;
      font-weight: 500;
      letter-spacing: 0.02em;
      transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }
    input::placeholder { color: rgba(138, 169, 192, 0.45); }
    input:focus, select:focus {
      outline: none;
      border-color: var(--accent);
      background: rgba(0, 229, 255, 0.035);
      box-shadow: 0 0 0 1px rgba(0, 229, 255, 0.72), 0 0 8px rgba(0, 229, 255, 0.18);
    }
    select {
      appearance: none;
      background-image:
        linear-gradient(45deg, transparent 50%, var(--accent) 50%),
        linear-gradient(135deg, var(--accent) 50%, transparent 50%);
      background-position: calc(100% - 18px) 50%, calc(100% - 13px) 50%;
      background-size: 5px 5px, 5px 5px;
      background-repeat: no-repeat;
      padding-right: 32px;
      cursor: pointer;
    }
    select option {
      background: #0a1424;
      color: var(--ink);
    }
    fieldset {
      margin: 0;
      padding: 0;
      border: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .checks {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .check {
      position: relative;
      display: inline-flex;
      align-items: center;
      min-height: 40px;
      border: 1px solid rgba(0,229,255,0.30);
      border-radius: 3px;
      background: rgba(4, 10, 22, 0.55);
      cursor: pointer;
      overflow: hidden;
      transition: all 0.15s ease;
    }
    .check input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .check span {
      padding: 10px 16px;
      font-family: "Orbitron", monospace;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.18em;
      line-height: 1;
      color: var(--muted-2);
      text-transform: uppercase;
      white-space: nowrap;
    }
    .check:hover {
      border-color: var(--accent);
      box-shadow: 0 0 12px rgba(0,229,255,0.25);
    }
    .check input:checked + span {
      color: var(--accent-strong);
      text-shadow: 0 0 10px rgba(0,229,255,0.6);
    }
    .check input:checked ~ * { }
    .check:has(input:checked) {
      background: rgba(0, 229, 255, 0.10);
      border-color: var(--accent);
      box-shadow: inset 0 0 14px rgba(0,229,255,0.18), 0 0 14px rgba(0,229,255,0.30);
    }
    .toggle {
      grid-column: span 12;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--ink);
      font-family: "Rajdhani", sans-serif;
      font-size: 14px;
      font-weight: 600;
      text-transform: none;
      letter-spacing: 0.02em;
      width: fit-content;
      cursor: pointer;
    }
    .toggle input {
      appearance: none;
      width: 18px; height: 18px;
      border: 1.5px solid var(--accent);
      border-radius: 2px;
      background: rgba(4,10,22,0.6);
      cursor: pointer;
      position: relative;
      transition: all 0.15s ease;
    }
    .toggle input:checked {
      background: var(--accent);
      box-shadow: 0 0 10px var(--accent);
    }
    .toggle input:checked::after {
      content: "";
      position: absolute;
      left: 5px; top: 1px;
      width: 5px; height: 10px;
      border: solid #04101a;
      border-width: 0 2px 2px 0;
      transform: rotate(45deg);
    }
    .toggle small {
      display: block;
      margin-top: 2px;
      font-family: "Share Tech Mono", monospace;
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 0;
      font-weight: 400;
    }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1.2fr 1fr;
      gap: 14px;
      margin-top: 24px;
      padding-top: 20px;
      border-top: 1px solid rgba(0,229,255,0.12);
      position: relative;
    }
    .actions::before {
      content: ""; position: absolute; left: 0; right: 0; top: -1px;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--accent-2) 30%, var(--accent) 70%, transparent);
      opacity: 0.5;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      min-height: 54px;
      padding: 0 18px;
      border: 1px solid var(--accent);
      background: rgba(0, 229, 255, 0.06);
      color: var(--accent-strong);
      font: inherit;
      font-family: "Orbitron", sans-serif;
      font-size: 12.5px;
      font-weight: 700;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      cursor: pointer;
      position: relative;
      clip-path: polygon(
        0 8px, 8px 0,
        calc(100% - 8px) 0, 100% 8px,
        100% calc(100% - 8px), calc(100% - 8px) 100%,
        8px 100%, 0 calc(100% - 8px)
      );
      transition: background 0.15s ease, color 0.15s ease, box-shadow 0.2s ease;
    }
    .btn:hover {
      background: rgba(0, 229, 255, 0.16);
      color: #fff;
      box-shadow: 0 0 20px rgba(0,229,255,0.45), inset 0 0 20px rgba(0,229,255,0.18);
    }
    .btn svg { width: 18px; height: 18px; filter: drop-shadow(0 0 4px currentColor); }
    .btn-primary {
      border: 1px solid var(--accent-2);
      color: #fff;
      background: linear-gradient(135deg, rgba(0,229,255,0.18), rgba(255,61,240,0.22));
      box-shadow: 0 0 22px rgba(0,229,255,0.30), 0 0 22px rgba(255,61,240,0.25);
    }
    .btn-primary:hover {
      background: linear-gradient(135deg, rgba(0,229,255,0.32), rgba(255,61,240,0.36));
      box-shadow: 0 0 30px rgba(0,229,255,0.55), 0 0 30px rgba(255,61,240,0.45);
    }
    .btn:disabled {
      cursor: wait;
      opacity: 0.5;
      box-shadow: none;
    }
    /* ---------- COMBO (símbolo) ---------- */
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
      background: #061224;
      border: 1px solid var(--accent);
      border-radius: 3px;
      box-shadow: 0 0 24px rgba(0,229,255,0.35), 0 12px 30px rgba(0,0,0,0.6);
    }
    .combo-list li {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 8px 14px;
      cursor: pointer;
      font-family: "Share Tech Mono", monospace;
      font-size: 13px;
      color: var(--muted-2);
      letter-spacing: 0.05em;
    }
    .combo-list li .asset-icon { position: static; }
    .combo-list li:hover,
    .combo-list li.active {
      background: rgba(0,229,255,0.12);
      color: var(--accent-strong);
    }
    .combo-list li.empty {
      color: var(--muted);
      cursor: default;
      font-style: italic;
    }
    .combo-list li.empty:hover { background: transparent; color: var(--muted); }
    .combo-list::-webkit-scrollbar { width: 8px; }
    .combo-list::-webkit-scrollbar-track { background: #02060e; }
    .combo-list::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.3); border-radius: 4px; }
    /* ---------- TF rows ---------- */
    .tf-rows { display: flex; flex-wrap: wrap; gap: 10px; }
    .tf-row {
      display: inline-flex;
      align-items: center;
      gap: 0;
      border: 1px solid rgba(0,229,255,0.28);
      border-radius: 3px;
      background: rgba(4, 10, 22, 0.55);
      overflow: hidden;
      transition: all 0.15s ease;
    }
    .tf-row .tf-chk {
      position: relative;
      display: inline-flex;
      align-items: center;
      min-height: 40px;
      cursor: pointer;
    }
    .tf-row .tf-chk input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .tf-row .tf-chk span {
      padding: 10px 14px;
      font-family: "Orbitron", monospace;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.12em;
      line-height: 1;
      color: var(--muted-2);
      text-transform: uppercase;
      white-space: nowrap;
    }
    .tf-row .tf-chk input:checked + span {
      color: var(--accent-strong);
      text-shadow: 0 0 10px rgba(0,229,255,0.6);
    }
    .tf-row:has(.tf-chk input:checked) {
      border-color: var(--accent);
      background: rgba(0,229,255,0.10);
      box-shadow: inset 0 0 12px rgba(0,229,255,0.18), 0 0 14px rgba(0,229,255,0.28);
    }
    .tf-row:hover {
      border-color: var(--accent);
      box-shadow: 0 0 10px rgba(0,229,255,0.18);
    }
    .tf-row .tf-sep {
      width: 1px;
      background: rgba(0,229,255,0.22);
      align-self: stretch;
    }
    .tf-row input[type="number"] {
      width: 64px;
      min-height: 40px;
      border: none;
      border-radius: 0;
      padding: 0 8px;
      font-family: "Share Tech Mono", monospace;
      font-size: 12px;
      color: var(--muted-2);
      background: transparent;
      letter-spacing: 0.04em;
    }
    .tf-row input[type="number"]:focus {
      outline: none;
      color: var(--accent-strong);
      background: rgba(0,229,255,0.06);
      box-shadow: none;
    }
    /* ---------- PROMPT PANEL ---------- */
    .prompt-panel {
      margin-top: 22px;
      padding: 18px;
      background: linear-gradient(180deg, rgba(8,16,30,0.85), rgba(4,10,22,0.95));
      border: 1px solid var(--panel-edge);
      box-shadow: 0 0 0 1px rgba(0,229,255,0.04) inset, 0 18px 50px rgba(0,0,0,0.5);
      clip-path: polygon(
        0 12px, 12px 0,
        calc(100% - 12px) 0, 100% 12px,
        100% calc(100% - 12px), calc(100% - 12px) 100%,
        12px 100%, 0 calc(100% - 12px)
      );
    }
    .prompt-panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .prompt-panel-header span {
      font-family: "Orbitron", monospace;
      font-size: 11px;
      font-weight: 700;
      color: var(--accent);
      letter-spacing: 0.28em;
      text-transform: uppercase;
      text-shadow: 0 0 8px rgba(0,229,255,0.5);
    }
    .prompt-textarea {
      width: 100%;
      height: 440px;
      font-family: "Share Tech Mono", "Fira Mono", monospace;
      font-size: 12.5px;
      line-height: 1.6;
      border: 1px solid rgba(0,229,255,0.18);
      border-radius: 3px;
      padding: 14px;
      resize: vertical;
      color: #b6e4ee;
      background: rgba(2, 6, 14, 0.85);
      box-sizing: border-box;
    }
    .prompt-textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 1px var(--accent), inset 0 0 20px rgba(0,229,255,0.08);
    }
    .copy-btn {
      min-height: 34px;
      padding: 0 16px;
      font-family: "Orbitron", monospace;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.18em;
      color: var(--accent);
      background: rgba(0,229,255,0.06);
      border: 1px solid var(--accent);
      border-radius: 3px;
      cursor: pointer;
      text-transform: uppercase;
      transition: all 0.15s ease;
    }
    .copy-btn:hover {
      background: rgba(0,229,255,0.18);
      box-shadow: 0 0 12px rgba(0,229,255,0.4);
      color: #fff;
    }
    /* ---------- AI LAUNCH STRIP ---------- */
    .ai-launch {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px dashed rgba(0,229,255,0.18);
      flex-wrap: wrap;
    }
    .ai-launch-label {
      font-family: "Share Tech Mono", monospace;
      font-size: 10.5px;
      font-weight: 600;
      color: var(--muted-2);
      letter-spacing: 0.22em;
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
      border: 1px solid rgba(0,229,255,0.25);
      border-radius: 3px;
      background: rgba(4,10,22,0.5);
      color: var(--ink);
      font-family: "Rajdhani", sans-serif;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.05em;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .ai-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 0 14px rgba(0,229,255,0.25);
    }
    .ai-btn svg { width: 18px; height: 18px; flex-shrink: 0; }
    .ai-btn.chatgpt:hover { border-color: #10a37f; box-shadow: 0 0 14px rgba(16,163,127,0.45); }
    .ai-btn.claude:hover  { border-color: #cc785c; box-shadow: 0 0 14px rgba(204,120,92,0.45); }
    .ai-btn.gemini:hover  { border-color: #4285f4; box-shadow: 0 0 14px rgba(66,133,244,0.45); }
    .ai-btn.deepseek:hover{ border-color: #4d6bfe; box-shadow: 0 0 14px rgba(77,107,254,0.45); }
    /* ---------- AI RESPONSE ---------- */
    .ai-response-panel {
      margin-top: 22px;
      padding: 18px;
      background: linear-gradient(180deg, rgba(8,16,30,0.95), rgba(4,10,22,0.98));
      border: 1px solid var(--accent-2);
      box-shadow: 0 0 0 1px rgba(255,61,240,0.10) inset, 0 0 28px rgba(255,61,240,0.18);
      clip-path: polygon(
        0 12px, 12px 0,
        calc(100% - 12px) 0, 100% 12px,
        100% calc(100% - 12px), calc(100% - 12px) 100%,
        12px 100%, 0 calc(100% - 12px)
      );
    }
    .ai-response-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid rgba(255,61,240,0.20);
    }
    .ai-response-header span:first-child {
      font-family: "Orbitron", monospace;
      font-size: 11px;
      font-weight: 700;
      color: var(--accent-2);
      text-transform: uppercase;
      letter-spacing: 0.30em;
      text-shadow: 0 0 8px rgba(255,61,240,0.6);
    }
    .ai-response-meta {
      font-family: "Share Tech Mono", monospace;
      font-size: 10.5px;
      color: var(--muted-2);
      letter-spacing: 0.05em;
    }
    .ai-response-body {
      color: #d4e8f0;
      font-family: "Share Tech Mono", "Fira Mono", monospace;
      font-size: 13px;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
    }
    /* ---------- TOAST ---------- */
    .toast {
      position: fixed;
      top: 22px;
      left: 50%;
      transform: translateX(-50%) translateY(-20px);
      z-index: 1000;
      min-width: 320px;
      max-width: 90vw;
      padding: 12px 22px;
      background: rgba(8, 16, 30, 0.95);
      color: var(--accent-strong);
      border: 1px solid var(--accent);
      box-shadow: 0 0 24px rgba(0,229,255,0.4), 0 14px 36px rgba(0,0,0,0.55);
      font-family: "Share Tech Mono", monospace;
      font-size: 13px;
      letter-spacing: 0.04em;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.22s ease, transform 0.22s ease;
      text-align: center;
      clip-path: polygon(0 8px, 8px 0, calc(100% - 8px) 0, 100% 8px, 100% calc(100% - 8px), calc(100% - 8px) 100%, 8px 100%, 0 calc(100% - 8px));
    }
    .toast.show {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
      pointer-events: auto;
    }
    .toast.error { border-color: var(--bad); color: var(--bad); box-shadow: 0 0 24px rgba(255,77,109,0.4), 0 14px 36px rgba(0,0,0,0.55); }
    .toast kbd {
      display: inline-block;
      padding: 2px 7px;
      margin: 0 2px;
      background: rgba(0,229,255,0.18);
      border: 1px solid rgba(0,229,255,0.4);
      border-radius: 2px;
      font-family: "Share Tech Mono", monospace;
      font-size: 12px;
      color: #fff;
    }
    /* ---------- SPINNER ---------- */
    .spinner {
      display: inline-block;
      width: 12px; height: 12px;
      border: 2px solid rgba(0,229,255,0.25);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    /* ---------- RESPONSIVE ---------- */
    @media (max-width: 980px) {
      .hud-shell { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .ai-core-label { display: none; }
    }
    @media (max-width: 760px) {
      .hud-shell { width: calc(100vw - 20px); padding-top: 14px; }
      .main-panel { padding: 16px; }
      .hud-header { flex-direction: column; align-items: flex-start; gap: 10px; }
      .status-pills { align-items: flex-start; flex-direction: row; }
      .status { text-align: left; }
      .field, .field.mid { grid-column: span 12; }
      .actions { grid-template-columns: 1fr; }
    }
    /* ---------- TUTORIAL MODAL ---------- */
    .btn-help {
      padding: 6px 14px;
      font-size: 10px;
      font-family: "Orbitron", monospace;
      font-weight: 700;
      letter-spacing: 0.14em;
      color: var(--accent);
      background: rgba(0,229,255,0.07);
      border: 1px solid rgba(0,229,255,0.35);
      cursor: pointer;
      transition: all 0.15s ease;
      white-space: nowrap;
    }
    .btn-help:hover { background: rgba(0,229,255,0.16); border-color: var(--accent); color: var(--accent-strong); }
    .tut-modal {
      position: fixed; inset: 0; z-index: 9999;
      display: flex; align-items: center; justify-content: center;
    }
    .tut-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.88); cursor: pointer; }
    .tut-panel {
      position: relative; z-index: 1;
      width: min(860px, 94vw); max-height: 88vh;
      display: flex; flex-direction: column;
      background: linear-gradient(160deg, #0a1627 0%, #050e1c 100%);
      border: 1px solid var(--accent);
      box-shadow: 0 0 40px rgba(0,229,255,0.14), inset 0 0 60px rgba(0,0,0,0.4);
    }
    .tut-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 16px 22px 12px; border-bottom: 1px solid var(--line); flex-shrink: 0;
    }
    .tut-title {
      font-family: "Orbitron", monospace; font-size: 12px; font-weight: 700;
      letter-spacing: 0.2em; color: var(--accent-strong); text-transform: uppercase;
    }
    .tut-close {
      background: none; border: 1px solid rgba(0,229,255,0.3); color: var(--muted-2);
      font-size: 14px; width: 30px; height: 30px; cursor: pointer;
      display: flex; align-items: center; justify-content: center; transition: all 0.15s;
    }
    .tut-close:hover { border-color: var(--bad); color: var(--bad); }
    .tut-tabs {
      display: flex; border-bottom: 1px solid var(--line); flex-shrink: 0; overflow-x: auto;
    }
    .tut-tab {
      padding: 11px 16px;
      font-family: "Orbitron", monospace; font-size: 10px; font-weight: 700;
      letter-spacing: 0.13em; text-transform: uppercase;
      color: var(--muted); background: none; border: none;
      border-bottom: 2px solid transparent; cursor: pointer; white-space: nowrap; transition: all 0.15s;
    }
    .tut-tab:hover { color: var(--ink); }
    .tut-tab.active { color: var(--accent-strong); border-bottom-color: var(--accent); background: rgba(0,229,255,0.06); }
    .tut-body {
      padding: 24px 26px; overflow-y: auto; flex: 1;
      color: var(--ink); font-family: "Rajdhani", sans-serif; font-size: 15px; line-height: 1.65;
    }
    .tut-page { display: none; }
    .tut-page.active { display: block; }
    .tut-body h2 {
      font-family: "Orbitron", monospace; font-size: 12px; font-weight: 700;
      letter-spacing: 0.18em; color: var(--accent); text-transform: uppercase; margin: 0 0 14px;
    }
    .tut-body h3 {
      font-family: "Orbitron", monospace; font-size: 10px; font-weight: 700;
      letter-spacing: 0.14em; color: var(--accent-2); text-transform: uppercase; margin: 22px 0 8px;
    }
    .tut-steps { list-style: none; padding: 0; margin: 0 0 16px; display: flex; flex-direction: column; gap: 10px; }
    .tut-steps li { display: flex; gap: 14px; align-items: flex-start; }
    .tut-step-n {
      flex-shrink: 0; width: 26px; height: 26px;
      border: 1px solid var(--accent); border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-family: "Orbitron", monospace; font-size: 11px; font-weight: 700;
      color: var(--accent); background: rgba(0,229,255,0.08);
    }
    .tut-highlight {
      background: rgba(0,229,255,0.05); border-left: 3px solid var(--accent);
      padding: 12px 16px; margin: 12px 0; font-size: 14px; color: var(--muted-2);
    }
    .tut-highlight strong { color: var(--ink); }
    .tut-warn {
      background: rgba(255,200,87,0.06); border-left: 3px solid var(--warn);
      padding: 12px 16px; margin: 12px 0; font-size: 14px; color: var(--warn);
    }
    .tut-table { width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 14px; }
    .tut-table th {
      text-align: left; padding: 8px 12px;
      font-family: "Orbitron", monospace; font-size: 9px; letter-spacing: 0.15em;
      color: var(--muted); border-bottom: 1px solid var(--line); text-transform: uppercase;
    }
    .tut-table td { padding: 9px 12px; border-bottom: 1px solid rgba(0,229,255,0.07); color: var(--ink); vertical-align: top; }
    .tut-table td:first-child { font-family: "Share Tech Mono", monospace; font-size: 13px; color: var(--accent-strong); white-space: nowrap; }
    .tut-table tr:hover td { background: rgba(0,229,255,0.03); }
    .tut-code {
      background: rgba(0,0,0,0.4); border: 1px solid rgba(0,229,255,0.2);
      padding: 12px 16px; font-family: "Share Tech Mono", monospace; font-size: 13px;
      color: var(--accent-strong); margin: 10px 0; word-break: break-all;
    }
    .tut-faq { display: flex; flex-direction: column; gap: 18px; }
    .tut-faq-q { font-weight: 700; color: var(--accent-strong); margin-bottom: 4px; font-size: 15px; }
    .tut-faq-a { color: var(--muted-2); font-size: 14px; }
    .tut-tag {
      display: inline-block; background: rgba(0,229,255,0.09);
      border: 1px solid rgba(0,229,255,0.28); color: var(--accent);
      font-family: "Share Tech Mono", monospace; font-size: 12px; padding: 1px 8px; margin: 2px;
    }
    .tut-flow {
      display: flex; gap: 0; margin: 20px 0; flex-wrap: wrap;
    }
    .tut-flow-step {
      flex: 1; min-width: 150px;
      background: rgba(0,229,255,0.04); border: 1px solid rgba(0,229,255,0.18);
      padding: 18px 12px; text-align: center; position: relative;
    }
    .tut-flow-step:not(:last-child)::after {
      content: "→"; position: absolute; right: -11px; top: 50%; transform: translateY(-50%);
      color: var(--accent); font-size: 18px; z-index: 2;
    }
    .tut-flow-n { font-family: "Orbitron", monospace; font-size: 26px; font-weight: 900; color: var(--accent); display: block; margin-bottom: 6px; }
    .tut-flow-label { font-size: 13px; color: var(--muted-2); }
    @media (max-width: 600px) {
      .tut-panel { width: 98vw; max-height: 95vh; }
      .tut-body { padding: 16px 14px; }
      .tut-tab { padding: 10px 10px; font-size: 9px; }
      .tut-flow { flex-direction: column; }
      .tut-flow-step:not(:last-child)::after { display: none; }
    }
  </style>
</head>
<body>
  <div id="toast" class="toast" role="status" aria-live="polite"></div>

  <div class="hud-shell">
    <aside class="sidebar">
      <div class="ai-core-label">AI CORE v2.4</div>

      <div class="avatar-wrap" aria-hidden="true">
        <span class="corner tl"></span><span class="corner tr"></span>
        <span class="corner bl"></span><span class="corner br"></span>
        <img class="avatar-gif" src="/static/ai-core.gif" alt="">
      </div>

      <div class="nucleo">
        <div class="nucleo-label"><span>NÚCLEO ACTIVO</span><span class="dot"></span></div>
        <svg class="mini-chart" viewBox="0 0 200 40" preserveAspectRatio="none">
          <polyline class="line2" points="0,28 12,24 24,30 36,18 48,22 60,12 72,20 84,10 96,18 108,8 120,16 132,6 144,14 156,4 168,12 180,8 192,16 200,10"/>
          <polyline class="line" points="0,32 12,30 24,34 36,26 48,30 60,22 72,28 84,18 96,24 108,16 120,22 132,14 144,20 156,12 168,18 180,14 192,22 200,16"/>
        </svg>
      </div>

      <div id="music-console" class="music-console">
        <button id="music-toggle" class="music-toggle" type="button" aria-pressed="false">
          <span class="music-led" aria-hidden="true"></span>
          <span>MÚSICA 24/7</span>
          <strong id="music-state">OFF</strong>
        </button>
        <audio id="music-audio" src="/static/music.mp3" loop preload="auto"></audio>
        <div class="music-meter" aria-hidden="true">
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
        </div>
      </div>

      <div class="stat-grid">
        <div class="stat-row"><label>INTELIGENCIA</label><span class="val ok">100%</span></div>
        <div class="stat-row"><label>ANÁLISIS</label><span id="stat-analysis" class="val ok">ACTIVO</span></div>
        <div class="stat-row"><label>MODELO</label><span id="stat-model" class="val">CLAUDE 4.6</span></div>
        <div class="stat-row"><label>LATENCIA</label><span id="stat-latency" class="val">— ms</span></div>
      </div>

      <div class="quote">"EL MERCADO NO DUERME.<br>NOSOTROS TAMPOCO."</div>
    </aside>

    <section class="main-panel">
      <header class="hud-header">
        <div class="title-block">
          <div class="title-row">
            <h1>AI <span class="a2">TRADER</span></h1>
            <span class="title-dots"><i></i><i></i><i></i><i></i></span>
          </div>
          <p class="subtitle">Terminal de Inteligencia de Mercados</p>
          <div id="status" class="status"></div>
        </div>
        <div class="status-pills">
          <span class="pill"><i class="dot"></i>Sistema Online</span>
          <span class="pill"><i class="dot"></i>Data Feed: En Vivo</span>
          <button id="help-btn" class="btn-help" type="button" title="Abrir tutorial de uso">? AYUDA</button>
        </div>
      </header>

    <form id="runner-form">
      <div class="grid">
        <div class="field combo-field">
          <label for="symbol-search">Símbolo</label>
          <div class="combo input-shell has-icon">
            <span id="selected-symbol-icon" class="asset-icon btc">B</span>
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
            <option value="bitget" selected>Bitget</option>
            <option value="binance">Binance</option>
            <option value="bybit">Bybit</option>
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
          <div class="input-shell has-icon">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="8" cy="14" r="4"/>
              <path d="M12 14h10m-3 0v4m-3-4v3"/>
            </svg>
            <input type="password" id="api_key" name="api_key" placeholder="sk-ant-..." autocomplete="off">
          </div>
        </div>

        <label class="toggle">
          <input type="checkbox" name="context" checked>
          <span>
            <strong>Contexto de mercado</strong>
            <small>Incluye datos y análisis del mercado actual en el prompt.</small>
          </span>
        </label>
        <label class="toggle">
          <input type="checkbox" name="ai_summary">
          <span>
            <strong>Resumen IA</strong>
            <small>Añade un resumen breve generado por IA con emojis al final del prompt.</small>
          </span>
        </label>
      </div>

      <div class="actions">
        <button id="btn-show" type="button" class="btn">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
          Mostrar Prompt
        </button>
        <button id="btn-claude" type="button" class="btn btn-primary">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3a4 4 0 00-4 4v1a4 4 0 00-2 7v1a4 4 0 004 4h4a4 4 0 004-4v-1a4 4 0 00-2-7V7a4 4 0 00-4-4z"/><path d="M9 11h6M9 14h6"/></svg>
          Analizar con Claude
        </button>
        <button id="submit" type="submit" class="btn">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 4v12m0 0l-5-5m5 5l5-5M4 20h16"/></svg>
          Descargar
        </button>
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
    </section>
  </div>

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
    const statModel = document.querySelector("#stat-model");
    const statLatency = document.querySelector("#stat-latency");
    const statAnalysis = document.querySelector("#stat-analysis");
    const musicConsole = document.querySelector("#music-console");
    const musicToggle = document.querySelector("#music-toggle");
    const musicState = document.querySelector("#music-state");
    const musicAudio = document.querySelector("#music-audio");

    let lastPrompt = null;
    let lastMode = null;
    const MUSIC_STORAGE_KEY = "ai_trader_music_enabled";

    function setMusicUi(isOn) {
      if (!musicConsole || !musicToggle || !musicState) return;
      musicConsole.classList.toggle("is-on", isOn);
      musicToggle.classList.toggle("is-on", isOn);
      musicToggle.setAttribute("aria-pressed", isOn ? "true" : "false");
      musicState.textContent = isOn ? "ON" : "OFF";
    }

    async function startMusic() {
      if (!musicAudio) return;
      musicAudio.loop = true;
      musicAudio.volume = 0.42;
      try {
        await musicAudio.play();
        setMusicUi(true);
        try { localStorage.setItem(MUSIC_STORAGE_KEY, "1"); } catch (_) {}
      } catch (_) {
        setMusicUi(false);
        showToast("Pon archivo <b>music.mp3</b> en <b>AI_trader/web/static/</b> y pulsa Música 24/7.", "error", 7000);
      }
    }

    function stopMusic() {
      if (musicAudio) {
        musicAudio.pause();
      }
      setMusicUi(false);
      try { localStorage.setItem(MUSIC_STORAGE_KEY, "0"); } catch (_) {}
    }

    if (musicToggle) {
      musicToggle.addEventListener("click", async () => {
        const isOn = musicToggle.getAttribute("aria-pressed") === "true";
        if (isOn) {
          stopMusic();
          return;
        }
        await startMusic();
      });
      try {
        if (localStorage.getItem(MUSIC_STORAGE_KEY) === "1") {
          musicState.textContent = "READY";
        }
      } catch (_) {}
    }

    function updateModelStat() {
      const sel = form.model;
      if (!sel || !statModel) return;
      const opt = sel.options[sel.selectedIndex];
      const label = (opt.text.split("—")[0] || "").trim();
      statModel.textContent = label.toUpperCase();
    }
    if (form.model) form.model.addEventListener("change", updateModelStat);
    updateModelStat();

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
      "SIREN/USDT","STO/USDT","TAO/USDT","TIA/USDT","TRUMP/USDT","XLM/USDT","XMR/USDT","ZEC/USDT",
      "XAU/USDT","XAG/USDT","XPD/USDT","XPT/USDT","COPPER/USDT","CL/USDT","BZ/USDT","NATGAS/USDT",
      "SPY/USDT","QQQ/USDT","TQQQ/USDT","SQQQ/USDT","SOXL/USDT","SOXS/USDT",
      "NVDA/USDT","TSLA/USDT","AAPL/USDT","GOOGL/USDT","AMZN/USDT","META/USDT","MSFT/USDT",
      "AMD/USDT","AVGO/USDT","TSM/USDT","ASML/USDT","ARM/USDT","INTC/USDT","MRVL/USDT",
      "PLTR/USDT","MSTR/USDT","COIN/USDT","HOOD/USDT","GME/USDT","RDDT/USDT","NFLX/USDT",
      "ORCL/USDT","GE/USDT","BA/USDT","WMT/USDT","COST/USDT","MCD/USDT","UNH/USDT","LLY/USDT",
      "BABA/USDT","JD/USDT","FUTU/USDT","RKLB/USDT","OKLO/USDT","IONQ/USDT","XOM/USDT","OXY/USDT"
    ].sort();

    const symbolInput  = document.querySelector("#symbol-search");
    const symbolHidden = document.querySelector("#symbol");
    const symbolList   = document.querySelector("#symbol-list");
    const selectedSymbolIcon = document.querySelector("#selected-symbol-icon");
    let activeIdx = -1;
    let filtered = SYMBOLS.slice();

    const ASSET_ICON_LABELS = {
      BTC: "B", ETH: "E", SOL: "S", BNB: "B", XRP: "X", DOGE: "D", ADA: "A", AVAX: "A",
      LINK: "L", DOT: "D", MATIC: "M", POL: "P", UNI: "U", ATOM: "A", LTC: "L", BCH: "B",
      NEAR: "N", OP: "O", ARB: "A", INJ: "I", TRX: "T", APT: "A", SUI: "S", TON: "T",
      PEPE: "P", SHIB: "S", ETC: "E", FIL: "F", AAVE: "A", ONDO: "O", RENDER: "R",
      RUNE: "R", TAO: "T", TIA: "T", XLM: "X", XMR: "X", ZEC: "Z",
      XAU: "Au", XAG: "Ag", XPD: "Pd", XPT: "Pt", COPPER: "Cu", CL: "Oil", BZ: "Br", NATGAS: "Gas",
      SPY: "SP", QQQ: "Q", NVDA: "NV", TSLA: "TS", AAPL: "AP", GOOGL: "GO", AMZN: "AM", META: "ME",
      MSFT: "MS", AMD: "AM", AVGO: "AV", TSM: "TS", ASML: "AS", ARM: "AR", INTC: "IN", PLTR: "PL",
      MSTR: "M", COIN: "C", HOOD: "H", NFLX: "NF", ORCL: "OR", GE: "GE", BA: "BA", XOM: "XO", OXY: "OX"
    };

    const BITGET_RWA_SYMBOLS = new Set([
      "PAXG/USDT","XAUT/USDT","XAU/USDT","XAG/USDT","XPD/USDT","XPT/USDT","COPPER/USDT",
      "CL/USDT","BZ/USDT","NATGAS/USDT","SPY/USDT","QQQ/USDT","TQQQ/USDT","SQQQ/USDT",
      "SOXL/USDT","SOXS/USDT","NVDA/USDT","TSLA/USDT","AAPL/USDT","GOOGL/USDT","AMZN/USDT",
      "META/USDT","MSFT/USDT","AMD/USDT","AVGO/USDT","TSM/USDT","ASML/USDT","ARM/USDT",
      "INTC/USDT","MRVL/USDT","PLTR/USDT","MSTR/USDT","COIN/USDT","HOOD/USDT","GME/USDT",
      "RDDT/USDT","NFLX/USDT","ORCL/USDT","GE/USDT","BA/USDT","WMT/USDT","COST/USDT",
      "MCD/USDT","UNH/USDT","LLY/USDT","BABA/USDT","JD/USDT","FUTU/USDT","RKLB/USDT",
      "OKLO/USDT","IONQ/USDT","XOM/USDT","OXY/USDT"
    ]);

    function assetBase(sym) {
      return (sym.split("/")[0] || sym).trim().toUpperCase();
    }

    function assetIconClass(sym) {
      return assetBase(sym).toLowerCase().replace(/[^a-z0-9_-]/g, "");
    }

    function assetIconText(sym) {
      const base = assetBase(sym);
      return ASSET_ICON_LABELS[base] || base.slice(0, 2);
    }

    function makeAssetIcon(sym) {
      const icon = document.createElement("span");
      icon.className = `asset-icon ${assetIconClass(sym)}`;
      icon.textContent = assetIconText(sym);
      icon.setAttribute("aria-hidden", "true");
      return icon;
    }

    function normalizeSymbolInput(value) {
      const raw = (value || "").trim().toUpperCase();
      if (!raw) return raw;
      if (raw.includes("/")) return raw;
      if (raw.endsWith("USDT")) return `${raw.slice(0, -4)}/USDT`;
      return raw;
    }

    function syncMarketSource(sym) {
      if (!BITGET_RWA_SYMBOLS.has(sym)) return;
      form.source.value = "ccxt";
      form.exchange.value = "bitget";
    }

    function updateSelectedSymbolIcon(sym) {
      if (!selectedSymbolIcon) return;
      selectedSymbolIcon.className = `asset-icon ${assetIconClass(sym)}`;
      selectedSymbolIcon.textContent = assetIconText(sym);
    }

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
        li.dataset.value = sym;
        li.appendChild(makeAssetIcon(sym));
        li.appendChild(document.createTextNode(sym));
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
      updateSelectedSymbolIcon(sym);
      syncMarketSource(sym);
      symbolList.hidden = true;
    }

    symbolInput.addEventListener("focus", () => {
      filterSymbols(symbolInput.value);
      symbolList.hidden = false;
    });
    symbolInput.addEventListener("input", () => {
      filterSymbols(symbolInput.value);
      symbolList.hidden = false;
      symbolHidden.value = normalizeSymbolInput(symbolInput.value);
      updateSelectedSymbolIcon(symbolHidden.value);
      syncMarketSource(symbolHidden.value);
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
        symbol: normalizeSymbolInput(form.symbol.value),
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
      if (statAnalysis) { statAnalysis.textContent = "ANALIZANDO"; statAnalysis.classList.remove("ok"); }

      const t0 = performance.now();
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
        const dt = Math.round(performance.now() - t0);
        if (statLatency) statLatency.textContent = dt + " ms";
        aiBody.textContent = data.response || "";
        aiMeta.textContent = fmtTokens(data.usage) + (data.model ? "  |  " + data.model : "");
        aiPanel.style.display = "block";
        aiPanel.scrollIntoView({ behavior: "smooth", block: "start" });
        statusEl.className = "status done";
        statusEl.textContent = "Análisis completo.";
      } catch (error) {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      } finally {
        setLoading(false);
        if (statAnalysis) { statAnalysis.textContent = "ACTIVO"; statAnalysis.classList.add("ok"); }
      }
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

    /* ===== TUTORIAL ===== */
    const tutModal = document.getElementById("tut-modal");
    const helpBtn  = document.getElementById("help-btn");
    const tutOverlay = document.getElementById("tut-overlay");
    const tutCloseBtn = document.getElementById("tut-close");
    const TUT_KEY = "tut_seen_v1";

    function openTutorial(tab) {
      tutModal.style.display = "flex";
      document.body.style.overflow = "hidden";
      if (tab) switchTutTab(tab);
    }
    function closeTutorial() {
      tutModal.style.display = "none";
      document.body.style.overflow = "";
      try { localStorage.setItem(TUT_KEY, "1"); } catch(_) {}
    }
    function switchTutTab(name) {
      document.querySelectorAll(".tut-tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
      document.querySelectorAll(".tut-page").forEach(p => p.classList.toggle("active", p.id === "tut-" + name));
    }

    helpBtn.addEventListener("click", () => openTutorial("inicio"));
    tutOverlay.addEventListener("click", closeTutorial);
    tutCloseBtn.addEventListener("click", closeTutorial);
    document.addEventListener("keydown", e => { if (e.key === "Escape" && tutModal.style.display === "flex") closeTutorial(); });
    document.querySelectorAll(".tut-tab").forEach(t => t.addEventListener("click", () => switchTutTab(t.dataset.tab)));

    try {
      if (!localStorage.getItem(TUT_KEY)) openTutorial("inicio");
    } catch(_) {}
  </script>

  <!-- TUTORIAL MODAL -->
  <div id="tut-modal" class="tut-modal" style="display:none" role="dialog" aria-modal="true" aria-label="Tutorial de uso">
    <div id="tut-overlay" class="tut-overlay"></div>
    <div class="tut-panel">
      <div class="tut-header">
        <span class="tut-title">// Manual de Usuario — AI Trader</span>
        <button id="tut-close" class="tut-close" type="button" aria-label="Cerrar">✕</button>
      </div>
      <div class="tut-tabs" role="tablist">
        <button class="tut-tab active" data-tab="inicio" role="tab">Inicio</button>
        <button class="tut-tab" data-tab="campos" role="tab">Campos</button>
        <button class="tut-tab" data-tab="indicadores" role="tab">Indicadores</button>
        <button class="tut-tab" data-tab="apikey" role="tab">API Key</button>
        <button class="tut-tab" data-tab="faq" role="tab">FAQ</button>
      </div>

      <div class="tut-body">

        <!-- TAB: INICIO -->
        <div id="tut-inicio" class="tut-page active">
          <h2>¿Qué es AI Trader?</h2>
          <p>AI Trader es una terminal de análisis de mercados. Descarga datos de precios en tiempo real, calcula indicadores técnicos avanzados y envía un análisis completo a Claude AI, que responde con zonas clave, setups de entrada y niveles de riesgo.</p>
          <div class="tut-highlight">
            <strong>AI Trader analiza — no opera.</strong> No ejecuta órdenes ni mueve dinero. Es una herramienta de análisis para ayudarte a tomar decisiones.
          </div>
          <h3>Flujo de 3 pasos</h3>
          <div class="tut-flow">
            <div class="tut-flow-step"><span class="tut-flow-n">1</span><span class="tut-flow-label">Escribe el símbolo que quieres analizar</span></div>
            <div class="tut-flow-step"><span class="tut-flow-n">2</span><span class="tut-flow-label">Selecciona timeframes e indicadores</span></div>
            <div class="tut-flow-step"><span class="tut-flow-n">3</span><span class="tut-flow-label">Haz clic en "Analizar con Claude"</span></div>
          </div>
          <h3>Qué necesitas</h3>
          <ul class="tut-steps">
            <li><div class="tut-step-n">✓</div><div>Conexión a internet (para descargar datos del mercado)</div></li>
            <li><div class="tut-step-n">✓</div><div>Una clave API de Anthropic para que Claude AI responda — ver pestaña <strong>API Key</strong></div></li>
          </ul>
          <h3>Activos soportados</h3>
          <p>Puedes analizar cualquiera de estos tipos de activos:</p>
          <p>
            <span class="tut-tag">BTC/USDT</span><span class="tut-tag">ETH/USDT</span><span class="tut-tag">SOL/USDT</span>
            <span style="color:var(--muted-2);font-size:13px;margin-left:6px;">criptomonedas</span>
          </p>
          <p>
            <span class="tut-tag">AAPL</span><span class="tut-tag">TSLA</span><span class="tut-tag">MSFT</span>
            <span style="color:var(--muted-2);font-size:13px;margin-left:6px;">acciones</span>
          </p>
          <p>
            <span class="tut-tag">EURUSD=X</span><span class="tut-tag">GBPUSD=X</span>
            <span style="color:var(--muted-2);font-size:13px;margin-left:6px;">forex</span>
          </p>
          <p>
            <span class="tut-tag">^SPX</span><span class="tut-tag">^FTSE</span>
            <span style="color:var(--muted-2);font-size:13px;margin-left:6px;">índices</span>
          </p>
        </div>

        <!-- TAB: CAMPOS -->
        <div id="tut-campos" class="tut-page">
          <h2>Guía de campos</h2>
          <table class="tut-table">
            <thead><tr><th>Campo</th><th>Qué hace</th><th>Recomendación</th></tr></thead>
            <tbody>
              <tr>
                <td>Símbolo</td>
                <td>El activo que vas a analizar. Escribe las primeras letras y elige de la lista.</td>
                <td>BTC/USDT para Bitcoin, AAPL para Apple</td>
              </tr>
              <tr>
                <td>Modo</td>
                <td><strong>mindset</strong>: análisis de contexto general del mercado.<br><strong>signal</strong>: búsqueda de señales de entrada concretas.</td>
                <td>Empieza con <strong>mindset</strong></td>
              </tr>
              <tr>
                <td>Fuente</td>
                <td>De dónde se descargan los datos de precios.</td>
                <td>Dejar en <strong>auto</strong> — detecta automáticamente</td>
              </tr>
              <tr>
                <td>Exchange</td>
                <td>Solo relevante para criptomonedas. Elige el exchange del que quieres los datos.</td>
                <td><strong>Bitget</strong> o Binance para la mayoría de pares</td>
              </tr>
              <tr>
                <td>Timeframes</td>
                <td>Los marcos temporales que se analizan. Cada vela representa ese período de tiempo (ej: 1h = 1 hora por vela).</td>
                <td>Activar <strong>1h, 4h, 1d</strong> para un análisis completo</td>
              </tr>
              <tr>
                <td>Velas por TF</td>
                <td>Cuántas velas históricas se descargan por timeframe.</td>
                <td>200 es suficiente para la mayoría de casos</td>
              </tr>
              <tr>
                <td>Indicadores</td>
                <td>Los análisis técnicos que se calculan. Ver pestaña <strong>Indicadores</strong> para detalles.</td>
                <td>Dejar todos activados</td>
              </tr>
              <tr>
                <td>Modelo Claude</td>
                <td>El modelo de IA que analiza el mercado.<br><strong>Sonnet</strong>: rápido y equilibrado.<br><strong>Opus</strong>: más profundo pero más lento.</td>
                <td><strong>Sonnet 4.6</strong> para uso diario</td>
              </tr>
              <tr>
                <td>API Key</td>
                <td>Tu clave personal de Anthropic. Sin ella, no se puede enviar a Claude AI.</td>
                <td>Ver pestaña <strong>API Key</strong> para obtenerla</td>
              </tr>
              <tr>
                <td>Contexto de mercado</td>
                <td>Incluye correlaciones con DXY, S&amp;P500, Bitcoin Dominance, etc.</td>
                <td>Activado — da más contexto al análisis</td>
              </tr>
            </tbody>
          </table>
          <h3>Botones de acción</h3>
          <table class="tut-table">
            <thead><tr><th>Botón</th><th>Qué hace</th></tr></thead>
            <tbody>
              <tr><td>Mostrar Prompt</td><td>Genera y muestra el texto que se le envía a la IA. Útil para revisar los datos antes de enviar.</td></tr>
              <tr><td>Analizar con Claude</td><td>Envía el análisis completo a Claude y muestra la respuesta con zonas clave, setups y niveles de riesgo.</td></tr>
              <tr><td>Descargar</td><td>Guarda el prompt generado como archivo de texto en tu computadora.</td></tr>
            </tbody>
          </table>
        </div>

        <!-- TAB: INDICADORES -->
        <div id="tut-indicadores" class="tut-page">
          <h2>Indicadores técnicos</h2>
          <p>Cada indicador mide un aspecto diferente del mercado. Juntos dan una visión completa antes de que Claude AI haga su análisis.</p>
          <table class="tut-table">
            <thead><tr><th>Indicador</th><th>Qué detecta</th><th>Útil para</th></tr></thead>
            <tbody>
              <tr>
                <td>WaveTrend</td>
                <td>Sobrecompra y sobreventa. Mide si el precio está demasiado alto o demasiado bajo en relación a su promedio reciente.</td>
                <td>Detectar reversiones y agotamiento del movimiento</td>
              </tr>
              <tr>
                <td>LuxAlgo AMO</td>
                <td>La fuerza y dirección del momentum. Muestra si el movimiento está acelerando o perdiendo fuerza. También detecta divergencias (cuando el precio sube pero la fuerza baja).</td>
                <td>Confirmar la fortaleza de una tendencia</td>
              </tr>
              <tr>
                <td>SMC Elite</td>
                <td>Zonas donde los grandes operadores (instituciones) han colocado órdenes. Detecta: bloques de órdenes (OB), vacíos de precio (FVG), rupturas de estructura (BOS) y cambios de carácter (CHoCH).</td>
                <td>Encontrar zonas de soporte/resistencia institucionales</td>
              </tr>
              <tr>
                <td>WAE</td>
                <td>Si hay una tendencia real o si el mercado está en rango (lateral). Combina dos indicadores para medir la calidad de la tendencia.</td>
                <td>Evitar operar en mercados sin dirección clara</td>
              </tr>
              <tr>
                <td>iTrend</td>
                <td>La dirección general del precio usando un filtro adaptativo. Similar a una media móvil pero más reactivo.</td>
                <td>Confirmar la dirección macro del mercado</td>
              </tr>
              <tr>
                <td>ICT Concepts</td>
                <td>Niveles de liquidez donde se acumulan stops de muchos traders. Los grandes operadores suelen "cazar" estos niveles antes de invertir.</td>
                <td>Anticipar movimientos de liquidación</td>
              </tr>
              <tr>
                <td>Trendlines</td>
                <td>Líneas de soporte y resistencia diagonales calculadas automáticamente a partir de máximos y mínimos relevantes.</td>
                <td>Identificar zonas de confluencia estructural</td>
              </tr>
            </tbody>
          </table>
          <div class="tut-highlight">
            <strong>Consejo:</strong> No necesitas entender cada indicador en detalle — Claude AI interpreta todos los datos juntos y te explica en lenguaje claro qué está pasando y qué zonas son relevantes.
          </div>
        </div>

        <!-- TAB: API KEY -->
        <div id="tut-apikey" class="tut-page">
          <h2>Cómo obtener tu API Key</h2>
          <p>Para que Claude AI analice el mercado necesitas una clave API de Anthropic. Es gratuita registrarse; los análisis se cobran por uso (muy bajo coste por análisis).</p>
          <h3>Pasos</h3>
          <ul class="tut-steps">
            <li><div class="tut-step-n">1</div><div>Abre tu navegador y ve a:<br><span class="tut-code">console.anthropic.com</span></div></li>
            <li><div class="tut-step-n">2</div><div>Crea una cuenta o inicia sesión con tu email.</div></li>
            <li><div class="tut-step-n">3</div><div>En el menú lateral, haz clic en <strong>"API Keys"</strong>.</div></li>
            <li><div class="tut-step-n">4</div><div>Haz clic en el botón <strong>"Create Key"</strong>.</div></li>
            <li><div class="tut-step-n">5</div><div>Copia la clave. Empieza con <span class="tut-code">sk-ant-</span></div></li>
            <li><div class="tut-step-n">6</div><div>Pégala en el campo <strong>"API Key"</strong> de esta página. Se guarda automáticamente en tu navegador.</div></li>
          </ul>
          <div class="tut-warn">
            ⚠ No compartas tu API Key con nadie. Es como una contraseña. Si crees que fue expuesta, elimínala en console.anthropic.com y crea una nueva.
          </div>
          <h3>Coste aproximado</h3>
          <table class="tut-table">
            <thead><tr><th>Modelo</th><th>Coste por análisis</th><th>Características</th></tr></thead>
            <tbody>
              <tr><td>Sonnet 4.6</td><td>~$0.01 – $0.05</td><td>Rápido, preciso, uso diario</td></tr>
              <tr><td>Opus 4.7</td><td>~$0.05 – $0.20</td><td>Análisis más profundo, más lento</td></tr>
              <tr><td>Haiku 4.5</td><td>~$0.001 – $0.01</td><td>El más económico, menos detalle</td></tr>
            </tbody>
          </table>
          <div class="tut-highlight">
            Anthropic ofrece créditos gratuitos al registrarse. Consulta el pricing actual en <strong>anthropic.com/pricing</strong>.
          </div>
        </div>

        <!-- TAB: FAQ -->
        <div id="tut-faq" class="tut-page">
          <h2>Preguntas frecuentes</h2>
          <div class="tut-faq">
            <div class="tut-faq-item">
              <div class="tut-faq-q">¿Qué formatos de símbolo acepta?</div>
              <div class="tut-faq-a">
                Criptomonedas: <span class="tut-tag">BTC/USDT</span><span class="tut-tag">ETH/USDT</span><span class="tut-tag">SOL/USDT</span><br>
                Acciones: <span class="tut-tag">AAPL</span><span class="tut-tag">TSLA</span><span class="tut-tag">NVDA</span><br>
                Forex: <span class="tut-tag">EURUSD=X</span><span class="tut-tag">GBPUSD=X</span><br>
                Índices: <span class="tut-tag">^SPX</span><span class="tut-tag">^FTSE</span><span class="tut-tag">^NDX</span>
              </div>
            </div>
            <div class="tut-faq-item">
              <div class="tut-faq-q">¿Cuánto tarda el análisis?</div>
              <div class="tut-faq-a">Entre 15 y 60 segundos. Depende del número de timeframes e indicadores activados, y del modelo Claude elegido. Sonnet es el más rápido.</div>
            </div>
            <div class="tut-faq-item">
              <div class="tut-faq-q">El análisis dice "Symbol not found" — ¿qué hago?</div>
              <div class="tut-faq-a">Verifica el formato exacto del símbolo. Para crypto usa la barra: <span class="tut-tag">BTC/USDT</span> no <span class="tut-tag">BTCUSDT</span>. Para acciones solo el ticker: <span class="tut-tag">AAPL</span> sin barras ni moneda.</div>
            </div>
            <div class="tut-faq-item">
              <div class="tut-faq-q">¿Es segura mi API Key?</div>
              <div class="tut-faq-a">La clave se guarda únicamente en tu navegador (localStorage). No se envía a ningún servidor de AI Trader — solo se usa directamente para comunicarse con la API de Anthropic.</div>
            </div>
            <div class="tut-faq-item">
              <div class="tut-faq-q">¿Puedo usar el análisis para operar en real?</div>
              <div class="tut-faq-a">El análisis es una herramienta de apoyo, no una señal automática. Siempre combínalo con tu propio criterio y gestión de riesgo. Trading conlleva riesgo de pérdida de capital.</div>
            </div>
            <div class="tut-faq-item">
              <div class="tut-faq-q">El botón "Analizar con Claude" no responde — ¿qué pasa?</div>
              <div class="tut-faq-a">Verifica que: (1) has introducido una API Key válida, (2) tienes créditos en tu cuenta Anthropic, (3) el símbolo es correcto. Si el error persiste, prueba con "Mostrar Prompt" primero para confirmar que los datos se descargan correctamente.</div>
            </div>
            <div class="tut-faq-item">
              <div class="tut-faq-q">¿Qué diferencia hay entre Mostrar Prompt y Analizar con Claude?</div>
              <div class="tut-faq-a">"Mostrar Prompt" solo genera el texto con los datos de mercado — no usa créditos de API. "Analizar con Claude" envía ese texto a la IA y consume créditos. Puedes usar "Mostrar Prompt" para revisar los datos sin coste.</div>
            </div>
          </div>
        </div>

      </div><!-- tut-body -->
    </div><!-- tut-panel -->
  </div><!-- tut-modal -->

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
