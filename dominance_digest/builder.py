"""Build the plain-text digest body for the scheduled dominance dispatch.

The output is a compact multi-section block (one section per symbol). Each
section lists one row per requested TF with WT / AMO / SMC last-bar values.
No buy/sell signals — only raw indicator outputs (per project policy).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

from pineforge_ai.usdt_dominance import tv_cache, usdt_indicators

log = logging.getLogger(__name__)

DOMINANCE_SYMBOLS: tuple[str, ...] = ("USDT.D", "BTC.D", "ETH.D", "SOL.D")

_SYMBOL_EMOJI = {
    "USDT.D": "💵",
    "BTC.D":  "🟧",
    "ETH.D":  "🟪",
    "SOL.D":  "🟦",
}


def _format_value(symbol: str, value: float) -> str:
    # All dominance series are percentages; use 3 decimals.
    return f"{value:.3f}%"


def _last_close(dfs: dict, tf: str) -> float | None:
    df = dfs.get(tf)
    if df is None or df.empty:
        return None
    try:
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def _change_pct(dfs: dict, tf: str, n_back: int = 1) -> float | None:
    df = dfs.get(tf)
    if df is None or len(df) <= n_back:
        return None
    try:
        cur = float(df["close"].iloc[-1])
        past = float(df["close"].iloc[-1 - n_back])
        if past == 0:
            return None
        return (cur / past - 1.0) * 100
    except Exception:
        return None


def _row(tf: str, wt: dict, lux: dict, smc: dict) -> str:
    """Compact single-line summary for a TF: WT · AMO · SMC."""
    wt_part = "WT —"
    if wt:
        osc = wt.get("osc", "—")
        trend = str(wt.get("trend", "—"))[:8]
        sig = str(wt.get("signal", "—"))[:12]
        wt_part = f"WT {osc:>5} {trend} ({sig})"

    lux_part = "AMO —"
    if lux:
        amo = lux.get("amo", "—")
        direction = str(lux.get("direction", "—"))[:7]
        div = lux.get("divergence", "—")
        div_tail = f" · {div}" if div and div != "—" else ""
        lux_part = f"AMO {amo:>7} {direction}{div_tail}"

    smc_part = "SMC —"
    if smc:
        event = str(smc.get("last_event", "—"))[:14]
        trend_map = {1: "↑", -1: "↓", 0: "→"}
        ms_t = trend_map.get(int(smc.get("ms_trend", 0)), "?")
        conf = smc.get("confluence", "—")
        smc_part = f"SMC {event} {ms_t} (Confl {conf})"

    return f"  {tf:<3} │ {wt_part} · {lux_part} · {smc_part}"


def _section_for_symbol(
    symbol: str,
    tfs: list[str],
    refresh: bool,
) -> str:
    if refresh:
        tv_cache.refresh_all(symbol, tfs=tfs)
    dfs = {tf: tv_cache.get_ohlcv(symbol, tf) for tf in tfs}
    summary = usdt_indicators.build_usdt_indicators_summary(dfs)

    emoji = _SYMBOL_EMOJI.get(symbol, "📊")
    # Use the most granular TF to report the "current" value.
    cur_tf = tfs[0]
    cur = _last_close(dfs, cur_tf)
    chg = _change_pct(dfs, cur_tf, n_back=1)
    cur_str = _format_value(symbol, cur) if cur is not None else "—"
    chg_str = f" ({chg:+.2f}%)" if chg is not None else ""

    lines = [f"{emoji} {symbol} · {cur_str}{chg_str}"]
    if not summary.get("available"):
        lines.append("  (sin datos)")
        return "\n".join(lines)
    for tf in tfs:
        lines.append(_row(tf, summary["wt"].get(tf, {}),
                              summary["lux"].get(tf, {}),
                              summary["smc"].get(tf, {})))
    return "\n".join(lines)


def build_digest_body(
    tfs: list[str],
    symbols: Iterable[str] = DOMINANCE_SYMBOLS,
    refresh: bool = True,
) -> str:
    """Compose the full digest text. `tfs` ordered most-granular → least."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks = [f"As of: {now}", f"TFs: {', '.join(tfs)}", ""]
    for sym in symbols:
        try:
            blocks.append(_section_for_symbol(sym, tfs, refresh=refresh))
        except Exception as e:
            log.warning("digest section failed sym=%s err=%s", sym, e)
            blocks.append(f"{sym}: error ({e})")
        blocks.append("")
    return "\n".join(blocks).rstrip()
