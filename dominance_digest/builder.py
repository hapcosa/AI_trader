"""Build the plain-text digest body for the scheduled dispatch.

Layout: one section per logical asset group. Each group can hold up to two
sub-rows (dominance and/or price feed). Sources are encoded as the FQ symbol
key (`EXCHANGE:SYMBOL`) so the cache routes to TradingView or ccxt Bitget
automatically.

No buy/sell signals are emitted — only WT / AMO / SMC last-bar values.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

from pineforge_ai.usdt_dominance import tv_cache, usdt_indicators

log = logging.getLogger(__name__)


# Logical layout: list of (group label, group emoji, list of feeds).
# A feed = (display_name, fq_symbol_key, value_decimals, value_suffix).
# Order shapes the message order.
ASSET_GROUPS: tuple[tuple[str, str, tuple[tuple[str, str, int, str], ...]], ...] = (
    ("USDT Dominance", "💵", (
        ("USDT.D", "CRYPTOCAP:USDT.D", 3, "%"),
    )),
    ("BTC", "🟧", (
        ("BTC.D",  "CRYPTOCAP:BTC.D", 2, "%"),
        ("BTCUSDT", "BITGET:BTCUSDT", 2, ""),
    )),
    ("ETH", "🟪", (
        ("ETH.D",  "CRYPTOCAP:ETH.D", 2, "%"),
        ("ETHUSDT", "BITGET:ETHUSDT", 2, ""),
    )),
    ("SOL", "🟦", (
        ("SOL.D",  "CRYPTOCAP:SOL.D", 2, "%"),
        ("SOLUSDT", "BITGET:SOLUSDT", 2, ""),
    )),
    ("Altcoins", "🌐", (
        ("OTHERS.D", "CRYPTOCAP:OTHERS.D", 2, "%"),
    )),
    ("Gold", "🪙", (
        ("XAU/USD", "OANDA:XAUUSD", 2, ""),
    )),
)


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

    return f"    {tf:<3} │ {wt_part} · {lux_part} · {smc_part}"


def _section_for_feed(
    label: str,
    fq_key: str,
    decimals: int,
    suffix: str,
    tfs: list[str],
    refresh: bool,
) -> list[str]:
    if refresh:
        tv_cache.refresh_all(fq_key, tfs=tfs)
    dfs = {tf: tv_cache.get_ohlcv(fq_key, tf) for tf in tfs}
    summary = usdt_indicators.build_usdt_indicators_summary(dfs)

    cur_tf = tfs[0]
    cur = _last_close(dfs, cur_tf)
    chg = _change_pct(dfs, cur_tf, n_back=1)
    cur_str = (f"{cur:,.{decimals}f}{suffix}" if cur is not None else "—")
    chg_str = f" ({chg:+.2f}%)" if chg is not None else ""

    lines = [f"  {label} · {cur_str}{chg_str}"]
    if not summary.get("available"):
        lines.append("    (sin datos)")
        return lines
    for tf in tfs:
        lines.append(_row(tf, summary["wt"].get(tf, {}),
                              summary["lux"].get(tf, {}),
                              summary["smc"].get(tf, {})))
    return lines


def build_digest_body(
    tfs: list[str],
    groups: Iterable[tuple] = ASSET_GROUPS,
    refresh: bool = True,
) -> str:
    """Compose the full digest text. `tfs` ordered most-granular → least."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks = [f"As of: {now}", f"TFs: {', '.join(tfs)}", ""]
    for label, emoji, feeds in groups:
        blocks.append(f"{emoji} [{label}]")
        for feed_label, fq, dec, suffix in feeds:
            try:
                blocks.extend(_section_for_feed(feed_label, fq, dec, suffix, tfs, refresh))
            except Exception as e:
                log.warning("digest feed failed key=%s err=%s", fq, e)
                blocks.append(f"  {feed_label}: error ({e})")
        blocks.append("")
    return "\n".join(blocks).rstrip()


# Legacy compatibility — DOMINANCE_SYMBOLS still referenced elsewhere.
DOMINANCE_SYMBOLS: tuple[str, ...] = (
    "CRYPTOCAP:USDT.D", "CRYPTOCAP:BTC.D", "CRYPTOCAP:ETH.D", "CRYPTOCAP:SOL.D",
)
