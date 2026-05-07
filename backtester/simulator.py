"""
Walk-forward trade simulator.

Given AI-generated entries and historical OHLCV, simulates each entry bar-by-bar:
- Entry triggered when price reaches entry_zone
- Closed at SL, TP1, TP2 or timeout
- Returns trade outcomes with PnL%, RR achieved, duration
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class TradeResult:
    id: int
    direction: str            # 'long' | 'short'
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    entry_bar: int            # offset from start of df
    exit_bar: int
    result: str               # 'win_full' | 'win_partial' | 'loss' | 'timeout' | 'never_triggered'
    pnl_pct: float            # PnL % of price change (long: (exit-entry)/entry, short: -)
    rr_achieved: float        # achieved R-multiples vs initial risk
    duration_bars: int
    notes: str = ""


def _entry_zone_lo_hi(zone) -> tuple[float, float]:
    if zone is None:
        return (np.nan, np.nan)
    if isinstance(zone, (list, tuple)) and len(zone) >= 2:
        return (float(min(zone[0], zone[1])), float(max(zone[0], zone[1])))
    if isinstance(zone, (int, float)):
        v = float(zone)
        return (v, v)
    return (np.nan, np.nan)


def simulate_entry(
    entry: dict,
    df: pd.DataFrame,
    start_idx: int,
    lookahead_bars: int = 200,
) -> TradeResult:
    """
    Simulate a single entry on `df` starting from `start_idx`.

    Args:
        entry:          AI-generated entry dict (from JSON response)
        df:             Execution-TF OHLCV (full visibility past start_idx allowed)
        start_idx:      Bar index in df where simulation begins (typically the prompt's "now")
        lookahead_bars: Max bars to wait for entry trigger + position outcome
    """
    eid = int(entry.get("id", 0))
    direction = entry.get("direction", "long").lower()
    zone = entry.get("entry_zone")
    sl = float(entry.get("stop_loss", np.nan))
    tp1 = float(entry.get("take_profit_1", np.nan))
    tp2 = float(entry.get("take_profit_2", tp1))

    elo, ehi = _entry_zone_lo_hi(zone)
    if np.isnan(elo) or np.isnan(sl) or np.isnan(tp1):
        return TradeResult(
            id=eid, direction=direction, entry_price=np.nan, stop_loss=sl,
            take_profit_1=tp1, take_profit_2=tp2,
            entry_bar=-1, exit_bar=-1, result="invalid",
            pnl_pct=0.0, rr_achieved=0.0, duration_bars=0,
            notes="missing entry_zone/SL/TP",
        )

    end = min(start_idx + lookahead_bars, len(df))
    triggered = False
    entry_bar = -1
    entry_price = np.nan
    tp1_hit = False

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    for i in range(start_idx, end):
        h, l, c = high[i], low[i], close[i]

        # Trigger
        if not triggered:
            # Long: bar low touched zone low, take entry at zone hi (worst fill)
            if direction == "long":
                if l <= ehi:
                    triggered = True
                    entry_bar = i
                    entry_price = min(ehi, max(elo, l))
            else:  # short
                if h >= elo:
                    triggered = True
                    entry_bar = i
                    entry_price = max(elo, min(ehi, h))
            if not triggered:
                continue

        # Position management (after trigger, possibly same bar)
        risk = abs(entry_price - sl)
        if risk <= 0:
            return TradeResult(
                id=eid, direction=direction, entry_price=entry_price, stop_loss=sl,
                take_profit_1=tp1, take_profit_2=tp2,
                entry_bar=entry_bar, exit_bar=i, result="invalid",
                pnl_pct=0.0, rr_achieved=0.0, duration_bars=i - entry_bar,
                notes="zero risk (entry==SL)",
            )

        if direction == "long":
            # Conservative: SL takes priority within same bar
            if l <= sl:
                pnl = (sl - entry_price) / entry_price * 100.0
                rr = -1.0 if not tp1_hit else (((tp1 - entry_price) / 2.0 + (sl - entry_price) / 2.0) / risk)
                return TradeResult(
                    id=eid, direction=direction, entry_price=entry_price,
                    stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                    entry_bar=entry_bar, exit_bar=i,
                    result=("win_partial" if tp1_hit else "loss"),
                    pnl_pct=pnl, rr_achieved=rr,
                    duration_bars=i - entry_bar,
                )
            if not tp1_hit and h >= tp1:
                tp1_hit = True
            if h >= tp2:
                pnl = (tp2 - entry_price) / entry_price * 100.0
                rr = (tp2 - entry_price) / risk
                return TradeResult(
                    id=eid, direction=direction, entry_price=entry_price,
                    stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                    entry_bar=entry_bar, exit_bar=i, result="win_full",
                    pnl_pct=pnl, rr_achieved=rr,
                    duration_bars=i - entry_bar,
                )
        else:  # short
            if h >= sl:
                pnl = (entry_price - sl) / entry_price * 100.0
                rr = -1.0 if not tp1_hit else 0.0
                return TradeResult(
                    id=eid, direction=direction, entry_price=entry_price,
                    stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                    entry_bar=entry_bar, exit_bar=i,
                    result=("win_partial" if tp1_hit else "loss"),
                    pnl_pct=pnl, rr_achieved=rr,
                    duration_bars=i - entry_bar,
                )
            if not tp1_hit and l <= tp1:
                tp1_hit = True
            if l <= tp2:
                pnl = (entry_price - tp2) / entry_price * 100.0
                rr = (entry_price - tp2) / risk
                return TradeResult(
                    id=eid, direction=direction, entry_price=entry_price,
                    stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                    entry_bar=entry_bar, exit_bar=i, result="win_full",
                    pnl_pct=pnl, rr_achieved=rr,
                    duration_bars=i - entry_bar,
                )

    # Timeout — close at last bar's close
    if not triggered:
        return TradeResult(
            id=eid, direction=direction, entry_price=np.nan,
            stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
            entry_bar=-1, exit_bar=end - 1, result="never_triggered",
            pnl_pct=0.0, rr_achieved=0.0,
            duration_bars=end - 1 - start_idx,
        )

    final_close = close[end - 1]
    if direction == "long":
        pnl = (final_close - entry_price) / entry_price * 100.0
        rr = (final_close - entry_price) / abs(entry_price - sl)
    else:
        pnl = (entry_price - final_close) / entry_price * 100.0
        rr = (entry_price - final_close) / abs(entry_price - sl)
    return TradeResult(
        id=eid, direction=direction, entry_price=entry_price,
        stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
        entry_bar=entry_bar, exit_bar=end - 1, result="timeout",
        pnl_pct=pnl, rr_achieved=rr,
        duration_bars=end - 1 - entry_bar,
    )


def simulate_signals(
    signals_json: dict,
    dfs: dict[str, pd.DataFrame],
    start_idx_per_tf: dict[str, int],
    lookahead_bars: int = 200,
) -> list[TradeResult]:
    """
    Simulate every entry in `signals_json["entries"]` on its execution_tf DF.

    Args:
        signals_json:     parsed JSON from send_to_ai()
        dfs:              {tf: ohlcv_df}
        start_idx_per_tf: {tf: index where "present" was when signal was generated}
        lookahead_bars:   max bars to look forward for outcome
    """
    entries = signals_json.get("entries", [])
    results: list[TradeResult] = []
    for entry in entries:
        tf = entry.get("execution_tf", "1h").lower()
        df = dfs.get(tf)
        if df is None or df.empty:
            continue
        start = start_idx_per_tf.get(tf, len(df) - 1)
        if start >= len(df):
            continue
        results.append(simulate_entry(entry, df, start, lookahead_bars=lookahead_bars))
    return results


def trades_to_dicts(trades: list[TradeResult]) -> list[dict]:
    return [asdict(t) for t in trades]
