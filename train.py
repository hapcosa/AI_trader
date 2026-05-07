"""
PineForge AI — Walk-forward pre-training CLI.

Usage:
    python -m pineforge_ai.train \
        --symbol BTC/USDT \
        --start 2024-01-01 \
        --end 2025-01-01 \
        --iterations 3 \
        --timeframes 1h,4h,1d \
        --api-key $ANTHROPIC_API_KEY

    # Dry run (no API calls):
    python -m pineforge_ai.train --symbol BTC/USDT --start 2025-01-01 --iterations 2 --dry-run
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone

import click

from pineforge_ai.config import (
    CANDLES_PER_DAY,
    DEFAULT_EXCHANGE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIMEFRAMES,
    VALID_TIMEFRAMES,
    WARMUP_BARS,
)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


@click.command()
@click.option("--symbol", required=True, help="Ticker: BTC/USDT, AAPL, ^FTSE")
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD (default: now)")
@click.option("--iterations", default=3, show_default=True, type=int)
@click.option("--timeframes", default=",".join(DEFAULT_TIMEFRAMES), show_default=True)
@click.option("--source", default="auto", type=click.Choice(["auto", "yfinance", "ccxt"]))
@click.option("--exchange", default=DEFAULT_EXCHANGE, show_default=True)
@click.option("--output", default="pineforge_ai/output/train", show_default=True)
@click.option("--api-key", default=None, help="Anthropic API key (or env ANTHROPIC_API_KEY)")
@click.option("--model", default="claude-opus-4-7", show_default=True)
@click.option("--lookahead-bars", default=200, show_default=True, type=int,
              help="Bars beyond iteration cutoff to wait for trade outcome")
@click.option("--dry-run", is_flag=True, default=False,
              help="Do not call API; emit prompts only")
def main(
    symbol, start, end, iterations, timeframes, source, exchange,
    output, api_key, model, lookahead_bars, dry_run,
):
    """Walk-forward training mode — let AI evaluate its own past performance."""

    start_dt = _parse_date(start)
    end_dt = _parse_date(end) if end else datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )

    tf_list = [t.strip().lower() for t in timeframes.split(",") if t.strip()]
    bad = [t for t in tf_list if t not in VALID_TIMEFRAMES]
    if bad:
        click.echo(f"ERROR: invalid TFs: {bad}", err=True)
        sys.exit(1)

    click.echo(f"\n{'═'*70}")
    click.echo(f"  PINEFORGE AI — WALK-FORWARD TRAINING")
    click.echo(f"  Symbol     : {symbol}")
    click.echo(f"  Range      : {start_dt.date()} → {end_dt.date()}")
    click.echo(f"  Iterations : {iterations}")
    click.echo(f"  Timeframes : {', '.join(tf_list)}")
    click.echo(f"  Mode       : {'DRY RUN' if dry_run else 'LIVE (API)'}")
    click.echo(f"{'═'*70}\n")

    # ── Fetch full historical range ──────────────────────────────────────────
    from pineforge_ai.data.fetcher import detect_source, fetch_multi_timeframe

    actual_source = source if source != "auto" else detect_source(symbol)
    total_days = max(1, (end_dt - start_dt).days + math.ceil(
        lookahead_bars / max(CANDLES_PER_DAY[tf_list[-1]], 1.0)
    ) + 30)  # +30 buffer for warmup

    click.echo(f"[1/3] Descargando datos OHLCV ({actual_source.upper()}, ~{total_days} días)...")
    try:
        dfs_full = fetch_multi_timeframe(
            symbol=symbol, timeframes=tf_list,
            days=total_days, source=actual_source, exchange=exchange,
        )
    except Exception as e:
        click.echo(f"ERROR descargando: {e}", err=True); sys.exit(1)

    if not dfs_full:
        click.echo("ERROR: empty dfs", err=True); sys.exit(1)

    # ── Indicator pipeline ───────────────────────────────────────────────────
    from pineforge_ai.indicators.wavetrend import wavetrend_all_timeframes, wavetrend_summary
    from pineforge_ai.indicators.luxalgo_amo import adaptive_momentum_all_timeframes, luxalgo_summary
    from pineforge_ai.indicators.smc_buda import smc_all_timeframes, smc_summary
    from pineforge_ai.indicators.wae import trend_quality_all_timeframes, trend_quality_summary
    from pineforge_ai.indicators.cybercycle import cybercycle_all_timeframes, cybercycle_summary
    from pineforge_ai.indicators.ict_concepts import ict_all_timeframes, ict_summary
    from pineforge_ai.indicators.trendlines import trendlines_all_timeframes, trendlines_summary

    def indicator_pipeline(dfs):
        try: wt = wavetrend_summary(wavetrend_all_timeframes(dfs))
        except Exception as e: click.echo(f"  WT skip: {e}"); wt = None
        try: lux = luxalgo_summary(adaptive_momentum_all_timeframes(dfs))
        except Exception as e: click.echo(f"  LUX skip: {e}"); lux = None
        try: smc = smc_summary(smc_all_timeframes(dfs))
        except Exception as e: click.echo(f"  SMC skip: {e}"); smc = None
        try: tq = trend_quality_summary(trend_quality_all_timeframes(dfs))
        except Exception as e: click.echo(f"  TQ skip: {e}"); tq = None
        try: cc = cybercycle_summary(cybercycle_all_timeframes(dfs))
        except Exception as e: click.echo(f"  CC skip: {e}"); cc = None
        try: ict = ict_summary(ict_all_timeframes(dfs))
        except Exception as e: click.echo(f"  ICT skip: {e}"); ict = None
        try: tl = trendlines_summary(trendlines_all_timeframes(dfs))
        except Exception as e: click.echo(f"  TL skip: {e}"); tl = None
        return {"wt": wt, "lux": lux, "smc": smc, "tq": tq, "cc": cc, "ict": ict, "tl": tl}

    # ── Context ──────────────────────────────────────────────────────────────
    from pineforge_ai.context.correlations import fetch_correlations
    from pineforge_ai.context.volatility import volatility_all_timeframes

    corrs_cache = None
    def get_correlations():
        nonlocal corrs_cache
        if corrs_cache is None:
            try: corrs_cache = fetch_correlations(skip_btc_for=symbol)
            except Exception as e: click.echo(f"  Corr skip: {e}"); corrs_cache = {}
        return corrs_cache

    # ── AI sender ────────────────────────────────────────────────────────────
    send_fn = None
    if not dry_run:
        from pineforge_ai.prompt_builder import send_to_ai
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            click.echo("WARN: no API key — switching to dry-run", err=True)
            dry_run = True
        else:
            send_fn = lambda prompt: send_to_ai(prompt, api_key=key, model=model)

    # ── Run engine ───────────────────────────────────────────────────────────
    from pineforge_ai.backtester.engine import run_walk_forward
    from pineforge_ai.prompt_builder import build_prompt, save_prompt

    click.echo(f"\n[2/3] Walk-forward orchestrator iniciado.")
    results = run_walk_forward(
        symbol=symbol,
        dfs_full=dfs_full,
        timeframes=tf_list,
        start=start_dt,
        end=end_dt,
        iterations=iterations,
        output_dir=output,
        build_prompt_fn=build_prompt,
        send_fn=send_fn,
        indicator_fn=indicator_pipeline,
        correlations_fn=get_correlations,
        volatility_fn=volatility_all_timeframes,
        lookahead_bars=lookahead_bars,
        source=actual_source,
        exchange=exchange,
        dry_run=dry_run,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    click.echo(f"\n[3/3] Walk-forward completado.")
    click.echo(f"\n{'═'*70}\n  RESUMEN DE ITERACIONES\n{'═'*70}")
    for r in results:
        click.echo(f"  Iter {r.iteration}: {r.summary_line}")

    # Save final iteration's prompt as LIVE prompt
    if results:
        live = results[-1]
        live_path = save_prompt(live.prompt, symbol=symbol, output_dir=output,
                                  dt_utc=datetime.now(tz=timezone.utc))
        click.echo(f"\n  LIVE PROMPT: {live_path}")
        click.echo(f"  LIVE REPORT: {live.report_path}")
    click.echo(f"\n{'═'*70}")


if __name__ == "__main__":
    main()
