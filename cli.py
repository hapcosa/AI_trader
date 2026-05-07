"""
PineForge AI CLI - elite trading indicator analysis prompt generator.

Usage:
    python -m pineforge_ai.cli --symbol BTC/USDT --timeframes 1h,4h,1d --days 60
    python -m pineforge_ai.cli --symbol BTC/USDT --mode mindset --timeframes 15m,1h,4h --days 7
    python -m pineforge_ai.cli --symbol BTC/USDT --timeframes 1h,4h,1d --candles 300
    python -m pineforge_ai.cli --symbol BTC/USDT --indicators wavetrend,smc --no-context
    python -m pineforge_ai.cli --symbol BTC/USDT --send-to-ai --api-key $KEY
"""

from __future__ import annotations

import click

from pineforge_ai.config import (
    ALL_INDICATORS,
    DEFAULT_DAYS,
    DEFAULT_EXCHANGE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIMEFRAMES,
)
from pineforge_ai.runner import generate_prompt


@click.command()
@click.option("--symbol", required=True, help="BTC/USDT, AAPL, ^FTSE, EURUSD=X")
@click.option(
    "--indicators",
    default="all",
    show_default=True,
    help=f"Comma list or 'all'. Available: {', '.join(ALL_INDICATORS)}",
)
@click.option("--timeframes", default=",".join(DEFAULT_TIMEFRAMES), show_default=True)
@click.option(
    "--days",
    default=None,
    type=int,
    help=f"History in days from now. Defaults to {DEFAULT_DAYS} when --candles is not set.",
)
@click.option(
    "--candles",
    default=None,
    type=int,
    help="Final candle count per selected timeframe. Mutually exclusive with --days.",
)
@click.option("--source", default="auto", type=click.Choice(["auto", "yfinance", "ccxt"]))
@click.option("--exchange", default=DEFAULT_EXCHANGE, show_default=True)
@click.option("--output", default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option("--no-save", is_flag=True, default=False, help="Print to stdout instead of saving")
@click.option("--no-context", is_flag=True, default=False, help="Skip correlations + volatility context")
@click.option("--send-to-ai", is_flag=True, default=False, help="Call Anthropic API after building prompt")
@click.option("--api-key", default=None, help="Anthropic API key (or env ANTHROPIC_API_KEY)")
@click.option("--model", default="claude-opus-4-7", show_default=True)
@click.option(
    "--mode",
    default="signal",
    show_default=True,
    type=click.Choice(["signal", "mindset"]),
    help="'signal': JSON signal analysis. 'mindset': Pre-NY Protocol checklist.",
)
def main(
    symbol,
    indicators,
    timeframes,
    days,
    candles,
    source,
    exchange,
    output,
    no_save,
    no_context,
    send_to_ai,
    api_key,
    model,
    mode,
):
    try:
        result = generate_prompt(
            symbol=symbol,
            indicators=indicators,
            timeframes=timeframes,
            days=days,
            candles=candles,
            source=source,
            exchange=exchange,
            output=output,
            no_context=no_context,
            send_to_ai=send_to_ai,
            api_key=api_key,
            model=model,
            mode=mode,
            save=not no_save,
            emit=click.echo,
        )
    except Exception as e:
        raise click.ClickException(str(e)) from e

    if no_save:
        click.echo(result.prompt)


if __name__ == "__main__":
    main()
