python -m pineforge_ai.cli --symbol BTC/USDT --indicators all \
  --timeframes 1h,4h,1d --days 30 --no-save

# Test backtester (modo dry-run sin API key)
python -m pineforge_ai.train --symbol BTC/USDT \
  --start 2024-01-01 --end 2025-01-01 \
  --iterations 3 --timeframes 1h,4h --dry-run

# Test con API (proveedores: anthropic, openai, gemini, deepseek)
python -m pineforge_ai.train --symbol BTC/USDT \
  --start 2024-06-01 --iterations 2 \
  --provider anthropic --api-key $ANTHROPIC_API_KEY --model claude-sonnet-4-6

python -m pineforge_ai.cli --symbol BTC/USDT \
  --send-to-ai --provider openai --api-key $OPENAI_API_KEY --model gpt-5.5
