"""
End-to-end wiring test for the BudAI oscillators in the prompt builder:
compute the 4 summaries from synthetic OHLCV and assert their MTF blocks land
in the generated prompt. Imports via the `pineforge_ai.*` namespace (see
conftest) exactly like production.
"""
import numpy as np
import pandas as pd

from pineforge_ai.prompt_builder import build_prompt
from pineforge_ai.indicators.budai_pulse import (
    budai_pulse_all_timeframes, budai_pulse_summary,
)
from pineforge_ai.indicators.budai_abyss import (
    budai_abyss_all_timeframes, budai_abyss_summary,
)
from pineforge_ai.indicators.budai_moneyflow_tide import (
    budai_moneyflow_tide_all_timeframes, budai_moneyflow_tide_summary,
)
from pineforge_ai.indicators.budai_athenea import (
    budai_athenea_all_timeframes, budai_athenea_summary,
)


def _df(n=400, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 100.0 + 8.0 * np.sin(t / 11.0) + 4.0 * np.sin(t / 3.3) + np.cumsum(rng.normal(0, 0.2, n))
    close = base + rng.normal(0, 0.4, n)
    high = np.maximum(close, base) + rng.uniform(0.1, 0.8, n)
    low = np.minimum(close, base) - rng.uniform(0.1, 0.8, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(50, 200, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


def test_budai_blocks_in_prompt():
    dfs = {"1h": _df(seed=1), "4h": _df(seed=2)}
    tfs = ["1h", "4h"]

    prompt = build_prompt(
        symbol="BTC/USDT",
        dfs=dfs,
        timeframes=tfs,
        pulse_summary=budai_pulse_summary(budai_pulse_all_timeframes(dfs)),
        abyss_summary=budai_abyss_summary(budai_abyss_all_timeframes(dfs)),
        tide_summary=budai_moneyflow_tide_summary(budai_moneyflow_tide_all_timeframes(dfs)),
        athenea_summary=budai_athenea_summary(budai_athenea_all_timeframes(dfs)),
    )

    assert "BUDAI PULSE FLOW OSCILLATOR — MTF" in prompt
    assert "BUDAI ABYSS WAVE OSCILLATOR — MTF" in prompt
    assert "BUDAI SMART MONEY FLOW TIDE — MTF" in prompt
    assert "BUDAI ATHENEA OSCILLATOR — MTF" in prompt
    # rows for both timeframes present in at least one block
    assert "1H" in prompt and "4H" in prompt


def test_budai_blocks_absent_when_no_summary():
    dfs = {"1h": _df(seed=1)}
    prompt = build_prompt(symbol="BTC/USDT", dfs=dfs, timeframes=["1h"])
    assert "BUDAI PULSE FLOW OSCILLATOR" not in prompt
    assert "BUDAI ATHENEA OSCILLATOR" not in prompt
