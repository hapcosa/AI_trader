"""Apply WaveTrend / LuxAlgo AMO / SMC Elite (smc_buda) to USDT.D multi-TF bars.

The cache layer (`tv_cache.py`) provides clean OHLCV per TF; this module runs
the same three indicators used by the prompt for normal symbols, producing
summary dicts ready to be rendered in the prompt.

Output shape mirrors the existing `wt_summary` / `lux_summary` / `smc_sum`
contracts, so prompt rendering can reuse the same conventions if desired.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from pineforge_ai.indicators.wavetrend import (
    wavetrend_all_timeframes,
    wavetrend_summary,
)
from pineforge_ai.indicators.luxalgo_amo import (
    adaptive_momentum_all_timeframes,
    luxalgo_summary,
)
from pineforge_ai.indicators.smc_buda import (
    smc_analysis,
    smc_summary,
    _build_mtf_state,
)

log = logging.getLogger(__name__)

# HTF → LTF priority for the SMC cascade (descending size).
_SMC_ORDER = ["1M", "1w", "1d", "4h", "1h"]


def _smc_cascade(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Run SMC analysis HTF→LTF, propagating MTFState.

    Replaces smc_buda.smc_all_timeframes because its `_tf_minutes` does not
    recognize the uppercase `1M` label.
    """
    results: dict[str, pd.DataFrame] = {}
    htf_state: Optional[object] = None
    for tf in _SMC_ORDER:
        df = dfs.get(tf)
        if df is None or df.empty:
            continue
        try:
            res = smc_analysis(df, htf_state=htf_state)
        except Exception as e:
            log.warning("smc_analysis failed tf=%s err=%s", tf, e)
            continue
        results[tf] = res
        try:
            htf_state = _build_mtf_state(res)
        except Exception:
            htf_state = None
    return results


def build_usdt_indicators_summary(
    dfs: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """Compute WT/AMO/SMC summaries for USDT.D MTF bars.

    Args:
        dfs: {tf: OHLCV DataFrame}  (typically from tv_cache.get_dfs())

    Returns:
        {
            "available": bool,
            "tfs": list[str],
            "wt":  {tf: {...}},
            "lux": {tf: {...}},
            "smc": {tf: {...}},
        }
    """
    clean = {tf: df for tf, df in dfs.items() if df is not None and not df.empty}
    if not clean:
        return {
            "available": False,
            "tfs": [],
            "wt": {}, "lux": {}, "smc": {},
        }

    try:
        wt = wavetrend_summary(wavetrend_all_timeframes(clean))
    except Exception as e:
        log.warning("wavetrend failed: %s", e)
        wt = {}

    try:
        lux = luxalgo_summary(adaptive_momentum_all_timeframes(clean))
    except Exception as e:
        log.warning("luxalgo failed: %s", e)
        lux = {}

    try:
        smc = smc_summary(_smc_cascade(clean))
    except Exception as e:
        log.warning("smc failed: %s", e)
        smc = {}

    tfs_present = sorted(
        set(clean.keys()) & (set(wt.keys()) | set(lux.keys()) | set(smc.keys())),
        key=lambda t: {"1h": 1, "4h": 2, "1d": 3, "1w": 4, "1M": 5}.get(t, 99),
    )

    return {
        "available": bool(tfs_present),
        "tfs": tfs_present,
        "wt": wt,
        "lux": lux,
        "smc": smc,
    }
