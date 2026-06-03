"""Shared PineForge prompt generation workflow for CLI and web."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

from pineforge_ai.config import (
    ALL_INDICATORS,
    DEFAULT_DAYS,
    DEFAULT_EXCHANGE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIMEFRAMES,
    VALID_TIMEFRAMES,
)


Emit = Callable[[str], None]


@dataclass(frozen=True)
class PromptGenerationResult:
    prompt: str
    file_path: str | None
    ai_response_path: str | None
    actual_source: str
    candle_counts: dict[str, int]


def parse_timeframes(timeframes: str | Iterable[str] | None) -> list[str]:
    if timeframes is None:
        tf_list = list(DEFAULT_TIMEFRAMES)
    elif isinstance(timeframes, str):
        tf_list = [t.strip().lower() for t in timeframes.split(",") if t.strip()]
    else:
        tf_list = [str(t).strip().lower() for t in timeframes if str(t).strip()]

    if not tf_list:
        raise ValueError("At least one timeframe is required")

    bad = [t for t in tf_list if t not in VALID_TIMEFRAMES]
    if bad:
        valid = ", ".join(sorted(VALID_TIMEFRAMES))
        raise ValueError(f"Invalid timeframes: {bad}. Valid: {valid}")

    return tf_list


def parse_indicators(indicators: str | Iterable[str] | None) -> list[str]:
    if indicators is None:
        return list(ALL_INDICATORS)

    if isinstance(indicators, str):
        if indicators.strip().lower() == "all":
            return list(ALL_INDICATORS)
        ind_list = [i.strip().lower() for i in indicators.split(",") if i.strip()]
    else:
        raw = [str(i).strip().lower() for i in indicators if str(i).strip()]
        if len(raw) == 1 and raw[0] == "all":
            return list(ALL_INDICATORS)
        ind_list = raw

    if not ind_list:
        raise ValueError("At least one indicator is required")

    bad = [i for i in ind_list if i not in ALL_INDICATORS]
    if bad:
        raise ValueError(f"Invalid indicators: {bad}. Valid: {ALL_INDICATORS}")

    return ind_list


def resolve_history(days: int | None, candles: int | None) -> tuple[int | None, int | None]:
    if days is not None and candles is not None:
        raise ValueError("Use --days or --candles, not both")
    if candles is not None:
        if candles < 1:
            raise ValueError("candles must be greater than 0")
        return None, candles
    if days is None:
        days = DEFAULT_DAYS
    if days < 1:
        raise ValueError("days must be greater than 0")
    return days, None


def _emit(emit: Emit | None, message: str) -> None:
    if emit is not None:
        emit(message)


def _visible_dfs(dfs: dict, candles: int | None, candles_per_tf: dict[str, int] | None = None) -> dict:
    if candles is None and candles_per_tf is None:
        return dfs
    result = {}
    for tf, df in dfs.items():
        c = candles_per_tf.get(tf) if candles_per_tf else None
        if c is None:
            c = candles
        result[tf] = df.iloc[-c:].copy() if c is not None else df
    return result


def _build_indicator_summaries(dfs: dict, ind_list: list[str], emit: Emit | None) -> dict[str, dict | None]:
    summaries: dict[str, dict | None] = {
        "wt": None,
        "lux": None,
        "smc": None,
        "tq": None,
        "it": None,
        "ict": None,
        "tl": None,
        "pulse": None,
        "abyss": None,
        "tide": None,
        "athenea": None,
    }

    if "wavetrend" in ind_list:
        try:
            from pineforge_ai.indicators.wavetrend import wavetrend_all_timeframes, wavetrend_summary

            summaries["wt"] = wavetrend_summary(wavetrend_all_timeframes(dfs))
            _emit(emit, "      WaveTrend OK")
        except Exception as e:
            _emit(emit, f"      WaveTrend FAIL: {e}")

    if "luxalgo" in ind_list:
        try:
            from pineforge_ai.indicators.luxalgo_amo import (
                adaptive_momentum_all_timeframes,
                luxalgo_summary,
            )

            summaries["lux"] = luxalgo_summary(adaptive_momentum_all_timeframes(dfs))
            _emit(emit, "      LuxAlgo AMO OK")
        except Exception as e:
            _emit(emit, f"      LuxAlgo FAIL: {e}")

    if "smc" in ind_list:
        try:
            from pineforge_ai.indicators.smc_buda import smc_all_timeframes, smc_summary

            summaries["smc"] = smc_summary(smc_all_timeframes(dfs))
            _emit(emit, "      SMC Buda OK")
        except Exception as e:
            _emit(emit, f"      SMC FAIL: {e}")

    if "wae" in ind_list:
        try:
            from pineforge_ai.indicators.wae import trend_quality_all_timeframes, trend_quality_summary

            summaries["tq"] = trend_quality_summary(trend_quality_all_timeframes(dfs))
            _emit(emit, "      WAE+Chop OK")
        except Exception as e:
            _emit(emit, f"      WAE FAIL: {e}")

    if "itrend" in ind_list:
        try:
            from pineforge_ai.indicators.itrend import itrend_all_timeframes, itrend_summary

            summaries["it"] = itrend_summary(itrend_all_timeframes(dfs))
            _emit(emit, "      Ehlers iTrend OK")
        except Exception as e:
            _emit(emit, f"      iTrend FAIL: {e}")

    if "ict" in ind_list:
        try:
            from pineforge_ai.indicators.ict_concepts import ict_all_timeframes, ict_summary

            summaries["ict"] = ict_summary(ict_all_timeframes(dfs))
            _emit(emit, "      ICT Concepts OK")
        except Exception as e:
            _emit(emit, f"      ICT FAIL: {e}")

    if "trendlines" in ind_list:
        try:
            from pineforge_ai.indicators.trendlines import trendlines_all_timeframes, trendlines_summary

            summaries["tl"] = trendlines_summary(trendlines_all_timeframes(dfs))
            _emit(emit, "      Trendlines OK")
        except Exception as e:
            _emit(emit, f"      Trendlines FAIL: {e}")

    if "pulse" in ind_list:
        try:
            from pineforge_ai.indicators.budai_pulse import (
                budai_pulse_all_timeframes, budai_pulse_summary,
            )

            summaries["pulse"] = budai_pulse_summary(budai_pulse_all_timeframes(dfs))
            _emit(emit, "      BudAI Pulse OK")
        except Exception as e:
            _emit(emit, f"      BudAI Pulse FAIL: {e}")

    if "abyss" in ind_list:
        try:
            from pineforge_ai.indicators.budai_abyss import (
                budai_abyss_all_timeframes, budai_abyss_summary,
            )

            summaries["abyss"] = budai_abyss_summary(budai_abyss_all_timeframes(dfs))
            _emit(emit, "      BudAI Abyss OK")
        except Exception as e:
            _emit(emit, f"      BudAI Abyss FAIL: {e}")

    if "tide" in ind_list:
        try:
            from pineforge_ai.indicators.budai_moneyflow_tide import (
                budai_moneyflow_tide_all_timeframes, budai_moneyflow_tide_summary,
            )

            summaries["tide"] = budai_moneyflow_tide_summary(
                budai_moneyflow_tide_all_timeframes(dfs))
            _emit(emit, "      BudAI Money Flow Tide OK")
        except Exception as e:
            _emit(emit, f"      BudAI Tide FAIL: {e}")

    if "athenea" in ind_list:
        try:
            from pineforge_ai.indicators.budai_athenea import (
                budai_athenea_all_timeframes, budai_athenea_summary,
            )

            summaries["athenea"] = budai_athenea_summary(
                budai_athenea_all_timeframes(dfs))
            _emit(emit, "      BudAI Athenea OK")
        except Exception as e:
            _emit(emit, f"      BudAI Athenea FAIL: {e}")

    return summaries


def generate_prompt(
    *,
    symbol: str,
    indicators: str | Iterable[str] | None = "all",
    timeframes: str | Iterable[str] | None = None,
    days: int | None = None,
    candles: int | None = None,
    source: str = "auto",
    exchange: str = DEFAULT_EXCHANGE,
    output: str = DEFAULT_OUTPUT_DIR,
    no_context: bool = False,
    send_to_ai: bool = False,
    api_key: str | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    mode: str = "signal",
    ai_summary: bool = False,
    candles_per_tf: dict[str, int] | None = None,
    save: bool = True,
    emit: Emit | None = None,
) -> PromptGenerationResult:
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("symbol is required")
    if source not in {"auto", "yfinance", "ccxt"}:
        raise ValueError("source must be auto, yfinance, or ccxt")
    if mode not in {"signal", "mindset"}:
        raise ValueError("mode must be signal or mindset")
    if not exchange.strip():
        raise ValueError("exchange is required")

    tf_list = parse_timeframes(timeframes)
    ind_list = parse_indicators(indicators)
    if candles_per_tf:
        days, candles = None, None
    else:
        days, candles = resolve_history(days, candles)
    dt_utc = datetime.now(tz=timezone.utc)

    if candles_per_tf:
        history_label = "Candles    : " + ", ".join(f"{tf}={c}" for tf, c in candles_per_tf.items())
    elif candles is not None:
        history_label = f"Candles    : {candles}"
    else:
        history_label = f"Days       : {days}"

    _emit(emit, f"\n{'=' * 70}")
    _emit(emit, "  AI TRADER v3")
    _emit(emit, f"  Symbol     : {symbol}")
    _emit(emit, f"  Mode       : {mode.upper()}")
    _emit(emit, f"  Indicators : {', '.join(ind_list)}")
    _emit(emit, f"  Timeframes : {', '.join(tf_list)}")
    _emit(emit, f"  {history_label}")
    _emit(emit, f"  Context    : {'OFF' if no_context else 'ON'}")
    _emit(emit, f"  Send to AI : {'YES' if send_to_ai else 'NO'}")
    _emit(emit, f"{'=' * 70}\n")

    from pineforge_ai.data.fetcher import detect_source, fetch_multi_timeframe

    actual_source = source if source != "auto" else detect_source(symbol)
    _emit(emit, f"[1/4] Descargando OHLCV ({actual_source.upper()})...")
    dfs_full = fetch_multi_timeframe(
        symbol=symbol,
        timeframes=tf_list,
        days=days,
        candles=candles,
        source=actual_source,
        exchange=exchange,
        candles_per_tf=candles_per_tf,
    )
    if not dfs_full:
        raise RuntimeError("empty dfs")

    dfs_prompt = _visible_dfs(dfs_full, candles, candles_per_tf=candles_per_tf)
    candle_counts = {tf: len(df) for tf, df in dfs_prompt.items()}

    _emit(emit, "[2/4] Calculando indicadores...")
    summaries = _build_indicator_summaries(dfs_full, ind_list, emit)

    correlations = None
    volatility = None
    if not no_context:
        _emit(emit, "[3/4] Contexto de mercado...")
        try:
            from pineforge_ai.context.correlations import fetch_correlations

            correlations = fetch_correlations(skip_btc_for=symbol)
            _emit(emit, "      Correlations OK")
        except Exception as e:
            _emit(emit, f"      Correlations FAIL: {e}")
        try:
            from pineforge_ai.context.volatility import volatility_all_timeframes

            volatility = volatility_all_timeframes(dfs_full)
            _emit(emit, "      Volatility OK")
        except Exception as e:
            _emit(emit, f"      Volatility FAIL: {e}")
    else:
        _emit(emit, "[3/4] Contexto OFF")

    usdt_data = None
    try:
        from pineforge_ai.usdt_dominance.reader import build_usdt_summary

        usdt_data = build_usdt_summary()
        if usdt_data.get("available"):
            _emit(emit, f"      USDT.D OK ({usdt_data['current']:.4f}%)")
        else:
            _emit(
                emit,
                "      USDT.D: daemon not running "
                "(start: python -m pineforge_ai.usdt_dominance.daemon)",
            )
    except Exception as e:
        _emit(emit, f"      USDT.D FAIL: {e}")

    usdt_alerts = None
    try:
        from pineforge_ai.usdt_dominance.alerts_reader import build_usdt_alerts_summary

        usdt_alerts = build_usdt_alerts_summary(hours=24)
        if usdt_alerts.get("available"):
            _emit(
                emit,
                f"      USDT.D alerts OK ({usdt_alerts['count']} en 24h, "
                f"last={usdt_alerts['last_signal']})",
            )
        else:
            _emit(emit, "      USDT.D alerts: sin eventos en 24h")
    except Exception as e:
        _emit(emit, f"      USDT.D alerts FAIL: {e}")

    dominance_mtf = None
    try:
        from pineforge_ai.usdt_dominance import tv_cache, usdt_indicators

        dominance_targets = [
            ("USDT.D",   "CRYPTOCAP:USDT.D"),
            ("BTC.D",    "CRYPTOCAP:BTC.D"),
            ("OTHERS.D", "CRYPTOCAP:OTHERS.D"),
        ]
        dominance_mtf = {}
        for label, fq in dominance_targets:
            try:
                tv_cache.refresh_all(fq)
                dfs_x = tv_cache.get_dfs(fq)
                dominance_mtf[label] = usdt_indicators.build_usdt_indicators_summary(dfs_x)
            except Exception as e:
                _emit(emit, f"      {label} MTF FAIL: {e}")
        ok_labels = [k for k, v in dominance_mtf.items() if v.get("available")]
        if ok_labels:
            _emit(emit, f"      Dominance MTF OK ({', '.join(ok_labels)})")
        else:
            dominance_mtf = None
    except Exception as e:
        _emit(emit, f"      Dominance MTF FAIL: {e}")

    market_cap_data = None
    try:
        from pineforge_ai.context.market_cap import build_market_cap_summary

        market_cap_data = build_market_cap_summary()
        if market_cap_data.get("available"):
            btc = market_cap_data.get("btc_dominance", 0.0)
            _emit(emit, f"      BTC.D OK ({btc:.2f}%)")
        else:
            _emit(emit, "      BTC.D: CoinGecko unavailable")
    except Exception as e:
        _emit(emit, f"      BTC.D FAIL: {e}")

    _emit(emit, f"[4/4] Ensamblando prompt ({mode.upper()})...")
    from pineforge_ai.prompt_builder import build_prompt, save_prompt, send_to_ai as send_fn

    prompt = build_prompt(
        symbol=symbol,
        dfs=dfs_prompt,
        timeframes=tf_list,
        wt_summary=summaries["wt"],
        lux_summary=summaries["lux"],
        smc_sum=summaries["smc"],
        tq_summary=summaries["tq"],
        cc_summary=None,
        ict_sum=summaries["ict"],
        tl_summary=summaries["tl"],
        it_summary=summaries["it"],
        pulse_summary=summaries["pulse"],
        abyss_summary=summaries["abyss"],
        tide_summary=summaries["tide"],
        athenea_summary=summaries["athenea"],
        correlations=correlations,
        volatility=volatility,
        usdt_data=usdt_data,
        usdt_alerts=usdt_alerts,
        usdt_mtf=dominance_mtf,
        market_cap_data=market_cap_data,
        source=actual_source,
        exchange=exchange,
        candle_counts=candle_counts,
        dt_utc=dt_utc,
        mode=mode,
        ai_summary=ai_summary,
    )

    file_path = None
    if save:
        file_path = save_prompt(prompt, symbol=symbol, output_dir=output, dt_utc=dt_utc)
        _emit(emit, f"\n{'=' * 70}\n  PROMPT GUARDADO\n  {file_path}\n{'=' * 70}")

    ai_response_path = None
    if send_to_ai:
        from pineforge_ai.ai_clients.registry import get_provider_spec

        provider_spec = get_provider_spec(provider)
        selected_model = model or provider_spec.default_model
        _emit(
            emit,
            f"\nLlamando a {provider_spec.name} ({selected_model}) - modo {mode.upper()}...",
        )
        data = send_fn(prompt, api_key=api_key, provider=provider, model=model, mode=mode)
        ai_response_path = (
            file_path.replace(".txt", "_ai.json")
            if file_path
            else f"./pineforge_ai_response_{dt_utc.strftime('%Y%m%d_%H%M')}.json"
        )
        with open(ai_response_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _emit(emit, f"  AI Response: {ai_response_path}")
        usage = data.get("_usage", {})
        if usage:
            _emit(
                emit,
                "  Tokens: "
                f"in={usage.get('input_tokens')} "
                f"out={usage.get('output_tokens')} "
                f"cache_create={usage.get('cache_creation_input_tokens')} "
                f"cache_read={usage.get('cache_read_input_tokens')}",
            )

    return PromptGenerationResult(
        prompt=prompt,
        file_path=file_path,
        ai_response_path=ai_response_path,
        actual_source=actual_source,
        candle_counts=candle_counts,
    )


def generate_prompt_file(**kwargs) -> PromptGenerationResult:
    kwargs["save"] = True
    return generate_prompt(**kwargs)
