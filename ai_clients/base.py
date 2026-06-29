"""Shared primitives for AI provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class AIProviderError(RuntimeError):
    """Raised when an AI provider request cannot be completed."""


@dataclass(frozen=True)
class AIResponse:
    provider: str
    model: str
    response: str
    usage: dict[str, Any]
    # True when the provider stopped because it hit the output-token cap, i.e.
    # the body is cut off mid-stream. Lets callers fail over / salvage instead
    # of choking on a truncated JSON blob (common on free-tier models).
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "response": self.response,
            "usage": self.usage,
            "truncated": self.truncated,
        }


# Finish/stop-reason markers that mean "ran out of output tokens" across the
# providers we speak to. Normalised to lowercase substrings:
#   Anthropic Messages → stop_reason == "max_tokens"
#   OpenAI Responses   → incomplete_details.reason == "max_output_tokens"
#   OpenAI ChatCompl.  → choices[].finish_reason == "length" (deepseek, moonshot)
#   Gemini             → candidates[].finish_reason == "MAX_TOKENS"
_TRUNCATION_MARKERS: tuple[str, ...] = ("max_tokens", "max_output_tokens", "length")


def is_truncated_reason(reason: Any) -> bool:
    """Whether a provider finish/stop reason signals an output-token cutoff.

    Tolerant of enums (Gemini returns a FinishReason enum whose str() is
    "FinishReason.MAX_TOKENS"), plain strings, and None.
    """
    if reason is None:
        return False
    text = str(reason).strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _TRUNCATION_MARKERS)


def obj_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def usage_from_openai_style(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}

    input_tokens = obj_value(usage, "input_tokens", obj_value(usage, "prompt_tokens", 0))
    output_tokens = obj_value(usage, "output_tokens", obj_value(usage, "completion_tokens", 0))
    total_tokens = obj_value(usage, "total_tokens", None)

    input_details = obj_value(usage, "input_tokens_details", None)
    if input_details is None:
        input_details = obj_value(usage, "prompt_tokens_details", None)
    cached = obj_value(input_details, "cached_tokens", 0)

    result = {
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "cache_read_input_tokens": cached or 0,
    }
    if total_tokens is not None:
        result["total_tokens"] = total_tokens
    return result


def require_api_key(api_key: str | None, env_var: str, provider_name: str) -> str:
    if api_key:
        return api_key
    raise ValueError(f"{env_var} not set in environment or API key missing for {provider_name}.")
