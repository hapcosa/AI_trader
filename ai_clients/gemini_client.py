"""Google Gemini API adapter."""

from __future__ import annotations

import os
from typing import Any

from pineforge_ai.ai_clients.base import AIResponse, obj_value, require_api_key
from pineforge_ai.ai_clients.registry import get_provider_spec


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "input_tokens": obj_value(usage, "prompt_token_count", 0) or 0,
        "output_tokens": obj_value(usage, "candidates_token_count", 0) or 0,
        "total_tokens": obj_value(usage, "total_token_count", 0) or 0,
    }


def call_raw(
    *,
    prompt: str,
    api_key: str | None,
    model: str,
    max_tokens: int,
    system_prompt: str,
) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError("Install with: pip install google-genai") from e

    spec = get_provider_spec("gemini")
    key = api_key or os.environ.get(spec.env_var)
    key = require_api_key(key, spec.env_var, spec.name)

    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
        ),
    )

    return AIResponse(
        provider=spec.id,
        model=model,
        response=(getattr(resp, "text", "") or "").strip(),
        usage=_usage_dict(getattr(resp, "usage_metadata", None)),
    ).as_dict()
