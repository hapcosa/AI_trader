"""DeepSeek OpenAI-compatible API adapter."""

from __future__ import annotations

import os
from typing import Any

from pineforge_ai.ai_clients.base import AIResponse, require_api_key, usage_from_openai_style
from pineforge_ai.ai_clients.registry import get_provider_spec

DEFAULT_BASE_URL = "https://api.deepseek.com"


def call_raw(
    *,
    prompt: str,
    api_key: str | None,
    model: str,
    max_tokens: int,
    system_prompt: str,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install with: pip install openai") from e

    spec = get_provider_spec("deepseek")
    key = api_key or os.environ.get(spec.env_var)
    key = require_api_key(key, spec.env_var, spec.name)

    client = OpenAI(
        api_key=key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    message = resp.choices[0].message if getattr(resp, "choices", None) else None
    response_text = getattr(message, "content", "") if message is not None else ""

    return AIResponse(
        provider=spec.id,
        model=model,
        response=(response_text or "").strip(),
        usage=usage_from_openai_style(getattr(resp, "usage", None)),
    ).as_dict()
