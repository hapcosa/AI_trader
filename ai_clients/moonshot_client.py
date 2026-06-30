"""Moonshot (Kimi) adapter — OpenAI-compatible Chat Completions API.

Moonshot exposes an OpenAI-compatible endpoint at https://api.moonshot.ai/v1,
so we reuse the `openai` SDK pointed at that base_url. Unlike OpenAI's newer
Responses API, Moonshot speaks the classic Chat Completions API, hence
`client.chat.completions.create(...)` with system+user messages.
"""

from __future__ import annotations

import os
from typing import Any

from pineforge_ai.ai_clients.base import (
    AIResponse,
    is_truncated_reason,
    require_api_key,
    usage_from_openai_style,
)
from pineforge_ai.ai_clients.registry import get_provider_spec

# Overridable so a self-hosted / .cn endpoint can be swapped without code change.
MOONSHOT_BASE_URL = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")


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

    spec = get_provider_spec("moonshot")
    key = api_key or os.environ.get(spec.env_var)
    key = require_api_key(key, spec.env_var, spec.name)

    client = OpenAI(api_key=key, base_url=MOONSHOT_BASE_URL)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )

    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    text = ""
    if choice is not None and getattr(choice, "message", None) is not None:
        text = (choice.message.content or "").strip()

    finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None

    return AIResponse(
        provider=spec.id,
        model=model,
        response=text,
        usage=usage_from_openai_style(getattr(resp, "usage", None)),
        truncated=is_truncated_reason(finish_reason),
    ).as_dict()
