"""Anthropic Claude API adapter."""

from __future__ import annotations

import os
from typing import Any

from pineforge_ai.ai_clients.base import AIResponse, require_api_key
from pineforge_ai.ai_clients.registry import get_provider_spec


def call_raw(
    *,
    prompt: str,
    api_key: str | None,
    model: str,
    max_tokens: int,
    system_prompt: str,
) -> dict[str, Any]:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Install with: pip install anthropic") from e

    spec = get_provider_spec("anthropic")
    key = api_key or os.environ.get(spec.env_var)
    key = require_api_key(key, spec.env_var, spec.name)

    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    response_text = "\n".join(text_parts).strip()

    usage = getattr(resp, "usage", None)
    usage_dict = {}
    if usage is not None:
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        }

    return AIResponse(
        provider=spec.id,
        model=model,
        response=response_text,
        usage=usage_dict,
    ).as_dict()
