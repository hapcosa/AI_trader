"""OpenAI Responses API adapter."""

from __future__ import annotations

import os
from typing import Any

from pineforge_ai.ai_clients.base import (
    AIResponse,
    is_truncated_reason,
    obj_value,
    require_api_key,
    usage_from_openai_style,
)
from pineforge_ai.ai_clients.registry import get_provider_spec


def _is_truncated(resp: Any) -> bool:
    """Responses API flags a token-cap cutoff via status='incomplete' +
    incomplete_details.reason='max_output_tokens'."""
    if str(getattr(resp, "status", "") or "").lower() != "incomplete":
        return False
    reason = obj_value(getattr(resp, "incomplete_details", None), "reason", None)
    # An incomplete response with no explicit reason is still a cutoff for our
    # purposes (the body is partial); treat the max_output_tokens reason and a
    # missing reason both as truncated.
    return reason is None or is_truncated_reason(reason)


def _response_text(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return str(output_text).strip()

    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = obj_value(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


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

    spec = get_provider_spec("openai")
    key = api_key or os.environ.get(spec.env_var)
    key = require_api_key(key, spec.env_var, spec.name)

    client = OpenAI(api_key=key)
    resp = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=prompt,
        max_output_tokens=max_tokens,
    )

    return AIResponse(
        provider=spec.id,
        model=model,
        response=_response_text(resp),
        usage=usage_from_openai_style(getattr(resp, "usage", None)),
        truncated=_is_truncated(resp),
    ).as_dict()
