"""Provider dispatch for raw AI completions."""

from __future__ import annotations

from typing import Any

from pineforge_ai.ai_clients.registry import get_provider_spec


def call_ai_raw(
    *,
    provider: str | None,
    prompt: str,
    api_key: str | None,
    model: str | None,
    max_tokens: int,
    system_prompt: str,
) -> dict[str, Any]:
    spec = get_provider_spec(provider)
    selected_model = (model or spec.default_model).strip() or spec.default_model

    if spec.id == "anthropic":
        from pineforge_ai.ai_clients.anthropic_client import call_raw
    elif spec.id == "openai":
        from pineforge_ai.ai_clients.openai_client import call_raw
    elif spec.id == "gemini":
        from pineforge_ai.ai_clients.gemini_client import call_raw
    elif spec.id == "deepseek":
        from pineforge_ai.ai_clients.deepseek_client import call_raw
    elif spec.id == "moonshot":
        from pineforge_ai.ai_clients.moonshot_client import call_raw
    else:
        raise ValueError(f"Unsupported AI provider '{provider}'.")

    return call_raw(
        prompt=prompt,
        api_key=api_key,
        model=selected_model,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
