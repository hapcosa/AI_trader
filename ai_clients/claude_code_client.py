"""Claude via a Pro/Max SUBSCRIPTION — through the headless Claude Code CLI.

No per-token API key: the "key" is a long-lived OAuth token a human mints once
with `claude setup-token` (prefix `sk-ant-oat01-...`). We run the headless CLI
`claude -p --output-format json` with CLAUDE_CODE_OAUTH_TOKEN in the env, which
bills against the subscription's quota instead of per-token API billing.

The CLI is agentic, so there's no `--max-tokens` for the final answer (it's
managed internally) — `max_tokens` is accepted for interface parity but not
forwarded. Truncation isn't surfaced the way the token-capped providers do, so
`truncated` stays False unless the CLI itself reports an error subtype.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from pineforge_ai.ai_clients.base import AIProviderError, AIResponse, require_api_key
from pineforge_ai.ai_clients.registry import get_provider_spec

# The `claude` binary. Overridable for non-standard installs.
CLAUDE_BIN = os.environ.get("CLAUDE_CODE_BIN", "claude")
# Hard ceiling on a single analysis call (the agentic CLI can run a few turns).
CLAUDE_TIMEOUT_S = int(os.environ.get("CLAUDE_CODE_TIMEOUT_S", "240"))


def _usage_from_cli(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    return {
        "input_tokens": usage.get("input_tokens", 0) or 0,
        "output_tokens": usage.get("output_tokens", 0) or 0,
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) or 0,
    }


def call_raw(
    *,
    prompt: str,
    api_key: str | None,
    model: str,
    max_tokens: int,  # accepted for interface parity; the CLI manages output budget
    system_prompt: str,
) -> dict[str, Any]:
    spec = get_provider_spec("claude-code")
    token = api_key or os.environ.get(spec.env_var)
    token = require_api_key(token, spec.env_var, spec.name)

    # Subscription auth via the OAuth token. Strip ANTHROPIC_API_KEY from the
    # child env: if both are set the CLI prefers the API key (per-token billing),
    # defeating the whole point of the subscription provider.
    env = dict(os.environ)
    env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    env.pop("ANTHROPIC_API_KEY", None)

    cmd = [
        CLAUDE_BIN, "-p",
        "--output-format", "json",
        "--model", model,
        "--append-system-prompt", system_prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            env=env,
            timeout=CLAUDE_TIMEOUT_S,
        )
    except FileNotFoundError as e:
        raise AIProviderError(
            f"'{CLAUDE_BIN}' not found — the Claude Code CLI is not installed in "
            "this container (see AI_trader Dockerfile)."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise AIProviderError(
            f"Claude Code CLI timed out after {CLAUDE_TIMEOUT_S}s."
        ) from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        # Surface auth failures clearly so the system pool can mark the token.
        raise AIProviderError(
            f"Claude Code CLI exited {proc.returncode}: {stderr[:300] or 'no stderr'}"
        )

    raw = (proc.stdout or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise AIProviderError(
            f"Claude Code CLI returned non-JSON output: {raw[:300]}"
        ) from e

    # `--output-format json` → {type, subtype, is_error, result, usage, ...}.
    if data.get("is_error") or data.get("subtype") not in (None, "success"):
        raise AIProviderError(
            f"Claude Code CLI error (subtype={data.get('subtype')}): "
            f"{str(data.get('result'))[:300]}"
        )

    response_text = str(data.get("result", "")).strip()

    return AIResponse(
        provider=spec.id,
        model=model,
        response=response_text,
        usage=_usage_from_cli(data.get("usage")),
        truncated=False,
    ).as_dict()
