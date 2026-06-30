"""Unit tests for the claude-code provider (Claude via subscription).

Mocks subprocess so no real `claude` CLI / token is needed: verifies the OAuth
token is injected (and ANTHROPIC_API_KEY stripped), the CLI output JSON is
mapped to the common shape, and error subtypes raise.
"""
import json

import pytest

from pineforge_ai.ai_clients import claude_code_client as cc
from pineforge_ai.ai_clients.base import AIProviderError
from pineforge_ai.ai_clients.registry import get_provider_spec


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_registry_has_claude_code():
    spec = get_provider_spec("claude-code")
    assert spec.env_var == "CLAUDE_CODE_OAUTH_TOKEN"
    assert spec.default_model == "sonnet"


def test_success_maps_result_and_usage(monkeypatch):
    captured = {}

    def fake_run(cmd, input=None, capture_output=None, text=None, env=None, timeout=None):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["env"] = env
        return _Proc(0, stdout=json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": '{"entries": []}',
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }))

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SHOULD-BE-STRIPPED")

    out = cc.call_raw(prompt="analyze BTC", api_key="sk-ant-oat01-TESTTOKEN",
                      model="opus", max_tokens=8000, system_prompt="be precise")

    assert out["provider"] == "claude-code"
    assert out["model"] == "opus"
    assert out["response"] == '{"entries": []}'
    assert out["usage"]["input_tokens"] == 100
    assert out["truncated"] is False
    # OAuth token injected; API key stripped from the child env.
    assert captured["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-TESTTOKEN"
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    # Prompt goes on stdin; system prompt + model on argv.
    assert captured["input"] == "analyze BTC"
    assert "--append-system-prompt" in captured["cmd"]
    assert "be precise" in captured["cmd"]
    assert "opus" in captured["cmd"]


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with pytest.raises(ValueError):
        cc.call_raw(prompt="x", api_key=None, model="sonnet",
                    max_tokens=8000, system_prompt="s")


def test_nonzero_exit_raises(monkeypatch):
    monkeypatch.setattr(cc.subprocess, "run",
                        lambda *a, **k: _Proc(1, stderr="invalid token"))
    with pytest.raises(AIProviderError) as ei:
        cc.call_raw(prompt="x", api_key="tok", model="sonnet",
                    max_tokens=8000, system_prompt="s")
    assert "invalid token" in str(ei.value)


def test_error_subtype_raises(monkeypatch):
    monkeypatch.setattr(cc.subprocess, "run",
                        lambda *a, **k: _Proc(0, stdout=json.dumps({
                            "subtype": "error_max_turns", "is_error": True,
                            "result": "ran out of turns"})))
    with pytest.raises(AIProviderError):
        cc.call_raw(prompt="x", api_key="tok", model="sonnet",
                    max_tokens=8000, system_prompt="s")


def test_cli_not_installed_raises(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(cc.subprocess, "run", boom)
    with pytest.raises(AIProviderError) as ei:
        cc.call_raw(prompt="x", api_key="tok", model="sonnet",
                    max_tokens=8000, system_prompt="s")
    assert "not installed" in str(ei.value).lower() or "not found" in str(ei.value).lower()


def test_non_json_output_raises(monkeypatch):
    monkeypatch.setattr(cc.subprocess, "run",
                        lambda *a, **k: _Proc(0, stdout="not json at all"))
    with pytest.raises(AIProviderError):
        cc.call_raw(prompt="x", api_key="tok", model="sonnet",
                    max_tokens=8000, system_prompt="s")
