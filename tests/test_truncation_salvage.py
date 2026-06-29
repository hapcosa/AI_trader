"""Truncation detection + partial-JSON salvage (PR-1 AI resilience).

Mirrors the dashboard frontend `salvageEntries` behaviour so a free-tier model
that cuts its output mid-array still yields the complete setups instead of
throwing away the whole response.
"""
from pineforge_ai.ai_clients.base import is_truncated_reason
from pineforge_ai.prompt_builder import salvage_entries


def test_is_truncated_reason_markers():
    # Provider-specific cutoff reasons across the adapters we speak to.
    assert is_truncated_reason("max_tokens")            # Anthropic
    assert is_truncated_reason("length")                # OpenAI ChatCompletions
    assert is_truncated_reason("max_output_tokens")     # OpenAI Responses
    assert is_truncated_reason("MAX_TOKENS")            # Gemini (string)
    assert is_truncated_reason("FinishReason.MAX_TOKENS")  # Gemini enum str()


def test_is_truncated_reason_non_cutoff():
    for reason in (None, "", "stop", "end_turn", "tool_use", "content_filter"):
        assert not is_truncated_reason(reason)


# A blob cut off mid-way through the SECOND entry (the real bug shape).
TRUNCATED = (
    '{"market_state":{"phase":"markdown","confidence":7},'
    '"entries":['
    '{"id":1,"direction":"short","entry_zone":[62819.44,63577.91],'
    '"stop_loss":64550,"take_profit_1":60800,"take_profit_2":60037.6,'
    '"execution_tf":"1H","confidence":7},'
    '{"id":2,"direction":"long","entry_zone":[60785.63,60885.97],'
    '"stop_loss":59900,"confluence_factors":["1H Bullish Order Block [60,785.63'
)


def test_salvage_recovers_complete_entries_from_truncated_blob():
    out = salvage_entries(TRUNCATED)
    assert len(out) == 1  # only entry #1 closed; entry #2 is the truncated tail
    assert out[0]["id"] == 1
    assert out[0]["direction"] == "short"


def test_salvage_handles_braces_inside_strings():
    # A string value containing { } [ ] must not confuse the brace walk.
    blob = (
        '{"entries":['
        '{"id":1,"note":"OB [60,785] and {curly} braces","direction":"long"},'
        '{"id":2,"note":"cut off here {'
    )
    out = salvage_entries(blob)
    assert len(out) == 1
    assert out[0]["id"] == 1
    assert "{curly}" in out[0]["note"]


def test_salvage_no_entries_key_returns_empty():
    assert salvage_entries('{"market_state":{"phase":"ranging"}}') == []
