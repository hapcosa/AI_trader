"""Registry coverage for the Moonshot/Kimi provider (free-tier pool primary)."""

from pineforge_ai.ai_clients.registry import (
    get_provider_spec,
    normalize_provider,
    supported_providers,
)


def test_kimi_normalizes_to_moonshot():
    assert normalize_provider("kimi") == "moonshot"
    assert normalize_provider("KIMI") == "moonshot"
    assert normalize_provider("moonshot") == "moonshot"


def test_moonshot_is_registered():
    assert "moonshot" in supported_providers()


def test_moonshot_spec_shape():
    spec = get_provider_spec("kimi")  # alias resolves to moonshot
    assert spec.id == "moonshot"
    assert spec.env_var == "MOONSHOT_API_KEY"
    assert spec.default_model.startswith("kimi-")
    assert spec.models  # at least one model offered


def test_existing_providers_still_present():
    providers = set(supported_providers())
    assert {"anthropic", "openai", "gemini", "deepseek"} <= providers
