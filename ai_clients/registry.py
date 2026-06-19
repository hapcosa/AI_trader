"""Provider and model registry for AI Trader."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PROVIDER = "anthropic"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label}


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    name: str
    env_var: str
    key_label: str
    key_placeholder: str
    default_model: str
    models: tuple[ModelSpec, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "env_var": self.env_var,
            "key_label": self.key_label,
            "key_placeholder": self.key_placeholder,
            "default_model": self.default_model,
            "models": [m.as_dict() for m in self.models],
        }


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        id="anthropic",
        name="Claude",
        env_var="ANTHROPIC_API_KEY",
        key_label="Anthropic API Key",
        key_placeholder="sk-ant-...",
        default_model="claude-sonnet-4-6",
        models=(
            ModelSpec("claude-sonnet-4-6", "Sonnet 4.6 - recomendado"),
            ModelSpec("claude-opus-4-7", "Opus 4.7 - analisis profundo"),
            ModelSpec("claude-haiku-4-5-20251001", "Haiku 4.5 - rapido"),
        ),
    ),
    "openai": ProviderSpec(
        id="openai",
        name="ChatGPT",
        env_var="OPENAI_API_KEY",
        key_label="OpenAI API Key",
        key_placeholder="sk-...",
        default_model="gpt-5.5",
        models=(
            ModelSpec("gpt-5.5", "GPT-5.5 - maximo razonamiento"),
            ModelSpec("gpt-5.4", "GPT-5.4 - balanceado"),
            ModelSpec("gpt-5.4-mini", "GPT-5.4 mini - rapido"),
            ModelSpec("gpt-5.4-nano", "GPT-5.4 nano - economico"),
        ),
    ),
    "gemini": ProviderSpec(
        id="gemini",
        name="Gemini",
        env_var="GEMINI_API_KEY",
        key_label="Gemini API Key",
        key_placeholder="AIza...",
        default_model="gemini-2.5-pro",
        models=(
            ModelSpec("gemini-2.5-pro", "Gemini 2.5 Pro - razonamiento"),
            ModelSpec("gemini-2.5-flash", "Gemini 2.5 Flash - rapido"),
            ModelSpec("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite - economico"),
        ),
    ),
    "deepseek": ProviderSpec(
        id="deepseek",
        name="DeepSeek",
        env_var="DEEPSEEK_API_KEY",
        key_label="DeepSeek API Key",
        key_placeholder="sk-...",
        default_model="deepseek-v4-flash",
        models=(
            ModelSpec("deepseek-v4-flash", "DeepSeek V4 Flash - recomendado"),
            ModelSpec("deepseek-v4-pro", "DeepSeek V4 Pro - mayor capacidad"),
        ),
    ),
    # Moonshot / Kimi — OpenAI-compatible endpoint (api.moonshot.ai/v1). Cheap
    # K2 models, strong reasoning; the natural primary for the system free-tier
    # pool. Key from platform.moonshot.ai (NOT the consumer Kimi app sub).
    "moonshot": ProviderSpec(
        id="moonshot",
        name="Kimi",
        env_var="MOONSHOT_API_KEY",
        key_label="Moonshot (Kimi) API Key",
        key_placeholder="sk-...",
        default_model="kimi-k2-0711-preview",
        models=(
            ModelSpec("kimi-k2-0711-preview", "Kimi K2 - recomendado"),
            ModelSpec("kimi-k2-turbo-preview", "Kimi K2 Turbo - rapido"),
            ModelSpec("moonshot-v1-128k", "Moonshot v1 128k - contexto largo"),
        ),
    ),
}


def supported_providers() -> tuple[str, ...]:
    return tuple(PROVIDERS)


def normalize_provider(provider: str | None) -> str:
    value = (provider or DEFAULT_PROVIDER).strip().lower()
    if value == "claude":
        value = "anthropic"
    if value == "chatgpt":
        value = "openai"
    if value == "kimi":
        value = "moonshot"
    return value


def get_provider_spec(provider: str | None) -> ProviderSpec:
    normalized = normalize_provider(provider)
    try:
        return PROVIDERS[normalized]
    except KeyError as e:
        valid = ", ".join(supported_providers())
        raise ValueError(f"Unsupported AI provider '{provider}'. Use one of: {valid}.") from e


def provider_options_payload() -> dict[str, object]:
    return {
        "default_provider": DEFAULT_PROVIDER,
        "providers": [PROVIDERS[key].as_dict() for key in supported_providers()],
    }


def provider_catalog() -> list[dict[str, object]]:
    """Flat list of providers, shape consumed by dashboard /api/ai/options.

    Each entry:
        {id, label, env_var, default_model,
         models: [{id, label}, ...]}

    The dashboard decorates each row with `available` (per-user) before
    serving the frontend; this catalog itself is user-agnostic and
    cacheable.
    """
    return [
        {
            "id": spec.id,
            "label": spec.name,
            "env_var": spec.env_var,
            "default_model": spec.default_model,
            "models": [m.as_dict() for m in spec.models],
        }
        for spec in (PROVIDERS[key] for key in supported_providers())
    ]
