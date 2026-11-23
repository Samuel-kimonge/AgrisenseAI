"""Configurable LLM provider factory. Returns None when no provider is configured."""
from __future__ import annotations

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini", "ollama": "llama3.2"}


class LLMConfigError(RuntimeError):
    pass


def get_llm(settings):
    provider = settings.llm_provider
    if provider in ("", "none"):
        return None
    model = settings.llm_model or DEFAULT_MODELS.get(provider)
    try:
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                raise LLMConfigError("ANTHROPIC_API_KEY is not set.")
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=settings.llm_temperature, max_tokens=900)
        if provider == "openai":
            if not settings.openai_api_key:
                raise LLMConfigError("OPENAI_API_KEY is not set.")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=settings.llm_temperature, max_tokens=900)
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=settings.llm_temperature)
    except ImportError as exc:
        pkg = {"anthropic": "langchain-anthropic", "openai": "langchain-openai", "ollama": "langchain-ollama"}[provider]
        raise LLMConfigError(f"LLM_PROVIDER={provider} needs `pip install {pkg}` ({exc}).") from exc
    raise LLMConfigError(f"Unknown LLM_PROVIDER: {provider!r}. Use none, anthropic, openai or ollama.")
