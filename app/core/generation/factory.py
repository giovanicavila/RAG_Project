from app.core.generation.base import BaseLLM
from config import settings

_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama":     "http://localhost:11434/v1",
    "openai":     None,
}


def _build_llm(provider: str, model: str, api_key: str, base_url: str) -> BaseLLM:
    """Internal helper — builds an OpenAICompatibleLLM with explicit parameters."""
    from app.core.generation.openai_compatible_llm import OpenAICompatibleLLM
    return OpenAICompatibleLLM(
        model=model,
        api_key=api_key or "no-key-needed",
        base_url=base_url or _BASE_URLS.get(provider),
    )


def get_llm() -> BaseLLM:
    """Returns the generation LLM (used for final answer)."""
    if settings.llm_provider in ("openrouter", "ollama", "openai"):
        return _build_llm(
            provider=settings.llm_provider,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    raise ValueError(f"Unknown LLM provider: '{settings.llm_provider}'")


def get_agent_llm() -> BaseLLM:
    """Returns the agent LLM (used for planner, grader, rewriter)."""
    if settings.agent_llm_provider in ("openrouter", "ollama", "openai"):
        return _build_llm(
            provider=settings.agent_llm_provider,
            model=settings.agent_llm_model,
            api_key=settings.agent_llm_api_key,
            base_url=settings.agent_llm_base_url,
        )
    raise ValueError(f"Unknown agent LLM provider: '{settings.agent_llm_provider}'")


# Singletons — instantiated once at startup.
llm: BaseLLM = get_llm()
agent_llm: BaseLLM = get_agent_llm()