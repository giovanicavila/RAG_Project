from app.core.generation.base import BaseLLM
from config import settings


def get_llm() -> BaseLLM:
    provider = settings.llm_provider

    if provider in ("openrouter", "ollama", "openai"):
        from app.core.generation.openai_compatible_llm import OpenAICompatibleLLM
        return OpenAICompatibleLLM()

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        f"Valid options: openrouter, ollama, openai"
    )


# Singleton — instantiated once at startup.
llm: BaseLLM = get_llm()
