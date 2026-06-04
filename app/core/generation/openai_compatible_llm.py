# Covers any provider that exposes an OpenAI-compatible chat completions API.
# This includes: OpenAI, OpenRouter, Ollama, LM Studio, vLLM, Groq, etc.
# The only difference between providers is the base_url and api_key.

from openai import OpenAI
from app.core.generation.base import BaseLLM
from config import settings

_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama":     "http://localhost:11434/v1",
    "openai":     None,   # None → uses the default OpenAI endpoint
}


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self):
        base_url = settings.llm_base_url or _BASE_URLS.get(settings.llm_provider)
        api_key  = settings.llm_api_key or "no-key-needed"
        # "no-key-needed" is a placeholder for local providers (ollama, lm-studio)
        # that require a non-empty string but ignore the actual value.

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model  = settings.llm_model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
        return response.choices[0].message.content
