# Covers any provider with an OpenAI-compatible chat completions API:
# OpenAI, OpenRouter, Ollama, LM Studio, vLLM, Groq, etc.

from openai import OpenAI
from app.core.generation.base import BaseLLM
from config import settings


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        # Accept explicit params (from factory) or fall back to settings.
        # This allows the same class to serve both llm and agent_llm
        # with different configs.
        self._model = model or settings.llm_model
        self._client = OpenAI(
            api_key=api_key or settings.llm_api_key or "no-key-needed",
            base_url=base_url,
        )

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
        return response.choices[0].message.content