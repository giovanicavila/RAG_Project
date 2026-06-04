# Works for any provider with an OpenAI-compatible embeddings endpoint.
# Examples: OpenAI, OpenRouter, local servers (LM Studio, vLLM).

from openai import OpenAI
from app.core.embeddings.base import BaseEmbedder
from config import settings


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self):
        self._client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.llm_base_url or None,
            # base_url=None → uses the default OpenAI endpoint.
            # Set EMBEDDING_BASE_URL in .env to point to any compatible server.
        )
        self._model = settings.embedding_model

    def embed_text(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # OpenAI API accepts a list directly — one request for the whole batch.
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        # Sort by index to guarantee order matches input list.
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
