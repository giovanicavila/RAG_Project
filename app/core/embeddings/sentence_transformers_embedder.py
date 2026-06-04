# Local embedding — no API key, no internet after first download.
# Model is cached in ~/.cache/huggingface/ after first run.

from app.core.embeddings.base import BaseEmbedder
from config import settings


class SentenceTransformersEmbedder(BaseEmbedder):
    def __init__(self):
        # Lazy import — only required when this provider is active.
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(settings.embedding_model)
        # Default: "all-MiniLM-L6-v2" → 384-dim vectors, ~90MB download.

    def embed_text(self, text: str) -> list[float]:
        # normalize_embeddings=True → unit-length vectors required for cosine similarity.
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # encode() supports batches natively — faster than looping.
        return self._model.encode(texts, normalize_embeddings=True).tolist()