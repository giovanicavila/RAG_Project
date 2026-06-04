from app.core.embeddings.base import BaseEmbedder
from config import settings


def get_embedder() -> BaseEmbedder:
    provider = settings.embedding_provider

    if provider == "sentence_transformers":
        from app.core.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder
        return SentenceTransformersEmbedder()

    elif provider in ("openai", "openrouter"):
        from app.core.embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()

    raise ValueError(
        f"Unknown embedding provider: '{provider}'. "
        f"Valid options: sentence_transformers, openai, openrouter"
    )


# Singleton — instantiated once at startup, reused everywhere.
embedder: BaseEmbedder = get_embedder()