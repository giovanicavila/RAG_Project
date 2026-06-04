from abc import ABC, abstractmethod


class BaseEmbedder(ABC):

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single query string. Called at search time."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents. Called at ingestion time."""
        ...