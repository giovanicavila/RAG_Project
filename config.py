from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["openrouter", "ollama", "openai"] = "openrouter"
    llm_model: str = "nousresearch/hermes-3-llama-3.1-405b:free"
    llm_api_key: str = ""
    llm_base_url: str = ""
    # llm_base_url is optional — used to point to custom endpoints.
    # openrouter: https://openrouter.ai/api/v1
    # ollama:     http://localhost:11434/v1  (auto-set if provider=ollama)

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_provider: Literal["sentence_transformers", "openai", "openrouter"] = (
        "sentence_transformers"
    )
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_api_key: str = ""
    # sentence_transformers: runs locally, no key needed.
    # openai / openrouter: api key required.

    # ── Vector Store ──────────────────────────────────────────────────────────
    vector_store_provider: Literal["chromadb"] = "chromadb"
    vector_store_persist_directory: str = "./chroma_db"
    vector_store_collection_name: str = "documents"
    vector_store_similarity_metric: Literal["cosine", "l2", "ip"] = "cosine"

    # ── RAG ───────────────────────────────────────────────────────────────────
    top_k: int = 4
    max_tokens: int = 1000
    temperature: float = 0.0

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        requires_key = {
            "llm": ["openrouter", "openai"],
            "embedding": ["openai", "openrouter"],
        }
        if self.llm_provider in requires_key["llm"] and not self.llm_api_key:
            raise ValueError(
                f"LLM_API_KEY is required for provider '{self.llm_provider}'"
            )
        if (
            self.embedding_provider in requires_key["embedding"]
            and not self.embedding_api_key
        ):
            raise ValueError(
                f"EMBEDDING_API_KEY is required for provider '{self.embedding_provider}'"
            )
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
