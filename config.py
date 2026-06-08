from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Literal


class Settings(BaseSettings):
    # ── LLM — generation ─────────────────────────────────────────────────────
    llm_provider: Literal["openrouter", "ollama", "openai"] = "openrouter"
    llm_model: str = "moonshotai/kimi-k2"
    llm_api_key: str = ""
    llm_base_url: str = ""

    # ── LLM — agent (planner, grader, rewriter) ───────────────────────────────
    agent_llm_provider: Literal["openrouter", "ollama", "openai"] = "openrouter"
    agent_llm_model: str = "nousresearch/hermes-3-llama-3.1-405b:free"
    agent_llm_api_key: str = ""
    agent_llm_base_url: str = ""

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_provider: Literal["sentence_transformers", "openai", "openrouter"] = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_api_key: str = ""

    # ── Vector Store ──────────────────────────────────────────────────────────
    vector_store_provider: Literal["chromadb"] = "chromadb"
    vector_store_persist_directory: str = "./chroma_db"
    vector_store_collection_name: str = "documents"
    vector_store_similarity_metric: Literal["cosine", "l2", "ip"] = "cosine"

    # ── Search / RRF ──────────────────────────────────────────────────────────
    search_mode: Literal["hybrid", "vector", "lexical"] = "hybrid"
    rrf_k: int = 60

    # ── RAG ───────────────────────────────────────────────────────────────────
    top_k: int = 4
    max_tokens: int = 1000
    temperature: float = 0.0

    @model_validator(mode="after")
    def apply_defaults_and_validate(self) -> "Settings":
        # If agent key is not set, reuse the main LLM key.
        if not self.agent_llm_api_key:
            self.agent_llm_api_key = self.llm_api_key

        # If agent base_url is not set, reuse the main LLM base_url.
        if not self.agent_llm_base_url:
            self.agent_llm_base_url = self.llm_base_url

        requires_key = {
            "llm": ["openrouter", "openai"],
            "embedding": ["openai", "openrouter"],
        }
        if self.llm_provider in requires_key["llm"] and not self.llm_api_key:
            raise ValueError(f"LLM_API_KEY is required for provider '{self.llm_provider}'")
        if self.agent_llm_provider in requires_key["llm"] and not self.agent_llm_api_key:
            raise ValueError(f"AGENT_LLM_API_KEY is required for provider '{self.agent_llm_provider}'")
        if self.embedding_provider in requires_key["embedding"] and not self.embedding_api_key:
            raise ValueError(f"EMBEDDING_API_KEY is required for provider '{self.embedding_provider}'")

        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()