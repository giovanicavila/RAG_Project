from pydantic_settings import BaseSettings

class Settings(BaseSettings):
     # ── LLM ──────────────────────────────────────────────
    gemini_api_key: str
    model_name: str = "gemini-2.5-flash"

    # ── Embeddings ───────────────────────────────────────
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768

     # ── Vector Store (ChromaDB local) ─────────────────────
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "gemini_collection"

    # ── RAG ──────────────────────────────────────────────
    top_k: int = 4
    max_tokens: int = 1000
    temperature: float = 0.0  # 0 = factual, 1 = creative       

    class Config:   
        env_file = ".env"


settings = Settings()
