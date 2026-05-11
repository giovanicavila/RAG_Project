from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "openai_collection"
    top_k: int = 4
    max_tokens = 1000
    temperature: float = 0.0

    class Config:
        env_file = ".env"


settings = Settings()
