import google.generativeai as genai
from config import settings

genai.configure(api_key=settings.gemini_api_key)

class Embedder:
    def __init__(self):
        self.model_name = settings.embedding_model # config.py: "models/text-embedding-004"

    def embed_query(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            text = text.replace("\n", " ")
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings


embedder = Embedder()