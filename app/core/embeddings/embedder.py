from google import genai
from google.genai import types
from config import settings

client = genai.Client(api_key=settings.gemini_api_key)

class Embedder:
    def __init__(self):
        self.model_name = settings.embedding_model  

    def embed_text(self, text: str) -> list[float]:
        response = client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            ),
        )
        return response.embeddings[0].values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                ),
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

embedder = Embedder()