# app/core/embeddings/embedder.py
import google.generativeai as genai
import os

class Embedder:
    def __init__(self):
        """
        Configures the Google Gemini API.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("The GEMINI_API_KEY environment variable was not found.")
            
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"

    def embed_query(self, text: str) -> list[float]:
        """
        Embedding of the QUERY (Question).
        """
        # Remove newlines for consistency
        text = text.replace("\n", " ")
        
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embedding of the DOCUMENTS.
        """
        # Gemini accepts lists of strings directly (batch)
        prepared_texts = [t.replace("\n", " ") for t in texts]
        
        result = genai.embed_content(
            model=self.model_name,
            content=prepared_texts,
            task_type="retrieval_document"
        )
        return result['embedding']

# Singleton
embedder = Embedder()
