from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from config import settings


llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.gemini_api_key,
        temperature=0,
    )
)

embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
    )
)