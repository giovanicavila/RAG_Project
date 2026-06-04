# app/core/agent/tools.py
# Already provider-agnostic — depends only on retriever, not on any LLM SDK.
# Minor improvements: type hints and docstring added.

from app.core.retrieval.retriever import retriever
from app.models.schemas import SourceDocument


class RetrievalTool:
    def run(self, query: str, top_k: int | None = None) -> list[SourceDocument]:
        """
        Searches ChromaDB for the most relevant chunks for a given query.
        Returns a list of SourceDocument with content, source, and score.
        """
        return retriever.search(query, top_k=top_k)


retrieval_tool = RetrievalTool()