from app.core.retrieval.retriever import retriever
from app.core.generation.generator import generator
from app.models.schemas import QueryResponse

def run_rag_pipeline(question: str, top_k: int | None = None) -> QueryResponse:
    """
    Orchestrates the full RAG flow:
    1. Retrieval  → finds relevant chunks in ChromaDB
    2. Generation → generates an answer with Gemini
    3. Returns a QueryResponse with the answer + source documents
    """

    # ── 1. Retrieval ──────────────────────────────────────
    sources = retriever.search(question, top_k=top_k)

    if not sources:
        # No context found → skip the LLM call entirely.
        # Avoids hallucination and wastes no tokens.
        return QueryResponse(
            answer="I could not find relevant information to answer your question.",
            sources=[],
            question=question,
        )

    # ── 2. Generation ─────────────────────────────────────
    answer = generator.generate(question=question, sources=sources)

    # ── 3. Return result ──────────────────────────────────
    return QueryResponse(answer=answer, sources=sources, question=question)