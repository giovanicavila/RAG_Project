import uuid
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse
)
from app.core.pipeline import run_rag_pipeline
from app.core.retrieval.retriever import retriever

router = APIRouter()
# APIRouter acts like a mini-app inside FastAPI.
# Routes are registered here and mounted in main.py with a prefix.

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Receives a question and returns the RAG-generated answer."""
    try:
        return run_rag_pipeline(
            question=request.question,
            top_k=request.top_k,   # None if not sent by the client
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse)
async def ingest_text(request: IngestRequest) -> IngestResponse:
    """Indexes a raw text string into ChromaDB via the API."""
    try:
        # Split text into overlapping chunks
        chunk_size, overlap = 500, 50
        # overlap ensures sentences at chunk boundaries appear in both
        # adjacent chunks, so nothing gets lost at the edges.
        chunks, start = [], 0
        while start < len(request.text):
            chunks.append(request.text[start:start + chunk_size])
            start += chunk_size - overlap

        metadatas = [
            {"source": request.source_name, "chunk_index": i}
            for i in range(len(chunks))
        ]
        ids = [f"{request.source_name}-{uuid.uuid4()}" for _ in chunks]
        # IDs must be unique across the entire collection.
        # uuid4() generates a random ID that is guaranteed to be unique.

        retriever.add_documents(texts=chunks, metadatas=metadatas, ids=ids)

        return IngestResponse(
            message="Text indexed successfully",
            chunks_created=len(chunks),
            source=request.source_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    # Simple liveness check. Used by monitoring tools and Docker health checks.
    return {"status": "ok"}