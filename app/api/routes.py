import uuid

from app.core.chunker.chunker import chunker
from fastapi import APIRouter, HTTPException

from app.core.pipeline import run_rag_pipeline
from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from scripts.ingestion.ingest import ingestion

router = APIRouter()
# APIRouter acts like a mini-app inside FastAPI.
# Routes are registered here and mounted in main.py with a prefix.


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Receives a question and returns the RAG-generated answer."""
    try:
        return run_rag_pipeline(
            question=request.question,
            top_k=request.top_k,  # None if not sent by the client
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse)
async def ingest_text(request: IngestRequest) -> IngestResponse:
    try:
        semantic_chunks = chunker.chunk(request.text)

        chunks = [chunk.text for chunk in semantic_chunks]

        metadatas = [
            {
                "source": request.source_name,
                "chunk_index": i,
                "token_count": semantic_chunks[i].token_count,
            }
            for i in range(len(chunks))
        ]

        ids = [f"{request.source_name}-{uuid.uuid4()}" for _ in chunks]
        # IDs must be unique across the entire collection.
        # uuid4() generates a random ID that is guaranteed to be unique.

        ingestion.add_documents(texts=chunks, metadatas=metadatas, ids=ids)

        return IngestResponse(
            message="Text indexed successfully",
            chunks_created=len(chunks),
            source=request.source_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    return {"status": "ok"}
