from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="RAG API — Gemini",
    description="RAG with FastAPI + ChromaDB + Google Gemini (free tier)",
    version="0.1.0",
    # These fields populate the auto-generated docs at /docs
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in the prod -> allow_origins=["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],    # Authorization, Content-Type, etc.
)

app.include_router(
    router,
    prefix="/api/v1",
    # All routes become /api/v1/query, /api/v1/ingest, etc.
    # The prefix makes future API versioning easy (/api/v2/...).
    tags=["RAG"]
    # Tags group endpoints in the /docs UI.
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    # reload=True → restarts the server on every file save.
    # Use only in development, never in production.