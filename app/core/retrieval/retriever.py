import chromadb
from app.core.embeddings.embedder import embedder
from app.models.schemas import SourceDocument
from config import settings

class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            # Persists vectors to disk at ./chroma_db.
            # Use chromadb.Client() for in-memory storage (tests only).
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
            # "cosine" = cosine similarity — standard for text embeddings.
            # hnsw = Hierarchical Navigable Small World, a fast approximate
            # nearest-neighbor search algorithm.
        )

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str]
    ) -> None:
        """Indexes chunks into ChromaDB. Called by the ingestion script."""
        embeddings = embedder.embed_documents(texts)
        # Generates all embeddings with task_type=RETRIEVAL_DOCUMENT.

        self.collection.upsert(
            documents=texts,        # raw text stored for retrieval
            embeddings=embeddings,  # vectors used for similarity search
            metadatas=metadatas,    # source file, page number, etc.
            ids=ids                 # unique ID per chunk
        )

    def search(self, query: str, top_k: int | None = None) -> list[SourceDocument]:
        """
        Finds the most relevant chunks for a given query.

        Internal flow:
        1. embed_text(query) with task_type=RETRIEVAL_QUERY
        2. ChromaDB computes cosine distance against all stored vectors
        3. Returns the top_k closest matches
        """
        k = top_k or settings.top_k
        query_embedding = embedder.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        sources = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            sources.append(SourceDocument(
                id=doc_id,
                content=doc,
                source=meta.get("source", "unknown"),
                score=round(1 - dist, 4),
                # ChromaDB returns cosine distance (0 = identical, 2 = opposite).
                # We convert to similarity score: score = 1 - distance
                # → 1.0 = identical, ~0.0 = irrelevant
            ))
        return sources

retriever = Retriever()