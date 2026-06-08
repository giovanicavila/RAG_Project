import logging

import chromadb
from rank_bm25 import BM25Okapi

from app.core.embeddings.factory import embedder
from app.models.schemas import SourceDocument
from config import settings

log = logging.getLogger(__name__)


RRF_K = settings.rrf_k


class VectorRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.vector_store_persist_directory,
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.vector_store_collection_name,
            metadata={"hnsw:space": settings.vector_store_similarity_metric},
        )

    def search(self, query: str, top_k: int | None = None) -> list[SourceDocument]:
        k = top_k or settings.top_k
        query_embedding = embedder.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return []

        sources = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sources.append(
                SourceDocument(
                    id=doc_id,
                    content=doc,
                    source=meta.get("source", "unknown"),
                    score=round(1 - dist, 4),
                )
            )
        return sources


class BM25Retriever:
    def __init__(self):
        self._documents: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._dirty = False

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def add_documents(
        self, texts: list[str], metadatas: list[dict], ids: list[str]
    ) -> None:
        for text, meta, id_ in zip(texts, metadatas, ids):
            self._documents.append({"id": id_, "text": text, "metadata": meta})
        self._dirty = True
        log.info("BM25: added %d documents (total: %d)", len(texts), len(self._documents))

    def load_from_vector_store(self, vector_retriever: VectorRetriever) -> None:
        all_docs = vector_retriever.collection.get(
            include=["documents", "metadatas"]
        )
        for doc_id, doc, meta in zip(
            all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
        ):
            self._documents.append({"id": doc_id, "text": doc, "metadata": meta})
        self._dirty = True
        log.info("BM25: loaded %d documents from vector store", len(self._documents))

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_index(self):
        corpus = [self._tokenize(doc["text"]) for doc in self._documents]
        self._bm25 = BM25Okapi(corpus)
        self._dirty = False
        log.info("BM25: index rebuilt with %d documents", len(self._documents))

    def search(self, query: str, top_k: int | None = None) -> list[SourceDocument]:
        if not self._documents:
            return []

        if self._dirty or self._bm25 is None:
            self._build_index()

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        k = top_k or len(self._documents)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        max_score = max(scores)
        if max_score <= 0:
            return []

        results = []
        for idx in top_indices:
            doc = self._documents[idx]
            results.append(
                SourceDocument(
                    id=doc["id"],
                    content=doc["text"],
                    source=doc["metadata"].get("source", "unknown"),
                    score=round(scores[idx] / max_score, 4),
                )
            )
        return results


class HybridRetriever:
    def __init__(self):
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()
        self.bm25.load_from_vector_store(self.vector)

    def add_documents(
        self, texts: list[str], metadatas: list[dict], ids: list[str]
    ) -> None:
        self.bm25.add_documents(texts, metadatas, ids)

    def search(self, query: str, top_k: int | None = None) -> list[SourceDocument]:
        k = top_k or settings.top_k
        if settings.search_mode == "vector":
            return self.vector.search(query, top_k=k)
        if settings.search_mode == "lexical":
            return self.bm25.search(query, top_k=k)
        vector_results = self.vector.search(query, top_k=k * 2)
        bm25_results = self.bm25.search(query, top_k=k * 2)
        return self._rrf_fuse(vector_results, bm25_results, k)

    def _rrf_fuse(
        self,
        vector_results: list[SourceDocument],
        bm25_results: list[SourceDocument],
        top_k: int,
    ) -> list[SourceDocument]:
        doc_map: dict[str, SourceDocument] = {}
        rrf_scores: dict[str, float] = {}

        all_ids = set()
        seen = set()
        for rank, doc in enumerate(vector_results):
            all_ids.add(doc.id)
            if doc.id not in seen:
                doc_map[doc.id] = doc
                seen.add(doc.id)
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + 1.0 / (RRF_K + rank + 1)

        seen.clear()
        for rank, doc in enumerate(bm25_results):
            all_ids.add(doc.id)
            if doc.id not in seen:
                doc_map[doc.id] = doc
                seen.add(doc.id)
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + 1.0 / (RRF_K + rank + 1)

        reranked = sorted(
            all_ids, key=lambda doc_id: rrf_scores[doc_id], reverse=True
        )

        result = []
        for doc_id in reranked[:top_k]:
            doc = doc_map[doc_id]
            doc.score = round(rrf_scores[doc_id], 4)
            result.append(doc)
        return result


retriever = HybridRetriever()
