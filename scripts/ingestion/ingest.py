from config import settings
from app.core.embeddings.factory import embedder  
import chromadb


class Ingestion:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.vector_store_persist_directory  
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.vector_store_collection_name,  
            metadata={"hnsw:space": settings.vector_store_similarity_metric}
        )

    def add_documents(
        self, texts: list[str], metadatas: list[dict], ids: list[str]
    ) -> None:
        embeddings = embedder.embed_batch(texts)          

        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )


ingestion = Ingestion()