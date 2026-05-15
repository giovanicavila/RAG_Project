from config import settings
from app.core.embeddings.embedder import embedder
import chromadb

class Ingestion:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self, texts: list[str], metadatas: list[dict], ids: list[str]
    ) -> None:
        # Generates all embeddings with task_type=RETRIEVAL_DOCUMENT.
        embeddings = embedder.embed_documents(texts)

        self.collection.upsert(
            documents=texts,  # raw text stored for retrieval
            embeddings=embeddings,  # vectors used for similarity search
            metadatas=metadatas,  # source file, page number, etc.
            ids=ids,  # unique ID per chunk
        )

ingestion = Ingestion()