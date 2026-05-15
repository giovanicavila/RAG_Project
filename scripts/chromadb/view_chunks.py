from pathlib import Path
import chromadb

path_db = Path(__file__).resolve().parents[2] / "chroma_db"

client = chromadb.PersistentClient(path=str(path_db))

collection = client.get_or_create_collection(
    name="gemini_collection",
    metadata={"hnsw:space": "cosine"}
)

results = collection.get(
    where={"source": "product-manual-v2"},
    include=["documents", "metadatas"]
)

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n--- Chunk {meta['chunk_index']} ---")
    print(f"Tokens: {meta.get('token_count', 'N/A')}")
    print(f"Texto: {doc[:120]}...")