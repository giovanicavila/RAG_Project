import chromadb 
from config import settings

print("=== chromadb smoke teste ===\n")

# 1. Create client
client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
print(f"✓ Client created — data will be saved to: {settings.chroma_persist_dir}")

# 2. Create collection
collection = client.get_or_create_collection(
    name="test_collection",
    metadata={"hnsw:space": "cosine"}
)
print(f"✓ Collection created: '{collection.name}'")

# 3. Add documents (with fake vectors — just to test the mechanics)
collection.add(
    documents=[
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "ChromaDB is a vector database.",
    ],
    embeddings=[
        [0.1, 0.2, 0.3],   # fake 3-dimensional vectors
        [0.9, 0.1, 0.2],
        [0.5, 0.8, 0.1],
    ],
    metadatas=[
        {"source": "ml-book.txt",     "topic": "AI"},
        {"source": "python-docs.txt", "topic": "programming"},
        {"source": "chroma-docs.txt", "topic": "databases"},
    ],
    ids=["doc-1", "doc-2", "doc-3"]
)
print("✓ 3 documents indexed")
print(f"✓ Total documents in collection: {collection.count()}")

# 4. Query by vector similarity
print("\n--- Similarity Search ---")
results = collection.query(
    query_embeddings=[[0.15, 0.25, 0.28]],   # close to doc-1
    n_results=2,
    include=["documents", "metadatas", "distances"]
)

for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
), start=1):
    score = round(1 - dist, 4)
    print(f"  [{i}] score={score} | source={meta['source']}")
    print(f"      {doc}")

# 5. Filter by metadata
print("\n--- Metadata Filter ---")
filtered = collection.get(where={"topic": "AI"})
print(f"  Documents with topic='AI': {filtered['documents']}")

# 6. Cleanup
client.delete_collection("test_collection")
print("\n✓ Test collection deleted")
print("\n=== All checks passed! ChromaDB is working. ===")