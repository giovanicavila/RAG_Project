from chonkie import SemanticChunker


chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-32M",  
    threshold=0.75,
    chunk_size=512,
    similarity_window=3
)
