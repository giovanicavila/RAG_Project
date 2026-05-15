from chonkie import SemanticChunker

# Basic initialization with default parameters
chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-32M",  # Default model
    threshold=0.8,                               # Similarity threshold (0-1)
    chunk_size=2048,                             # Maximum tokens per chunk
    similarity_window=3,                         # Window for similarity calculation
    skip_window=0                                # Skip-and-merge window (0=disabled)
)
