from chonkie import SemanticChunker
from config import settings


chunker = SemanticChunker(
    embedding_model=settings.embedding_model,  
    threshold=0.75,
    chunk_size=512,
    similarity_window=3
)
