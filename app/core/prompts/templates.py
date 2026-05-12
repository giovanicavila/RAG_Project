RAG_SYSTEM_PROMPT = """You are a helpful and precise assistant.
Answer ONLY based on the context provided below.
If the answer is not in the context, clearly state that
you could not find the information — never make things up.

Context:
{context}"""
# {context} is dynamically filled with the retrieved chunks.


def build_context(sources: list) -> str:
    """Formats retrieved chunks into a numbered context block."""
    parts = []
    for i, source in enumerate(sources, start=1):
        parts.append(f"[{i}] Source: {source.source}\n{source.content}")
    return "\n\n---\n\n".join(parts)
    # The "---" separator makes it clear to the LLM where one chunk ends
    # and the next begins.
