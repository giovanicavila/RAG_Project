RAG_SYSTEM_PROMPT = """You are a helpful and precise assistant. Your main goal is to answer the user's question accurately.

CRITICAL INSTRUCTIONS:
1. First, search for the answer inside the "Context" section below. If the information is there, rely ONLY on the context.
2. If the context is empty, missing, or insufficient to answer the question, you MUST use your parametric memory (your own internal knowledge) to answer.
3. When using your parametric memory, you are strictly required to prefix your response with one of the specific phrases shown in the examples below.

Review these examples to understand how to format your response when using parametric memory:

<example_1>
User Question: When did World War II begin?
Context: No relevant documents were found in the database.
Assistant Response: I could not find a related document, but I know that World War II officially began on September 1, 1939, when Germany invaded Poland.
</example_1>

<example_2>
User Question: What is the capital of France?
Context: [1] Source: document_42.txt
The economy of France is highly developed and free-market-oriented.
Assistant Response: I didn't find an adequate document to answer, however, I know that the capital of France is Paris.
</example_2>

Context:
{context}"""


def build_context(sources: list) -> str:
    """Formats retrieved chunks into a numbered context block."""
    if not sources:
        return "No relevant documents were found in the database."
    
    parts = []
    for i, source in enumerate(sources, start=1):
        parts.append(f"[{i}] Source: {source.source}\n{source.content}")
    return "\n\n---\n\n".join(parts)
