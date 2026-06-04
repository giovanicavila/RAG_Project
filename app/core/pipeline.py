from app.core.agent.grader import grader
from app.core.agent.rewriter import rewriter
from app.core.agent.tools import retrieval_tool
from app.core.generation.factory import llm          # ✅ era: generator
from app.core.prompts.templates import RAG_SYSTEM_PROMPT, build_context
from app.models.schemas import QueryResponse

MAX_RETRIEVAL_ATTEMPTS = 3


def run_rag_pipeline(question: str, top_k: int | None = None) -> QueryResponse:

    current_query = question
    sources = []

    for attempt in range(MAX_RETRIEVAL_ATTEMPTS):
        sources = retrieval_tool.run(current_query, top_k)

        if not sources:
            current_query = rewriter.rewrite(current_query)
            continue

        context = "\n".join(doc.content for doc in sources)

        if grader.grade(question, context):
            break

        current_query = rewriter.rewrite(current_query)

    if not sources:
        return QueryResponse(
            answer="No relevant information found.", sources=[], question=question
        )

    # Build prompt and generate — llm.generate() takes a plain string
    prompt = f"{RAG_SYSTEM_PROMPT.format(context=build_context(sources))}\n\nQuestion: {question}"
    answer = llm.generate(prompt)                    

    return QueryResponse(answer=answer, sources=sources, question=question)