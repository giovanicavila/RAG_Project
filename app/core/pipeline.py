from app.core.agent.grader import grader
from app.core.agent.rewriter import rewriter
from app.core.agent.tools import retrieval_tool
from app.core.generation.generator import generator
from app.models.schemas import QueryResponse

MAX_RETRIEVAL_ATTEMPTS = 3


def run_rag_pipeline(question: str, top_k: int | None = None):

    current_query = question
    sources = []

    for attempt in range(MAX_RETRIEVAL_ATTEMPTS):
        sources = retrieval_tool.run(current_query, top_k)

        if not sources:
            current_query = rewriter.rewrite(current_query)
            continue

        context = "\n".join(doc.content for doc in sources)

        relevant = grader.grade(question, context)

        if relevant:
            break

        current_query = rewriter.rewrite(current_query)

    if not sources:
        return QueryResponse(
            answer="No relevant information found.", sources=[], question=question
        )

    answer = generator.generate(question=question, sources=sources)

    return QueryResponse(answer=answer, sources=sources, question=question)
