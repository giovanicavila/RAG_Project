import logging

from app.core.agent.grader import grader
from app.core.agent.planner import planner
from app.core.agent.rewriter import rewriter
from app.core.agent.tools import retrieval_tool
from app.core.generation.factory import llm
from app.core.prompts.templates import RAG_SYSTEM_PROMPT, build_context
from app.models.schemas import QueryResponse

log = logging.getLogger(__name__)

MAX_RETRIEVAL_ATTEMPTS = 3

DIRECT_PROMPT = """You are a helpful assistant.
Answer the question below using your own knowledge.

Question: {question}"""


def run_rag_pipeline(question: str, top_k: int | None = None) -> QueryResponse:

    # ── 1. Planner decides: retrieve from DB or answer directly ───────────────
    action = planner.decide(question)
    log.info(f"Planner decision: '{action}' | question='{question}'")

    # ── 2. Direct answer — uses the model's parametric memory ─────────────────
    if action == "direct":
        answer = llm.generate(DIRECT_PROMPT.format(question=question))
        log.info("Answered directly from model parametric memory")
        return QueryResponse(answer=answer, sources=[], question=question)

    # ── 3. RAG — retrieval + grading loop ─────────────────────────────────────
    current_query = question
    sources = []

    for attempt in range(MAX_RETRIEVAL_ATTEMPTS):
        sources = retrieval_tool.run(current_query, top_k)
        log.info(f"Attempt {attempt + 1} | query='{current_query}' | chunks={len(sources)}")

        if not sources:
            # No chunks found — rewrite and try again
            current_query = rewriter.rewrite(current_query)
            continue

        context = "\n".join(doc.content for doc in sources)
        relevant = grader.grade(question, context)
        log.info(f"Grader: relevant={relevant}")

        if relevant:
            break

        # Context found but not relevant — rewrite and try again
        current_query = rewriter.rewrite(current_query)

    # ── 4. Fallback — nothing relevant found after all attempts ───────────────
    if not sources:
        log.warning("No relevant sources found — falling back to parametric memory")
        answer = llm.generate(DIRECT_PROMPT.format(question=question))
        return QueryResponse(answer=answer, sources=[], question=question)

    # ── 5. Generate answer grounded in retrieved context ─────────────────────
    prompt = f"{RAG_SYSTEM_PROMPT.format(context=build_context(sources))}\n\nQuestion: {question}"
    answer = llm.generate(prompt)
    return QueryResponse(answer=answer, sources=sources, question=question)