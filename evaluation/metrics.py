"""
scripts/evaluation/metrics.py

Define the RAGA metrics groups available for the project.

Groups:
  - RETRIEVAL_METRICS   → evaluates the quality of retrieved chunks
  - GENERATION_METRICS  → evaluates the generated response quality
  - FULL_METRICS        → all above combined (requires ground_truth)
  - QUICK_METRICS       → metrics without need for ground_truth (reference)

Usage:
  from scripts.evaluation.metrics import FULL_METRICS
  result = evaluate(dataset, metrics=FULL_METRICS)
"""

from __future__ import annotations

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerAccuracy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
    ResponseRelevancy,
)

from config import settings

# ── Embedding evaluator ─────────────────────────────────────────────────────
_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
    )
)

# ── LLM evaluator ─────────────────────────────────────────────────────────────
# RAGAs uses a separate LLM to calculate the metrics.
# We use the same Gemini model already configured in the project.
_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.gemini_api_key,
        temperature=0,  # temperature 0 for deterministic evaluations
    )
)

# ── Configured metric instances with the LLM ────────────────────────────

context_precision = ContextPrecision(llm=_llm)
context_recall = ContextRecall(llm=_llm)  # requires ground_truth
faithfulness = Faithfulness(llm=_llm)
response_relevancy = ResponseRelevancy(llm=_llm)
factual_correctness = FactualCorrectness(llm=_llm)  # requires ground_truth
answer_accuracy = AnswerAccuracy(llm=_llm)  # requires ground_truth


# ── Metric groups ────────────────────────────────────────────────────────

# Evaluates only the retriever (no need for ground_truth, except context_recall)
RETRIEVAL_METRICS = [
    context_precision,
    context_recall,  # ← requires ground_truth
]

# Evaluates only the generator
GENERATION_METRICS = [
    faithfulness,
    response_relevancy,
]

# Complete evaluation — requires ground_truth in all samples
FULL_METRICS = [
    context_precision,
    context_recall,
    faithfulness,
    response_relevancy,
    factual_correctness,
    answer_accuracy,
]

# Quick — no need for ground_truth, good for initial tests
QUICK_METRICS = [
    faithfulness,
    response_relevancy,
    context_precision,
]

EMBEDDINGS = _embeddings
