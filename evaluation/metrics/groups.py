from .registry import (
    answer_accuracy,
    context_precision,
    context_recall,
    faithfulness,
    factual_correctness,
    response_relevancy,
)

QUICK = [
    faithfulness,
    response_relevancy,
    context_precision,
]

RETRIEVAL = [
    context_precision,
    context_recall,
]

GENERATION = [
    faithfulness,
    response_relevancy,
]

FULL = [
    context_precision,
    context_recall,
    faithfulness,
    response_relevancy,
    factual_correctness,
    answer_accuracy,
]

MODES = {
    "quick": QUICK,
    "retrieval": RETRIEVAL,
    "generation": GENERATION,
    "full": FULL,
}