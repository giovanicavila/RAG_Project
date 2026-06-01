from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample

from app.core.generation.generator import generator
from app.core.retrieval.retriever import retriever

from .loader import load_questions


def build_dataset(path: str | Path) -> EvaluationDataset:
    questions = load_questions(path)

    samples = []

    for item in questions:
        question = item["question"]
        ground_truth = item.get("ground_truth")

        sources = retriever.search(question)

        response = generator.generate(
            question=question,
            sources=sources,
        )

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=[s.content for s in sources],
                response=response,
                reference=ground_truth,
            )
        )

    return EvaluationDataset(samples=samples)