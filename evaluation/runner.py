from pathlib import Path

from ragas import evaluate

from evaluation.datasets.builder import build_dataset
from evaluation.metrics.groups import MODES
from evaluation.metrics.llm import embeddings
from evaluation.serializers import save_results


def run_evaluation(
    questions_path: str | Path,
    mode: str,
    output_dir: str | Path,
):
    metrics = MODES[mode]

    dataset = build_dataset(questions_path)

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        embeddings=embeddings,
    )

    output = save_results(
        results=results,
        output_dir=Path(output_dir),
        mode=mode,
    )

    return {
        "results": results,
        "output": output,
    }