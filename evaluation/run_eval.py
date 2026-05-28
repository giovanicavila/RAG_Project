"""
scripts/evaluation/run_eval.py

Main evaluation script for RAGAs.

Basic usage:
  python -m scripts.evaluation.run_eval

Options:
  python -m scripts.evaluation.run_eval \
    --questions scripts/evaluation/data/questions.json \
    --mode full \
    --output scripts/evaluation/results/

Available modes (--mode):
  quick     → faithfulness + response_relevancy + context_precision (without ground_truth)
  retrieval → context_precision + context_recall
  generation→ faithfulness + response_relevancy
  full      → all metrics (require ground_truth in all samples)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ragas import evaluate

from scripts.evaluation.dataset import build_evaluation_dataset
from scripts.evaluation.metrics import (
    EMBEDDINGS,
    FULL_METRICS,
    GENERATION_METRICS,
    QUICK_METRICS,
    RETRIEVAL_METRICS,
)

# ── Configurations default ──────────────────────────────────────────────────────

DEFAULT_QUESTIONS_PATH = Path("scripts/evaluation/data/questions.json")
DEFAULT_OUTPUT_DIR = Path("scripts/evaluation/results")

MODES = {
    "quick": QUICK_METRICS,
    "retrieval": RETRIEVAL_METRICS,
    "generation": GENERATION_METRICS,
    "full": FULL_METRICS,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _save_results(results, output_dir: Path, mode: str) -> Path:
    """Saves the results in JSON with timestamp in file name."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{mode}_{timestamp}.json"

    # Convert RAGAs evaluation results to serializable dict
    scores = results.scores  # list of dicts per sample
    agg = results.to_pandas().mean(numeric_only=True).to_dict()

    payload = {
        "mode": mode,
        "timestamp": timestamp,
        "aggregate_scores": {k: round(v, 4) for k, v in agg.items()},
        "per_sample_scores": [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
            for s in scores
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path


def _print_summary(results, mode: str) -> None:
    """Prints a readable summary of the aggregated scores."""
    df = results.to_pandas()
    agg = df.mean(numeric_only=True)

    print("\n" + "=" * 50)
    print(f"  RAGAs — Results ({mode.upper()})")
    print("=" * 50)
    for metric, score in agg.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<25} {score:.4f}  {bar}")
    print("=" * 50 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def run(
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    mode: str = "quick",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
) -> dict:
    """
    Executes the full RAGAs evaluation.

    Args:
        questions_path: path to the .json or .csv file with the questions.
        mode: group of metrics to use (quick | retrieval | generation | full).
        output_dir: directory where the results will be saved.
        verbose: prints progress in terminal.

    Returns:
        dict with aggregate_scores and saved file path.
    """
    if mode not in MODES:
        raise ValueError(f"Invalid mode: '{mode}'. Choose between: {list(MODES)}")

    metrics = MODES[mode]

    if verbose:
        print(f"\n→ Mode: {mode.upper()} | Metrics: {[m.name for m in metrics]}")
        print(f"→ Questions: {questions_path}\n")

    # 1. Build dataset
    dataset = build_evaluation_dataset(questions_path, verbose=verbose)

    # 2. Run evaluation
    if verbose:
        print("\n→ Running ragas.evaluate()...\n")

    results = evaluate(dataset=dataset, metrics=metrics, embeddings=EMBEDDINGS)

    # 3. Show and save
    _print_summary(results, mode)

    output_path = _save_results(results, Path(output_dir), mode)

    if verbose:
        print(f"→ Results saved in: {output_path}\n")

    agg = results.to_pandas().mean(numeric_only=True).to_dict()
    return {"aggregate_scores": agg, "output_path": str(output_path)}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAs evaluation script for the project RAG"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(DEFAULT_QUESTIONS_PATH),
        help="Path to the .json or .csv file with the questions",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=list(MODES),
        help="Group of metrics: quick | retrieval | generation | full",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save the results",
    )
    args = parser.parse_args()

    run(
        questions_path=args.questions,
        mode=args.mode,
        output_dir=args.output,
    )
