import argparse

from evaluation.config import QUESTIONS_PATH, RESULTS_PATH
from evaluation.runner import run_evaluation


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        default=str(QUESTIONS_PATH),
    )

    parser.add_argument(
        "--mode",
        default="quick",
    )

    parser.add_argument(
        "--output",
        default=str(RESULTS_PATH),
    )

    args = parser.parse_args()

    run_evaluation(
        questions_path=args.questions,
        mode=args.mode,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()