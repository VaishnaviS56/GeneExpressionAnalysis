from __future__ import annotations

import argparse

from evaluators.utils import DEFAULT_RESULTS_FILE
from evaluators.multi_turn import run_multi_turn_eval
from evaluators.single_turn import run_single_turn_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEA agent LangSmith evaluations.")
    parser.add_argument(
        "suite",
        choices=("single", "multi", "all"),
        help="Evaluation suite to run.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Use existing LangSmith datasets without creating/updating examples.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=0,
        help="LangSmith evaluation concurrency. Use 0 for sequential runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Save results after this many tests. Use 0 to run the whole dataset at once.",
    )
    parser.add_argument(
        "--results-file",
        default=str(DEFAULT_RESULTS_FILE),
        help="Text file where batch results are appended.",
    )
    args = parser.parse_args()

    if args.suite in {"single", "all"}:
        single = run_single_turn_eval(
            seed_dataset=not args.no_seed,
            max_concurrency=args.max_concurrency,
            batch_size=args.batch_size,
            results_file=args.results_file,
        )
        if isinstance(single, list):
            print(f"Single-turn experiments: {', '.join(result.experiment_name for result in single)}")
        else:
            print(f"Single-turn experiment: {single.experiment_name}")

    if args.suite in {"multi", "all"}:
        multi = run_multi_turn_eval(
            seed_dataset=not args.no_seed,
            max_concurrency=args.max_concurrency,
            batch_size=args.batch_size,
            results_file=args.results_file,
        )
        if isinstance(multi, list):
            print(f"Multi-turn experiments: {', '.join(result.experiment_name for result in multi)}")
        else:
            print(f"Multi-turn experiment: {multi.experiment_name}")


if __name__ == "__main__":
    main()
