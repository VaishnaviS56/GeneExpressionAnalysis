from __future__ import annotations

import argparse

from evaluators.utils import DEFAULT_RESULTS_FILE
from evaluators.multi_turn import run_multi_turn_eval
from evaluators.single_turn import run_single_turn_eval


def _case_prefix(value: str | None) -> str:
    return str(value or "").strip().split("-", 1)[0].upper()


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
    parser.add_argument(
        "--case",
        help="Run only one dataset example by id, such as ST-007 or MT-003.",
    )
    parser.add_argument(
        "--from-case",
        help="Run dataset examples from this id onward, such as ST-007 or MT-003.",
    )
    args = parser.parse_args()

    if args.case and args.from_case:
        parser.error("Use either --case or --from-case, not both.")

    requested_prefix = _case_prefix(args.case or args.from_case)
    run_single = args.suite in {"single", "all"} and requested_prefix != "MT"
    run_multi = args.suite in {"multi", "all"} and requested_prefix != "ST"

    if run_single:
        single = run_single_turn_eval(
            seed_dataset=not args.no_seed,
            max_concurrency=args.max_concurrency,
            batch_size=args.batch_size,
            results_file=args.results_file,
            case_id=args.case,
            from_case_id=args.from_case,
        )
        if isinstance(single, list):
            print(f"Single-turn experiments: {', '.join(result.experiment_name for result in single)}")
        else:
            print(f"Single-turn experiment: {single.experiment_name}")

    if run_multi:
        multi = run_multi_turn_eval(
            seed_dataset=not args.no_seed,
            max_concurrency=args.max_concurrency,
            batch_size=args.batch_size,
            results_file=args.results_file,
            case_id=args.case,
            from_case_id=args.from_case,
        )
        if isinstance(multi, list):
            print(f"Multi-turn experiments: {', '.join(result.experiment_name for result in multi)}")
        else:
            print(f"Multi-turn experiment: {multi.experiment_name}")


if __name__ == "__main__":
    main()
