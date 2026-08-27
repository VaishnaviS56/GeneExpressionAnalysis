from __future__ import annotations

import argparse
from typing import Any, Callable

from langsmith.evaluation import evaluate

from evaluators.hallucination import (
    evaluate_evidence_validity,
    evaluate_response_groundedness,
)
from evaluators.human_review import add_experiment_to_queue
from evaluators.multi_turn import run_multi_turn_example
from evaluators.single_turn import run_single_turn_example
from evaluators.utils import (
    DEFAULT_RESULTS_FILE,
    MULTI_TURN_DATASET,
    SINGLE_TURN_DATASET,
    append_batch_results,
    chunked,
    ensure_dataset,
    list_dataset_examples,
    load_multi_turn_specs,
    load_single_turn_specs,
    select_dataset_examples,
)


QUALITY_EVALUATORS = [
    evaluate_response_groundedness,
    evaluate_evidence_validity,
]


def run_quality_suite(
    *,
    suite_name: str,
    target: Callable[[dict[str, Any]], dict[str, Any]],
    dataset_name: str,
    seed_specs: list[dict[str, Any]],
    seed_dataset: bool,
    experiment_prefix: str,
    max_concurrency: int,
    batch_size: int,
    results_file: str,
    add_to_human_review: bool,
    case_id: str | None = None,
    from_case_id: str | None = None,
):
    if seed_dataset:
        ensure_dataset(dataset_name, seed_specs)

    examples, first_position = select_dataset_examples(
        list_dataset_examples(dataset_name),
        case_id=case_id,
        from_case_id=from_case_id,
    )

    if batch_size > 0:
        results = []
        total_examples = len(examples)
        for batch_number, batch in enumerate(chunked(examples, batch_size), start=1):
            batch_results = evaluate(
                target,
                data=batch,
                evaluators=QUALITY_EVALUATORS,
                experiment_prefix=f"{experiment_prefix}-quality-batch-{batch_number}",
                max_concurrency=max_concurrency,
            )
            rows = list(batch_results)
            append_batch_results(
                results_file=results_file,
                suite_name=f"{suite_name}-quality",
                experiment_name=batch_results.experiment_name,
                batch_number=batch_number,
                batch_start=first_position + ((batch_number - 1) * batch_size),
                batch_end=first_position + ((batch_number - 1) * batch_size) + len(batch) - 1,
                total_examples=total_examples,
                rows=rows,
            )
            if add_to_human_review:
                add_experiment_to_queue(batch_results.experiment_name)
            results.append(batch_results)
        return results

    results = evaluate(
        target,
        data=examples,
        evaluators=QUALITY_EVALUATORS,
        experiment_prefix=f"{experiment_prefix}-quality",
        max_concurrency=max_concurrency,
    )
    rows = list(results)
    append_batch_results(
        results_file=results_file,
        suite_name=f"{suite_name}-quality",
        experiment_name=results.experiment_name,
        batch_number=1,
        batch_start=1,
        batch_end=len(rows),
        total_examples=len(rows),
        rows=rows,
    )
    if add_to_human_review:
        add_experiment_to_queue(results.experiment_name)
    return results


def _case_prefix(value: str | None) -> str:
    return str(value or "").strip().split("-", 1)[0].upper()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hallucination/groundedness evals and add runs to human review."
    )
    parser.add_argument(
        "suite",
        choices=("single", "multi", "all"),
        help="Quality evaluation suite to run.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Use existing LangSmith datasets without creating/updating examples.",
    )
    parser.add_argument(
        "--no-human-review",
        action="store_true",
        help="Run judge evaluators without adding experiment runs to the annotation queue.",
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
        help="Save and queue results after this many tests. Use 0 for one experiment.",
    )
    parser.add_argument(
        "--results-file",
        default=str(DEFAULT_RESULTS_FILE.with_name("quality_results.txt")),
        help="Text file where quality batch results are appended.",
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

    add_to_human_review = not args.no_human_review
    requested_prefix = _case_prefix(args.case or args.from_case)
    run_single = args.suite in {"single", "all"} and requested_prefix != "MT"
    run_multi = args.suite in {"multi", "all"} and requested_prefix != "ST"

    if run_single:
        single = run_quality_suite(
            suite_name="single-turn",
            target=run_single_turn_example,
            dataset_name=SINGLE_TURN_DATASET,
            seed_specs=load_single_turn_specs(),
            seed_dataset=not args.no_seed,
            experiment_prefix="target-discovery-single-turn",
            max_concurrency=args.max_concurrency,
            batch_size=args.batch_size,
            results_file=args.results_file,
            add_to_human_review=add_to_human_review,
            case_id=args.case,
            from_case_id=args.from_case,
        )
        if isinstance(single, list):
            print(f"Single-turn quality experiments: {', '.join(result.experiment_name for result in single)}")
        else:
            print(f"Single-turn quality experiment: {single.experiment_name}")

    if run_multi:
        multi = run_quality_suite(
            suite_name="multi-turn",
            target=run_multi_turn_example,
            dataset_name=MULTI_TURN_DATASET,
            seed_specs=load_multi_turn_specs(),
            seed_dataset=not args.no_seed,
            experiment_prefix="target-discovery-multi-turn",
            max_concurrency=args.max_concurrency,
            batch_size=args.batch_size,
            results_file=args.results_file,
            add_to_human_review=add_to_human_review,
            case_id=args.case,
            from_case_id=args.from_case,
        )
        if isinstance(multi, list):
            print(f"Multi-turn quality experiments: {', '.join(result.experiment_name for result in multi)}")
        else:
            print(f"Multi-turn quality experiment: {multi.experiment_name}")


if __name__ == "__main__":
    main()
