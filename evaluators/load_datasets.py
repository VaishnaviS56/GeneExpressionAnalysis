from __future__ import annotations

import argparse

try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def find_dotenv(*args, **kwargs) -> str:
        return ""

    def load_dotenv(*args, **kwargs) -> bool:
        return False

from evaluators.utils import (
    MULTI_TURN_DATASET,
    SINGLE_TURN_DATASET,
    ensure_dataset,
    load_multi_turn_specs,
    load_single_turn_specs,
)


def load_single(*, prune_removed: bool) -> None:
    specs = load_single_turn_specs()
    ensure_dataset(SINGLE_TURN_DATASET, specs, prune_removed=prune_removed)
    print(f"Loaded {len(specs)} examples into {SINGLE_TURN_DATASET}.")


def load_multi(*, prune_removed: bool) -> None:
    specs = load_multi_turn_specs()
    ensure_dataset(MULTI_TURN_DATASET, specs, prune_removed=prune_removed)
    print(f"Loaded {len(specs)} examples into {MULTI_TURN_DATASET}.")


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)

    parser = argparse.ArgumentParser(
        description="Create/update LangSmith datasets from evaluator question files."
    )
    parser.add_argument(
        "suite",
        choices=("single", "multi", "all"),
        help="Dataset suite to load.",
    )
    parser.add_argument(
        "--prune-removed",
        action="store_true",
        help="Delete LangSmith examples whose metadata id is no longer in the local question file.",
    )
    args = parser.parse_args()

    if args.suite in {"single", "all"}:
        load_single(prune_removed=args.prune_removed)

    if args.suite in {"multi", "all"}:
        load_multi(prune_removed=args.prune_removed)


if __name__ == "__main__":
    main()
