from __future__ import annotations

import uuid
from typing import Any

from langsmith.evaluation import evaluate

from evaluators.single_turn import (
    TOOL_EVALUATORS,
    get_compiled_graph,
)
from evaluators.utils import (
    DEFAULT_RESULTS_FILE,
    MULTI_TURN_DATASET,
    append_batch_results,
    chunked,
    coerce_human_message,
    ensure_dataset,
    list_dataset_examples,
    load_multi_turn_specs,
    sanitize_agent_output,
)


def run_multi_turn_example(inputs: dict[str, Any]) -> dict[str, Any]:
    """Replay a conversation one turn at a time with one LangGraph thread."""

    messages = [coerce_human_message(message) for message in inputs.get("messages", [])]
    thread_id = str(uuid.uuid4())
    history = []
    final_state: dict[str, Any] = {}

    for message in messages:
        history.append(message)
        final_state = get_compiled_graph().invoke(
            {
                "query": str(message.content),
                "messages": list(history),
                "memory_summary": "",
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        history = list(final_state.get("messages") or history)

    return sanitize_agent_output(final_state)


def run_multi_turn_eval(
    *,
    seed_dataset: bool = True,
    experiment_prefix: str = "target-discovery-multi-turn",
    max_concurrency: int = 0,
    batch_size: int = 10,
    results_file: str = str(DEFAULT_RESULTS_FILE),
):
    if seed_dataset:
        ensure_dataset(MULTI_TURN_DATASET, load_multi_turn_specs())

    if batch_size > 0:
        examples = list_dataset_examples(MULTI_TURN_DATASET)
        results = []
        total_examples = len(examples)
        for batch_number, batch in enumerate(chunked(examples, batch_size), start=1):
            batch_results = evaluate(
                run_multi_turn_example,
                data=batch,
                evaluators=TOOL_EVALUATORS,
                experiment_prefix=f"{experiment_prefix}-batch-{batch_number}",
                max_concurrency=max_concurrency,
            )
            rows = list(batch_results)
            append_batch_results(
                results_file=results_file,
                suite_name="multi-turn",
                experiment_name=batch_results.experiment_name,
                batch_number=batch_number,
                batch_start=((batch_number - 1) * batch_size) + 1,
                batch_end=min(batch_number * batch_size, total_examples),
                total_examples=total_examples,
                rows=rows,
            )
            results.append(batch_results)
        return results

    return evaluate(
        run_multi_turn_example,
        data=MULTI_TURN_DATASET,
        evaluators=TOOL_EVALUATORS,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )


if __name__ == "__main__":
    results = run_multi_turn_eval()
    if isinstance(results, list):
        print(f"Experiments: {', '.join(result.experiment_name for result in results)}")
    else:
        print(f"Experiment: {results.experiment_name}")
