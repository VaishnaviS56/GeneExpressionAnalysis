from __future__ import annotations

from functools import lru_cache
from typing import Any

from langsmith.evaluation import evaluate

from gea_agent.agent.graph import build_app
from evaluators.utils import (
    DEFAULT_RESULTS_FILE,
    SINGLE_TURN_DATASET,
    append_batch_results,
    chunked,
    coerce_human_message,
    ensure_dataset,
    extract_tool_calls,
    list_dataset_examples,
    load_single_turn_specs,
    sanitize_agent_output,
    select_dataset_examples,
    tool_ordering_score,
    tool_precision_score,
    tool_recall_score,
    tool_selection_score,
)


API_PROVIDERS = {"anthropic", "google", "google_ai_studio", "gemini", "gemma", "gemma4", "groq", "mistral", "ollama"}


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_app()


def evaluate_tool_selection(run, example):
    called_tools = extract_tool_calls(run)
    score = tool_selection_score(
        called_tools=called_tools,
        required_tools=example.outputs.get("required_tools", []),
        required_tool_groups=example.outputs.get("required_tool_groups", []),
    )
    return {"key": "tool_selection", "score": score}


def evaluate_tool_precision(run, example):
    called_tools = extract_tool_calls(run)
    score = tool_precision_score(
        called_tools=called_tools,
        required_tools=example.outputs.get("required_tools", []),
        optional_tools=example.outputs.get("optional_tools", []),
    )
    return {"key": "tool_precision", "score": score}


def evaluate_tool_recall(run, example):
    called_tools = extract_tool_calls(run)
    score = tool_recall_score(
        called_tools=called_tools,
        required_tools=example.outputs.get("required_tools", []),
        optional_tools=example.outputs.get("optional_tools", []),
        required_tool_groups=example.outputs.get("required_tool_groups", []),
    )
    return {"key": "tool_recall", "score": score}


def evaluate_tool_ordering(run, example):
    if not example.outputs.get("order_matters", True):
        return {"key": "tool_ordering", "score": 1}

    called_tools = extract_tool_calls(run)
    score = tool_ordering_score(
        called_tools=called_tools,
        required_tools=example.outputs.get("required_tools", []),
        optional_tools=example.outputs.get("optional_tools", []),
        required_tool_groups=example.outputs.get("required_tool_groups", []),
    )
    return {"key": "tool_ordering", "score": score}


def evaluate_latency(run, example):
    if not run.start_time or not run.end_time:
        latency_seconds = None
    else:
        latency_seconds = (run.end_time - run.start_time).total_seconds()
    return {"key": "latency_seconds", "score": latency_seconds}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            dumped = dict_method()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    return {}


def _numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _token_count_from_mapping(data: dict[str, Any]) -> int | None:
    for key in (
        "total_tokens",
        "total_token_count",
        "totalTokens",
        "total_billable_characters",
    ):
        value = _numeric(data.get(key))
        if value is not None:
            return int(value)

    input_tokens = (
        _numeric(data.get("input_tokens"))
        or _numeric(data.get("prompt_tokens"))
        or _numeric(data.get("prompt_token_count"))
    )
    output_tokens = (
        _numeric(data.get("output_tokens"))
        or _numeric(data.get("completion_tokens"))
        or _numeric(data.get("candidates_token_count"))
    )
    if input_tokens is not None or output_tokens is not None:
        return int((input_tokens or 0) + (output_tokens or 0))

    return None


def _cost_from_mapping(data: dict[str, Any]) -> float | None:
    for key in ("total_cost", "cost", "totalCost"):
        value = _numeric(data.get(key))
        if value is not None:
            return value

    input_cost = _numeric(data.get("input_cost"))
    output_cost = _numeric(data.get("output_cost"))
    if input_cost is not None or output_cost is not None:
        return (input_cost or 0.0) + (output_cost or 0.0)

    return None


def _iter_mappings(value: Any):
    data = _as_dict(value)
    if data:
        yield data
        iterable = data.values()
    elif isinstance(value, list):
        iterable = value
    else:
        iterable = []

    for item in iterable:
        if isinstance(item, (dict, list)) or _as_dict(item):
            yield from _iter_mappings(item)


def _direct_run_tokens(run: Any) -> int:
    candidates: list[int] = []
    for source in (
        getattr(run, "extra", None),
        getattr(run, "outputs", None),
        getattr(run, "events", None),
    ):
        for mapping in _iter_mappings(source):
            tokens = _token_count_from_mapping(mapping)
            if tokens is not None:
                candidates.append(tokens)

    direct = _numeric(getattr(run, "total_tokens", None))
    if direct is not None:
        candidates.append(int(direct))

    # Usage can be duplicated in several output locations for one LLM call.
    return max(candidates) if candidates else 0


def _trace_total_tokens(run: Any) -> int:
    children = list(getattr(run, "child_runs", []) or [])
    child_total = sum(_trace_total_tokens(child) for child in children)
    direct_total = _direct_run_tokens(run)
    if children and child_total:
        return child_total
    return direct_total


def _trace_total_cost(run: Any) -> float | None:
    candidates: list[float] = []
    direct = _numeric(getattr(run, "total_cost", None))
    if direct is not None:
        candidates.append(direct)

    for source in (
        getattr(run, "extra", None),
        getattr(run, "outputs", None),
        getattr(run, "events", None),
    ):
        for mapping in _iter_mappings(source):
            cost = _cost_from_mapping(mapping)
            if cost is not None:
                candidates.append(cost)

    children = list(getattr(run, "child_runs", []) or [])
    child_costs = [_trace_total_cost(child) for child in children]
    child_total = sum(cost for cost in child_costs if cost is not None)

    if children and any(cost is not None for cost in child_costs):
        return child_total
    return max(candidates) if candidates else None


def evaluate_total_tokens(run, example):
    total_tokens = _trace_total_tokens(run)
    return {"key": "total_tokens", "score": total_tokens}


def evaluate_cost(run, example):
    return {"key": "cost_usd", "score": _trace_total_cost(run)}


TOOL_EVALUATORS = [
    evaluate_tool_selection,
    evaluate_tool_precision,
    evaluate_tool_recall,
    evaluate_tool_ordering,
    evaluate_latency,
    evaluate_total_tokens,
    evaluate_cost,
]


def run_single_turn_example(inputs: dict[str, Any]) -> dict[str, Any]:
    query = str(inputs.get("query") or "")
    result = get_compiled_graph().invoke(
        {
            "query": query,
            "messages": [coerce_human_message(query)],
            "memory_summary": "",
        }
    )
    return sanitize_agent_output(result)


def run_single_turn_eval(
    *,
    seed_dataset: bool = True,
    experiment_prefix: str = "target-discovery-single-turn",
    max_concurrency: int = 0,
    batch_size: int = 10,
    results_file: str = str(DEFAULT_RESULTS_FILE),
    case_id: str | None = None,
    from_case_id: str | None = None,
):
    if seed_dataset:
        ensure_dataset(SINGLE_TURN_DATASET, load_single_turn_specs())

    examples, first_position = select_dataset_examples(
        list_dataset_examples(SINGLE_TURN_DATASET),
        case_id=case_id,
        from_case_id=from_case_id,
    )

    if batch_size > 0:
        results = []
        total_examples = len(examples)
        for batch_number, batch in enumerate(chunked(examples, batch_size), start=1):
            batch_results = evaluate(
                run_single_turn_example,
                data=batch,
                evaluators=TOOL_EVALUATORS,
                experiment_prefix=f"{experiment_prefix}-batch-{batch_number}",
                max_concurrency=max_concurrency,
            )
            rows = list(batch_results)
            append_batch_results(
                results_file=results_file,
                suite_name="single-turn",
                experiment_name=batch_results.experiment_name,
                batch_number=batch_number,
                batch_start=first_position + ((batch_number - 1) * batch_size),
                batch_end=first_position + ((batch_number - 1) * batch_size) + len(batch) - 1,
                total_examples=total_examples,
                rows=rows,
            )
            results.append(batch_results)
        return results

    return evaluate(
        run_single_turn_example,
        data=examples,
        evaluators=TOOL_EVALUATORS,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )


if __name__ == "__main__":
    results = run_single_turn_eval()
    if isinstance(results, list):
        print(f"Experiments: {', '.join(result.experiment_name for result in results)}")
    else:
        print(f"Experiment: {results.experiment_name}")
