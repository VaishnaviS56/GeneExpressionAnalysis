from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Iterable, List, Set

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langsmith import Client


SINGLE_TURN_DATASET = "TargetDiscovery-SingleTurn"
MULTI_TURN_DATASET = "TargetDiscovery-MultiTurn"
DEFAULT_RESULTS_FILE = Path(__file__).with_name("langsmith_results.txt")


def _coerce_tool_name(value: Any) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _normalize_tools(tools: List[str]) -> Set[str]:
    """Convert a tool list to a clean set."""
    return {_coerce_tool_name(tool) for tool in tools or [] if _coerce_tool_name(tool)}


def _normalize_tool_groups(groups: Iterable[Iterable[str]] | None) -> list[set[str]]:
    normalized: list[set[str]] = []
    for group in groups or []:
        tools = _normalize_tools(list(group))
        if tools:
            normalized.append(tools)
    return normalized


def extract_tool_calls(run_or_output: Any) -> list[str]:
    """Extract tool names from a LangSmith run or sanitized target output."""

    called_tools: list[str] = []

    def add(name: Any) -> None:
        tool_name = _coerce_tool_name(name)
        if tool_name:
            called_tools.append(tool_name)

    def visit_message(message: Any) -> None:
        for call in getattr(message, "tool_calls", []) or []:
            if isinstance(call, dict):
                add(call.get("name"))

    def visit_output(output: Any) -> None:
        if isinstance(output, dict):
            for row in output.get("tool_history") or []:
                if isinstance(row, dict):
                    add(row.get("tool_name") or row.get("name") or row.get("tool"))

            meta = output.get("meta")
            if isinstance(meta, dict):
                visit_output(meta)

            for message in output.get("messages") or []:
                visit_message(message)
                if isinstance(message, dict):
                    for call in message.get("tool_calls") or []:
                        if isinstance(call, dict):
                            add(call.get("name"))
            return

        if isinstance(output, list):
            for item in output:
                visit_output(item)
                visit_message(item)

    def visit_run(node: Any) -> None:
        if getattr(node, "run_type", None) == "tool":
            add(getattr(node, "name", ""))

        outputs = getattr(node, "outputs", None)
        if isinstance(outputs, dict):
            visit_output(outputs)

        extra = getattr(node, "extra", None)
        if isinstance(extra, dict):
            tool_calls = extra.get("tool_calls") or extra.get("invocation_params", {}).get("tool_calls")
            for call in tool_calls or []:
                if isinstance(call, dict):
                    add(call.get("name") or call.get("function", {}).get("name"))

        for child in getattr(node, "child_runs", []) or []:
            visit_run(child)

    visit_run(run_or_output)

    deduped: list[str] = []
    for tool in called_tools:
        if tool not in deduped:
            deduped.append(tool)
    return deduped


def tool_selection_score(
    called_tools: List[str],
    required_tools: List[str],
    required_tool_groups: list[list[str]] | None = None,
) -> float:
    """Score required tools, with support for one-of tool groups."""

    called = _normalize_tools(called_tools)
    required = _normalize_tools(required_tools)
    required_groups = _normalize_tool_groups(required_tool_groups)

    denominator = len(required) + len(required_groups)
    if not denominator:
        return 1.0

    matched = len(called & required)
    matched += sum(1 for group in required_groups if called & group)
    return matched / denominator


def tool_precision_score(
    called_tools: List[str],
    required_tools: List[str],
    optional_tools: List[str],
) -> float:
    """Score how many called tools were expected or explicitly allowed."""

    called = _normalize_tools(called_tools)
    allowed = _normalize_tools(required_tools) | _normalize_tools(optional_tools)

    if not called:
        return 1.0

    return len(called & allowed) / len(called)


def tool_recall_score(
    called_tools: List[str],
    required_tools: List[str],
    optional_tools: List[str],
    required_tool_groups: list[list[str]] | None = None,
) -> float:
    """Score coverage of required and allowed tools."""

    called = _normalize_tools(called_tools)
    allowed = _normalize_tools(required_tools) | _normalize_tools(optional_tools)
    required_groups = _normalize_tool_groups(required_tool_groups)

    if required_groups:
        required_score = tool_selection_score(called_tools, required_tools, required_tool_groups)
        optional = _normalize_tools(optional_tools)
        if optional:
            return (required_score + (len(called & optional) / len(optional))) / 2
        return required_score

    if not allowed and not called:
        return 1.0
    if not allowed:
        return 0.0

    return len(called & allowed) / len(allowed)


def tool_ordering_score(
    called_tools: List[str],
    required_tools: List[str],
    optional_tools: List[str],
    required_tool_groups: list[list[str]] | None = None,
) -> int:
    """Binary ordering evaluator, with support for one-of tool groups."""

    required_groups = _normalize_tool_groups(required_tool_groups)
    if not required_tools and not required_groups and not (
        _normalize_tools(required_tools) | _normalize_tools(optional_tools)
    ):
        return 1

    current_position = -1
    for required_tool in required_tools:
        tool = _coerce_tool_name(required_tool)
        if not tool:
            continue
        found = False
        for idx in range(current_position + 1, len(called_tools)):
            if called_tools[idx] == tool:
                current_position = idx
                found = True
                break
        if not found:
            return 0

    called = _normalize_tools(called_tools)
    for group in required_groups:
        if not called & group:
            return 0

    return 1


def _parse_expected_tools(raw: str) -> tuple[list[str], list[str], list[list[str]]]:
    required_tools: list[str] = []
    optional_tools: list[str] = []
    required_tool_groups: list[list[str]] = []

    text = (raw or "").strip()
    if not text:
        return required_tools, optional_tools, required_tool_groups

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None

    if parsed is None:
        group_spans: list[tuple[int, int]] = []
        for match in re.finditer(r'"([^"]+)"\s*/\s*"([^"]+)"', text):
            group = [_coerce_tool_name(match.group(1)), _coerce_tool_name(match.group(2))]
            required_tool_groups.append(group)
            optional_tools.extend(group)
            group_spans.append(match.span())

        def in_group_span(start: int) -> bool:
            return any(span_start <= start < span_end for span_start, span_end in group_spans)

        for match in re.finditer(r'"([^"]*)"', text):
            if in_group_span(match.start()):
                continue
            tool = _coerce_tool_name(match.group(1))
            if tool:
                required_tools.append(tool)

        return required_tools, optional_tools, required_tool_groups

    for item in parsed if isinstance(parsed, list) else []:
        if isinstance(item, tuple):
            group = [_coerce_tool_name(tool) for tool in item if _coerce_tool_name(tool)]
        else:
            tool = _coerce_tool_name(item)
            group = [_coerce_tool_name(part) for part in tool.split("/") if _coerce_tool_name(part)] if "/" in tool else []
            if not group and tool:
                required_tools.append(tool)
                continue

        if len(group) == 1:
            required_tools.append(group[0])
        elif group:
            required_tool_groups.append(group)
            optional_tools.extend(group)

    return required_tools, optional_tools, required_tool_groups


def _parse_blocks(path: Path) -> list[tuple[str, list[str], str]]:
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^(ST|MT)-\d+\s*$", text))
    blocks: list[tuple[str, list[str], str]] = []

    for index, match in enumerate(matches):
        example_id = match.group(0).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        user_messages = [
            line.split("User:", 1)[1].strip()
            for line in body.splitlines()
            if line.strip().startswith("User:")
        ]
        expected_match = re.search(r"(?ms)^Expected Tools:\s*(.*?)(?:\n\s*\n|\Z)", body)
        expected_tools = expected_match.group(1).strip() if expected_match else "[]"
        blocks.append((example_id, user_messages, expected_tools))

    return blocks


def load_single_turn_specs(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or Path(__file__).with_name("singleQs.txt")
    specs: list[dict[str, Any]] = []
    for example_id, user_messages, raw_tools in _parse_blocks(path):
        if not user_messages:
            continue
        required, optional, groups = _parse_expected_tools(raw_tools)
        specs.append(
            {
                "id": example_id,
                "inputs": {"query": user_messages[0]},
                "outputs": {
                    "required_tools": required,
                    "optional_tools": optional,
                    "required_tool_groups": groups,
                    "order_matters": True,
                },
            }
        )
    return specs


def load_multi_turn_specs(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or Path(__file__).with_name("MultiTurnQs.txt")
    specs: list[dict[str, Any]] = []
    for example_id, user_messages, raw_tools in _parse_blocks(path):
        if not user_messages:
            continue
        required, optional, groups = _parse_expected_tools(raw_tools)
        specs.append(
            {
                "id": example_id,
                "inputs": {"messages": user_messages},
                "outputs": {
                    "required_tools": required,
                    "optional_tools": optional,
                    "required_tool_groups": groups,
                    "order_matters": True,
                },
            }
        )
    return specs


def ensure_dataset(
    dataset_name: str,
    specs: list[dict[str, Any]],
    *,
    prune_removed: bool = False,
) -> None:
    client = Client()
    try:
        client.read_dataset(dataset_name=dataset_name)
    except Exception:
        client.create_dataset(
            dataset_name=dataset_name,
            description="Tool-routing regression tests for the GEA target discovery agent.",
        )

    existing_by_id = {
        str(example.metadata.get("id")): example
        for example in client.list_examples(dataset_name=dataset_name)
        if example.metadata and example.metadata.get("id")
    }
    spec_ids = {spec["id"] for spec in specs}

    if prune_removed:
        for example_id, example in existing_by_id.items():
            if example_id not in spec_ids:
                client.delete_example(example.id)

    for spec in specs:
        metadata = {"id": spec["id"]}
        existing = existing_by_id.get(spec["id"])
        if existing is None:
            client.create_example(
                dataset_name=dataset_name,
                inputs=spec["inputs"],
                outputs=spec["outputs"],
                metadata=metadata,
            )
        else:
            client.update_example(
                existing.id,
                inputs=spec["inputs"],
                outputs=spec["outputs"],
                metadata=metadata,
            )


def list_dataset_examples(dataset_name: str) -> list[Any]:
    client = Client()
    examples = list(client.list_examples(dataset_name=dataset_name))
    return sorted(examples, key=lambda example: str((example.metadata or {}).get("id") or example.id))


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items]
    return [items[index : index + size] for index in range(0, len(items), size)]


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _format_row_result(row: Any) -> str:
    example = _row_value(row, "example")
    run = _row_value(row, "run")
    evaluation_results = _row_value(row, "evaluation_results")

    example_id = str((getattr(example, "metadata", None) or {}).get("id") or getattr(example, "id", "unknown"))
    run_id = str(getattr(run, "id", "unknown"))

    score_parts: list[str] = []
    for result in _row_value(evaluation_results, "results", []) or []:
        key = _result_value(result, "key", "")
        score = _result_value(result, "score", None)
        comment = _result_value(result, "comment", "")
        if key:
            if comment and score is None:
                score_parts.append(f"{key}=ERROR ({comment})")
            else:
                score_parts.append(f"{key}={score}")

    scores = ", ".join(score_parts) if score_parts else "no evaluator scores"
    return f"- {example_id} | run={run_id} | {scores}"


def append_batch_results(
    *,
    results_file: str | Path,
    suite_name: str,
    experiment_name: str,
    batch_number: int,
    batch_start: int,
    batch_end: int,
    total_examples: int,
    rows: list[Any],
) -> None:
    path = Path(results_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "",
        f"[{suite_name}] batch {batch_number}: tests {batch_start}-{batch_end} of {total_examples}",
        f"Experiment: {experiment_name}",
    ]
    lines.extend(_format_row_result(row) for row in rows)
    path.write_text(
        (path.read_text(encoding="utf-8") if path.exists() else "")
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def coerce_human_message(value: Any) -> HumanMessage:
    if isinstance(value, HumanMessage):
        return value
    if isinstance(value, dict):
        return HumanMessage(content=str(value.get("content") or value.get("text") or ""))
    return HumanMessage(content=str(value or ""))


def messages_to_text(messages: list[BaseMessage]) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for message in messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        serialized.append({"role": role, "content": str(getattr(message, "content", "") or "")})
    return serialized


def sanitize_agent_output(output: dict[str, Any]) -> dict[str, Any]:
    meta = output.get("meta") if isinstance(output.get("meta"), dict) else {}
    return {
        "answer": str(output.get("answer") or ""),
        "analysis_arm": str(output.get("analysis_arm") or ""),
        "meta": meta,
        "tool_history": list(output.get("tool_history") or meta.get("tool_history") or []),
        "messages": messages_to_text(list(output.get("messages") or [])),
    }
