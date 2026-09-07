from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QUALITY_GLOB = str(Path(__file__).with_name("quality_*.txt"))
DEFAULT_TRACE_GLOBS = [
    str(Path(__file__).with_name("eval_traces_*.jsonl")),
    str(Path(__file__).with_name("current_eval_traces_*.jsonl")),
    str(Path(__file__).with_name("current_judge_traces_*.jsonl")),
]

METRIC_KEYS = ("response_groundedness", "evidence_validity")

FIELDNAMES = [
    "source_quality_file",
    "source_trace_file",
    "trace_account",
    "suite",
    "turn_type",
    "model",
    "batch_number",
    "batch_start",
    "batch_end",
    "batch_total",
    "test_number",
    "test_id",
    "experiment",
    "target_run_id",
    "target_trace_id",
    "target_status",
    "target_error",
    "target_start_time",
    "target_end_time",
    "user_query",
    "messages_json",
    "final_answer",
    "called_tools_json",
    "tool_history_json",
    "tool_outputs_text",
    "required_tools_json",
    "optional_tools_json",
    "required_tool_groups_json",
    "order_matters",
    "response_groundedness_score",
    "response_groundedness_comment",
    "response_groundedness_evaluator_run_id",
    "response_groundedness_trace_url",
    "evidence_validity_score",
    "evidence_validity_comment",
    "evidence_validity_evaluator_run_id",
    "evidence_validity_trace_url",
    "quality_scores_json",
    "evaluator_costs_json",
]


def _compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except TypeError:
        return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return ""


def _test_number_from_id(test_id: str) -> str:
    match = re.search(r"(\d+)$", str(test_id).strip())
    return str(int(match.group(1))) if match else ""


def _infer_turn_type(path: Path, suite: str) -> str:
    text = f"{path.name} {suite}".lower()
    if "multi" in text or "multiple" in text:
        return "multi"
    if "single" in text:
        return "single"
    return ""


def _infer_model(path: Path) -> str:
    name = path.stem.lower()
    for model in ("gemini", "gemma", "claude", "mistral", "groq", "ollama"):
        if model in name:
            return model
    return ""


def _parse_metric_summary(text: str) -> dict[str, str]:
    scores: dict[str, str] = {}
    for part in re.split(r",\s*", text.strip()):
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        scores[key.strip()] = value.strip()
    return scores


def parse_quality_file(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    suite = ""
    batch_number = ""
    batch_start = ""
    batch_end = ""
    batch_total = ""
    experiment = ""

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue

            header_match = re.match(
                r"^\[([^\]]+)\]\s+batch\s+(\d+):\s+tests\s+(\d+)-(\d+)\s+of\s+(\d+)",
                raw,
            )
            if header_match:
                suite = header_match.group(1)
                batch_number = header_match.group(2)
                batch_start = header_match.group(3)
                batch_end = header_match.group(4)
                batch_total = header_match.group(5)
                continue

            experiment_match = re.match(r"^Experiment:\s+(.+)$", raw)
            if experiment_match:
                experiment = experiment_match.group(1).strip()
                continue

            row_match = re.match(r"^-\s+([^|]+?)\s+\|\s+run=([^|\s]+)(?:\s+\|\s+(.*))?$", raw)
            if not row_match:
                continue

            test_id = row_match.group(1).strip()
            target_run_id = row_match.group(2).strip()
            scores = _parse_metric_summary(row_match.group(3) or "")
            row = {
                "source_quality_file": str(path),
                "source_trace_file": "",
                "trace_account": "",
                "suite": suite,
                "turn_type": _infer_turn_type(path, suite),
                "model": _infer_model(path),
                "batch_number": batch_number,
                "batch_start": batch_start,
                "batch_end": batch_end,
                "batch_total": batch_total,
                "test_number": _test_number_from_id(test_id) or batch_start,
                "test_id": test_id,
                "experiment": experiment,
                "target_run_id": target_run_id,
                "quality_scores_json": _compact_json(scores),
            }
            for metric in METRIC_KEYS:
                row[f"{metric}_score"] = scores.get(metric, "")
            rows.append(row)

    return rows


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return str(value or "")
    content = value.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or item)
            for item in content
            if item not in (None, "", [], {})
        )
    return str(value.get("text") or content or "")


def _message_role(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("role") or value.get("type") or "").lower()


def _messages_to_query(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    lines: list[str] = []
    for message in messages:
        if isinstance(message, str):
            lines.append(message)
            continue
        role = _message_role(message)
        if role and role not in {"user", "human"}:
            continue
        text = _message_text(message).strip()
        if text:
            lines.append(text)
    return "\n\n".join(lines)


def _latest_assistant_message(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if _message_role(message) in {"assistant", "ai"}:
            text = _message_text(message).strip()
            if text:
                return text
    return ""


def _extract_final_answer(outputs: Any) -> str:
    outputs = _as_dict(outputs)
    nested = _as_dict(outputs.get("output"))
    answer = _first_present(
        outputs.get("answer"),
        outputs.get("final_answer"),
        outputs.get("response"),
        outputs.get("result"),
        nested.get("answer"),
        nested.get("final_answer"),
        nested.get("response"),
        nested.get("result"),
        _latest_assistant_message(outputs.get("messages")),
        _latest_assistant_message(nested.get("messages")),
    )
    if answer:
        return str(answer)
    output = outputs.get("output")
    return _compact_json(output) if output not in (None, "", [], {}) else ""


def _extract_tool_history(outputs: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for row in _as_list(value.get("tool_history")):
                if isinstance(row, dict):
                    found.append(row)
            for key in ("meta", "output", "outputs"):
                nested = value.get(key)
                if isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(outputs)
    return found


def _extract_tool_outputs_text(tool_history: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for row in tool_history:
        name = row.get("tool") or row.get("tool_name") or row.get("name") or ""
        result = row.get("result")
        chunks.append(f"Tool: {name}\nOutput:\n{result}")
    return "\n\n".join(chunks)


def _extract_called_tools(tool_history: list[dict[str, Any]]) -> list[str]:
    called: list[str] = []
    for row in tool_history:
        name = str(row.get("tool") or row.get("tool_name") or row.get("name") or "").strip()
        if name and name not in called:
            called.append(name)
    return called


def _trace_account_from_path(path: Path) -> str:
    name = path.name.lower()
    if "sais" in name:
        return "sais"
    if "evaluators" in name:
        return "evaluators"
    return ""


def _trace_url(trace: dict[str, Any]) -> str:
    app_path = str(trace.get("app_path") or "")
    if not app_path:
        return ""
    if app_path.startswith("http"):
        return app_path
    return f"https://smith.langchain.com{app_path}"


def _target_run_id(trace: dict[str, Any]) -> str:
    metadata = _as_dict(_as_dict(trace.get("extra")).get("metadata"))
    target_run = _as_dict(_as_dict(trace.get("inputs")).get("run"))
    return str(
        _first_present(
            metadata.get("target_run_id"),
            metadata.get("reference_run_id"),
            target_run.get("id"),
        )
    )


def _quality_metric_key(trace: dict[str, Any]) -> str:
    outputs = _as_dict(trace.get("outputs"))
    key = str(outputs.get("key") or trace.get("name") or "")
    for metric in METRIC_KEYS:
        if metric in key:
            return metric
    return key


def _quality_enrichment(trace: dict[str, Any], path: Path) -> dict[str, str]:
    inputs = _as_dict(trace.get("inputs"))
    target_run = _as_dict(inputs.get("run"))
    example = _as_dict(inputs.get("example"))
    example_inputs = _as_dict(example.get("inputs"))
    example_outputs = _as_dict(example.get("outputs"))
    target_inputs = _as_dict(target_run.get("inputs"))
    target_outputs = target_run.get("outputs")
    target_outputs_dict = _as_dict(target_outputs)
    tool_history = _extract_tool_history(target_outputs_dict)
    messages = _first_present(target_inputs.get("messages"), example_inputs.get("messages"))
    query = _first_present(
        target_inputs.get("query"),
        example_inputs.get("query"),
        _messages_to_query(messages),
    )
    metric = _quality_metric_key(trace)
    outputs = _as_dict(trace.get("outputs"))

    costs = {
        "prompt_tokens": trace.get("prompt_tokens"),
        "completion_tokens": trace.get("completion_tokens"),
        "total_tokens": trace.get("total_tokens"),
        "prompt_cost": trace.get("prompt_cost"),
        "completion_cost": trace.get("completion_cost"),
        "total_cost": trace.get("total_cost"),
    }

    row = {
        "source_trace_file": str(path),
        "trace_account": _trace_account_from_path(path),
        "target_trace_id": str(target_run.get("trace_id") or ""),
        "target_status": str(target_run.get("status") or ""),
        "target_error": str(target_run.get("error") or ""),
        "target_start_time": str(target_run.get("start_time") or ""),
        "target_end_time": str(target_run.get("end_time") or ""),
        "user_query": str(query or ""),
        "messages_json": _compact_json(messages),
        "final_answer": _extract_final_answer(target_outputs),
        "called_tools_json": _compact_json(_extract_called_tools(tool_history)),
        "tool_history_json": _compact_json(tool_history),
        "tool_outputs_text": _extract_tool_outputs_text(tool_history),
        "required_tools_json": _compact_json(example_outputs.get("required_tools")),
        "optional_tools_json": _compact_json(example_outputs.get("optional_tools")),
        "required_tool_groups_json": _compact_json(example_outputs.get("required_tool_groups")),
        "order_matters": str(example_outputs.get("order_matters") or ""),
        "evaluator_costs_json": _compact_json({key: value for key, value in costs.items() if value not in (None, "")}),
    }

    if metric in METRIC_KEYS:
        row[f"{metric}_score"] = str(outputs.get("score") if outputs.get("score") is not None else "")
        row[f"{metric}_comment"] = str(outputs.get("comment") or "")
        row[f"{metric}_evaluator_run_id"] = str(trace.get("id") or "")
        row[f"{metric}_trace_url"] = _trace_url(trace)

    return row


def _target_enrichment(trace: dict[str, Any], path: Path) -> dict[str, str]:
    inputs = _as_dict(trace.get("inputs"))
    outputs = trace.get("outputs")
    outputs_dict = _as_dict(outputs)
    metadata = _as_dict(_as_dict(trace.get("extra")).get("metadata"))
    tool_history = _extract_tool_history(outputs_dict)
    messages = inputs.get("messages")
    query = _first_present(
        inputs.get("query"),
        _messages_to_query(messages),
    )

    return {
        "source_trace_file": str(path),
        "trace_account": _trace_account_from_path(path),
        "target_trace_id": str(trace.get("trace_id") or trace.get("id") or ""),
        "target_status": str(trace.get("status") or ""),
        "target_error": str(trace.get("error") or ""),
        "target_start_time": str(trace.get("start_time") or ""),
        "target_end_time": str(trace.get("end_time") or ""),
        "user_query": str(query or ""),
        "messages_json": _compact_json(messages),
        "final_answer": _extract_final_answer(outputs),
        "called_tools_json": _compact_json(_extract_called_tools(tool_history)),
        "tool_history_json": _compact_json(tool_history),
        "tool_outputs_text": _extract_tool_outputs_text(tool_history),
        "test_id": str(metadata.get("ls_example_id") or ""),
    }


def _expand_patterns(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.is_file() and path not in paths:
                paths.append(path)
    return sorted(paths)


def load_trace_enrichment(trace_files: list[Path], wanted_run_ids: set[str]) -> dict[str, dict[str, str]]:
    enrichment: dict[str, dict[str, str]] = {}
    if not trace_files or not wanted_run_ids:
        return enrichment

    for path in trace_files:
        print(f"Scanning {path}...", flush=True)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    trace = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(f"Skipping invalid JSON at {path}:{line_number}: {exc}", flush=True)
                    continue
                if not isinstance(trace, dict):
                    continue
                direct_run_id = str(trace.get("id") or "")
                if direct_run_id in wanted_run_ids:
                    existing = enrichment.setdefault(direct_run_id, {})
                    existing.update(
                        {
                            key: value
                            for key, value in _target_enrichment(trace, path).items()
                            if value != "" and key != "test_id"
                        }
                    )
                    continue
                run_id = _target_run_id(trace)
                if run_id not in wanted_run_ids:
                    continue
                existing = enrichment.setdefault(run_id, {})
                existing.update({key: value for key, value in _quality_enrichment(trace, path).items() if value != ""})

    return enrichment


def write_csv(rows: list[dict[str, str]], output_path: Path, enrichment: dict[str, dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            merged = {field: "" for field in FIELDNAMES}
            merged.update(row)
            merged.update(enrichment.get(row.get("target_run_id", ""), {}))
            writer.writerow(merged)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert quality_*.txt LangSmith batch summaries into review CSVs, "
            "optionally enriched with downloaded evaluator trace JSONL files."
        )
    )
    parser.add_argument(
        "quality_files",
        nargs="*",
        help="quality_*.txt files. Defaults to evaluators/quality_*.txt.",
    )
    parser.add_argument(
        "--trace-files",
        nargs="*",
        default=DEFAULT_TRACE_GLOBS,
        help="Downloaded LangSmith evaluator trace JSONL files or glob patterns from either account.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for per-file CSVs. Defaults to writing beside each TXT file.",
    )
    parser.add_argument(
        "--combined-output",
        default="",
        help="Optional path for one combined CSV containing all quality files.",
    )
    parser.add_argument(
        "--no-trace-enrichment",
        action="store_true",
        help="Only parse the TXT summaries; do not scan JSONL traces.",
    )
    args = parser.parse_args()

    quality_patterns = args.quality_files or [DEFAULT_QUALITY_GLOB]
    quality_files = _expand_patterns(quality_patterns)
    if not quality_files:
        raise SystemExit(f"No quality files matched: {quality_patterns}")

    rows_by_file = {path: parse_quality_file(path) for path in quality_files}
    wanted_run_ids = {
        row["target_run_id"]
        for rows in rows_by_file.values()
        for row in rows
        if row.get("target_run_id")
    }
    trace_files = [] if args.no_trace_enrichment else _expand_patterns(args.trace_files)
    enrichment = load_trace_enrichment(trace_files, wanted_run_ids)

    output_dir = Path(args.output_dir) if args.output_dir else None
    all_rows: list[dict[str, str]] = []
    for path, rows in rows_by_file.items():
        output_path = (output_dir / path.with_suffix(".csv").name) if output_dir else path.with_suffix(".csv")
        write_csv(rows, output_path, enrichment)
        all_rows.extend(rows)
        print(f"Saved {len(rows)} row(s) to {output_path}", flush=True)

    if args.combined_output:
        combined_path = Path(args.combined_output)
        write_csv(all_rows, combined_path, enrichment)
        print(f"Saved {len(all_rows)} combined row(s) to {combined_path}", flush=True)


if __name__ == "__main__":
    main()
