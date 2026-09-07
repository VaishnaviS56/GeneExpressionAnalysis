from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def find_dotenv(*args, **kwargs) -> str:
        return ""

    def load_dotenv(*args, **kwargs) -> bool:
        return False


DEFAULT_QUALITY_GLOB = str(Path(__file__).with_name("quality_*.csv"))
SKIP_BASENAMES = {"quality_results_review_combined.csv"}
RERUN_FIELDNAMES = [
    "quality_rerun_at",
    "quality_rerun_provider",
    "quality_rerun_experiment",
    "quality_rerun_error",
]


def _infer_provider(path: Path) -> str:
    name = path.stem.lower()
    if "claude" in name:
        return "claude"
    if "gemma" in name:
        return "gemma"
    if "gemini" in name:
        return "gemini"
    return str(os.getenv("LLM_PROVIDER") or "auto").strip().lower() or "auto"


def _expand_files(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.is_file() and path.name not in SKIP_BASENAMES and path not in paths:
                paths.append(path)
    return sorted(paths)


def _plain_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return _plain_value(value.model_dump(mode="json"))
        except TypeError:
            return _plain_value(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _plain_value(value.dict())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _plain_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item) for item in value]
    return value


def _compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    for field in RERUN_FIELDNAMES:
        if field not in fieldnames:
            fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _needs_rerun(row: dict[str, str]) -> bool:
    if not str(row.get("final_answer") or "").strip():
        return True
    retry_generated = bool(str(row.get("final_answer_retry_at") or "").strip())
    has_target_error = bool(str(row.get("target_error") or "").strip())
    if retry_generated and has_target_error:
        return True
    retry_error = str(row.get("final_answer_retry_error") or "").strip().lower()
    return retry_generated and "not a final-stage-only failure" in retry_error


def _case_ids_to_retry(rows: list[dict[str, str]], limit: int | None) -> list[str]:
    case_ids: list[str] = []
    for row in rows:
        if not _needs_rerun(row):
            continue
        case_id = str(row.get("test_id") or "").strip()
        if not case_id or case_id in case_ids:
            continue
        case_ids.append(case_id)
        if limit is not None and len(case_ids) >= limit:
            break
    return case_ids


def _is_single_case(case_id: str) -> bool:
    return case_id.upper().startswith("ST-")


def _load_examples_for_cases(case_ids: list[str]) -> dict[str, Any]:
    from evaluators.utils import (
        MULTI_TURN_DATASET,
        SINGLE_TURN_DATASET,
        example_identifier,
        list_dataset_examples,
    )

    need_single = any(_is_single_case(case_id) for case_id in case_ids)
    need_multi = any(not _is_single_case(case_id) for case_id in case_ids)
    examples: dict[str, Any] = {}
    if need_single:
        for example in list_dataset_examples(SINGLE_TURN_DATASET):
            examples[example_identifier(example)] = example
    if need_multi:
        for example in list_dataset_examples(MULTI_TURN_DATASET):
            examples[example_identifier(example)] = example
    return examples


def _evaluate_case(case_id: str, *, provider: str, max_concurrency: int):
    from langsmith.evaluation import evaluate

    from evaluators.multi_turn import run_multi_turn_example
    from evaluators.run_quality_evals import QUALITY_EVALUATORS
    from evaluators.single_turn import run_single_turn_example

    examples = _load_examples_for_cases([case_id])
    example = examples.get(case_id)
    if example is None:
        raise RuntimeError(f"No LangSmith dataset example found for {case_id}.")

    target = run_single_turn_example if _is_single_case(case_id) else run_multi_turn_example
    prefix = (
        f"target-discovery-{'single' if _is_single_case(case_id) else 'multi'}-turn"
        f"-quality-rerun-{provider}-{case_id.lower()}"
    )
    results = evaluate(
        target,
        data=[example],
        evaluators=QUALITY_EVALUATORS,
        experiment_prefix=prefix,
        max_concurrency=max_concurrency,
    )
    result_rows = list(results)
    if not result_rows:
        raise RuntimeError(f"LangSmith evaluate returned no rows for {case_id}.")
    return results, result_rows[0]


def _update_from_result(row: dict[str, str], result_row: Any, experiment_name: str, provider: str) -> bool:
    from evaluators.quality_results_to_csv import (
        _extract_called_tools,
        _extract_final_answer,
        _extract_tool_history,
        _extract_tool_outputs_text,
    )

    run = _row_value(result_row, "run")
    run_data = _plain_value(run)
    if not isinstance(run_data, dict):
        run_data = {}
    outputs = run_data.get("outputs")
    final_answer = _extract_final_answer(outputs)
    if not final_answer:
        raise RuntimeError("Rerun completed but target run still has no final_answer.")

    evaluation_results = _row_value(result_row, "evaluation_results")
    for result in _row_value(evaluation_results, "results", []) or []:
        key = str(_result_value(result, "key", "") or "")
        if key not in {"response_groundedness", "evidence_validity"}:
            continue
        score = _result_value(result, "score", "")
        comment = _result_value(result, "comment", "")
        row[f"{key}_score"] = "" if score is None else str(score)
        row[f"{key}_comment"] = "" if comment is None else str(comment)

    tool_history = _extract_tool_history(outputs if isinstance(outputs, dict) else {})
    row["final_answer"] = final_answer
    row["target_run_id"] = str(run_data.get("id") or row.get("target_run_id") or "")
    row["target_trace_id"] = str(run_data.get("trace_id") or row.get("target_trace_id") or "")
    row["target_status"] = str(run_data.get("status") or "")
    row["target_error"] = str(run_data.get("error") or "")
    row["target_start_time"] = str(run_data.get("start_time") or "")
    row["target_end_time"] = str(run_data.get("end_time") or "")
    row["called_tools_json"] = _compact_json(_extract_called_tools(tool_history))
    row["tool_history_json"] = _compact_json(tool_history)
    row["tool_outputs_text"] = _extract_tool_outputs_text(tool_history)
    row["experiment"] = experiment_name
    row["quality_rerun_at"] = datetime.now(timezone.utc).isoformat()
    row["quality_rerun_provider"] = provider
    row["quality_rerun_experiment"] = experiment_name
    row["quality_rerun_error"] = ""
    return True


def process_file(
    path: Path,
    *,
    provider: str,
    limit: int | None,
    max_concurrency: int,
    dry_run: bool,
) -> tuple[int, int, int]:
    fieldnames, rows = _load_rows(path)
    for field in RERUN_FIELDNAMES:
        if field not in fieldnames:
            fieldnames.append(field)

    case_ids = _case_ids_to_retry(rows, limit)
    attempted = updated = failed = 0
    for case_id in case_ids:
        attempted += 1
        matching_rows = [row for row in rows if row.get("test_id") == case_id and _needs_rerun(row)]
        try:
            results, result_row = _evaluate_case(case_id, provider=provider, max_concurrency=max_concurrency)
            experiment_name = str(getattr(results, "experiment_name", "") or "")
            for row in matching_rows:
                _update_from_result(row, result_row, experiment_name, provider)
                updated += 1
            print(f"{path.name}: {case_id} updated from {experiment_name}", flush=True)
        except Exception as exc:
            failed += len(matching_rows) or 1
            message = str(exc)
            for row in matching_rows:
                row["quality_rerun_at"] = datetime.now(timezone.utc).isoformat()
                row["quality_rerun_provider"] = provider
                row["quality_rerun_error"] = message
            print(f"{path.name}: {case_id} failed: {message}", flush=True)

    if not dry_run and (updated or failed):
        _write_rows(path, fieldnames, rows)
    return attempted, updated, failed


def _run_provider_child(provider: str, files: list[Path], args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "evaluators.rerun_quality_missing",
        "--worker",
        "--provider",
        provider,
        "--max-concurrency",
        str(args.max_concurrency),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.dry_run:
        command.append("--dry-run")
    command.extend(str(path) for path in files)

    env = os.environ.copy()
    env["LLM_PROVIDER"] = provider
    return subprocess.run(command, cwd=Path.cwd(), env=env, check=False).returncode


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)

    parser = argparse.ArgumentParser(
        description="Rerun full LangSmith quality evals for CSV rows with empty final_answer and update the CSVs."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="quality_*.csv files. Defaults to evaluators/quality_*.csv, excluding the combined CSV.",
    )
    parser.add_argument("--provider", help="LLM provider to use. Defaults to inferring from each CSV filename.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum distinct empty test IDs to retry per file.")
    parser.add_argument("--max-concurrency", type=int, default=0, help="LangSmith evaluate concurrency.")
    parser.add_argument("--dry-run", action="store_true", help="Run evals but do not write CSV changes.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than 0.")

    files = _expand_files(args.files or [DEFAULT_QUALITY_GLOB])
    if not files:
        raise SystemExit("No quality CSV files matched.")

    if not args.worker:
        files_by_provider: dict[str, list[Path]] = {}
        for path in files:
            provider = args.provider or _infer_provider(path)
            files_by_provider.setdefault(provider, []).append(path)

        exit_code = 0
        for provider, provider_files in files_by_provider.items():
            print(f"Running {len(provider_files)} file(s) with LLM_PROVIDER={provider}", flush=True)
            code = _run_provider_child(provider, provider_files, args)
            exit_code = exit_code or code
        raise SystemExit(exit_code)

    provider = args.provider or str(os.getenv("LLM_PROVIDER") or "auto").strip().lower() or "auto"
    os.environ["LLM_PROVIDER"] = provider

    total_attempted = total_updated = total_failed = 0
    for path in files:
        attempted, updated, failed = process_file(
            path,
            provider=provider,
            limit=args.limit,
            max_concurrency=args.max_concurrency,
            dry_run=args.dry_run,
        )
        total_attempted += attempted
        total_updated += updated
        total_failed += failed
        print(f"{path.name}: attempted={attempted}, updated={updated}, failed={failed}", flush=True)

    print(f"Total: attempted={total_attempted}, updated={total_updated}, failed={total_failed}", flush=True)


if __name__ == "__main__":
    main()
