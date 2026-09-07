from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "suite",
    "batch_number",
    "batch_start",
    "batch_end",
    "test_number",
    "test_id",
    "run_id",
    "trace_id",
    "evaluator_run_id",
    "evaluator_name",
    "evaluator_key",
    "score",
    "comment",
    "metrics",
    "experiment",
    "reference_example_id",
    "source_file",
]


def _get(data: Any, *path: str, default: Any = "") -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _row_from_trace(trace: dict[str, Any], source_file: Path) -> dict[str, str]:
    inputs = trace.get("inputs") if isinstance(trace.get("inputs"), dict) else {}
    example = inputs.get("example") if isinstance(inputs.get("example"), dict) else {}
    target_run = inputs.get("run") if isinstance(inputs.get("run"), dict) else {}
    outputs = trace.get("outputs") if isinstance(trace.get("outputs"), dict) else {}

    test_id = (
        _get(example, "metadata", "id")
        or _get(trace, "extra", "metadata", "reference_example_id")
        or _get(trace, "reference_example_id")
    )
    run_id = (
        target_run.get("id")
        or _get(trace, "extra", "metadata", "reference_run_id")
        or trace.get("id")
    )
    experiment = (
        _get(trace, "extra", "metadata", "experiment")
        or target_run.get("session_name")
        or trace.get("session_name")
    )

    return {
        "test_id": _as_text(test_id),
        "suite": "",
        "batch_number": "",
        "batch_start": "",
        "batch_end": "",
        "test_number": _as_text(_test_number_from_id(_as_text(test_id))),
        "run_id": _as_text(run_id),
        "trace_id": _as_text(trace.get("trace_id")),
        "evaluator_run_id": _as_text(trace.get("id")),
        "evaluator_name": _as_text(trace.get("name")),
        "evaluator_key": _as_text(outputs.get("key")),
        "score": _as_text(outputs.get("score")),
        "comment": _as_text(outputs.get("comment")),
        "metrics": "",
        "experiment": _as_text(experiment),
        "reference_example_id": _as_text(target_run.get("reference_example_id") or trace.get("reference_example_id")),
        "source_file": str(source_file),
    }


def _test_number_from_id(test_id: str) -> str:
    match = re.search(r"(\d+)$", test_id.strip())
    return str(int(match.group(1))) if match else ""


def _row_from_text_line(
    line: str,
    source_file: Path,
    *,
    suite: str = "",
    batch_number: str = "",
    batch_start: str = "",
    batch_end: str = "",
    experiment: str = "",
    row_index_in_batch: int = 0,
) -> dict[str, str] | None:
    match = re.search(r"-\s+([^|]+?)\s+\|\s+run=([^|\s]+)(?:\s+\|\s+(.*))?$", line)
    if not match:
        return None

    test_id = match.group(1).strip()
    test_number = _test_number_from_id(test_id)
    if not test_number and batch_start:
        test_number = str(int(batch_start) + row_index_in_batch)

    return {
        "suite": suite,
        "batch_number": batch_number,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "test_number": test_number,
        "test_id": test_id,
        "run_id": match.group(2).strip(),
        "trace_id": "",
        "evaluator_run_id": "",
        "evaluator_name": "",
        "evaluator_key": "",
        "score": "",
        "comment": "",
        "metrics": (match.group(3) or "").strip(),
        "experiment": experiment,
        "reference_example_id": "",
        "source_file": str(source_file),
    }


def _iter_rows(path: Path, *, result_batch_size: int | None = None):
    suite = ""
    batch_number = ""
    batch_start = ""
    batch_end = ""
    experiment = ""
    row_index_in_batch = 0

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue

            header_match = re.search(
                r"^\[([^\]]+)\]\s+batch\s+(\d+):\s+tests\s+(\d+)-(\d+)\s+of\s+\d+",
                raw,
            )
            if header_match:
                suite = header_match.group(1)
                batch_number = header_match.group(2)
                batch_start = header_match.group(3)
                batch_end = header_match.group(4)
                row_index_in_batch = 0
                continue

            experiment_match = re.search(r"^Experiment:\s+(.+)$", raw)
            if experiment_match:
                experiment = experiment_match.group(1).strip()
                continue

            try:
                trace = json.loads(raw)
            except json.JSONDecodeError:
                if result_batch_size and batch_number and not batch_start:
                    start = ((int(batch_number) - 1) * result_batch_size) + 1
                    batch_start = str(start)
                    batch_end = str(start + result_batch_size - 1)
                row = _row_from_text_line(
                    raw,
                    path,
                    suite=suite,
                    batch_number=batch_number,
                    batch_start=batch_start,
                    batch_end=batch_end,
                    experiment=experiment,
                    row_index_in_batch=row_index_in_batch,
                )
                if row is not None:
                    row_index_in_batch += 1
                    yield row
                continue

            if not isinstance(trace, dict):
                continue
            yield _row_from_trace(trace, path)


def convert_file(path: Path, *, result_batch_size: int | None, write_buffer_size: int) -> Path:
    output_path = path.with_suffix(".csv")
    total = 0
    batch: list[dict[str, str]] = []

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in _iter_rows(path, result_batch_size=result_batch_size):
            batch.append(row)
            if len(batch) >= write_buffer_size:
                writer.writerows(batch)
                total += len(batch)
                print(f"Wrote {total} row(s) to {output_path}")
                batch = []

        if batch:
            writer.writerows(batch)
            total += len(batch)

    print(f"Saved {total} row(s) to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert saved LangSmith trace JSONL/TXT files into CSV rows with test and run ids."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Trace file path(s). Wildcards are supported by PowerShell before Python receives them.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size used when the txt results were created, such as 1 or 5. Used to infer test numbers if needed.",
    )
    parser.add_argument(
        "--write-buffer-size",
        type=int,
        default=1000,
        help="Number of rows to buffer before writing to CSV.",
    )
    args = parser.parse_args()

    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0.")
    if args.write_buffer_size <= 0:
        raise SystemExit("--write-buffer-size must be greater than 0.")

    for filename in args.files:
        path = Path(filename)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        convert_file(
            path,
            result_batch_size=args.batch_size,
            write_buffer_size=args.write_buffer_size,
        )


if __name__ == "__main__":
    main()
