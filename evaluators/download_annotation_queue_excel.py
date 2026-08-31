from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import zipfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID
from xml.sax.saxutils import escape

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

from evaluators.human_review import QUEUE_NAME


DEFAULT_OUTPUT = Path(__file__).with_name("annotation_queue.xlsx")
DEFAULT_ROWS_PER_FILE = 1000

try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional dependency
    find_dotenv = None
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None or find_dotenv is None:
        return
    load_dotenv(find_dotenv(usecwd=True), override=False)


def _has_langsmith_key() -> bool:
    return bool(str(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip())


def _plain_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _plain_value(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _plain_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (UUID, Decimal)):
        return str(value)
    return value


def _compact_json(value: Any) -> str:
    value = _plain_value(value)
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except TypeError:
        return str(value)


def _excel_text(value: Any) -> str:
    text = _compact_json(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text[:32767]


def _get_field(row: Any, key: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _run_url(run: Any) -> str:
    app_path = _get_field(run, "app_path")
    host_url = str(_get_field(run, "_host_url") or "https://smith.langchain.com").rstrip("/")
    if app_path:
        return f"{host_url}{app_path}"
    run_id = _get_field(run, "id")
    return f"{host_url}/public/{run_id}/r" if run_id else ""


def _row_from_run(index: int, queue: Any, run: Any, feedback_by_run_id: dict[str, list[Any]]) -> dict[str, Any]:
    run_id = str(_get_field(run, "id") or "")
    feedback = feedback_by_run_id.get(run_id, [])
    return {
        "queue_name": _get_field(queue, "name"),
        "queue_id": _get_field(queue, "id"),
        "queue_index": index,
        "queue_added_at": _get_field(run, "added_at"),
        "last_reviewed_time": _get_field(run, "last_reviewed_time"),
        "run_id": run_id,
        "trace_id": _get_field(run, "trace_id"),
        "project_name": _get_field(run, "project_name") or _get_field(run, "session_name"),
        "name": _get_field(run, "name"),
        "run_type": _get_field(run, "run_type"),
        "status": _get_field(run, "status"),
        "start_time": _get_field(run, "start_time"),
        "end_time": _get_field(run, "end_time"),
        "latency_seconds": _get_field(run, "latency"),
        "error": _get_field(run, "error"),
        "inputs": _get_field(run, "inputs"),
        "outputs": _get_field(run, "outputs"),
        "reference_example_id": _get_field(run, "reference_example_id"),
        "reference_example": _get_field(run, "reference_example"),
        "feedback_stats": _get_field(run, "feedback_stats"),
        "feedback": feedback,
        "url": _run_url(run),
    }


def _resolve_queue(client: Client, queue_name: str):
    try:
        queues = list(client.list_annotation_queues(name=queue_name, limit=1))
    except Exception as exc:
        text = str(exc)
        if "401" in text or "unauthorized" in text.lower():
            raise SystemExit(
                "LangSmith rejected the API request with 401 Unauthorized. "
                "Set a valid LANGSMITH_API_KEY for the workspace that owns the annotation queue, "
                "or refresh the key in your .env/PowerShell environment."
            ) from exc
        raise
    if not queues:
        raise SystemExit(f"Annotation queue not found: {queue_name!r}")
    return queues[0]


def _queue_exhausted(exc: Exception) -> bool:
    if isinstance(exc, LangSmithNotFoundError):
        return True
    text = str(exc).lower()
    return any(token in text for token in ("404", "not found", "out of range", "index"))


def _get_queue_run(client: Client, queue_id: Any, index: int) -> tuple[int, Any | None, Exception | None]:
    try:
        return index, client.get_run_from_annotation_queue(queue_id, index=index), None
    except Exception as exc:
        if _queue_exhausted(exc):
            return index, None, None
        return index, None, exc


def _download_queue_runs(
    client: Client,
    queue_id: Any,
    *,
    limit: int | None,
    workers: int,
    batch_size: int,
) -> list[Any]:
    runs_by_index: dict[int, Any] = {}
    next_index = 0
    workers = max(1, int(workers or 1))
    batch_size = max(1, int(batch_size or workers))

    while limit is None or next_index < limit:
        remaining = None if limit is None else max(0, limit - next_index)
        current_batch_size = batch_size if remaining is None else min(batch_size, remaining)
        if current_batch_size <= 0:
            break

        indices = list(range(next_index, next_index + current_batch_size))
        print(f"Fetching queue items {indices[0]}-{indices[-1]}...", flush=True)
        exhausted_at: int | None = None

        with ThreadPoolExecutor(max_workers=min(workers, len(indices))) as executor:
            futures = [executor.submit(_get_queue_run, client, queue_id, index) for index in indices]
            for future in as_completed(futures):
                index, run, exc = future.result()
                if exc is not None:
                    raise exc
                if run is None:
                    exhausted_at = index if exhausted_at is None else min(exhausted_at, index)
                    continue
                runs_by_index[index] = run

        if exhausted_at is not None:
            break
        next_index += current_batch_size

    return [runs_by_index[index] for index in sorted(runs_by_index)]


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _feedback_by_run_id(client: Client, runs: list[Any]) -> dict[str, list[Any]]:
    feedback_by_run_id: dict[str, list[Any]] = {}
    run_ids = [str(_get_field(run, "id")) for run in runs if _get_field(run, "id")]
    for chunk in _chunks(run_ids, 100):
        print(f"Fetching feedback for {len(chunk)} runs...", flush=True)
        for feedback in client.list_feedback(run_ids=chunk):
            run_id = str(_get_field(feedback, "run_id") or "")
            if run_id:
                feedback_by_run_id.setdefault(run_id, []).append(_plain_value(feedback))
    return feedback_by_run_id


def _rows_from_runs(
    runs: list[Any],
    *,
    queue: Any,
    start_index: int,
    feedback_by_run_id: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    return [
        _row_from_run(start_index + offset, queue, run, feedback_by_run_id)
        for offset, run in enumerate(runs)
    ]


def _download_queue_run_chunk(
    client: Client,
    queue_id: Any,
    *,
    start_index: int,
    row_count: int,
    workers: int,
    batch_size: int,
) -> tuple[list[Any], bool]:
    runs_by_index: dict[int, Any] = {}
    next_index = start_index
    workers = max(1, int(workers or 1))
    batch_size = max(1, int(batch_size or workers))
    exhausted = False

    while len(runs_by_index) < row_count:
        remaining = row_count - len(runs_by_index)
        current_batch_size = min(batch_size, remaining)
        indices = list(range(next_index, next_index + current_batch_size))
        print(f"Fetching queue items {indices[0]}-{indices[-1]}...", flush=True)
        exhausted_at: int | None = None

        with ThreadPoolExecutor(max_workers=min(workers, len(indices))) as executor:
            futures = [executor.submit(_get_queue_run, client, queue_id, index) for index in indices]
            for future in as_completed(futures):
                index, run, exc = future.result()
                if exc is not None:
                    raise exc
                if run is None:
                    exhausted_at = index if exhausted_at is None else min(exhausted_at, index)
                    continue
                runs_by_index[index] = run

        if exhausted_at is not None:
            exhausted = True
            break
        next_index += current_batch_size

    runs = [runs_by_index[index] for index in sorted(runs_by_index)]
    return runs, exhausted


def download_queue_rows(
    *,
    queue_name: str = QUEUE_NAME,
    limit: int | None = None,
    include_feedback: bool = True,
    workers: int = 8,
    batch_size: int = 25,
) -> list[dict[str, Any]]:
    _load_env()
    if not _has_langsmith_key():
        raise SystemExit(
            "LANGSMITH_API_KEY is not set. Add it to .env or set it in PowerShell before running this exporter."
        )
    client = Client()
    queue = _resolve_queue(client, queue_name)

    runs = _download_queue_runs(
        client,
        queue.id,
        limit=limit,
        workers=workers,
        batch_size=batch_size,
    )

    feedback_by_run_id: dict[str, list[Any]] = {}
    if include_feedback and runs:
        feedback_by_run_id = _feedback_by_run_id(client, runs)

    return _rows_from_runs(runs, queue=queue, start_index=0, feedback_by_run_id=feedback_by_run_id)


def _column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _sheet_xml(rows: list[dict[str, Any]], columns: list[str]) -> str:
    xml_rows: list[str] = []
    for row_index, row in enumerate([dict.fromkeys(columns, "")] + rows, start=1):
        cells: list[str] = []
        values = columns if row_index == 1 else [_excel_text(row.get(column, "")) for column in columns]
        for column_index, value in enumerate(values, start=1):
            cell_ref = f"{_column_letter(column_index)}{row_index}"
            cells.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">'
                f"{escape(str(value))}"
                "</t></is></c>"
            )
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        "</worksheet>"
    )


def write_xlsx(rows: list[dict[str, Any]], output_path: Path) -> None:
    columns = [
        "queue_name",
        "queue_id",
        "queue_index",
        "queue_added_at",
        "last_reviewed_time",
        "run_id",
        "trace_id",
        "project_name",
        "name",
        "run_type",
        "status",
        "start_time",
        "end_time",
        "latency_seconds",
        "error",
        "inputs",
        "outputs",
        "reference_example_id",
        "reference_example",
        "feedback_stats",
        "feedback",
        "url",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Annotation Queue" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        archive.writestr("xl/worksheets/sheet1.xml", _sheet_xml(rows, columns))


def _chunk_rows(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(size or DEFAULT_ROWS_PER_FILE))
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _part_output_path(output_path: Path, part_number: int, part_count: int) -> Path:
    if part_count <= 1:
        return output_path
    return output_path.with_name(f"{output_path.stem}_part_{part_number:03d}{output_path.suffix}")


def write_xlsx_parts(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    rows_per_file: int = DEFAULT_ROWS_PER_FILE,
) -> list[Path]:
    chunks = _chunk_rows(rows, rows_per_file)
    if not chunks:
        chunks = [[]]

    written_paths: list[Path] = []
    part_count = len(chunks)
    for part_number, chunk in enumerate(chunks, start=1):
        part_path = _part_output_path(output_path, part_number, part_count)
        write_xlsx(chunk, part_path)
        written_paths.append(part_path)
        print(f"Wrote part {part_number}/{part_count}: {part_path} ({len(chunk)} rows)", flush=True)
    return written_paths


def stream_queue_to_xlsx_parts(
    *,
    queue_name: str = QUEUE_NAME,
    output_path: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
    include_feedback: bool = True,
    workers: int = 8,
    batch_size: int = 25,
    rows_per_file: int = DEFAULT_ROWS_PER_FILE,
) -> list[Path]:
    _load_env()
    if not _has_langsmith_key():
        raise SystemExit(
            "LANGSMITH_API_KEY is not set. Add it to .env or set it in PowerShell before running this exporter."
        )

    client = Client()
    queue = _resolve_queue(client, queue_name)
    rows_per_file = max(1, int(rows_per_file or DEFAULT_ROWS_PER_FILE))

    written_paths: list[Path] = []
    start_index = 0
    part_number = 1
    while limit is None or start_index < limit:
        requested_rows = rows_per_file if limit is None else min(rows_per_file, limit - start_index)
        if requested_rows <= 0:
            break

        runs, exhausted = _download_queue_run_chunk(
            client,
            queue.id,
            start_index=start_index,
            row_count=requested_rows,
            workers=workers,
            batch_size=batch_size,
        )
        if not runs:
            break

        feedback_by_run_id = _feedback_by_run_id(client, runs) if include_feedback else {}
        rows = _rows_from_runs(
            runs,
            queue=queue,
            start_index=start_index,
            feedback_by_run_id=feedback_by_run_id,
        )
        part_path = _part_output_path(output_path, part_number, part_number + 1)
        write_xlsx(rows, part_path)
        written_paths.append(part_path)
        print(f"Wrote part {part_number}: {part_path} ({len(rows)} rows)", flush=True)

        start_index += len(runs)
        part_number += 1
        if exhausted or len(runs) < requested_rows:
            break

    if not written_paths:
        write_xlsx([], output_path)
        written_paths.append(output_path)
        print(f"Wrote empty annotation queue workbook: {output_path}", flush=True)

    return written_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a LangSmith annotation queue to an Excel .xlsx file.")
    parser.add_argument("--queue-name", default=QUEUE_NAME, help=f"LangSmith annotation queue name. Default: {QUEUE_NAME!r}.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Output .xlsx path. Default: {DEFAULT_OUTPUT}.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of queue items to download.")
    parser.add_argument("--no-feedback", action="store_true", help="Skip downloading feedback for queued runs.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel LangSmith queue item fetches. Default: 8.")
    parser.add_argument("--batch-size", type=int, default=25, help="Queue indices to request per batch. Default: 25.")
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=DEFAULT_ROWS_PER_FILE,
        help=f"Maximum queue rows per workbook. Default: {DEFAULT_ROWS_PER_FILE}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    written_paths = stream_queue_to_xlsx_parts(
        queue_name=args.queue_name,
        output_path=output_path,
        limit=args.limit,
        include_feedback=not args.no_feedback,
        workers=args.workers,
        batch_size=args.batch_size,
        rows_per_file=args.rows_per_file,
    )
    print(f"Wrote {len(written_paths)} Excel file(s).")


if __name__ == "__main__":
    main()
