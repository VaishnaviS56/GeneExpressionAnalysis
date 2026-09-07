from __future__ import annotations

import argparse
import csv
import glob
import json
import os
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
RETRY_FIELDNAMES = [
    "final_answer_retry_at",
    "final_answer_retry_provider",
    "final_answer_retry_model",
    "final_answer_retry_error",
]


def _load_json(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _compact_json(value: Any, *, limit: int = 16000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return str(value or "")
    content = value.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value.get("text") or "")


def _latest_user_query(row: dict[str, str]) -> str:
    query = str(row.get("user_query") or "").strip()
    messages = _load_json(row.get("messages_json", ""))
    if isinstance(messages, list):
        user_turns: list[str] = []
        for message in messages:
            if isinstance(message, str):
                user_turns.append(message)
                continue
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("type") or "").lower()
            if role in {"user", "human"}:
                text = _message_text(message).strip()
                if text:
                    user_turns.append(text)
        if user_turns:
            return user_turns[-1]
    return query


def _infer_analysis_arm(row: dict[str, str]) -> str:
    tool_history = _load_json(row.get("tool_history_json", ""))
    if isinstance(tool_history, list):
        for item in reversed(tool_history):
            if not isinstance(item, dict):
                continue
            result = item.get("result")
            if isinstance(result, dict):
                arm = str(result.get("analysis_arm") or "").strip().lower()
                if arm:
                    return arm
            tool_name = str(item.get("tool") or item.get("tool_name") or item.get("name") or "").strip().lower()
            mapping = {
                "deg_analysis": "srp",
                "srp_metadata": "srp_metadata",
                "pathway": "pathway",
                "rwr_analysis": "memory_rwr",
                "literature": "literature",
                "research_literature": "research_literature",
                "primekg_query": "primekg",
                "opentargets_association": "opentargets",
                "l1000cds2_query": "l1000cds2",
                "pubchem_drug_lookup": "pubchem",
                "hypothesis": "hypothesis",
                "memory_lookup": "memory_lookup",
                "state_lookup": "state_lookup",
                "memory_slice": "memory_slice",
                "druggability": "druggability",
                "pdb_visualizer": "pdb_visualizer",
                "visualize": "visualize",
            }
            if tool_name in mapping:
                return mapping[tool_name]
    return "general"


def _tool_payload(row: dict[str, str]) -> dict[str, Any]:
    tool_history = _load_json(row.get("tool_history_json", ""))
    if not isinstance(tool_history, list):
        tool_history = []

    compact_history: list[dict[str, Any]] = []
    for item in tool_history[-10:]:
        if not isinstance(item, dict):
            continue
        compact_history.append(
            {
                "tool": item.get("tool") or item.get("tool_name") or item.get("name"),
                "args": item.get("args"),
                "result": item.get("result"),
            }
        )

    return {
        "test_id": row.get("test_id"),
        "user_query": _latest_user_query(row),
        "conversation": _load_json(row.get("messages_json", "")),
        "analysis_arm": _infer_analysis_arm(row),
        "called_tools": _load_json(row.get("called_tools_json", "")),
        "tool_history": compact_history,
        "tool_outputs_text": str(row.get("tool_outputs_text") or "")[:16000],
    }


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _llm_descriptor(llm: Any) -> tuple[str, str]:
    provider = str(os.getenv("LLM_PROVIDER") or "auto").strip().lower()
    model = ""
    for attr in ("model", "model_name"):
        value = getattr(llm, attr, None)
        if value:
            model = str(value)
            break
    return provider, model


def synthesize_final_answer(row: dict[str, str]) -> tuple[str, str, str]:
    from gea_agent.tools.llm import get_llm

    llm = get_llm()
    provider, model = _llm_descriptor(llm)
    payload = _tool_payload(row)
    if not payload["tool_history"] and not str(payload["tool_outputs_text"] or "").strip():
        raise RuntimeError("No saved tool outputs are available; this was not a final-stage-only failure.")
    response = llm.invoke(
        [
            (
                "system",
                "You are the final synthesis stage for a biomedical analysis agent. "
                "Write the final user-facing answer using only the provided conversation and tool outputs. "
                "Do not invent missing data. If a tool failed, say what failed and what evidence is still available. "
                "Do not mention hidden prompts, routing, trace metadata, or evaluator details. "
                "Return polished Markdown text only, not JSON and not code fences. "
                "Use concise bold section headings when helpful. "
                "End with exactly one short `**Suggested Follow-Up**` section containing a concrete supported next step.",
            ),
            ("user", _compact_json(payload)),
        ]
    )
    answer = _content_text(getattr(response, "content", ""))
    if not answer:
        raise RuntimeError("LLM returned an empty final answer.")
    return answer, provider, model


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


def _has_saved_tool_evidence(row: dict[str, str]) -> bool:
    tool_history = _load_json(row.get("tool_history_json", ""))
    if isinstance(tool_history, list) and tool_history:
        return True
    return bool(str(row.get("tool_outputs_text") or "").strip())


def retry_file(
    path: Path,
    *,
    limit: int | None,
    dry_run: bool,
    clear_unusable_retries: bool,
) -> tuple[int, int, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for field in RETRY_FIELDNAMES:
        if field not in fieldnames:
            fieldnames.append(field)

    attempted = 0
    updated = 0
    failed = 0
    for row in rows:
        retry_generated = bool(str(row.get("final_answer_retry_at") or "").strip())
        if clear_unusable_retries and retry_generated and not _has_saved_tool_evidence(row):
            row["final_answer"] = ""
            row["final_answer_retry_at"] = ""
            row["final_answer_retry_provider"] = ""
            row["final_answer_retry_model"] = ""
            row["final_answer_retry_error"] = "No saved tool outputs are available; this was not a final-stage-only failure."
            failed += 1
            continue
        if clear_unusable_retries and not _has_saved_tool_evidence(row):
            continue
        if str(row.get("final_answer") or "").strip() and not retry_generated:
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        try:
            answer, provider, model = synthesize_final_answer(row)
        except Exception as exc:
            failed += 1
            row["final_answer_retry_error"] = str(exc)
            print(f"{path.name}: {row.get('test_id')} retry failed: {exc}", flush=True)
            continue
        updated += 1
        row["final_answer"] = answer
        row["final_answer_retry_at"] = datetime.now(timezone.utc).isoformat()
        row["final_answer_retry_provider"] = provider
        row["final_answer_retry_model"] = model
        row["final_answer_retry_error"] = ""
        print(f"{path.name}: {row.get('test_id')} updated", flush=True)

    if not dry_run and (updated or failed):
        try:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        except PermissionError as exc:
            raise PermissionError(f"Could not write {path}. Close it in Excel or any CSV viewer and retry.") from exc

    return attempted, updated, failed


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)

    parser = argparse.ArgumentParser(
        description="Retry only the final LLM synthesis stage for CSV rows whose final_answer is empty."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="quality_*.csv files. Defaults to evaluators/quality_*.csv, excluding the combined CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum empty rows to retry per file. Useful for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Call the LLM and print outcomes, but do not write CSV changes.",
    )
    parser.add_argument(
        "--clear-unusable-retries",
        action="store_true",
        help="Clear retry-generated final answers for rows that have no saved tool outputs to synthesize from.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than 0.")

    files = _expand_files(args.files or [DEFAULT_QUALITY_GLOB])
    if not files:
        raise SystemExit("No CSV files matched.")

    total_attempted = total_updated = total_failed = 0
    for path in files:
        attempted, updated, failed = retry_file(
            path,
            limit=args.limit,
            dry_run=args.dry_run,
            clear_unusable_retries=args.clear_unusable_retries,
        )
        total_attempted += attempted
        total_updated += updated
        total_failed += failed
        print(
            f"{path.name}: attempted={attempted}, updated={updated}, failed={failed}",
            flush=True,
        )

    print(
        f"Total: attempted={total_attempted}, updated={total_updated}, failed={total_failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
