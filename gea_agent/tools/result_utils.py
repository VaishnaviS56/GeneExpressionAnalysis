from __future__ import annotations

from typing import Any


SUCCESS_STATUSES = {
    "ok",
    "not_found",
    "no_results",
    "no_query",
    "missing_field",
    "not_list_like",
}


def sanitize_exception_message(exc: Exception, *, fallback: str = "The operation failed.") -> str:
    text = " ".join(str(exc or "").split()).strip()
    if not text:
        return fallback
    if len(text) > 300:
        text = text[:297] + "..."
    return text


def tool_error_result(
    tool_name: str,
    message: str,
    *,
    answer: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    clean_message = " ".join(str(message or "").split()).strip() or f"{tool_name} failed."
    return {
        "status": "error",
        "message": clean_message,
        "answer": answer if answer is not None else clean_message,
        **extra,
    }


def normalize_tool_result(tool_name: str, result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {"status": "ok", "answer": str(result or "").strip()}

    status = str(payload.get("status") or "").strip().lower()
    if not status:
        payload["status"] = "ok"
        status = "ok"

    message = " ".join(str(payload.get("message") or "").split()).strip()
    answer = str(payload.get("answer") or "").strip()

    if status in SUCCESS_STATUSES:
        if not message and answer:
            payload["message"] = answer
        elif message:
            payload["message"] = message
        return payload

    if not message:
        message = answer or f"{tool_name} failed."
    payload["message"] = message
    if not answer:
        payload["answer"] = message
    return payload

