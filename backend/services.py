from __future__ import annotations

from datetime import datetime
from typing import Any

from gea_agent.agent.graph import build_app

from backend.db import fetch_all, fetch_one, execute, init_db, json_dumps, json_loads, now_iso
from backend.security import hash_password, hash_token, new_token, verify_password


_AGENT_APP = None


def _normalize_agent_type(agent_type: str | None) -> str:
    value = str(agent_type or "general").strip().lower()
    return "literature" if value == "literature" else "general"


def get_agent_app():
    global _AGENT_APP
    if _AGENT_APP is None:
        _AGENT_APP = build_app()
    return _AGENT_APP


def ensure_initialized() -> None:
    init_db()


def register_user(email: str, password: str, display_name: str | None = None) -> dict[str, Any]:
    display_name = (display_name or email.split("@")[0]).strip() or "User"
    existing = fetch_one("SELECT id FROM users WHERE email = ?", (email.lower().strip(),))
    if existing:
        raise ValueError("Email already exists")

    salt, password_hash = hash_password(password)
    user_id = execute(
        """
        INSERT INTO users (email, display_name, password_salt, password_hash, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (email.lower().strip(), display_name, salt, password_hash, now_iso()),
    )
    token = create_session(user_id)
    return {
        "token": token,
        "user": get_user(user_id),
    }


def login_user(email: str, password: str) -> dict[str, Any]:
    user = fetch_one("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
    if not user:
        raise ValueError("Invalid credentials")
    if not verify_password(password, user["password_salt"], user["password_hash"]):
        raise ValueError("Invalid credentials")
    token = create_session(int(user["id"]))
    return {"token": token, "user": get_user(int(user["id"]))}


def create_session(user_id: int) -> str:
    token = new_token()
    execute(
        """
        INSERT INTO sessions (user_id, token_hash, created_at, last_used_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, hash_token(token), now_iso(), now_iso()),
    )
    return token


def get_user_by_token(token: str) -> dict[str, Any] | None:
    session = fetch_one("SELECT user_id FROM sessions WHERE token_hash = ?", (hash_token(token),))
    if not session:
        return None
    execute(
        "UPDATE sessions SET last_used_at = ? WHERE token_hash = ?",
        (now_iso(), hash_token(token)),
    )
    return get_user(int(session["user_id"]))


def get_user(user_id: int) -> dict[str, Any] | None:
    return fetch_one(
        "SELECT id, email, display_name, created_at FROM users WHERE id = ?",
        (user_id,),
    )


def create_chat(user_id: int, title: str | None = None, agent_type: str | None = None) -> dict[str, Any]:
    resolved_title = (title or "New chat").strip() or "New chat"
    resolved_agent_type = _normalize_agent_type(agent_type)
    initial_arm = "literature" if resolved_agent_type == "literature" else "general"
    chat_id = execute(
        """
        INSERT INTO chats (
            user_id, title, agent_type, analysis_arm, srp_ids_json, memory_deg_genes_json,
            memory_deg_analysis_json, memory_control_name, memory_test_name, memory_enrichr_json, memory_rwr_seed_genes_json, memory_rwr_genes_json, memory_disease_name, memory_openalex_genes_json,
            last_meta_json,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, '[]', '[]', '{}', '', '', '{}', '[]', '[]', '', '[]', '{}', ?, ?)
        """,
        (user_id, resolved_title, resolved_agent_type, initial_arm, now_iso(), now_iso()),
    )
    return get_chat(user_id, chat_id)


def get_chat(user_id: int, chat_id: int) -> dict[str, Any] | None:
    chat = fetch_one(
        "SELECT * FROM chats WHERE id = ? AND user_id = ?",
        (chat_id, user_id),
    )
    if not chat:
        return None
    return _enrich_chat(chat)


def list_chats(user_id: int) -> list[dict[str, Any]]:
    chats = fetch_all(
        "SELECT * FROM chats WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
        (user_id,),
    )
    return [_enrich_chat(chat) for chat in chats]


def list_messages(user_id: int, chat_id: int) -> list[dict[str, Any]]:
    chat = get_chat(user_id, chat_id)
    if not chat:
        raise ValueError("Chat not found")
    return fetch_all(
        "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    )


def append_message(chat_id: int, role: str, content: str) -> None:
    execute(
        "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, now_iso()),
    )
    execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now_iso(), chat_id))


def update_chat_memory(chat_id: int, *, result_meta: dict[str, Any] | None) -> None:
    if not isinstance(result_meta, dict):
        return

    current = fetch_one("SELECT * FROM chats WHERE id = ?", (chat_id,))
    if not current:
        return

    analysis_arm = str(result_meta.get("analysis_arm") or current.get("analysis_arm") or "general")
    srp_ids = result_meta.get("srp_ids", json_loads(current.get("srp_ids_json"), []))
    memory_deg_genes = result_meta.get("deg_genes", json_loads(current.get("memory_deg_genes_json"), []))
    memory_deg_analysis = result_meta.get("deg_analysis", json_loads(current.get("memory_deg_analysis_json"), {}))
    memory_control_name = result_meta.get("control_name", current.get("memory_control_name", ""))
    memory_test_name = result_meta.get("test_name", current.get("memory_test_name", ""))
    memory_enrichr = result_meta.get("enrichr", json_loads(current.get("memory_enrichr_json"), {}))
    memory_rwr_seed_genes = result_meta.get("rwr_seed_genes", json_loads(current.get("memory_rwr_seed_genes_json"), []))
    memory_rwr_genes = result_meta.get("rwr_genes", json_loads(current.get("memory_rwr_genes_json"), []))
    memory_disease_name = result_meta.get("disease_name", current.get("memory_disease_name", ""))
    memory_openalex_genes = result_meta.get("openalex_genes", json_loads(current.get("memory_openalex_genes_json"), []))
    last_meta = result_meta if isinstance(result_meta, dict) else json_loads(current.get("last_meta_json"), {})

    execute(
        """
        UPDATE chats
        SET analysis_arm = ?,
            srp_ids_json = ?,
            memory_deg_genes_json = ?,
            memory_deg_analysis_json = ?,
            memory_control_name = ?,
            memory_test_name = ?,
            memory_enrichr_json = ?,
            memory_rwr_seed_genes_json = ?,
            memory_rwr_genes_json = ?,
            memory_disease_name = ?,
            memory_openalex_genes_json = ?,
            last_meta_json = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            analysis_arm,
            json_dumps(srp_ids if isinstance(srp_ids, list) else []),
            json_dumps(memory_deg_genes if isinstance(memory_deg_genes, list) else []),
            json_dumps(memory_deg_analysis if isinstance(memory_deg_analysis, dict) else {}),
            memory_control_name if isinstance(memory_control_name, str) else "",
            memory_test_name if isinstance(memory_test_name, str) else "",
            json_dumps(memory_enrichr if isinstance(memory_enrichr, dict) else {}),
            json_dumps(memory_rwr_seed_genes if isinstance(memory_rwr_seed_genes, list) else []),
            json_dumps(memory_rwr_genes if isinstance(memory_rwr_genes, list) else []),
            memory_disease_name if isinstance(memory_disease_name, str) else "",
            json_dumps(memory_openalex_genes if isinstance(memory_openalex_genes, list) else []),
            json_dumps(last_meta if isinstance(last_meta, dict) else {}),
            now_iso(),
            chat_id,
        ),
    )


def build_memory_summary(chat: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    if chat.get("memory_deg_genes"):
        parts.append(f"Stored DEG genes: {len(chat['memory_deg_genes'])}.")
    if chat.get("memory_control_name") or chat.get("memory_test_name"):
        parts.append(
            f"Stored DEG comparison: control={chat.get('memory_control_name', '') or 'NA'}, "
            f"test={chat.get('memory_test_name', '') or 'NA'}."
        )
    if chat.get("memory_enrichr"):
        libraries = chat["memory_enrichr"].get("libraries", {}) if isinstance(chat["memory_enrichr"], dict) else {}
        if isinstance(libraries, dict) and libraries:
            parts.append(f"Stored pathway results: {len(libraries)} libraries.")
    if chat.get("memory_rwr_seed_genes"):
        parts.append(f"Stored RWR seed genes: {len(chat['memory_rwr_seed_genes'])}.")
    if chat.get("memory_rwr_genes"):
        parts.append(f"Stored RWR targets: {len(chat['memory_rwr_genes'])}.")
    if chat.get("memory_disease_name"):
        parts.append(f"Last disease context: {chat['memory_disease_name']}.")
    if chat.get("memory_openalex_genes"):
        parts.append(f"Stored disease literature genes: {len(chat['memory_openalex_genes'])}.")

    recent = messages[-4:]
    if recent:
        transcript = []
        for message in recent:
            role = message.get("role", "")
            content = " ".join(str(message.get("content", "")).split())
            if content:
                transcript.append(f"{role}: {content[:160]}")
        if transcript:
            parts.append("Recent turns: " + " | ".join(transcript))

    return "\n".join(parts)


def _chat_preview_fields(chat_id: int) -> tuple[int, str, str]:
    count_row = fetch_one(
        "SELECT COUNT(*) AS message_count FROM messages WHERE chat_id = ?",
        (chat_id,),
    ) or {}
    latest = fetch_one(
        """
        SELECT role, content
        FROM messages
        WHERE chat_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (chat_id,),
    ) or {}
    preview = " ".join(str(latest.get("content", "")).split())
    return (
        int(count_row.get("message_count") or 0),
        str(latest.get("role") or ""),
        preview[:180],
    )


def _enrich_chat(chat: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(chat)
    enriched["agent_type"] = _normalize_agent_type(enriched.get("agent_type"))
    enriched["srp_ids"] = json_loads(enriched.get("srp_ids_json"), [])
    enriched["memory_deg_genes"] = json_loads(enriched.get("memory_deg_genes_json"), [])
    enriched["memory_deg_analysis"] = json_loads(enriched.get("memory_deg_analysis_json"), {})
    enriched["memory_control_name"] = enriched.get("memory_control_name", "")
    enriched["memory_test_name"] = enriched.get("memory_test_name", "")
    enriched["memory_enrichr"] = json_loads(enriched.get("memory_enrichr_json"), {})
    enriched["memory_rwr_seed_genes"] = json_loads(enriched.get("memory_rwr_seed_genes_json"), [])
    enriched["memory_rwr_genes"] = json_loads(enriched.get("memory_rwr_genes_json"), [])
    enriched["memory_openalex_genes"] = json_loads(enriched.get("memory_openalex_genes_json"), [])
    enriched["last_meta"] = json_loads(enriched.get("last_meta_json"), {})
    message_count, last_message_role, last_message_preview = _chat_preview_fields(int(enriched["id"]))
    enriched["message_count"] = message_count
    enriched["last_message_role"] = last_message_role
    enriched["last_message_preview"] = last_message_preview
    return enriched


def _memory_value_from_chat(chat: dict[str, Any], key: str, default: Any) -> Any:
    if key in chat and chat.get(key) not in (None, "", [], {}):
        return chat.get(key)
    last_meta = chat.get("last_meta")
    if isinstance(last_meta, dict):
        value = last_meta.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _invoke_agent_for_chat(chat: dict[str, Any], content: str, memory_summary: str) -> dict[str, Any]:
    agent = get_agent_app()
    return agent.invoke(
        {
            "query": content,
            "memory_summary": memory_summary,
            "memory_deg_genes": _memory_value_from_chat(chat, "memory_deg_genes", []),
            "memory_upregulated_genes": _memory_value_from_chat(chat, "memory_upregulated_genes", []),
            "memory_downregulated_genes": _memory_value_from_chat(chat, "memory_downregulated_genes", []),
            "memory_deg_analysis": _memory_value_from_chat(chat, "memory_deg_analysis", {}),
            "memory_deg_gene_records": _memory_value_from_chat(chat, "memory_deg_gene_records", []),
            "memory_control_name": _memory_value_from_chat(chat, "memory_control_name", ""),
            "memory_test_name": _memory_value_from_chat(chat, "memory_test_name", ""),
            "memory_enrichr": _memory_value_from_chat(chat, "memory_enrichr", {}),
            "memory_rwr_seed_genes": _memory_value_from_chat(chat, "memory_rwr_seed_genes", []),
            "memory_rwr_genes": _memory_value_from_chat(chat, "memory_rwr_genes", []),
            "memory_disease_name": _memory_value_from_chat(chat, "memory_disease_name", ""),
            "memory_openalex_genes": _memory_value_from_chat(chat, "memory_openalex_genes", []),
            "memory_slice_result": _memory_value_from_chat(chat, "memory_slice_result", {}),
        }
    )


def handle_chat_message(user_id: int, chat_id: int, content: str) -> dict[str, Any]:
    chat = get_chat(user_id, chat_id)
    if not chat:
        raise ValueError("Chat not found")

    append_message(chat_id, "user", content)
    messages = list_messages(user_id, chat_id)
    memory_summary = build_memory_summary(chat, messages)
    result = _invoke_agent_for_chat(chat, content, memory_summary)

    answer = result.get("answer", "")
    append_message(chat_id, "assistant", answer)
    update_chat_memory(chat_id, result_meta=result.get("meta"))

    return {
        "answer": answer,
        "meta": result.get("meta", {}),
        "chat": get_chat(user_id, chat_id),
        "messages": list_messages(user_id, chat_id),
    }
