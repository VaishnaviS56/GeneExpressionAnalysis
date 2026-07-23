from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
import networkx as nx

from gea_agent.agent.graph import build_app

from backend.db import fetch_all, fetch_one, execute, init_db, json_dumps, json_loads, now_iso
from backend.security import hash_password, hash_token, new_token, verify_password


_AGENT_APP = None
TECHNICAL_ASSET_DIR = Path("backend/data/technical_assets")


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
            memory_upregulated_genes_json, memory_downregulated_genes_json,
            memory_deg_analysis_json, memory_deg_gene_records_json,
            memory_srp_metadata_result_json,
            memory_control_name, memory_test_name, memory_enrichr_json,
            memory_rwr_seed_genes_json, memory_rwr_genes_json,
            memory_disease_name, memory_openalex_genes_json,
            memory_opentargets_results_json, memory_l1000cds2_result_json,
            memory_pubchem_result_json, memory_hypothesis_result_json,
            memory_slice_result_json,
            last_meta_json,
            created_at, updated_at
        )
        VALUES (
            ?, ?, ?, ?, '[]', '[]', '[]', '[]', '{}', '[]', '{}', '', '', '{}',
            '[]', '[]', '', '[]', '[]', '{}', '{}', '{}', '{}', '{}', ?, ?
        )
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
    rows = fetch_all(
        "SELECT role, content, meta_json, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    )
    messages: list[dict[str, Any]] = []
    for row in rows:
        message = {
            "role": row.get("role"),
            "content": row.get("content"),
            "created_at": row.get("created_at"),
        }
        meta = json_loads(row.get("meta_json"), {})
        if isinstance(meta, dict) and meta:
            message["meta"] = meta
        messages.append(message)
    return messages


def append_message(chat_id: int, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
    execute(
        "INSERT INTO messages (chat_id, role, content, meta_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (chat_id, role, content, json_dumps(meta if isinstance(meta, dict) else {}), now_iso()),
    )
    execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now_iso(), chat_id))


def _first_non_empty(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return default


def update_chat_memory(chat_id: int, *, result_meta: dict[str, Any] | None) -> None:
    if not isinstance(result_meta, dict):
        return

    current = fetch_one("SELECT * FROM chats WHERE id = ?", (chat_id,))
    if not current:
        return

    analysis_arm = str(result_meta.get("analysis_arm") or current.get("analysis_arm") or "general")
    srp_ids = _first_non_empty(
        result_meta.get("srp_ids"),
        json_loads(current.get("srp_ids_json"), []),
        default=[],
    )
    memory_deg_genes = _first_non_empty(
        result_meta.get("deg_genes"),
        result_meta.get("memory_deg_genes"),
        json_loads(current.get("memory_deg_genes_json"), []),
        default=[],
    )
    memory_upregulated_genes = _first_non_empty(
        result_meta.get("upregulated_genes"),
        result_meta.get("memory_upregulated_genes"),
        json_loads(current.get("memory_upregulated_genes_json"), []),
        default=[],
    )
    memory_downregulated_genes = _first_non_empty(
        result_meta.get("downregulated_genes"),
        result_meta.get("memory_downregulated_genes"),
        json_loads(current.get("memory_downregulated_genes_json"), []),
        default=[],
    )
    memory_deg_analysis = _first_non_empty(
        result_meta.get("deg_analysis"),
        result_meta.get("memory_deg_analysis"),
        json_loads(current.get("memory_deg_analysis_json"), {}),
        default={},
    )
    memory_deg_gene_records = _first_non_empty(
        result_meta.get("deg_gene_records"),
        result_meta.get("memory_deg_gene_records"),
        json_loads(current.get("memory_deg_gene_records_json"), []),
        default=[],
    )
    memory_srp_metadata_result = _first_non_empty(
        result_meta.get("srp_metadata_result"),
        result_meta.get("memory_srp_metadata_result"),
        json_loads(current.get("memory_srp_metadata_result_json"), {}),
        default={},
    )
    memory_control_name = _first_non_empty(
        result_meta.get("control_name"),
        result_meta.get("memory_control_name"),
        current.get("memory_control_name", ""),
        default="",
    )
    memory_test_name = _first_non_empty(
        result_meta.get("test_name"),
        result_meta.get("memory_test_name"),
        current.get("memory_test_name", ""),
        default="",
    )
    memory_enrichr = _first_non_empty(
        result_meta.get("enrichr"),
        result_meta.get("memory_enrichr"),
        json_loads(current.get("memory_enrichr_json"), {}),
        default={},
    )
    memory_rwr_seed_genes = _first_non_empty(
        result_meta.get("rwr_seed_genes"),
        result_meta.get("memory_rwr_seed_genes"),
        json_loads(current.get("memory_rwr_seed_genes_json"), []),
        default=[],
    )
    memory_rwr_genes = _first_non_empty(
        result_meta.get("rwr_genes"),
        result_meta.get("memory_rwr_genes"),
        json_loads(current.get("memory_rwr_genes_json"), []),
        default=[],
    )
    memory_disease_name = _first_non_empty(
        result_meta.get("disease_name"),
        result_meta.get("memory_disease_name"),
        current.get("memory_disease_name", ""),
        default="",
    )
    memory_openalex_genes = _first_non_empty(
        result_meta.get("openalex_genes"),
        result_meta.get("memory_openalex_genes"),
        json_loads(current.get("memory_openalex_genes_json"), []),
        default=[],
    )
    memory_opentargets_results = _first_non_empty(
        result_meta.get("memory_opentargets_results"),
        json_loads(current.get("memory_opentargets_results_json"), []),
        default=[],
    )
    memory_l1000cds2_result = _first_non_empty(
        result_meta.get("l1000cds2_result"),
        result_meta.get("memory_l1000cds2_result"),
        json_loads(current.get("memory_l1000cds2_result_json"), {}),
        default={},
    )
    memory_pubchem_result = _first_non_empty(
        result_meta.get("pubchem_result"),
        result_meta.get("memory_pubchem_result"),
        json_loads(current.get("memory_pubchem_result_json"), {}),
        default={},
    )
    memory_hypothesis_result = _first_non_empty(
        result_meta.get("hypothesis_result"),
        result_meta.get("memory_hypothesis_result"),
        json_loads(current.get("memory_hypothesis_result_json"), {}),
        default={},
    )
    memory_slice_result = _first_non_empty(
        result_meta.get("memory_slice_result"),
        json_loads(current.get("memory_slice_result_json"), {}),
        default={},
    )
    last_meta = result_meta if isinstance(result_meta, dict) else json_loads(current.get("last_meta_json"), {})

    execute(
        """
        UPDATE chats
        SET analysis_arm = ?,
            srp_ids_json = ?,
            memory_deg_genes_json = ?,
            memory_upregulated_genes_json = ?,
            memory_downregulated_genes_json = ?,
            memory_deg_analysis_json = ?,
            memory_deg_gene_records_json = ?,
            memory_srp_metadata_result_json = ?,
            memory_control_name = ?,
            memory_test_name = ?,
            memory_enrichr_json = ?,
            memory_rwr_seed_genes_json = ?,
            memory_rwr_genes_json = ?,
            memory_disease_name = ?,
            memory_openalex_genes_json = ?,
            memory_opentargets_results_json = ?,
            memory_l1000cds2_result_json = ?,
            memory_pubchem_result_json = ?,
            memory_hypothesis_result_json = ?,
            memory_slice_result_json = ?,
            last_meta_json = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            analysis_arm,
            json_dumps(srp_ids if isinstance(srp_ids, list) else []),
            json_dumps(memory_deg_genes if isinstance(memory_deg_genes, list) else []),
            json_dumps(memory_upregulated_genes if isinstance(memory_upregulated_genes, list) else []),
            json_dumps(memory_downregulated_genes if isinstance(memory_downregulated_genes, list) else []),
            json_dumps(memory_deg_analysis if isinstance(memory_deg_analysis, dict) else {}),
            json_dumps(memory_deg_gene_records if isinstance(memory_deg_gene_records, list) else []),
            json_dumps(memory_srp_metadata_result if isinstance(memory_srp_metadata_result, dict) else {}),
            memory_control_name if isinstance(memory_control_name, str) else "",
            memory_test_name if isinstance(memory_test_name, str) else "",
            json_dumps(memory_enrichr if isinstance(memory_enrichr, dict) else {}),
            json_dumps(memory_rwr_seed_genes if isinstance(memory_rwr_seed_genes, list) else []),
            json_dumps(memory_rwr_genes if isinstance(memory_rwr_genes, list) else []),
            memory_disease_name if isinstance(memory_disease_name, str) else "",
            json_dumps(memory_openalex_genes if isinstance(memory_openalex_genes, list) else []),
            json_dumps(memory_opentargets_results if isinstance(memory_opentargets_results, list) else []),
            json_dumps(memory_l1000cds2_result if isinstance(memory_l1000cds2_result, dict) else {}),
            json_dumps(memory_pubchem_result if isinstance(memory_pubchem_result, dict) else {}),
            json_dumps(memory_hypothesis_result if isinstance(memory_hypothesis_result, dict) else {}),
            json_dumps(memory_slice_result if isinstance(memory_slice_result, dict) else {}),
            json_dumps(last_meta if isinstance(last_meta, dict) else {}),
            now_iso(),
            chat_id,
        ),
    )


def build_memory_summary(chat: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    if chat.get("memory_deg_genes"):
        parts.append(f"Stored DEG genes: {len(chat['memory_deg_genes'])}.")
    if chat.get("memory_upregulated_genes"):
        parts.append(f"Stored up-regulated genes: {len(chat['memory_upregulated_genes'])}.")
    if chat.get("memory_downregulated_genes"):
        parts.append(f"Stored down-regulated genes: {len(chat['memory_downregulated_genes'])}.")
    if chat.get("memory_deg_gene_records"):
        parts.append(f"Stored DEG gene records: {len(chat['memory_deg_gene_records'])}.")
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
    if chat.get("memory_opentargets_results"):
        parts.append(f"Stored OpenTargets results: {len(chat['memory_opentargets_results'])}.")
    if isinstance(chat.get("memory_l1000cds2_result"), dict) and chat["memory_l1000cds2_result"]:
        top_drugs = chat["memory_l1000cds2_result"].get("top_drugs", [])
        if isinstance(top_drugs, list):
            parts.append(f"Stored L1000CDS2 hits: {len(top_drugs)}.")
    if isinstance(chat.get("memory_pubchem_result"), dict) and chat["memory_pubchem_result"]:
        cid = chat["memory_pubchem_result"].get("cid")
        parts.append(f"Stored PubChem result{f' for CID {cid}' if cid else ''}.")
    hypothesis_result = chat.get("memory_hypothesis_result") or (chat.get("last_meta") or {}).get("hypothesis_result")
    if isinstance(hypothesis_result, dict):
        hypotheses = hypothesis_result.get("hypotheses")
        if isinstance(hypotheses, list) and hypotheses:
            parts.append(f"Stored experimental hypotheses: {len(hypotheses)}.")
    if isinstance(chat.get("memory_slice_result"), dict) and chat["memory_slice_result"]:
        selected = chat["memory_slice_result"].get("selected_values", [])
        field = chat["memory_slice_result"].get("field")
        if isinstance(selected, list):
            parts.append(f"Stored memory slice{f' from {field}' if field else ''}: {len(selected)} selected items.")
    druggability_result = (chat.get("last_meta") or {}).get("druggability_result")
    if isinstance(druggability_result, dict) and druggability_result:
        pockets = druggability_result.get("top_pockets")
        gene = druggability_result.get("gene")
        if gene:
            parts.append(
                f"Stored druggability result for {gene}"
                + (f": {len(pockets)} pockets." if isinstance(pockets, list) else ".")
            )
    pdb_visualization_result = (chat.get("last_meta") or {}).get("pdb_visualization_result")
    if isinstance(pdb_visualization_result, dict) and pdb_visualization_result:
        gene = pdb_visualization_result.get("gene")
        uniprot = pdb_visualization_result.get("uniprot_id")
        if gene or uniprot:
            parts.append(f"Stored PDB visualization for {gene or uniprot}.")

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
    last_meta = json_loads(enriched.get("last_meta_json"), {})
    enriched["agent_type"] = _normalize_agent_type(enriched.get("agent_type"))
    enriched["srp_ids"] = json_loads(enriched.get("srp_ids_json"), [])
    enriched["memory_deg_genes"] = json_loads(enriched.get("memory_deg_genes_json"), [])
    enriched["memory_upregulated_genes"] = json_loads(enriched.get("memory_upregulated_genes_json"), [])
    enriched["memory_downregulated_genes"] = json_loads(enriched.get("memory_downregulated_genes_json"), [])
    enriched["memory_deg_analysis"] = json_loads(enriched.get("memory_deg_analysis_json"), {})
    enriched["memory_deg_gene_records"] = json_loads(enriched.get("memory_deg_gene_records_json"), [])
    enriched["memory_srp_metadata_result"] = json_loads(enriched.get("memory_srp_metadata_result_json"), {})
    enriched["memory_control_name"] = enriched.get("memory_control_name", "")
    enriched["memory_test_name"] = enriched.get("memory_test_name", "")
    enriched["memory_enrichr"] = json_loads(enriched.get("memory_enrichr_json"), {})
    enriched["memory_rwr_seed_genes"] = json_loads(enriched.get("memory_rwr_seed_genes_json"), [])
    enriched["memory_rwr_genes"] = json_loads(enriched.get("memory_rwr_genes_json"), [])
    enriched["memory_disease_name"] = enriched.get("memory_disease_name", "")
    enriched["memory_openalex_genes"] = json_loads(enriched.get("memory_openalex_genes_json"), [])
    enriched["memory_opentargets_results"] = json_loads(enriched.get("memory_opentargets_results_json"), [])
    enriched["memory_l1000cds2_result"] = json_loads(enriched.get("memory_l1000cds2_result_json"), {})
    enriched["memory_pubchem_result"] = json_loads(enriched.get("memory_pubchem_result_json"), {})
    enriched["memory_hypothesis_result"] = json_loads(enriched.get("memory_hypothesis_result_json"), {})
    enriched["memory_slice_result"] = json_loads(enriched.get("memory_slice_result_json"), {})
    enriched["last_meta"] = last_meta
    if isinstance(last_meta, dict):
        fallback_pairs = {
            "memory_deg_genes": ("deg_genes", "memory_deg_genes"),
            "memory_upregulated_genes": ("upregulated_genes", "memory_upregulated_genes"),
            "memory_downregulated_genes": ("downregulated_genes", "memory_downregulated_genes"),
            "memory_deg_analysis": ("deg_analysis", "memory_deg_analysis"),
            "memory_deg_gene_records": ("deg_gene_records", "memory_deg_gene_records"),
            "memory_srp_metadata_result": ("srp_metadata_result", "memory_srp_metadata_result"),
            "memory_enrichr": ("enrichr", "memory_enrichr"),
            "memory_rwr_seed_genes": ("rwr_seed_genes", "memory_rwr_seed_genes"),
            "memory_rwr_genes": ("rwr_genes", "memory_rwr_genes"),
            "memory_disease_name": ("disease_name", "memory_disease_name"),
            "memory_openalex_genes": ("openalex_genes", "memory_openalex_genes"),
            "memory_opentargets_results": ("memory_opentargets_results",),
            "memory_l1000cds2_result": ("l1000cds2_result", "memory_l1000cds2_result"),
            "memory_pubchem_result": ("pubchem_result", "memory_pubchem_result"),
            "memory_hypothesis_result": ("hypothesis_result", "memory_hypothesis_result"),
            "memory_slice_result": ("memory_slice_result",),
            "memory_druggability_result": ("druggability_result", "memory_druggability_result"),
            "memory_pdb_visualization_result": ("pdb_visualization_result", "memory_pdb_visualization_result"),
        }
        for target_key, meta_keys in fallback_pairs.items():
            if enriched.get(target_key) not in (None, "", [], {}):
                continue
            enriched[target_key] = _first_non_empty(
                *(last_meta.get(meta_key) for meta_key in meta_keys),
                default=enriched.get(target_key),
            )
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


def _history_to_langchain_messages(messages: list[dict[str, Any]]) -> list[Any]:
    history: list[Any] = []
    for row in messages:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip().lower()
        content = str(row.get("content") or "")
        if not content.strip():
            continue
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    return history


def _invoke_agent_for_chat(chat: dict[str, Any], content: str, memory_summary: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    agent = get_agent_app()
    return agent.invoke(
        {
            "query": content,
            "messages": _history_to_langchain_messages(messages),
            "memory_summary": memory_summary,
            "memory_deg_genes": _memory_value_from_chat(chat, "memory_deg_genes", []),
            "memory_upregulated_genes": _memory_value_from_chat(chat, "memory_upregulated_genes", []),
            "memory_downregulated_genes": _memory_value_from_chat(chat, "memory_downregulated_genes", []),
            "memory_deg_analysis": _memory_value_from_chat(chat, "memory_deg_analysis", {}),
            "memory_deg_gene_records": _memory_value_from_chat(chat, "memory_deg_gene_records", []),
            "memory_srp_metadata_result": _memory_value_from_chat(chat, "memory_srp_metadata_result", {}),
            "memory_control_name": _memory_value_from_chat(chat, "memory_control_name", ""),
            "memory_test_name": _memory_value_from_chat(chat, "memory_test_name", ""),
            "memory_enrichr": _memory_value_from_chat(chat, "memory_enrichr", {}),
            "memory_rwr_seed_genes": _memory_value_from_chat(chat, "memory_rwr_seed_genes", []),
            "memory_rwr_genes": _memory_value_from_chat(chat, "memory_rwr_genes", []),
            "memory_disease_name": _memory_value_from_chat(chat, "memory_disease_name", ""),
            "memory_openalex_genes": _memory_value_from_chat(chat, "memory_openalex_genes", []),
            "memory_opentargets_results": _memory_value_from_chat(chat, "memory_opentargets_results", []),
            "memory_l1000cds2_result": _memory_value_from_chat(chat, "memory_l1000cds2_result", {}),
            "memory_pubchem_result": _memory_value_from_chat(chat, "memory_pubchem_result", {}),
            "memory_hypothesis_result": _memory_value_from_chat(chat, "memory_hypothesis_result", {}),
            "memory_druggability_result": _memory_value_from_chat(chat, "memory_druggability_result", {}),
            "memory_pdb_visualization_result": _memory_value_from_chat(chat, "memory_pdb_visualization_result", {}),
            "memory_slice_result": _memory_value_from_chat(chat, "memory_slice_result", {}),
        }
    )


def _attach_graphml_download(chat_id: int, meta: dict[str, Any], graph: Any) -> dict[str, Any]:
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return meta

    TECHNICAL_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    graphml_path = TECHNICAL_ASSET_DIR / f"chat_{chat_id}_{timestamp}_string_network.graphml"
    nx.write_graphml(graph, graphml_path)
    enriched = dict(meta)
    enriched["graphml_path"] = str(graphml_path)
    return enriched


def _message_display_meta(meta: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {}

    analysis_arm = str(meta.get("analysis_arm") or "").strip().lower()
    display: dict[str, Any] = {
        "analysis_arm": analysis_arm,
        "route_rationale": meta.get("route_rationale", ""),
    }

    common_visual_keys = ("pyvis_html_path", "kegg_pathway_path", "volcano_plot_path", "graphml_path", "network")
    allowed_by_arm: dict[str, tuple[str, ...]] = {
        "srp": (
            "deg_analysis",
            "deg_gene_records",
            "deg_genes",
            "upregulated_genes",
            "downregulated_genes",
            "volcano_plot_path",
        ),
        "srp_metadata": ("srp_metadata_result",),
        "pathway": ("enrichr",),
        "memory_rwr": ("rwr_genes", "network", "graphml_path"),
        "disease": (
            "disease_name",
            "openalex_papers",
            "ranked_openalex_papers",
            "literature_key_points",
            "literature_references",
            "literature_summary",
        ),
        "research_literature": (
            "disease_name",
            "openalex_papers",
            "ranked_openalex_papers",
            "literature_key_points",
            "literature_references",
            "literature_summary",
        ),
        "literature": (
            "disease_name",
            "openalex_papers",
            "ranked_openalex_papers",
            "literature_key_points",
            "literature_references",
            "literature_summary",
        ),
        "visualize": common_visual_keys,
        "primekg": ("primekg_result",),
        "opentargets": ("opentargets_result",),
        "l1000cds2": ("l1000cds2_result",),
        "pubchem": ("pubchem_result",),
        "hypothesis": ("hypothesis_result",),
        "druggability": ("druggability_result",),
        "pdb_visualizer": ("pdb_visualization_result",),
    }

    for key in allowed_by_arm.get(analysis_arm, ()):
        value = meta.get(key)
        if value not in (None, "", [], {}):
            display[key] = value

    tool_history = meta.get("tool_history")
    tool_names = {
        str(entry.get("tool") or "").strip()
        for entry in tool_history
        if isinstance(entry, dict)
    } if isinstance(tool_history, list) else set()
    if tool_names.intersection({"top_rwr_genes", "rwr_analysis"}) and meta.get("rwr_genes"):
        display["rwr_genes"] = meta.get("rwr_genes")
        display["rwr_result_is_current"] = True
        network = meta.get("network")
        if network not in (None, "", [], {}):
            display["network"] = network
        graphml_path = meta.get("graphml_path")
        if graphml_path:
            display["graphml_path"] = graphml_path

    return display if len(display) > 2 else {}


def handle_chat_message(user_id: int, chat_id: int, content: str) -> dict[str, Any]:
    chat = get_chat(user_id, chat_id)
    if not chat:
        raise ValueError("Chat not found")

    append_message(chat_id, "user", content)
    messages = list_messages(user_id, chat_id)
    memory_summary = build_memory_summary(chat, messages)
    result = _invoke_agent_for_chat(chat, content, memory_summary, messages)

    answer = result.get("answer", "")
    result_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    result_meta = _attach_graphml_download(chat_id, result_meta, result.get("graph"))
    append_message(chat_id, "assistant", answer, _message_display_meta(result_meta))
    update_chat_memory(chat_id, result_meta=result_meta)

    return {
        "answer": answer,
        "meta": result_meta,
        "chat": get_chat(user_id, chat_id),
        "messages": list_messages(user_id, chat_id),
    }
