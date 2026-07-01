from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    configured = os.getenv("GEA_DB_PATH", "")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("backend/data/gea_agent.sqlite3").resolve()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    path = _db_path()
    _ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                analysis_arm TEXT DEFAULT 'general',
                srp_ids_json TEXT NOT NULL DEFAULT '[]',
                memory_deg_genes_json TEXT NOT NULL DEFAULT '[]',
                memory_deg_analysis_json TEXT NOT NULL DEFAULT '{}',
                memory_control_name TEXT NOT NULL DEFAULT '',
                memory_test_name TEXT NOT NULL DEFAULT '',
                memory_enrichr_json TEXT NOT NULL DEFAULT '{}',
                memory_rwr_seed_genes_json TEXT NOT NULL DEFAULT '[]',
                memory_rwr_genes_json TEXT NOT NULL DEFAULT '[]',
                memory_disease_name TEXT NOT NULL DEFAULT '',
                memory_openalex_genes_json TEXT NOT NULL DEFAULT '[]',
                last_meta_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(chats)").fetchall()}
        if "memory_control_name" not in columns:
            conn.execute("ALTER TABLE chats ADD COLUMN memory_control_name TEXT NOT NULL DEFAULT ''")
        if "memory_test_name" not in columns:
            conn.execute("ALTER TABLE chats ADD COLUMN memory_test_name TEXT NOT NULL DEFAULT ''")
        if "memory_enrichr_json" not in columns:
            conn.execute("ALTER TABLE chats ADD COLUMN memory_enrichr_json TEXT NOT NULL DEFAULT '{}'")
        if "memory_rwr_seed_genes_json" not in columns:
            conn.execute("ALTER TABLE chats ADD COLUMN memory_rwr_seed_genes_json TEXT NOT NULL DEFAULT '[]'")
        if "memory_rwr_genes_json" not in columns:
            conn.execute("ALTER TABLE chats ADD COLUMN memory_rwr_genes_json TEXT NOT NULL DEFAULT '[]'")
        if "last_meta_json" not in columns:
            conn.execute("ALTER TABLE chats ADD COLUMN last_meta_json TEXT NOT NULL DEFAULT '{}'")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def fetch_one(query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(query, params).fetchone()
        return row_to_dict(row)


def fetch_all(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def execute(query: str, params: tuple[Any, ...]) -> int:
    with get_conn() as conn:
        cursor = conn.execute(query, params)
        return int(cursor.lastrowid)


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def now_iso() -> str:
    return _utc_now()
