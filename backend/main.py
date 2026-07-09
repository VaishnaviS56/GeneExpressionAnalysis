from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from backend.services import (
    create_chat,
    ensure_initialized,
    get_chat,
    get_user_by_token,
    handle_chat_message,
    list_chats,
    list_messages,
    login_user,
    register_user,
)


app = FastAPI(title="GEA Agent API")
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuthPayload(BaseModel):
    email: str
    password: str
    display_name: str | None = None

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized:
            raise ValueError("Invalid email")
        return normalized


class ChatCreatePayload(BaseModel):
    title: str | None = None
    agent_type: str | None = None


class MessagePayload(BaseModel):
    content: str


def _extract_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return authorization[len(prefix) :].strip()


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, object]:
    token = _extract_token(authorization)
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    return user


def _get_user_from_header_or_query(
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> dict[str, object]:
    if token:
        user = get_user_by_token(token.strip())
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
        return user
    return get_current_user(authorization)


def _resolve_asset_path(raw_path: str) -> Path:
    candidate = Path(str(raw_path or "").strip())
    if not candidate:
        raise HTTPException(status_code=400, detail="Missing asset path")

    resolved = candidate if candidate.is_absolute() else (WORKSPACE_ROOT / candidate)
    try:
        normalized = resolved.resolve(strict=True)
        normalized.relative_to(WORKSPACE_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Asset path is outside the workspace") from exc

    if not normalized.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return normalized


@app.on_event("startup")
def _startup() -> None:
    ensure_initialized()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/assets")
def get_asset(path: str, _user=Depends(_get_user_from_header_or_query)):
    asset_path = _resolve_asset_path(path)
    media_type, _ = mimetypes.guess_type(asset_path.name)
    return FileResponse(asset_path, media_type=media_type or "application/octet-stream")


@app.post("/api/auth/register")
def register(payload: AuthPayload):
    try:
        return register_user(payload.email, payload.password, payload.display_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/auth/login")
def login(payload: AuthPayload):
    try:
        return login_user(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/auth/me")
def me(user=Depends(get_current_user)):
    return user


@app.get("/api/chats")
def get_chats(user=Depends(get_current_user)):
    return {"chats": list_chats(int(user["id"]))}


@app.post("/api/chats")
def create_new_chat(payload: ChatCreatePayload | None = None, user=Depends(get_current_user)):
    title = payload.title if payload else None
    agent_type = payload.agent_type if payload else None
    return create_chat(int(user["id"]), title=title, agent_type=agent_type)


@app.get("/api/chats/{chat_id}")
def get_chat_by_id(chat_id: int, user=Depends(get_current_user)):
    chat = get_chat(int(user["id"]), chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.get("/api/chats/{chat_id}/messages")
def get_chat_messages(chat_id: int, user=Depends(get_current_user)):
    try:
        return {"messages": list_messages(int(user["id"]), chat_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chats/{chat_id}/messages")
def post_chat_message(chat_id: int, payload: MessagePayload, user=Depends(get_current_user)):
    try:
        return handle_chat_message(int(user["id"]), chat_id, payload.content)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
