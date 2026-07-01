from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
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


@app.on_event("startup")
def _startup() -> None:
    ensure_initialized()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    return create_chat(int(user["id"]), title=title)


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
