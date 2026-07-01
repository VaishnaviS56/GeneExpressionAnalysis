from __future__ import annotations

import hashlib
import hmac
import secrets


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = salt_hex or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    )
    return salt, digest.hex()


def verify_password(password: str, salt_hex: str, password_hash_hex: str) -> bool:
    _, computed = hash_password(password, salt_hex)
    return hmac.compare_digest(computed, password_hash_hex)


def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
