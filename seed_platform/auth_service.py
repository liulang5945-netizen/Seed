"""Application-facing authentication service owned by Seed."""

from __future__ import annotations

from .auth import AuthManager


def _auth() -> AuthManager:
    return AuthManager()


def login(username: str, password: str) -> str | None:
    return _auth().login(username, password)


def change_password(old_password: str, new_password: str) -> bool:
    auth = _auth()
    if auth.password_hash and not auth.verify_password(old_password):
        return False
    auth.set_password(new_password)
    return True


def get_status() -> dict:
    return _auth().get_status()


def enable_auth(username: str, password: str) -> None:
    _auth().enable_auth(username, password)


def disable_auth() -> None:
    _auth().disable_auth()


def get_audit_logs(limit: int = 50, days: int = 7) -> list[dict]:
    return _auth().audit.get_recent_events(limit=limit, days=days)


def refresh_token(token: str) -> str | None:
    return _auth().jwt.refresh_token(token)


__all__ = [
    "change_password",
    "disable_auth",
    "enable_auth",
    "get_audit_logs",
    "get_status",
    "login",
    "refresh_token",
]
