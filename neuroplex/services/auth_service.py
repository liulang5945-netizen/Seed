"""Compatibility exports for the platform-owned authentication service."""

from seed_platform.auth_service import (
    change_password,
    disable_auth,
    enable_auth,
    get_audit_logs,
    get_status,
    login,
    refresh_token,
)

__all__ = [
    "change_password",
    "disable_auth",
    "enable_auth",
    "get_audit_logs",
    "get_status",
    "login",
    "refresh_token",
]
