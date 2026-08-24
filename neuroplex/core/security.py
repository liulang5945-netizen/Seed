"""Compatibility exports for the platform-owned authentication module."""

from seed_platform.auth import AuditLogger, AuthManager, JWTManager, SecureStorage

__all__ = ["AuditLogger", "AuthManager", "JWTManager", "SecureStorage"]
