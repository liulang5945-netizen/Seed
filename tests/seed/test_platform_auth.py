from __future__ import annotations

from seed_platform.auth import AuthManager, JWTManager
from seed_platform import auth_service


def test_jwt_roundtrip_and_refresh_are_platform_owned():
    manager = JWTManager(secret_key="s" * 64, token_expire_hours=1)
    token = manager.create_token("alice")

    payload = manager.verify_token(token)
    refreshed = manager.refresh_token(token)

    assert payload is not None
    assert payload["sub"] == "alice"
    assert refreshed is not None
    assert manager.verify_token(refreshed)["sub"] == "alice"


def test_auth_service_delegates_to_platform_manager(monkeypatch):
    class FakeAudit:
        def get_recent_events(self, limit: int, days: int):
            return [{"limit": limit, "days": days}]

    class FakeJWT:
        def refresh_token(self, token: str):
            return f"refreshed:{token}"

    class FakeAuth:
        password_hash = ""
        audit = FakeAudit()
        jwt = FakeJWT()

        def login(self, username: str, password: str):
            return f"token:{username}:{password}"

        def get_status(self):
            return {"enabled": False, "username": "admin", "has_password": False}

        def enable_auth(self, username: str, password: str):
            self.enabled = True

        def disable_auth(self):
            self.enabled = False

        def set_password(self, password: str):
            self.password_hash = password

    fake = FakeAuth()
    monkeypatch.setattr(auth_service, "_auth", lambda: fake)

    assert auth_service.login("alice", "secret") == "token:alice:secret"
    assert auth_service.get_status()["username"] == "admin"
    assert auth_service.change_password("", "new-secret") is True
    assert auth_service.refresh_token("old") == "refreshed:old"
    assert auth_service.get_audit_logs(limit=3, days=2) == [{"limit": 3, "days": 2}]


def test_legacy_auth_import_is_compatibility_export():
    from neuroplex.core.security import AuthManager as LegacyAuthManager

    assert LegacyAuthManager is AuthManager
