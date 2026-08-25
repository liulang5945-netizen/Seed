"""R5: seed_platform.auth 单元测试——JWT/审计/密码哈希核心路径。

注意：全部用例避免触碰全局单例与磁盘密钥文件——
- JWTManager 用显式 secret_key 构造（跳过 _load_or_generate_secret 落盘）
- AuditLogger 用 tmp_path 作为 log_dir
"""

import time

import pytest

from seed_platform.auth import AuditLogger, AuthManager, JWTManager


@pytest.fixture()
def jwt():
    return JWTManager(secret_key="unit-test-secret-key-0123456789abcdef", token_expire_hours=1)


# ======================== JWT ========================


def test_create_and_verify_token(jwt):
    token = jwt.create_token("admin")
    assert token.count(".") == 2
    payload = jwt.verify_token(token)
    assert payload is not None
    assert payload["sub"] == "admin"
    assert payload["exp"] > payload["iat"]


def test_create_token_with_extra_claims(jwt):
    token = jwt.create_token("admin", {"role": "tester"})
    payload = jwt.verify_token(token)
    assert payload["role"] == "tester"


def test_verify_rejects_tampered_token(jwt):
    token = jwt.create_token("admin")
    header, payload, signature = token.split(".")
    forged = f"{header}.{payload}.{'A' * len(signature)}"
    assert jwt.verify_token(forged) is None


def test_verify_rejects_malformed_token(jwt):
    assert jwt.verify_token("not-a-jwt") is None
    assert jwt.verify_token("a.b") is None


def test_verify_rejects_other_secret(jwt):
    token = jwt.create_token("admin")
    other = JWTManager(secret_key="another-secret-key-fedcba9876543210")
    assert other.verify_token(token) is None


def test_expired_token_rejected():
    jwt = JWTManager(secret_key="expire-test-secret-0123456789abcdef", token_expire_hours=-1)
    token = jwt.create_token("admin")
    assert jwt.verify_token(token) is None


def test_refresh_token_window():
    # 30 分钟后过期 → 处于「过期前 2 小时」刷新窗口内
    jwt = JWTManager(secret_key="refresh-test-secret-0123456789abcdef", token_expire_hours=1)
    token = jwt.create_token("admin", {"role": "x"})
    refreshed = jwt.refresh_token(token)
    assert refreshed is not None
    payload = jwt.verify_token(refreshed)
    assert payload["sub"] == "admin"
    assert payload.get("role") == "x"
    assert "iat" in payload


def test_refresh_denied_for_long_lived_token():
    # 48 小时有效期 → 不在刷新窗口内
    jwt = JWTManager(secret_key="fresh-test-secret-fedcba9876543210", token_expire_hours=48)
    token = jwt.create_token("admin")
    assert jwt.refresh_token(token) is None


def test_refresh_invalid_token(jwt):
    assert jwt.refresh_token("garbage") is None


# ======================== 审计日志 ========================


def test_audit_log_event_and_read_back(tmp_path):
    audit = AuditLogger(log_dir=str(tmp_path))
    audit.log_event("login_success", {"user": "admin"})
    audit.log_event("login_failed", {"user": "eve"}, user="eve", ip="10.0.0.1")

    events = audit.get_recent_events(limit=10, days=1)
    assert len(events) == 2
    types = {e["type"] for e in events}
    assert types == {"login_success", "login_failed"}
    # 按时间降序：最新的在前
    assert events[0]["timestamp"] >= events[-1]["timestamp"]
    failed = next(e for e in events if e["type"] == "login_failed")
    assert failed["user"] == "eve"
    assert failed["ip"] == "10.0.0.1"
    assert failed["detail"] == {"user": "eve"}


def test_audit_log_event_without_detail(tmp_path):
    audit = AuditLogger(log_dir=str(tmp_path))
    audit.log_event("auth_disabled")
    events = audit.get_recent_events()
    assert events[0]["detail"] == {}


def test_audit_recent_events_skips_corrupt_lines(tmp_path):
    audit = AuditLogger(log_dir=str(tmp_path))
    audit.log_event("ok_event")
    # 注入一行损坏数据
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    with open(tmp_path / f"audit_{today}.jsonl", "a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    events = audit.get_recent_events()
    assert [e["type"] for e in events] == ["ok_event"]


# ======================== 密码哈希 ========================


def test_hash_password_salt_and_verify():
    hashed = AuthManager._hash_password("s3cret")
    assert "$" in hashed
    salt, _ = hashed.split("$", 1)
    # 相同盐重算结果一致
    assert AuthManager._hash_password("s3cret", salt=salt) == hashed
    # 不同密码不同结果
    assert AuthManager._hash_password("wrong", salt=salt) != hashed


def test_hash_password_random_salt_differs():
    assert AuthManager._hash_password("same") != AuthManager._hash_password("same")


# ======================== 时间窗口稳定性 ========================


def test_token_iat_close_to_now(jwt):
    payload = jwt.verify_token(jwt.create_token("admin"))
    assert abs(payload["iat"] - time.time()) < 10
