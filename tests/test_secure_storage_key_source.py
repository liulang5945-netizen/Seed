"""SecureStorage 密钥来源修正的契约测试（2026-09-19）。

背景：旧实现用「机器指纹 + 盐」经 PBKDF2 派生 Fernet 密钥，而机器指纹的四个分量
（platform.node / platform.machine / USERNAME / processor）都能从仓库内容或运行环境
推得 ⇒ 盐一旦泄漏即可离线重放（本仓库确实发生过盐泄漏）。修正后：

* 正式密钥 = **随机生成并持久化**在 `<security>/.fernet_key`（`Fernet.generate_key()`）；
* 旧 PBKDF2 派生只保留为**解密回退**（第三级是更早的 XOR）。

本文件钉住三件事：
1. 密钥来自 `.fernet_key`（随机、持久化、跨实例可复用），且**与机器指纹无关**；
2. **篡改机器指纹后既有密文仍可解** —— 证明密文安全性不再依赖可推得的指纹；
3. **用旧 PBKDF2 派生密钥加密的密文仍可解** —— 证明不会丢失历史数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import seed_platform.auth as auth  # noqa: E402


@pytest.fixture()
def security_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把安全目录重定向到临时目录，避免污染真实 security/。"""

    monkeypatch.setattr(auth, "_security_dir", lambda: str(tmp_path))
    return tmp_path


def test_key_file_is_created_and_is_a_valid_fernet_key(security_dir: Path) -> None:
    from cryptography.fernet import Fernet

    store = auth.SecureStorage()
    key_path = security_dir / ".fernet_key"
    assert key_path.is_file(), "应当生成 .fernet_key"
    raw = key_path.read_bytes().strip()
    Fernet(raw)  # 非法会抛异常
    assert len(raw) == 44, f"Fernet key 应为 44 字节 base64，实得 {len(raw)}"
    assert store.encrypt("x")  # 可正常加密


def test_key_is_reused_across_instances(security_dir: Path) -> None:
    first = auth.SecureStorage()
    token = first.encrypt("跨实例")
    second = auth.SecureStorage()
    assert second.decrypt(token) == "跨实例"
    assert (security_dir / ".fernet_key").read_bytes() == (
        security_dir / ".fernet_key"
    ).read_bytes()


def test_ciphertext_survives_a_different_machine_fingerprint(
    security_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """核心断言：密文安全性不再依赖可推得的机器指纹。"""

    store = auth.SecureStorage()
    token = store.encrypt("secret-payload")

    # 换一个完全不同的"机器指纹"，既有密文必须仍可解
    monkeypatch.setattr(
        auth.SecureStorage,
        "_machine_fingerprint",
        lambda self: "OTHER-HOST|OTHER-ARCH|OTHER-USER|OTHER-CPU",
    )
    reopened = auth.SecureStorage()
    assert reopened.decrypt(token) == "secret-payload"


def test_legacy_pbkdf2_ciphertext_still_decryptable(security_dir: Path) -> None:
    """不丢历史数据：旧「指纹 + 盐」派生的密钥加密的密文仍可解。"""

    from cryptography.fernet import Fernet

    store = auth.SecureStorage()
    legacy_key = store._derive_fernet_key_pbkdf2()
    legacy_token = Fernet(legacy_key).encrypt(b"legacy-value").decode("ascii")

    fresh = auth.SecureStorage()  # 主密钥是随机的 ⇒ 必须走回退
    assert fresh.decrypt(legacy_token) == "legacy-value"


def test_roundtrip_and_empty_string(security_dir: Path) -> None:
    store = auth.SecureStorage()
    assert store.decrypt(store.encrypt("带中文与符号 !@#")) == "带中文与符号 !@#"
    assert store.encrypt("") == ""
    assert store.decrypt("") == ""
    assert store.decrypt("not-a-valid-token") == ""


def test_gitignore_covers_the_new_key_file() -> None:
    """新密钥文件必须被 ignore（否则会重演 .storage_salt 被提交的事故）。"""

    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v", "security/.fernet_key"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, "security/.fernet_key 未被 .gitignore 命中"
    assert ".fernet_key" in (result.stdout or ""), result.stdout
