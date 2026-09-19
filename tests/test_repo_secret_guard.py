"""仓库卫生守卫：运行期生成的凭据与检查点**不得被跟踪**。

今天真实发生的事（不是假想敌）：打包运行会在自己的数据目录里生成一对
`security/.jwt_secret` / `security/.storage_salt`（`seed_platform/auth.py` 写明"随机生成并持久化"），
而 `.gitignore` 的三条 Security 规则写成 `security/.jwt_secret` 这种**带斜杠的形式 ⇒ 锚定在仓库根**，
覆盖不到 `output/p6-1d-packaged-data/security/` ⇒ 那一对凭据随 `8f7fc6fa` 进了历史，且已在
`origin/main` 里。根目录那对值与打包那份**不同**，所以被泄的是打包运行自己生成的那一份。

因此这里分两条测：① 跟踪清单里不许出现凭据/审计日志/`*.pt`；② 用 `git check-ignore --no-index`
**实测规则能否命中嵌套路径** —— 只看 `.gitignore` 里有没有某个字符串是测不出这个 bug 的。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: 相对仓库根的嵌套样例：与今天实际泄漏的那条同形。
NESTED_SECRET_PATHS = (
    "output/p6-1d-packaged-data/security/.jwt_secret",
    "output/p6-1d-packaged-data/security/.storage_salt",
    "somewhere/deep/security/audit_logs/2026-09-19.log",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False
    )


def _tracked() -> list[str]:
    result = _git("ls-files")
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {result.stderr[:200]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_runtime_credentials_and_checkpoints_are_not_tracked() -> None:
    tracked = _tracked()
    secrets = [
        path
        for path in tracked
        if path.endswith(("/security/.jwt_secret", "/security/.storage_salt"))
        or "/security/audit_logs/" in path
    ]
    assert secrets == [], f"运行期凭据被跟踪：{secrets}"
    assert not [path for path in tracked if path.endswith(".pt")], "*.pt 不应进 git"


def test_ignore_rules_reach_nested_packaged_data() -> None:
    """规则必须**命中嵌套路径**（用 --no-index，否则已跟踪的文件会掩盖规则本身的错）。"""

    for relative in NESTED_SECRET_PATHS:
        result = _git("check-ignore", "--no-index", "-v", relative)
        assert result.returncode == 0, (
            f"{relative} 没有被任何 .gitignore 规则命中 —— 带斜杠的模式锚定在仓库根，"
            f"覆盖不到 output/... 下的打包数据目录（这正是今天泄漏的根因）"
        )
