"""仓库卫生守卫：运行期生成的凭据与检查点**不得被跟踪**。

今天真实发生的事（不是假想敌）：打包运行会在自己的数据目录里生成一对
`security/.jwt_secret` / `security/.storage_salt`（`seed_platform/auth.py` 写明"随机生成并持久化"），
而 `.gitignore` 的三条 Security 规则写成 `security/.jwt_secret` 这种**带斜杠的形式 ⇒ 锚定在仓库根**，
覆盖不到 `output/p6-1d-packaged-data/security/` ⇒ 那一对凭据随 `8f7fc6fa` 进了历史，且已在
`origin/main` 里。根目录那对值与打包那份**不同**，所以被泄的是打包运行自己生成的那一份。

因此这里分几条测：① 跟踪清单里不许出现凭据/审计日志/`*.pt`；② 用 `git check-ignore --no-index`
**实测规则能否命中嵌套路径** —— 只看 `.gitignore` 里有没有某个字符串是测不出这个 bug 的；
③ 按命名约定覆盖打包目录；④ 反向钉住通配不误伤有意入库的产物；⑤ 在位凭据的值不得等于历史 blob；
⑥ 规范（R3）列出的凭据名 ⇔ 本文件盯的名字，必须一致（防"文档加了、守卫没跟上"）。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: **单一事实源**：`test_runtime_credentials_and_checkpoints_are_not_tracked` 与
#: `test_guard_watches_every_credential_name_the_rules_list` 都读这份清单。
#: 刻意做成同一个常量，是为了堵住一种假绿 —— 若漂移断言只比对一个"装饰性常量"，
#: 那么只改常量就能让它变绿，而真正的跟踪清单检查仍然只盯旧的两个名字。
#: `audit_logs/` 以 `/` 结尾，按**目录**匹配（其余按文件名 fnmatch）。
CREDENTIAL_PATTERNS = (
    ".jwt_secret",
    ".storage_salt",
    ".fernet_key",
    "audit_logs/",
    ".env*",
    "*.pem",
    "*.key",
    "id_rsa*",
    ".netrc",
)

#: 相对仓库根的嵌套样例：与今天实际泄漏的那条同形。
NESTED_SECRET_PATHS = (
    "output/p6-1d-packaged-data/security/.jwt_secret",
    "output/p6-1d-packaged-data/security/.storage_salt",
    "output/p6-1d-packaged-data/security/.fernet_key",
    "somewhere/deep/security/audit_logs/2026-09-19.log",
)

#: 打包运行的数据目录按**命名约定**覆盖，而不是逐个写实例名。
#: 2026-09-19 加固：`output/p6-1d-packaged-data/` 当年漏网，正因为上一版规则逐目录列举。
PACKAGED_DATA_DIRS = (
    "output/p6-1d-packaged-data/",
    "output/anything-packaged-data/",
)

#: 反向钉住：`output/` 下**有意入库**的产物，通配不得误伤。
#: （`manual-r5-*/README.md` 与 `output/playwright/*.png` 是版本化的验收证据。）
INTENTIONALLY_TRACKED_UNDER_OUTPUT = (
    "output/manual-r5-canary/README.md",
    "output/manual-r5-s0/README.md",
    "output/manual-r5-successor/a.txt",
    "output/playwright/seed-s2-packaged-workspace.png",
)

#: 需要比对"在位值 vs 历史 blob"的凭据文件（相对仓库根）。
#: 2026-09-19 补：只测"规则是否命中"与"跟踪清单是否为空"**测不到这一条** ——
#: `security/.storage_salt` 的在位值曾与已推送历史里的 blob `5bfda9cd` 逐位相同，
#: 即"当前在用的盐可以从远端历史里取出"，而当时 4 条守卫全绿。
CREDENTIAL_FILES = (
    "security/.jwt_secret",
    "security/.storage_salt",
    "security/.fernet_key",
    "dist/Seed/security/.jwt_secret",
    "dist/Seed/security/.storage_salt",
)

RULES_DOC = REPO / "docs" / "REPO_HYGIENE_RULES.md"


def _sha16(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()[:16]


def _is_credential_path(path: str) -> bool:
    """路径是否命中 R3 的凭据清单（目录项按前缀，其余按 basename fnmatch）。"""

    if "audit_logs/" in path:
        return True
    import fnmatch

    name = path.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(name, pattern)
        for pattern in CREDENTIAL_PATTERNS
        if not pattern.endswith("/")
    )


def _history_blob_fingerprints(relative: str) -> list[tuple[str, str]]:
    """**逐路径**取历史 blob 指纹。

    必须逐路径查询：`git log -- a b c` 会把多个路径混在一起，无法逐路径归因
    （这正是 2026-09-19 那次误判的成因）。
    """

    commits = [c for c in _git("log", "--all", "--format=%H", "--", relative).stdout.split() if c]
    out: list[tuple[str, str]] = []
    for commit in commits:
        blob = subprocess.run(
            ["git", "cat-file", "-p", f"{commit}:{relative}"],
            cwd=REPO,
            capture_output=True,
        )
        if blob.returncode == 0:
            out.append((commit[:8], _sha16(blob.stdout)))
    return out


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False
    )


def _tracked() -> list[str]:
    result = _git("ls-files")
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {result.stderr[:200]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _documented_credential_names() -> tuple[str, ...]:
    """从 R3 的"凭据类"一句里解析文件名清单。

    解析结果为空必须**响**：文档一改写就静默返回空集合的话，下面的断言会永远绿 ——
    那正是这批守卫要防的那种"永不红的检查"。
    """

    text = RULES_DOC.read_text(encoding="utf-8")
    start = text.index("**凭据类**")
    segment = text[start : text.index("。", start)]
    names = tuple(re.findall(r"`([^`]+)`", segment))
    assert names, (
        "REPO_HYGIENE_RULES.md 的『凭据类』清单解析为空 —— 文档被改写，"
        "解析器需同步修改，否则这条守卫会静默全绿"
    )
    return names


def test_guard_watches_every_credential_name_the_rules_list() -> None:
    """规范列出的凭据名 ⇔ 守卫盯着的名字，**必须相等**（双向）。

    只查一侧就会漏：文档加了新凭据而守卫没跟上（今天真实发生的那次：`SecureStorage` 换成随机密钥
    后多了 `.fernet_key`，`.gitignore` 已覆盖而守卫没盯），或守卫自己扩了范围而规范没写
    （于是下一次"精简守卫"会静默删掉一条真实防线）。
    """

    documented = set(_documented_credential_names())
    covered = set(CREDENTIAL_PATTERNS)
    assert documented == covered, (
        "R3 凭据清单与守卫覆盖不一致："
        f"文档有而守卫不盯={sorted(documented - covered)}；"
        f"守卫盯而文档未列={sorted(covered - documented)}"
    )


#: 发布脚本的"随包复制"清单。`scripts/release.py` 的 `postprocess()` 用
#: `shutil.copytree(ROOT/<name>, dist/Seed/<name>)` 把这些目录打进发布包 ——
#: 2026-09-19 实测：其中 `security/` 装着**在用的** JWT 签名密钥与存储盐（与开发机逐字节相同，
#: 也正是 `dist/Seed/security/.jwt_secret` 泄漏的来源），`user_data/` 装着真实聊天记录。
#: 它们是**运行期可写状态**，不是应用资源 ⇒ 只能建空目录，让每个安装自行生成凭据。
PACKAGING_STATE_DIRS = ("security", "user_data")


def _copied_dirs_in_release_script(source: str) -> tuple[str, ...]:
    """取 `postprocess()` 里 `for extra_dir in (...)` 的那个元组。"""

    match = re.search(r"for extra_dir in \(([^)]*)\)", source)
    assert match, "release.py 的 postprocess() 结构变了（找不到 extra_dir 循环），解析器需同步修改"
    return tuple(re.findall(r'"([^"]+)"', match.group(1)))


def test_release_packaging_does_not_ship_runtime_credentials_or_user_data() -> None:
    """发布包不得携带凭据与用户数据目录。

    前面几条守卫视的是"仓库里跟踪了什么"，**看不见打包这条外泄通路**：凭据可以完全不被跟踪，
    却由 `release.py` 从工作目录 `copytree` 进 dist 随包发出去。
    """

    source = (REPO / "scripts" / "release.py").read_text(encoding="utf-8")
    copied = _copied_dirs_in_release_script(source)
    shipped = [name for name in PACKAGING_STATE_DIRS if name in copied]
    assert shipped == [], (
        f"release.py 会把运行期状态打进发布包：{shipped} —— "
        "应改为只建空目录（`JWTManager`/`SecureStorage` 缺失即随机重建）"
    )


def test_runtime_credentials_and_checkpoints_are_not_tracked() -> None:
    tracked = _tracked()
    secrets = [path for path in tracked if _is_credential_path(path)]
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


def test_ignore_rules_cover_packaged_data_dirs_by_convention() -> None:
    """按**命名约定**覆盖打包运行目录，而不是逐个写实例名。

    2026-09-19 加固：只把 `p6-1d-packaged-data` 写进规则，等于等下一个打包目录再次漏网。
    这里同时验证目录本身与其下的 security 子路径。
    """

    probes = PACKAGED_DATA_DIRS + tuple(
        parent + "security/.jwt_secret" for parent in PACKAGED_DATA_DIRS
    )
    for relative in probes:
        result = _git("check-ignore", "--no-index", "-v", relative)
        assert (
            result.returncode == 0
        ), f"{relative} 未被命中 —— 打包数据目录需按 `<name>-packaged-data` 约定通配覆盖"


def test_wildcards_do_not_swallow_intentionally_tracked_output() -> None:
    """反向钉住：通配不得误伤 `output/` 下有意入库的产物。

    只查一侧会接受"为了不再泄漏而把验收证据一起 ignore 掉"的坏中间态。
    """

    tracked = set(_tracked())
    swallowed = [path for path in INTENTIONALLY_TRACKED_UNDER_OUTPUT if path not in tracked]
    assert swallowed == [], f"通配把有意入库的产物一起吞掉了（这些路径应仍在跟踪中）：{swallowed}"


def test_live_credentials_do_not_match_any_historical_blob() -> None:
    """**在位凭据的值不得等于任何一个已提交的历史 blob。**

    这是前面几条守卫的**结构性缺口**补丁。它们测的是「规则是否命中」与「跟踪清单是否为空」，
    因此拦不住下面这件事：文件**从未被跟踪**（清单为空 ✓）、规则**也确实命中**（嵌套路径 ✓），
    但**当前在用的值**与仓库历史里某个 blob **逐位相同** —— 也就是「在用的凭据能直接从历史里取出」。

    2026-09-19 实测踩中的正是这一条：`security/.storage_salt` 的在位值指纹 `f1d2ed7b…`
    等于已推送历史里的 blob `5bfda9cd`，而当时四条守卫全绿。
    """

    offenders: list[str] = []
    for relative in CREDENTIAL_FILES:
        live = REPO / relative
        if not live.is_file():
            continue
        live_fp = _sha16(live.read_bytes())
        for commit, blob_fp in _history_blob_fingerprints(relative):
            if blob_fp == live_fp:
                offenders.append(f"{relative}: 在位值({live_fp}) == 历史 blob {commit}({blob_fp})")
    assert (
        offenders == []
    ), (
        "当前在用的凭据值可以从仓库历史里取出 —— 仅切断「继续被跟踪」不够，需要轮换：\n  "
        + "\n  ".join(offenders)
    )
