"""目录结构守卫：仓库根出现的**每一个目录**都必须在台账里有名有姓。

为什么要有它（不是假想敌）：2026-09-21 一次清理删掉 **15,754 个文件 / 135 个空目录
（约 3.5 GB）**——`.tmp-m3-*`、`.m0-checkpoint-*`、`direct-*` 这些散落的实验目录之所以
能积累到那个规模，正因为「仓库根多了个目录」这件事**没有任何机制会响**。
本守卫把 `docs/FOLDER_STRUCTURE_RULES.md` 的 S2 台账变成会响的门。

台账的**单一事实源是那篇文档的 S2 节**（不是本文件）：守卫解析 S2 里所有以
`` `…/` `` 形式出现的目录名。于是有两种合法的"新增根级目录"路径：
① 在 S2 台账里加一行（改文档）；② 属于命名族（`.tmp-<主题>/` 等，改 `.gitignore`）。
之外的任何根级目录都会让本守卫变红，红消息给出这两条出路。

四条测（按 R4 双向钉住）：
① 正向：磁盘上的每个根级目录都在台账或命名族里；
② 反向钉住：台账必须列全核心模块 —— 防台账被静默删条目（删了某模块的条目，
   该模块就会在 ① 里变"未知"而触发 ①，但核心模块若**整个目录暂时不存在**
   （如 `dist/`、`build/`），① 管不到，需这条显式钉住）；
③ 正向（R1）：命名族的 ignore 规则必须能命中**假想的新目录** —— 只看
   `.gitignore` 里有没有字符串测不出规则写错；
④ 文档本体存在且含 S2 台账（解析器的数据源不能丢）。
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "FOLDER_STRUCTURE_RULES.md"

#: 命名族：匹配即视为已知（无需进台账），但删除仍须走 clean_worktree 的豁免逻辑。
#: `_pytest_*` / `.pytest-*` 与 `direct-*` 分别对应 .gitignore 既有规则与 S2「特殊」节。
FAMILY_PREFIXES = (
    ".tmp-",
    ".m0-checkpoint-",
    "_pytest_",
    ".pytest-",
    "direct-",
)

#: 反向钉住：这些模块是仓库的骨架，台账**必须**始终列有它们 ——
#: 即使某个模块目录暂时不存在（如被清理的构建目录、或尚未 clone 的子树）。
CORE_MODULES = (
    "api",
    "frontend",
    "instruments",
    "neuroplex",
    "seed",
    "seed_platform",
    "taiji",
    "tests",
    "scripts",
    "desktop",
    "desktop-electron",
    "taiji-harness",
    "plans",
    "docs",
    "reports",
    "design",
    "artifacts",
)


def _doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def _s2_section(text: str) -> str:
    return text.split("## S2", 1)[1].split("\n## S3", 1)[0]


def _ledger_dir_names(text: str) -> frozenset[str]:
    """S2 节里所有 `` `…/` `` 形式的目录名，取首段路径（顶层名）。"""
    section = _s2_section(text)
    names: set[str] = set()
    for token in re.findall(r"`([^`]+)`", section):
        cleaned = token.strip()
        if not cleaned.endswith("/"):
            continue
        names.add(cleaned.rstrip("/").split("/")[0])
    return frozenset(names)


def _is_known(name: str, ledger: frozenset[str]) -> bool:
    if name in ledger:
        return True
    return any(name.startswith(prefix) for prefix in FAMILY_PREFIXES)


def test_structure_rules_doc_exists_and_defines_s2_ledger():
    """④ 文档本体存在、含 S2 台账，且解析结果非空（解析器的数据源不能丢）。"""
    assert DOC.is_file(), f"缺少目录结构规范文档: {DOC}"
    text = _doc_text()
    assert "## S2" in text, "规范文档缺少 S2 台账节"
    ledger = _ledger_dir_names(text)
    assert ledger, "S2 台账解析结果为空 —— 文档结构变了，守卫与文档需要同步"


def test_ledger_covers_every_core_module():
    """② 反向钉住：核心模块必须始终在台账里。

    若有人从 S2 删掉某个模块的条目，而该模块目录又恰好不在盘上（被清理/未构建），
    ① 不会响 —— 这条显式断言补上那个盲区。
    """
    ledger = _ledger_dir_names(_doc_text())
    missing = [name for name in CORE_MODULES if name not in ledger]
    assert not missing, (
        "S2 台账缺少以下模块的条目（或文档格式变了导致解析不到）: "
        + ", ".join(missing)
        + " —— 请在 docs/FOLDER_STRUCTURE_RULES.md 的 S2 节补回，而不是删守卫。"
    )


def test_every_root_directory_is_listed_in_the_ledger():
    """① 正向：磁盘上的每个根级目录都必须在台账或命名族里。"""
    ledger = _ledger_dir_names(_doc_text())
    unknown = sorted(
        entry.name
        for entry in REPO.iterdir()
        if entry.is_dir() and entry.name != ".git" and not _is_known(entry.name, ledger)
    )
    assert not unknown, (
        "仓库根出现台账之外的目录: " + ", ".join(unknown) + "。两条出路："
        "① 它是要长期存在的 —— 在 docs/FOLDER_STRUCTURE_RULES.md 的 S2 台账加一行并说明归属类别；"
        "② 它是临时/实验物 —— 按规范 S3 改名为 `.tmp-<主题>/`（已被 ignore 命名族覆盖），"
        "或直接删除；运行时可写数据按 S6 走 get_external_path() 的数据根，不进仓库。"
    )


def test_scratch_family_ignore_rules_bite_on_hypothetical_names():
    """③ 正向（R1）：命名族的 ignore 规则必须能命中**假想的新目录**。

    只看 `.gitignore` 里有没有某个字符串，测不出"规则写错/被删"。
    这里用 `--no-index` 实测：连不存在的假想目录都必须命中。
    """
    hypothetical = (
        ".tmp-future-topic/nested/probe.txt",
        ".m0-checkpoint-future/probe.txt",
        ".pytest-future/probe.txt",
        "_pytest_future/probe.txt",
        ".mypy_cache/probe.bin",
        ".ruff_cache/probe.bin",
    )
    misses = []
    for rel in hypothetical:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-v", rel],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            misses.append(rel)
    assert not misses, (
        "命名族的 ignore 规则未命中假想路径: " + ", ".join(misses) +
        " —— .gitignore 的 /.tmp-*/ 、/.m0-checkpoint-*/ 、_pytest_*/ 、.pytest-*/ 或"
        "工具缓存规则被改坏，见 docs/FOLDER_STRUCTURE_RULES.md S3。"
    )
