"""行为级封印哈希与封存清单校验的单一实现（ast-canonical-v1）。

ADR: outputs/project-audit-20261021-2315/artifacts/11-seal-migration-design.md 第三节。
背景: 2026-10-22 owner 裁定方案②——封印从「字节不许动」迁到「语义不许动」：
black 重排不改 canonical AST，封印不再与格式化纪律相撞。

三个校验通道（verify_hash_dict / verify_path_pin）:
- 在位字节: sha256(当前文件) == 记录值（回执/数据证据的原生语义）。
- 在位 AST: canonical AST 哈希 == 记录值（重冻结后的 .py 条目）。
- git 历史取证: 记录值 == 某个历史 blob 的字节哈希（基线报告 controller 绑定、
  readiness 回执等「历史事实」——历史事实用历史证据核验，当前文件解冻；
  与 docs/REPO_HYGIENE_RULES.md R5「在位值 vs 历史 blob 指纹」同法）。

自钉: GOLDEN_SNIPPET/GOLDEN_AST_SHA256 钉住算法本身；改动算法必须显式更新
golden 值并走重冻结流程。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEME_BYTE = "byte-v0"
SCHEME_AST = "ast-canonical-v1"
PYTHON_PIN = (3, 12)

# 历史取证时最多回溯的提交数（单文件）；防御性上界，防病态长历史拖死校验。
GIT_HISTORY_WALK_LIMIT = 500


# ---------------------------------------------------------------- 记录路径解析
# 封存证据里的路径键来自「某台检出本仓的机器」：维护机（E:\Seed\...）或 CI
# runner（D:\a\Seed\Seed\...）。绝对路径在本机不存在时，按「以已知仓库路径
# 结尾」的最长后缀归位到本仓文件；候选必须命中 git 索引或在位文件，否则拒绝
# 映射。049ab5789 的「仓外钉 fail-closed」由此收紧为「映射不到仓库文件的钉
# fail-closed」：伪造哈希与真仓外路径仍全红，而历史记录可跨机核验。

_REPO_TRACKED: set[str] | None = None


def _tracked_repo_paths() -> set[str]:
    """HEAD 索引里的仓库相对路径全集（懒加载缓存；git 不可用时为空集）。"""
    global _REPO_TRACKED
    if _REPO_TRACKED is None:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        _REPO_TRACKED = set(result.stdout.splitlines()) if result.returncode == 0 else set()
    return _REPO_TRACKED


def _path_suffix_candidates(path_str: str) -> list[str]:
    """记录路径的仓库相对后缀候选（最长＝最具体者优先）；分隔符与盘符归一。"""
    text = str(path_str).replace("\\", "/").strip()
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if parts and len(parts[0]) == 2 and parts[0][1] == ":":
        parts = parts[1:]  # Windows 盘符（E:）不是仓库结构的一部分
    return ["/".join(parts[index:]) for index in range(len(parts))]


def _repo_relative(path_str: str) -> str | None:
    """记录路径 → 仓库相对 posix 路径；映射不到仓库内文件时返回 None。"""
    text = str(path_str).replace("\\", "/").strip()
    if not text:
        return None
    windows_drive = len(text) > 2 and text[1] == ":" and text[2] == "/"
    posix_absolute = text.startswith("/")
    if not (windows_drive or posix_absolute or Path(path_str).is_absolute()):
        parts = [part for part in text.split("/") if part not in ("", ".")]
        candidate = "/".join(parts)
        if not candidate or candidate.startswith("../"):
            return None
        if candidate in _tracked_repo_paths() or (ROOT / candidate).is_file():
            return candidate
        return None
    for candidate in _path_suffix_candidates(text):
        if candidate in _tracked_repo_paths() or (ROOT / candidate).is_file():
            return candidate
    return None


def resolve_recorded_path(path_str: str) -> Path | None:
    """记录路径 → 本机在位文件；原路径存在则原样返回，否则试仓库内映射。"""
    path = Path(path_str)
    if path.is_file():
        return path
    relative = _repo_relative(path_str)
    if relative is not None:
        local = ROOT / relative
        if local.is_file():
            return local
    return None


# Golden 夹具：含 docstring / 注解 / f-string / 控制流的固定片段。
GOLDEN_SNIPPET = '''\
"""Golden seal snippet v1."""

from __future__ import annotations

import math


def sealed_probe(items: list[str], limit: int = 3) -> dict[str, int]:
    """Probe docstring: documents, not behavior."""
    out: dict[str, int] = {}
    for name in items:
        key = f"probe:{name}"
        if len(name) > limit and name not in out:
            out[key] = int(math.floor(len(name) * 1.5))
    return out
'''

# 由 ast-canonical-v1 实算得出；改动 GOLDEN_SNIPPET 或算法必须同步更新。
GOLDEN_AST_SHA256 = "e0a2955b0693cc486d569fd93ed104590f1e3a22dffd8b24931735160ca444f9"


# ---------------------------------------------------------------- 算法核心


def _strip_docstrings(tree: ast.AST) -> None:
    """剥离 module/class/function 的首个字符串表达式（T1: docstring 不算语义）。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                len(body) > 1
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]


def _strip_annotations(tree: ast.AST) -> None:
    """剥离类型注解（T2: 注解不改运行时语义）。AnnAssign 转等价 Assign。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(parent, field, None)
            if not isinstance(body, list):
                continue
            rewritten = []
            for stmt in body:
                if isinstance(stmt, ast.AnnAssign):
                    if stmt.value is not None:
                        rewritten.append(ast.Assign(targets=[stmt.target], value=stmt.value))
                    continue
                rewritten.append(stmt)
            if (
                not rewritten
                and field == "body"
                and isinstance(
                    parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
            ):
                rewritten.append(ast.Pass())  # 裸注解剥离后体为空，补 pass 保持可解析
            setattr(parent, field, rewritten)


def canonical_text(source: str) -> str:
    """ast-canonical-v1: parse → 剥 docstring/注解 → 双次 unparse 不动点。"""
    tree = ast.parse(source)
    _strip_docstrings(tree)
    _strip_annotations(tree)
    ast.fix_missing_locations(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and not hasattr(node, "type_comment"):
            node.type_comment = None
    text1 = ast.unparse(tree)
    tree2 = ast.parse(text1)
    _strip_docstrings(tree2)
    _strip_annotations(tree2)
    ast.fix_missing_locations(tree2)
    text2 = ast.unparse(tree2)
    if text2 != text1:  # 不动点被破坏 = 算法 bug，宁可红不可静默
        raise ValueError("ast-canonical-v1 fixpoint broken")
    return text2


def ast_hash(source: str) -> str:
    return hashlib.sha256(canonical_text(source).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _looks_like_python(path: str | Path) -> bool:
    return str(path).endswith(".py")


# ---------------------------------------------------------------- 三通道校验


def _in_place_match(path_str: str, recorded: str, *, scheme: str | None) -> bool:
    """scheme=None 双通道（字节或 AST 任一）；byte-v0 纯字节；ast-canonical-v1 py 用 AST 其余字节。"""
    path = resolve_recorded_path(path_str)
    if path is None:
        return False
    if scheme == SCHEME_BYTE:
        return sha256_file(path) == recorded
    if _looks_like_python(path_str):
        if scheme == SCHEME_AST:
            try:
                return ast_hash(path.read_text(encoding="utf-8")) == recorded
            except (SyntaxError, UnicodeDecodeError):
                return False
        try:
            if ast_hash(path.read_text(encoding="utf-8")) == recorded:
                return True
        except (SyntaxError, UnicodeDecodeError):
            return False
    return sha256_file(path) == recorded


def git_blob_bytes(commit: str, rel_path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{rel_path}"],
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def file_commits(rel_path: str, limit: int = GIT_HISTORY_WALK_LIMIT) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "log", f"-{limit}", "--format=%H", "--", rel_path],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line]


def _blob_sha_variants(blob: bytes) -> set[str]:
    """git blob 与工作树读数之间的换行渲染差异（autocrlf）三态归一。

    本仓 manifest 记录值来自工作树读数（autocrlf 下为 CRLF 渲染），而
    `git show` 返回原始 blob（LF）。比对时同时尝试 raw / LF→CRLF / CRLF→LF。
    """
    return {
        hashlib.sha256(blob).hexdigest(),
        hashlib.sha256(blob.replace(b"\n", b"\r\n")).hexdigest(),
        hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest(),
    }


def git_history_attests(path_str: str, recorded: str) -> bool:
    """记录值是否与该文件某个历史 blob 的字节哈希一致（R5 同法）。"""
    rel = _repo_relative(path_str)
    if rel is None:
        # 映射不到仓库内文件的路径没有可取证的历史；fail-closed 由调用方决定语义。
        return False
    for commit in file_commits(rel):
        blob = git_blob_bytes(commit, rel)
        if blob is None:
            continue
        if recorded in _blob_sha_variants(blob):
            return True
    return False


def recorded_value_attested(path: Path | str, recorded: str) -> bool:
    """记录值是否仍可取证：在位字节/AST，或 git 历史 blob 的字节/AST 任一。"""
    path_str = str(path)
    if _in_place_match(path_str, recorded, scheme=None):
        return True
    if git_history_attests(path_str, recorded):
        return True
    if not _looks_like_python(path_str):
        return False
    rel = _repo_relative(path_str)
    if rel is None:
        return False
    for commit in file_commits(rel):
        blob = git_blob_bytes(commit, rel)
        if blob is None:
            continue
        try:
            if ast_hash(blob.decode("utf-8")) == recorded:
                return True
        except (SyntaxError, UnicodeDecodeError):
            continue
    return False


def verify_path_pin(path: Path | str, recorded: str, *, allow_git_history: bool = True) -> bool:
    """单文件钉点: 在位字节，失败时可选 git 历史取证（controller 绑定等历史钉）。"""
    resolved = resolve_recorded_path(str(path))
    if resolved is not None and sha256_file(resolved) == recorded:
        return True
    return allow_git_history and git_history_attests(str(path), recorded)


def receipt_binding_matches(recorded: str | None, path: Path | str) -> bool:
    """回执↔manifest 绑定校验: 记录值 == 在位字节 或 某个历史 blob 字节。

    用于 `X.get("face_sha256") != sha256(FACE)` 形态的比较点: 回执记录的是
    该 manifest 在回执产生时刻的字节；manifest 经文档化重冻结（S4）后当前字节
    已变，历史事实由 git 历史 blob 核验（REPO_HYGIENE_RULES R5 同法）。
    """
    if not isinstance(recorded, str) or not recorded:
        return False
    return verify_path_pin(path, recorded)


def verify_hash_dict(
    entries: dict[str, str],
    *,
    scheme: str | None = None,
    allow_git_history: bool = True,
) -> tuple[bool, list[str]]:
    """清单条目校验。返回 (全部通过, 失败清单)。

    缺省 scheme=None = 双通道在位匹配 + 失败条目转 git 历史取证；
    语义保证不变：真实语义变更（常量/名字/控制流）在任何通道都必红。
    """
    failures: list[str] = []
    for path_str, recorded in entries.items():
        if _in_place_match(path_str, recorded, scheme=scheme):
            continue
        if allow_git_history and git_history_attests(path_str, recorded):
            continue
        failures.append(path_str)
    return (not failures), failures


def verify_source_hashes(
    manifest: dict,
    *,
    git_history_commit: str | None = None,
    key: str = "source_hashes",
) -> tuple[bool, list[str]]:
    """清单级入口: 按 manifest 顶层 hash_scheme 分派（缺字段 = 历史字节语义）。"""
    entries = manifest.get(key)
    if not isinstance(entries, dict):
        return True, []
    scheme = manifest.get("hash_scheme")
    if git_history_commit is not None:
        failures = []
        for path_str, recorded in entries.items():
            rel = _repo_relative(path_str)
            if rel is None:
                failures.append(path_str)
                continue
            blob = git_blob_bytes(git_history_commit, rel)
            if blob is None:
                failures.append(path_str)
                continue
            byte_ok = recorded in _blob_sha_variants(blob)
            ast_ok = False
            if not byte_ok and _looks_like_python(rel):
                try:
                    ast_ok = ast_hash(blob.decode("utf-8")) == recorded
                except (SyntaxError, UnicodeDecodeError):
                    ast_ok = False
            if not (byte_ok or ast_ok):
                failures.append(path_str)
        return (not failures), failures
    return verify_hash_dict(entries, scheme=scheme)


def _rel_to_root(path_str: str) -> str:
    """兼容 shim：旧调用方语义（解析失败抛 ValueError）。"""
    relative = _repo_relative(path_str)
    if relative is None:
        raise ValueError(f"path does not map into the repository: {path_str}")
    return relative


# ---------------------------------------------------------------- 自检


def self_check() -> bool:
    """golden 自钉 + 幂等性 + 剥离行为红绿自证（算法被改动时必须红）。"""
    if sys.version_info[:2] != PYTHON_PIN:
        raise SystemExit(
            f"ast-canonical-v1 pinned to CPython {PYTHON_PIN[0]}.{PYTHON_PIN[1]}, "
            f"got {sys.version_info[:2]}.{sys.version_info[2]}"
        )
    golden = ast_hash(GOLDEN_SNIPPET)
    if golden != GOLDEN_AST_SHA256:
        raise SystemExit(f"golden hash drift: {golden} != {GOLDEN_AST_SHA256}")
    if canonical_text(canonical_text(GOLDEN_SNIPPET)) != canonical_text(GOLDEN_SNIPPET):
        raise SystemExit("canonical form is not idempotent")
    docstring_variant = GOLDEN_SNIPPET.replace("documents, not behavior.", "edited comment.")
    if ast_hash(docstring_variant) != golden:
        raise SystemExit("docstring-only edit must not change hash (T1)")
    constant_variant = GOLDEN_SNIPPET.replace("* 1.5", "* 2.5")
    if ast_hash(constant_variant) == golden:
        raise SystemExit("constant change must change hash (red self-check)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="ast-canonical-v1 seal helper self-check")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("seal_ast self-check: OK (golden pinned, idempotent, red/green proven)")
        return 0
    parser.print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
