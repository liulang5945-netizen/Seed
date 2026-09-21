"""工作区清理工具（目录结构规范 S5 的执行器，见 docs/FOLDER_STRUCTURE_RULES.md）。

默认 **dry-run**：只列出将删项，不动任何文件。`--apply` 才实际删除。

安全带（与 2026-09-21 手工清理一致，勿删）：
1. 只删「C 运行时产物 / D 构建产物 / E 工具缓存 / F 实验草稿」四类（见规范 S1/S2）；
2. 豁免清单硬编码：frontend/dist（后端在服务的前端）、desktop-electron/dist（TS 产物）、
   desktop-electron/release/SeedSetup-*（最终安装包）、.m0-checkpoint-*（保守保留）、
   direct-*（tracked，须人工逐个确认）；
3. 每个目标先断言 `git check-ignore`，不满足即跳过并报告；
4. 本机批量删除大目录可能很慢（间歇性 I/O），属正常，勿中途中断。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_FOREVER = {".git", ".workbuddy", "node_modules"}

# F 类豁免：即使匹配删除清单也不删（理由见 docs/FOLDER_STRUCTURE_RULES.md S2/S7）
EXEMPT = {
    os.path.join("frontend", "dist"),
    os.path.join("desktop-electron", "dist"),
    ".m0-checkpoint-a4x_ye4_",
}
# 特殊：tracked 的实验工作区，删除须人工逐个 git ls-files 确认（S2「特殊」节）
HUMAN_ONLY_PREFIXES = ("direct-",)

# C/D/E/F 类中「整目录可删」的项（相对 ROOT；目录本身可由构建/运行重建）。
#
# 刻意**不在**此清单里的（交由 OWNER_DECISION 报告，永不自动删除）：
# - data/            dev 模式的运行数据根（S6；dry-run 曾量出 63.5 GB，内含训练数据）
# - .codex/          含 git worktree 副本（plans/tests/scripts/frontend 的镜像），
#                    删它等于销毁 worktree（.gitignore 注释有言在先）
# - taiji_data/      PyInstaller datas 的来源（seed*.spec 引用 taiji_data/final）
# - .local/          opencode/copilot 的会话状态
# - checkpoints/ security/  S2 标注「谨慎/不删」
#
# dist/ 仍是 electron-builder extraFiles 的输入：删了它，下次 `npm run dist` 前
# 需先跑 `release.py --electron`（它会重建）。
DELETABLE_DIRS = [
    "build",
    "dist",
    ".seed_test_tmp",
    "_pytest_m2aa_full",
    "_pytest_m2aa_runtime",
    "_pytest_m2aa_runtime_2",
    ".black_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".npm-cache",
    "__pycache__",
    "neuroplex.egg-info",
    "_libs",
    "logs",
    "rag_data",
    "user_data",
    "agent_workspace",
    "update_frontend",
]
# 前缀族：.tmp-<主题>/（S3 命名约定）；.m0-checkpoint-*/ 在 EXEMPT 中保守保留
DELETABLE_PREFIXES = (".tmp-",)

# 只报告、永不删除：体积大或语义特殊，删不删由所有者决定
OWNER_DECISION = ["data", ".codex", "taiji_data", ".local", "checkpoints", "security"]


def git_ignored(rel: str) -> bool:
    return subprocess.run(
        ["git", "check-ignore", "-q", rel], cwd=ROOT, capture_output=True, text=True
    ).returncode == 0


def tree_count(path: str) -> tuple[int, float]:
    n, total = 0, 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            n += 1
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return n, total / 1e6


def exempt(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    if normalized in {e.replace("\\", "/") for e in EXEMPT}:
        return True
    base = os.path.basename(normalized)
    if base.startswith("SeedSetup-"):
        return True
    if normalized.startswith(HUMAN_ONLY_PREFIXES):
        return True
    return False


def plan_dirs() -> list[tuple[str, int, float, str]]:
    plan: list[tuple[str, int, float, str]] = []
    for rel in DELETABLE_DIRS:
        full = os.path.join(ROOT, rel)
        if not os.path.isdir(full):
            continue
        if exempt(rel):
            plan.append((rel, *tree_count(full), "豁免，跳过"))
            continue
        if not git_ignored(rel):
            plan.append((rel, *tree_count(full), "非 git-ignored，跳过"))
            continue
        plan.append((rel, *tree_count(full), "删除"))
    for entry in sorted(os.listdir(ROOT)):
        if not entry.startswith(DELETABLE_PREFIXES):
            continue
        full = os.path.join(ROOT, entry)
        if not os.path.isdir(full) or exempt(entry):
            continue
        if not git_ignored(entry):
            plan.append((entry, *tree_count(full), "非 git-ignored，跳过"))
            continue
        plan.append((entry, *tree_count(full), "删除"))
    return plan


def plan_release() -> list[tuple[str, int, float, str]]:
    plan: list[tuple[str, int, float, str]] = []
    release = os.path.join(ROOT, "desktop-electron", "release")
    if not os.path.isdir(release):
        return plan
    for entry in sorted(os.listdir(release)):
        rel = os.path.join("desktop-electron", "release", entry)
        if entry.startswith("SeedSetup-"):
            plan.append((rel, *tree_count(os.path.join(release, entry)), "保留（最终安装包）"))
            continue
        if not git_ignored(rel):
            plan.append((rel, *tree_count(os.path.join(release, entry)), "非 git-ignored，跳过"))
            continue
        n, mb = tree_count(os.path.join(release, entry))
        plan.append((rel, n, mb, "删除"))
    return plan


def plan_empty_dirs(apply: bool) -> list[str]:
    """空目录：git 不跟踪，零数据损失（S5）。apply=False 时也会把上一轮删空后
    留下的父目录一并列出，故 dry-run 的清单可能略多于实际可删数。"""
    empty: list[str] = []
    for dp, dn, fn in os.walk(ROOT, topdown=True):
        dn[:] = [d for d in dn if d not in SKIP_FOREVER]
        if fn or dn:
            continue
        rel = os.path.relpath(dp, ROOT)
        if exempt(rel):
            continue
        empty.append(rel)
    return sorted(empty, key=len, reverse=True)


def remove_dir(rel: str) -> bool:
    import shutil

    full = os.path.join(ROOT, rel)
    if not os.path.isdir(full):
        return False
    shutil.rmtree(full, ignore_errors=True)
    return not os.path.isdir(full)


def main() -> None:
    parser = argparse.ArgumentParser(description="清理工作区：构建产物/运行残留/空目录")
    parser.add_argument("--apply", action="store_true", help="实际删除（默认 dry-run）")
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== 工作区清理（{mode}）===\n")

    total_files = 0
    for rel, n, mb, verdict in plan_dirs() + plan_release():
        mark = "x" if verdict == "删除" else " "
        print(f"  [{mark}] {rel:52s} {n:6d} files {mb:9.1f} MB  {verdict}")
        if verdict == "删除":
            total_files += n

    empties = plan_empty_dirs(args.apply)
    print(f"  [x] <空目录>                          {len(empties):6d} 个            删除")

    print("\n=== 需所有者裁定（只报告，永不自动删除）===")
    for rel in OWNER_DECISION:
        full = os.path.join(ROOT, rel)
        if not os.path.isdir(full):
            continue
        n, mb = tree_count(full)
        print(f"  {rel:26s} {n:6d} files {mb:9.1f} MB")

    print(f"\n合计将删文件（不含空目录）: {total_files}")
    if not args.apply:
        print("\ndry-run 结束。确认无误后加 --apply 执行。")
        return

    import shutil  # noqa: F401  （remove_dir 内使用）

    done = 0
    for rel, _n, _mb, verdict in plan_dirs() + plan_release():
        if verdict != "删除":
            continue
        if remove_dir(rel):
            done += 1
            print(f"  已删: {rel}", flush=True)

    removed_empty = 0
    for rel in empties:
        full = os.path.join(ROOT, rel)
        try:
            os.rmdir(full)
            removed_empty += 1
        except OSError:
            pass
    print(f"\n完成：删除目录 {done} 个、空目录 {removed_empty} 个。", flush=True)


if __name__ == "__main__":
    main()
