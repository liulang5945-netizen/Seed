"""M7 收束（2026-10-07，㊵-469）：验证 CI-only deselect 清单每条都真的指向一个已收集的用例。

**为什么需要这道门**：pytest 对指向不存在路径的 --deselect **静默通过**。
㊵-467 写清单时把 tests/test_taiji_runtime_attachment.py 误写成
tests/seed/test_taiji_runtime_attachment.py（真实路径在 tests/ 根，不在 tests/seed/）
⇒ 那一轮的 5 条清单实际只排掉 4 条，被写错的那条在 Windows 腿照旧红
（run 37566941440 读数 **7 deselected 而非 5**；junit 报
FileNotFoundError: D:/a/Seed/Seed/checkpoints/seed_a31self_with_circuit.pt）。

**判据：清单必须自证每条都真的指向一个已收集的用例**，否则「已处置」会静默退化
为「未处置」——而且没有任何报错提示你。

被 ci.yml 的 `Verify CI-only deselect list is not silently empty` 步调用。

R-S2a 改写（2026-10-26，ADR-S2 §3.3 C-3）——语义校正记录：
1. **旧版把「收集失败」与「条目落空」报同一种红**（`(该文件里没有同名用例)`）。
   新版把两者分开（三态）：同一事实不会从红变黄——收集失败一定使 pytest step
   非零退出并被汇总判红（S6a），守卫只是不再把它重复算作「清单腐化」。
2. **旧版判定是子串包含**（`if method in collected.stdout`）：只要某条用例名是
   另一条 node id 的子串，落空就会被静默判成命中——与本守卫自己要防的
   「静默失效」反向同源。新版把 `--co -q` 输出解析成 node-id 集合，做**精确
   成员判定**（含 `::Class::` 与参数化 `[...]` 后缀的整体匹配），无任何 str-in-str。
3. **旧版每条起一个 pytest --co 子进程**（172 条，实测 5m19s–6m24s，run
   40207103896/40185913504）。新版**一次**调用收集全部清单文件（去重后 56 个）。
4. **过渡态**：数据源仍是扫 ci.yml（`listed_entries()` 保留）——两腿 env 各 172 行，
   去重后 86 条；**86 才是清单条数，172 不是**（CI 历史上打过 186＝93×2 的先例）。
   JSON 单数据源是 R-S2b 的事，本刀不动。
5. fail-closed：清单被删/为空 ⇒ exit 2，不允许退化为「没清单＝全通过」。

退出码：0 全命中；1 存在 MISS/NOFILE；2 数据源不可读；3 pytest --co 子进程无法启动。
三态：MISS（红）＝条目 node ∉ 收集集合且其文件被成功收集；
NOFILE（红）＝条目文件在磁盘上不存在；
UNCOLLECTABLE（黄，::warning:: 不判红）＝条目所在文件本次收集产出 collection error
（该事实由 pytest step 自己的非零退出承担，红不被放宽，只是不重复计一次）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
KEY = "DSH_CI_DESELECT"


def listed_entries() -> list[str]:
    """从两个 job 的 env 里读出全部清单条目（不硬编码清单本身）。

    ⚠️ 只认**行首**就是 `--deselect` 的行：注释里出现的同形文本（㊵-467 的说明
    里就写了字面的 `--deselect`）必须跳过，否则会被当成条目并报「缺方法名」。
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    entries: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("--deselect"):
            continue
        entry = stripped[len("--deselect") :].strip()
        if entry:
            entries.append(entry)
    return entries


def dedupe_preserving_order(entries: list[str]) -> list[str]:
    """两腿 env 各写一份同一清单 ⇒ 读到 172 条；清单条数是去重后的 86，不是 172。"""
    seen: set[str] = set()
    out: list[str] = []
    for entry in entries:
        if entry not in seen:
            seen.add(entry)
            out.append(entry)
    return out


@dataclass
class Verdict:
    """三态判定结果。entries 为去重后的清单条目。"""

    total: int
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    nofile: list[str] = field(default_factory=list)
    uncollectable: list[str] = field(default_factory=list)

    @property
    def red(self) -> bool:
        return bool(self.misses) or bool(self.nofile)


def judge(
    entries: list[str],
    collected_nodes: set[str],
    error_files: set[str],
    repo: Path = REPO,
) -> Verdict:
    """三态判定。**精确成员判定**：`entry in collected_nodes` 整体匹配，
    禁止任何子串包含（含 `::Class::` 与参数化 `[...]` 后缀的整体匹配）。"""
    verdict = Verdict(total=len(entries))
    for entry in entries:
        path, sep, method = entry.partition("::")
        if not sep or not method:
            # 条目缺 ::方法名 = 清单腐化，按 MISS 判红（fail-closed）。
            verdict.misses.append(f"{entry}(条目缺 ::方法名)")
            continue
        target = repo / path
        if not target.is_file():
            verdict.nofile.append(f"{entry}(路径不存在: {path})")
            continue
        if path in error_files:
            # 收集失败 ≠ 清单落空：pytest step 自己会非零退出并被汇总判红。
            verdict.uncollectable.append(f"{entry}(所在文件本次收集失败: {path})")
            continue
        if entry in collected_nodes:
            verdict.hits.append(entry)
        else:
            verdict.misses.append(f"{entry}(该文件里没有同名用例)")
    return verdict


def _node_id_of(line: str) -> str | None:
    """`--co -q` 的收集行：形如 `tests/x.py::Class::test_y[param]`。
    判据：含 `::` 且首个 `::` 前的段以 `.py` 结尾（排除 traceback 回显行、
    `=== ... ===` 汇总行、`ERROR collecting ...` 错误行）。"""
    stripped = line.strip()
    if "::" not in stripped:
        return None
    head = stripped.split("::", 1)[0]
    if not head.endswith(".py"):
        return None
    return stripped


def _error_file_of(line: str) -> str | None:
    """从错误行提取失败文件路径。
    两种形态：`_____ ERROR collecting tests/x.py _____`（错误节标题）与
    short summary 的 `ERROR tests/x.py - ImportError: ...`。"""
    stripped = line.strip()
    marker = "ERROR collecting "
    idx = stripped.find(marker)
    if idx >= 0:
        rest = stripped[idx + len(marker) :]
        token = rest.split()[0].rstrip(":")
        if token.endswith(".py"):
            return token.replace("\\", "/")
    if stripped.startswith("ERROR "):
        token = stripped[len("ERROR ") :].split()[0].rstrip(":")
        if token.endswith(".py"):
            return token.replace("\\", "/")
    return None


def parse_collection_output(stdout: str) -> tuple[set[str], set[str]]:
    """把 `pytest --co -q --continue-on-collection-errors` 的 stdout 解析成
    （收集到的 node-id 集合，产出 collection error 的文件集合）。"""
    nodes: set[str] = set()
    error_files: set[str] = set()
    for line in stdout.splitlines():
        err_file = _error_file_of(line)
        if err_file is not None:
            error_files.add(err_file)
            continue
        node = _node_id_of(line)
        if node is not None:
            nodes.add(node)
    return nodes, error_files


def collect_listed_files(files: list[str], repo: Path = REPO) -> tuple[set[str], set[str]] | int:
    """**一次**调用收集全部清单文件（取代旧版每条一个子进程）。
    返回 (nodes, error_files)；子进程无法启动 ⇒ 返回 3（调用方转 exit code）。"""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--co",
        "-q",
        "-p",
        "no:cacheprovider",
        "--continue-on-collection-errors",
        *files,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(repo), encoding="utf-8", errors="replace"
        )
    except OSError:
        return 3
    if "No module named pytest" in (proc.stderr or ""):
        return 3
    return parse_collection_output(proc.stdout or "")


def listed_files(entries: list[str]) -> list[str]:
    """清单去重后的文件路径（排序、固定顺序）。
    **只收集清单文件，不全量收集 tests/**（全量更慢，且把无关收集错误掺进判定面）。"""
    files = sorted({entry.partition("::")[0] for entry in entries})
    return files


def write_step_summary(verdict: Verdict, source: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "### deselect 守卫（R-S2a 三态判定）",
        "",
        f"清单条数 {verdict.total}（来自 `{source}`）；"
        f"MISS {len(verdict.misses)}；NOFILE {len(verdict.nofile)}；"
        f"UNCOLLECTABLE {len(verdict.uncollectable)}",
        "",
        "| 状态 | 条目 | 说明 |",
        "|---|---|---|",
    ]
    for miss in verdict.misses:
        node, _, why = miss.partition("(")
        lines.append(f"| MISS | `{node}` | {why.rstrip(')')} |")
    for nofile in verdict.nofile:
        node, _, why = nofile.partition("(")
        lines.append(f"| NOFILE | `{node}` | {why.rstrip(')')} |")
    for warn in verdict.uncollectable:
        node, _, why = warn.partition("(")
        lines.append(f"| UNCOLLECTABLE | `{node}` | {why.rstrip(')')} |")
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    if not WORKFLOW.is_file():
        print(f"::error::数据源不可读：{WORKFLOW.name} 不存在")
        return 2
    raw_entries = listed_entries()
    entries = dedupe_preserving_order(raw_entries)
    if not entries:
        # fail-closed：清单被删/为空 ⇒ 判输入不可用，不允许退化为「没清单＝全通过」。
        print(f"::error::{WORKFLOW.name} 里没读到任何 {KEY} 条目——清单被清空或改名了")
        return 2
    files = listed_files(entries)
    collected = collect_listed_files(files)
    if collected == 3:
        print("::error::pytest --co 子进程无法启动（pytest 缺失或环境不可用）")
        return 3
    nodes, error_files = collected
    verdict = judge(entries, nodes, error_files)

    print(
        f"deselect 清单 {verdict.total} 条（来自 {WORKFLOW.as_posix()}）；"
        f"MISS {len(verdict.misses)}；NOFILE {len(verdict.nofile)}；"
        f"UNCOLLECTABLE {len(verdict.uncollectable)}"
    )
    for miss in verdict.misses:
        print(f"::error::清单条目落空，deselect 会静默失效：{miss}")
    for nofile in verdict.nofile:
        print(f"::error::清单条目指向不存在的文件（㊵-467 原病）：{nofile}")
    for warn in verdict.uncollectable:
        # 黄：收集失败由 pytest step 自己判红，守卫只提醒，不重复计红。
        print(f"::warning::{warn}")
    write_step_summary(verdict, WORKFLOW.as_posix())
    return 1 if verdict.red else 0


if __name__ == "__main__":
    raise SystemExit(main())
