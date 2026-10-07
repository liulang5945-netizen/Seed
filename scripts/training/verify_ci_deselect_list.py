"""M7 收束（2026-10-07，㊵-469）：验证 CI-only deselect 清单每条都真的指向一个已收集的用例。

**为什么需要这道门**：pytest 对指向不存在路径的 --deselect **静默通过**。
㊵-467 写清单时把 tests/test_taiji_runtime_attachment.py 误写成
tests/seed/test_taiji_runtime_attachment.py（真实路径在 tests/ 根，不在 tests/seed/）
⇒ 那一轮的 5 条清单实际只排掉 4 条，被写错的那条在 Windows 腿照旧红
（run 37566941440 读数 **7 deselected 而非 5**；junit 报
FileNotFoundError: D:/a/Seed/Seed/checkpoints/seed_a31self_with_circuit.pt）。

**判据：清单必须自证每条都命中**，否则「已处置」会静默退化为「未处置」，
而且**没有任何报错提示你**——这与文件里已记的「过期基线让门永久红」是同一类静默失效，
只是方向相反（这次是排除项本身失效）。

被 ci.yml 的 `Verify CI-only deselect list is not silently empty` 步调用。
退出码 0＝全部命中；1＝有条目落空（并在 GitHub Actions 上打 ::error::）。
"""

from __future__ import annotations

import subprocess
import sys
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


def main() -> int:
    entries = listed_entries()
    if not entries:
        print(f"::error::{WORKFLOW.name} 里没读到任何 {KEY} 条目——清单被清空或改名了")
        return 1
    hits: list[str] = []
    misses: list[str] = []
    for entry in entries:
        path, sep, method = entry.partition("::")
        if not sep:
            misses.append(f"{entry}(条目缺 ::方法名)")
            continue
        target = REPO / path
        if not target.is_file():
            misses.append(f"{entry}(路径不存在: {path})")
            continue
        collected = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                path,
                "--co",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
        if method in collected.stdout:
            hits.append(entry)
        else:
            misses.append(f"{entry}(该文件里没有同名用例)")
    print(f"deselect 清单 {len(entries)} 条，命中 {len(hits)} 条")
    for miss in misses:
        print(f"::error::清单条目落空，deselect 会静默失效：{miss}")
    return 1 if misses else 0


if __name__ == "__main__":
    raise SystemExit(main())
