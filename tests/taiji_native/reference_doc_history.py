"""DEBT-G83② 的共用取文器：件在树里就读树里的，被蒸馏删掉就从 git 历史取回**同一份字节**。

为什么要它：2026-10-07 的 M5 族蒸馏收束（`678e35fa2`）删掉 158 份实验过程文档，
留下一批仍有效的断言在扫这些文档的**文本纪律**（例：「K2 阈值必须先写在 §5.1 里」、
「每个 rule_revision=0 的档必须自标版本」）。把断言改成"该件已删"＝把内容检查做成水
（DEBT-G75 明确禁止），所以这里改的是**取法**不是判据：树里没有就取删除提交父辈的那一份，
一字不差，判别力与建档当天相同。

两条自带的守卫：
* 件既不在树里、历史里也查不到删除记录 ⇒ 响亮失败（不许静默返回空串——那是假通过）；
* 件在树里却同时有删除记录 ⇒ 取树里那份，本器不猜哪份更新。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - 仓内固定命令，参数不含用户输入
        ["git", *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return proc.stdout


def deletion_commit(relpath: str) -> str:
    """返回删除该件的提交 sha；没删过 ⇒ 空串。"""

    return _git("log", "--diff-filter=D", "-1", "--format=%H", "--", relpath).strip()


def reference_text(path: Path) -> str:
    """读一份计划文档：树里优先，被删则从历史取回原文。"""

    relpath = path.relative_to(REPO).as_posix()
    if path.is_file():
        return path.read_text(encoding="utf-8")
    deleted = deletion_commit(relpath)
    if not deleted:
        raise AssertionError(
            f"{relpath} 既不在树里、历史里也没有删除记录 ⇒ 要么被改名（那要改的是测试的取用路径），"
            "要么从没入库（那这条测本来就该红在源头）"
        )
    shown = _git("show", f"{deleted}^:{relpath}")
    if not shown.strip():
        raise AssertionError(f"git show {deleted}^:{relpath} 取回空文 ⇒ 取法失效，不许当「该件无内容」")
    return shown
