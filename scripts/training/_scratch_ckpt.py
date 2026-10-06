"""并发安全的 scratch checkpoint 助手（DEBT-B4-5）。

多条 w7/p2/p4/m3 gate 把临时 checkpoint 直接写进**仓库共享的** `checkpoints/`，文件名按
case/seed 固定。两个进程同时跑套件时，A 写完还没撒手、B 就 `unlink()` ⇒ Windows 抛
`PermissionError [WinError 32] 另一个程序正在使用此文件`，整条 lane 当场红，
顺带把核心路径覆盖率读数压低（有次 ensemble 少 203 行就是这么来的，一度被误判成"随机抖动"）。
崩溃/被强杀的运行还会在 `checkpoints/` 留下 `.p2-9-*.pt` 一类残留。

放置策略：`checkpoints/.scratch/<pid>/<原文件名>` —— **文件名一字不改**，只按 PID 分目录。
两条理由：
* gate 自己的判据里有 `restored_checkpoint_name == "seed:.p2-9-semantic-grounding-11.pt"`
  这种**字面文件名**断言（p2-8 / p2-9 / p2-10 三条），把 PID 塞进文件名就等于改动判据；
  `api/seed_runtime.py` 的 `name` 只取 `checkpoint_path.name`，所以换目录不影响读数。
* `eval_taiji_cap0_inventory.py::_checkpoint_inventory()` 用**非递归** `glob("*.pt")` 盘点
  `checkpoints/`，且 `tests/taiji_native/test_cap0_inventory_contract.py` 拿现场重采与封存样本
  逐叶比较 ⇒ scratch 平铺在那一层时，一次并发跑就能把普查的字段面挪红。收进子目录后永远看不见。

约定：
* `scratch_checkpoint(".p2-9-semantic-grounding-11")` ⇒ 同名但**按 PID 分目录**的路径；
* `fresh_scratch(stem)` ⇒ 取路径并保证起点为空（删不掉就报错，不静默跑在上一次运行的陈旧件上）；
* `discard(path)` ⇒ 收尾清理，容忍被别的进程持有（记 warning，不再让 lane 因清理而红）；
* `sweep((".p2-9-", ...))` ⇒ 运行开始时清掉**过期**残留（只看时间戳，不碰还活着的进程的文件）。

为什么不用 `tempfile`（`tests/_scratch.py` 的 `artifact_scratch_root()` 那条仓库外的路）：
托管 Windows runner 能在 TemporaryDirectory 里建目录，却**拒绝** Python 在其中独占创建文件
（见 `eval_taiji_terminal_three_domain_governance.py` 里同一段说明）⇒ 这些 lane 的 scratch 只能
留在仓库内那个"已确认可写"的 `checkpoints/` 下，隔离靠子目录而不是换根。
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

_logger = logging.getLogger("taiji.scratch_ckpt")

# 残留超过这个秒数才扫（远大于一次 gate 的运行时长，避免碰到并发进程的活文件）
STALE_AFTER_SECONDS = 900.0


def scratch_root() -> Path:
    """scratch 存放根目录。故意不做模块级常量：测试要能整体改址，不往仓库里写。"""

    return CHECKPOINT_DIR / ".scratch"


def scratch_checkpoint(stem: str) -> Path:
    """返回 `checkpoints/.scratch/<pid>/` 下与旧实现同名的 scratch 路径。"""

    target = scratch_root() / str(os.getpid())
    target.mkdir(parents=True, exist_ok=True)
    return target / f"{stem}.pt"


def discard(path: Path) -> bool:
    """删除 scratch 文件；被并发进程持有时只记日志，不再让整条 lane 因清理而红。"""

    try:
        Path(path).unlink(missing_ok=True)
        return True
    except PermissionError as exc:  # Windows: 另一进程仍持有句柄
        _logger.warning("scratch checkpoint 暂无法删除（并发持有，跳过清理）: %s (%s)", path, exc)
        return False


def fresh_scratch(stem: str) -> Path:
    """取 scratch 路径并保证起点为空。

    删不掉意味着有别的进程还持有同名件（PID 重用后的崩溃残留）。此时 gate 会 `load` 到**上一次
    运行**的陈旧状态，读数失去意义 ⇒ 宁可当场报错，也不静默跑在旧 checkpoint 上。
    """

    path = scratch_checkpoint(stem)
    if not discard(path):
        raise RuntimeError(f"scratch checkpoint 无法清空，拒绝在陈旧状态上跑 gate: {path}")
    return path


def sweep(stems: tuple[str, ...], stale_after: float = STALE_AFTER_SECONDS) -> list[Path]:
    """清掉过期残留，返回真正被删掉的路径。

    只按文件名前缀匹配本轮 gate 的 scratch，且只删 mtime 老于 `stale_after` 的
    ⇒ 并发运行中进程的活文件（刚刚写过）不会被误删。空掉的 PID 目录一并收掉。
    """

    root = scratch_root()
    if not root.exists():
        return []
    cutoff = time.time() - stale_after
    removed: list[Path] = []
    emptied: set[Path] = set()
    for path in sorted(root.glob("*/*.pt")):
        if not any(path.name.startswith(stem) for stem in stems):
            continue
        try:
            if path.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        if discard(path):
            removed.append(path)
            emptied.add(path.parent)
    # 空 PID 目录两条路都要收，否则会像 2026-09-21 那次清理前的 135 个空目录一样只增不减：
    # ① 本轮刚被自己清空的（下面 emitted）——Windows 在删掉目录里最后一个文件时会把该目录 mtime
    #    顶成当下，所以这一类**不能**靠时间戳判弃用，时间戳此刻永远"很新"；
    # ② 别轮运行留下、如今确实老了的空目录 —— 时间戳规则只对这一类有效。
    # 跳过陌生目录的时间戳下限恰是需要的保护：并发进程可能刚 mkdir 还没落笔。
    for pid_dir in sorted(emptied):
        _rmdir_if_empty(pid_dir)
    for pid_dir in sorted(root.glob("*")):
        if pid_dir in emptied:
            continue
        try:
            if pid_dir.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        _rmdir_if_empty(pid_dir)
    return removed


def _rmdir_if_empty(path: Path) -> None:
    """用 rmdir 而非 rmtree：非空时系统调用本身就删不动，不为本轮清理冒毁别人内容的险。"""

    # 仍非空（并发进程刚落笔）或已被别人收掉 —— 残留一个空壳不值得让 lane 红
    with contextlib.suppress(OSError):
        path.rmdir()
