"""DEBT-B4-5 的守卫：scratch checkpoint 必须按进程隔离、清理必须容忍并发持有。

多条 gate 此前把临时件写进共享的 `checkpoints/` 且文件名固定 ⇒ 两个进程同时跑套件时
`unlink()` 撞 `PermissionError [WinError 32]`，lane 当场红（并被误读成"覆盖率随机抖动"）。
本件直接测那几件事：进程隔离、**文件名不变**（多条 gate 的判据字面钉死 `seed:<name>`）、
被持有时的容错、残留扫描不碰活文件。

`CHECKPOINT_DIR` 全程指向 tmp_path ⇒ 测试自己也不往仓库的 checkpoints/ 里写。
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_helper():
    path = PROJECT_ROOT / "scripts" / "training" / "_scratch_ckpt.py"
    spec = importlib.util.spec_from_file_location("_scratch_ckpt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_helper()

STEM = ".p2-9-semantic-grounding-11"


@pytest.fixture()
def ckdir(tmp_path, monkeypatch):
    target = tmp_path / "checkpoints"
    target.mkdir()
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", target)
    return target


def test_scratch_is_per_process_but_keeps_the_exact_file_name(ckdir, monkeypatch) -> None:
    """按 PID 隔离，但**文件名必须与旧实现逐字相同**。

    后半条不是洁癖：`api/seed_runtime.py` 报的 `name` 是 `f"seed:{path.name}"`，而 p2-8/p2-9/p2-10
    三条 gate 的判据把这个字面值钉进了断言。名字一变就得改判据，那是本仓明令禁止的放宽路径。
    """

    seen: dict[str, Path] = {}
    for pid in (1111, 2222, 3333):
        monkeypatch.setattr(os, "getpid", lambda p=pid: p)
        path = mod.scratch_checkpoint(STEM)
        assert path.name == f"{STEM}.pt", f"文件名不能带 PID 后缀，实得 {path.name}"
        assert path.parent == ckdir / ".scratch" / str(pid), path
        assert path.parent.parent == mod.scratch_root(), "仍在 checkpoints/ 之下"
        seen[str(pid)] = path
    assert len({str(p) for p in seen.values()}) == 3, f"不同进程必须拿到不同路径：{seen}"


def test_discard_removes_files_and_tolerates_a_missing_target(ckdir) -> None:
    path = ckdir / ".scratch" / "1" / ".m3-x.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    assert mod.discard(path) is True and not path.exists()
    assert mod.discard(path) is True, "missing_ok 语义：已不存在也算清理成功，不该抛"


def test_discard_survives_a_file_held_by_another_handle(ckdir) -> None:
    """核心那条：文件被别的句柄持有时，清理只能记日志，不能把 lane 炸掉。"""

    path = ckdir / ".p4-12-terminal-recovery-47.999.pt"
    path.write_bytes(b"data")
    handle = path.open("rb")  # 模拟并发进程的打开句柄（Windows 下 unlink 会 PermissionError）
    try:
        removed = mod.discard(path)  # 不许抛：抛了就等于把整条 lane 炸掉
        # 返回值必须与实际结果一致——别把"没删掉"说成"清理成功"（POSIX 能删、Windows 不能）
        assert removed == (not path.exists()), (removed, path.exists())
    finally:
        handle.close()
    assert mod.discard(path) is True and not path.exists(), "句柄释放后必须能清掉"


def test_fresh_scratch_refuses_to_start_on_a_stale_checkpoint(ckdir, monkeypatch) -> None:
    """起点件删不掉时必须当场报错：静默跑在上一次运行的状态上，读数就没有意义了。

    这里注入"删不掉"而不是真去持句柄 ⇒ POSIX 与 Windows 语义一致（OS 锁定只在上一条测试里测）。
    """

    monkeypatch.setattr(mod, "discard", lambda path: False)
    with pytest.raises(RuntimeError, match="拒绝在陈旧状态上跑 gate"):
        mod.fresh_scratch(STEM)

    monkeypatch.setattr(mod, "discard", lambda path: True)
    path = mod.fresh_scratch(STEM)
    assert path.name == f"{STEM}.pt" and path.parent.parent == mod.scratch_root(), path


def test_sweep_removes_only_stale_matching_files(ckdir) -> None:
    live_pid, dead_pid, other_pid, idle_pid, abandoned_pid = (
        ckdir / ".scratch" / "4242",
        ckdir / ".scratch" / "4243",
        ckdir / ".scratch" / "4244",
        ckdir / ".scratch" / "4245",
        ckdir / ".scratch" / "4246",
    )
    for d in (live_pid, dead_pid, other_pid, idle_pid, abandoned_pid):
        d.mkdir(parents=True)
    stale = dead_pid / ".p2-11-language-chain-11.pt"
    fresh = live_pid / ".p2-11-language-chain-11.pt"
    other = other_pid / ".unrelated-something.pt"
    for p in (stale, fresh, other):
        p.write_bytes(b"x")
    old = time.time() - 4000
    os.utime(stale, (old, old))
    os.utime(abandoned_pid, (old, old))  # 别轮运行留下的、确实老了的空目录
    (ckdir / "seed_beta.pt").write_bytes(b"x")  # 真正的训练件：永远不在扫描面里

    removed = mod.sweep((".p2-11-language-chain-",), stale_after=900)
    assert removed == [stale], removed
    assert fresh.exists(), "并发进程刚写的活文件不能删"
    assert other.exists() and (ckdir / "seed_beta.pt").exists(), "不匹配前缀的一律不碰"
    assert not dead_pid.exists(), "本轮刚清空的 PID 目录要一并收掉，否则 .scratch/ 只增不减"
    assert not abandoned_pid.exists(), "别轮运行弃用的空目录也要收（时间戳那条路）"
    assert other_pid.exists(), "目录里还有活文件时不能连目录一起删"
    assert idle_pid.exists(), "并发进程刚 mkdir 还没落笔 ⇒ 空目录也不能被别人收走"


def test_sweep_is_safe_when_the_directory_is_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path / "nope")
    assert mod.sweep((".p2-9-",)) == []
