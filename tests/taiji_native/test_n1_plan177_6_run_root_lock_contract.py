"""N177.6 run-root 单实例互斥的契约册。

来历：03 队首登记的缺口——「runner 无 run-root 单实例互斥，任何后续新包须先补门」。双跑会把
同一份冻结预算（2.4M process-CPU 秒）烧两遍，而且两边的 checkpoint 会互相顶掉取证面。

这里**不 patch 被检的判据**：验活用真实 psutil＋真实子进程，所以每条断言都能为假——
占着不放就拒起、pid 已死或已被复用就接管并留取证、锁读不出内容也拒起（fail-closed）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import run_taiji_n1_plan177_6_t1_pilot as runner  # noqa: E402


def _spawn_runner_like() -> subprocess.Popen:
    """真起一个活进程，其命令行里带 runner 的文件名——这就是"另一个实例正在跑"的可见形状。"""

    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep(120)  # {runner.RUNNER_STEM}"]
    )


def _write_lock(path: Path, payload: object) -> Path:
    lock = path / runner.RUN_LOCK_NAME
    lock.write_text(
        payload if isinstance(payload, str) else json.dumps(payload),
        encoding="utf-8",
    )
    return lock


def test_first_acquire_writes_the_lock_and_release_removes_it(tmp_path: Path) -> None:
    lock, adopted = runner.acquire_run_lock(tmp_path)
    assert adopted is None
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["runner"] == runner.RUNNER_STEM
    runner.release_run_lock(lock)
    assert not lock.exists()


def test_a_live_other_instance_refuses_a_second_start(tmp_path: Path) -> None:
    child = _spawn_runner_like()
    try:
        _write_lock(tmp_path, {"pid": child.pid, "runner": runner.RUNNER_STEM})
        with pytest.raises(RuntimeError, match="已被另一实例占用"):
            runner.acquire_run_lock(tmp_path)
        assert runner._owner_alive(child.pid) is True
    finally:
        child.kill()
        child.wait(timeout=30)


def test_a_dead_owner_lock_is_adopted_and_kept_for_forensics(tmp_path: Path) -> None:
    child = _spawn_runner_like()
    child.kill()
    dead_pid = child.pid
    child.wait(timeout=30)
    assert runner._owner_alive(dead_pid) is False

    _write_lock(tmp_path, {"pid": dead_pid, "runner": runner.RUNNER_STEM})
    lock, adopted = runner.acquire_run_lock(tmp_path)
    assert adopted is not None and adopted.startswith(f"{runner.RUN_LOCK_NAME}.stale-")
    stale = tmp_path / adopted
    assert json.loads(stale.read_text(encoding="utf-8"))["pid"] == dead_pid
    assert json.loads(lock.read_text(encoding="utf-8"))["adopted_stale_lock"] == adopted


def test_a_reused_pid_running_something_else_is_not_an_owner(tmp_path: Path) -> None:
    #: 进程还在但命令行里已没有本 runner＝pid 复用——它不是"另一个实例"，不能永远占着门。
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        _write_lock(tmp_path, {"pid": child.pid, "runner": runner.RUNNER_STEM})
        assert runner._owner_alive(child.pid) is False
        lock, adopted = runner.acquire_run_lock(tmp_path)
        assert adopted is not None
        runner.release_run_lock(lock)
    finally:
        child.kill()
        child.wait(timeout=30)


def test_an_unreadable_lock_fails_closed(tmp_path: Path) -> None:
    #: 锁内容坏掉不许退化成"没人占着"——判不了就拒起。
    _write_lock(tmp_path, "{这不是 json")
    with pytest.raises(RuntimeError, match="已被另一实例占用"):
        runner.acquire_run_lock(tmp_path)


def test_release_never_removes_another_instance_lock(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, {"pid": os.getpid() + 987654321, "runner": runner.RUNNER_STEM})
    runner.release_run_lock(lock)
    assert lock.is_file(), "别人的锁不是我的清理对象"
