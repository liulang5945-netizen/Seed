"""N177.6 训练监督器（`monitor_taiji_n1_plan177_6_t1_pilot.py`）的契约册。

监督器只读 run root 并出版一份带时刻的读数件；这里用合成 run root 钉住三件会为假的事：

① 逐臂成本必须由该臂自己的首末 `update` 事件算出（样本不足的臂不进成本表）；
② 投影只按**有成本样本**的模式求和，没有样本的模式必须被点名而不是当 0；
③ 异常面与锁归属照实出版（陈旧锁的 pid 判死）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
INSTRUMENT = REPO / "scripts/training/monitor_taiji_n1_plan177_6_t1_pilot.py"
CAPS = {
    "process_cpu_seconds": 2_400_000,
    "wall_seconds": 220_000,
    "artifact_bytes": 8 * 1024**3,
}


def _module():
    spec = importlib.util.spec_from_file_location("n1776_monitor", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def monitor():
    return _module()


def _update(arm: str, update: int, cpu: float, wall: float, revision: str = "r1") -> dict[str, Any]:
    return {
        "event": "update",
        "arm_id": arm,
        "update": update,
        "cumulative_process_cpu_seconds": cpu,
        "cumulative_wall_seconds": wall,
        "implementation_revision": revision,
    }


def _root(
    tmp_path: Path,
    *,
    cursors: dict[str, int],
    completed: list[str],
    rows: list[dict[str, Any]],
    lock_pid: int,
    latest_paths: dict[str, str] | None = None,
) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").write_text(
        json.dumps({"hard_caps": CAPS}), encoding="utf-8", newline="\n"
    )
    (root / "run_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "optimizer_updates": sum(cursors.values()),
                "completed_arms": completed,
                "latest_checkpoint_by_arm": {
                    arm: {
                        "update_cursor": cursor,
                        "completed_updates": cursor,
                        "path": (latest_paths or {}).get(arm, str(root / arm / "final.pt")),
                    }
                    for arm, cursor in cursors.items()
                },
                "artifact_bytes": CAPS["artifact_bytes"] // 10,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    (root / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8", newline="\n"
    )
    (root / "run_root.lock").write_text(
        json.dumps({"pid": lock_pid, "marker": "plan177_6_t1_pilot"}),
        encoding="utf-8",
        newline="\n",
    )
    return root


def test_per_arm_cost_comes_from_that_arm_only(monitor, tmp_path: Path) -> None:
    rows = [
        _update("correct__seed_0", 1, 100.0, 10.0),
        _update("correct__seed_0", 161, 700.0, 70.0),
        _update("no_update__seed_0", 1, 0.0, 0.0),
        _update("no_update__seed_0", 11, 90.0, 9.0),  # 样本不足 MIN_SAMPLE
    ]
    root = _root(
        tmp_path,
        cursors={"correct__seed_0": 2560},
        completed=["correct__seed_0"],
        rows=rows,
        lock_pid=999_999_999,
    )
    report = monitor.build(root)
    cost = report["per_arm_cost"]
    assert set(cost) == {"correct__seed_0"}
    assert cost["correct__seed_0"]["cpu_per_update"] == pytest.approx(600.0 / 160)
    assert cost["correct__seed_0"]["wall_per_update"] == pytest.approx(60.0 / 160)


def test_projection_names_modes_without_a_cost_sample(monitor, tmp_path: Path) -> None:
    rows = [
        _update("correct__seed_0", 1, 0.0, 0.0),
        _update("correct__seed_0", 101, 440.0, 38.0),
    ]
    root = _root(
        tmp_path,
        cursors={"correct__seed_0": 2560},
        completed=["correct__seed_0"],
        rows=rows,
        lock_pid=999_999_999,
    )
    report = monitor.build(root)
    projection = report["projection"]
    #: correct 还剩 seed1/seed2 两臂 ⇒ 2×2560 步按 4.4 CPU 秒/步投影。
    assert report["per_mode_remaining"]["correct"]["remaining_updates"] == 2 * 2560
    assert projection["remaining_process_cpu_seconds"] == pytest.approx(2 * 2560 * 4.4)
    assert projection["remaining_wall_seconds"] == pytest.approx(2 * 2560 * 0.38)
    #: 其余模式一步样本都没有 ⇒ 必须点名，且不得当 0 摊掉。
    assert set(projection["modes_without_cost_sample"]) == {
        "parser_permuted",
        "no_update",
        "readout_only",
        "responsible_owner_frozen",
    }


def test_a_completed_arm_without_its_final_file_is_named(monitor, tmp_path: Path) -> None:
    rows = [
        _update("correct__seed_0", 1, 0.0, 0.0),
        _update("correct__seed_0", 101, 440.0, 38.0),
    ]
    root = _root(
        tmp_path,
        cursors={"correct__seed_0": 2560, "no_update__seed_0": 2560},
        completed=["correct__seed_0", "no_update__seed_0"],
        rows=rows,
        lock_pid=999_999_999,
    )
    (root / "correct__seed_0").mkdir()
    final = root / "correct__seed_0" / "final.pt"
    final.write_bytes(b"x" * 128)
    report = monitor.build(root)
    #: 在位的完成臂不进缺件名单，且字节被数到；不在位的必须点名。
    assert report["checkpoints"]["completed_arms_with_missing_final"] == ["no_update__seed_0"]
    assert report["checkpoints"]["per_arm"]["correct__seed_0"]["bytes"] == 128
    assert report["checkpoints"]["per_arm"]["correct__seed_0"]["latest_exists"] is True


def test_verdicts_pass_on_a_consistent_state(monitor, tmp_path: Path) -> None:
    #: 五种模式都要有成本样本，否则投影那格会因为"有剩余臂但无速率"判 False（那是对的）。
    rows: list[dict[str, Any]] = []
    cursors: dict[str, int] = {}
    completed: list[str] = []
    for index, mode in enumerate(monitor.MODES):
        arm = f"{mode}__seed_0"
        rows.append(_update(arm, 1, 0.0, 0.0))
        rows.append(_update(arm, 101, 400.0 + index, 40.0 + index))
        cursors[arm] = 2560
        completed.append(arm)
    root = _root(tmp_path, cursors=cursors, completed=completed, rows=rows, lock_pid=os.getpid())
    for arm in completed:
        (root / arm).mkdir()
        (root / arm / "final.pt").write_bytes(b"x")
    report = monitor.build(root)
    assert report["all_verdicts_ok"] is True, report["verdicts"]
    assert report["verdicts"]["single_lock_owner_alive"] is True


def test_a_clean_exit_without_a_lock_passes_the_lock_verdict(monitor, tmp_path: Path) -> None:
    rows = [
        _update("correct__seed_0", 1, 0.0, 0.0),
        _update("correct__seed_0", 101, 440.0, 38.0),
    ]
    root = _root(
        tmp_path,
        cursors={"correct__seed_0": 2560},
        completed=["correct__seed_0"],
        rows=rows,
        lock_pid=0,
    )
    (root / "run_root.lock").unlink()
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    state["status"] = "training_complete_evaluation_pending"
    (root / "run_state.json").write_bytes(json.dumps(state).encode("utf-8"))
    report = monitor.build(root)
    #: 终点后锁已被释放 ⇒ 没有锁是干净形状；若还挂着 running 却没锁，则必须是红。
    assert report["verdicts"]["single_lock_owner_alive"] is True


def test_an_active_state_without_a_lock_fails_the_lock_verdict(monitor, tmp_path: Path) -> None:
    rows = [
        _update("correct__seed_0", 1, 0.0, 0.0),
        _update("correct__seed_0", 101, 440.0, 38.0),
    ]
    root = _root(
        tmp_path,
        cursors={"correct__seed_0": 2560},
        completed=["correct__seed_0"],
        rows=rows,
        lock_pid=0,
    )
    (root / "run_root.lock").unlink()
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    state["status"] = "running"
    (root / "run_state.json").write_bytes(json.dumps(state).encode("utf-8"))
    report = monitor.build(root)
    #: 还在跑却没有锁 ⇒ 有人可能正裸跑，必须响。
    assert report["verdicts"]["single_lock_owner_alive"] is False


def test_a_cursor_mismatch_fails_the_consistency_verdict(monitor, tmp_path: Path) -> None:
    rows = [
        _update("correct__seed_0", 1, 0.0, 0.0),
        _update("correct__seed_0", 101, 440.0, 38.0),
    ]
    root = _root(
        tmp_path,
        cursors={"correct__seed_0": 2560},
        completed=["correct__seed_0"],
        rows=rows,
        lock_pid=os.getpid(),
    )
    (root / "correct__seed_0").mkdir()
    (root / "correct__seed_0" / "final.pt").write_bytes(b"x")
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    #: 全局游标比各臂之和大 1 ⇒ 账不自洽，必须是显式的 False 而不是被四舍五入掉。
    state["optimizer_updates"] += 1
    (root / "run_state.json").write_text(json.dumps(state), encoding="utf-8", newline="\n")
    report = monitor.build(root)
    assert report["verdicts"]["completed_arms_at_exact_multiples"] is False
    assert report["all_verdicts_ok"] is False


def test_anomalies_and_stale_lock_are_published(monitor, tmp_path: Path) -> None:
    rows = [
        _update("correct__seed_0", 1, 0.0, 0.0),
        #: 历史异常（旧 revision）与续训后异常分开计数，不许混成一锅。
        {"event": "checkpoint_resume_rollback", "implementation_revision": "r1"},
        {"event": "engineering_failure_recovered", "implementation_revision": "r1"},
        {"event": "engineering_failure_recovered", "implementation_revision": "r2"},
    ]
    root = _root(tmp_path, cursors={}, completed=[], rows=rows, lock_pid=999_999_999)
    report = monitor.build(root)
    anomalies = report["anomalies"]
    assert anomalies["rollback_events"] == 1
    assert anomalies["recovered_failure_events"] == 2
    assert anomalies["rollback_events_since_latest_revision"] == 0
    assert anomalies["recovered_failure_events_since_latest_revision"] == 1
    assert anomalies["latest_implementation_revision"] == "r2"
    assert report["lock"]["pid_alive"] is False
