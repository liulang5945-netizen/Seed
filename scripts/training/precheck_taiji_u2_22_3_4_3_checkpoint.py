"""22.3.4.3 开卡前的“检查点能落盘且能读回”预检（0 optimizer step）。

Usage: python scripts/training/precheck_taiji_u2_22_3_4_3_checkpoint.py \
           --report reports/<new>.json

只做机制核查：父存档写盘→读回、截断/篡改拒载、两次故障与一次准入拒绝后父保持
不变且父文件仍可读回、一次参数迁移提交后的存档读回。全程 monkeypatch
`torch.optim.Adam.step` / `torch.optim.SGD.step` 为抛错，证明 0 次优化器更新。
不读 confirm、不读既有 checkpoint、不改 taiji/ 模块、不写冻结判据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_recovery import ReferenceExecutor, U2Runtime, atomic_load, run_conditions
from taiji.u2_subject import U2Config, U2Packet, U2Subject, synthetic_fixture

PINNED = (
    "taiji/u2_recovery.py",
    "taiji/u2_history.py",
    "taiji/u2_learning_transaction.py",
    "taiji/u2_subject.py",
)

CRITERIA = {
    "parent_roundtrip": "父存档 save→load 后 state/参数/optimizer moments/历史引用逐项相等",
    "truncated_rejected": "截掉尾部 8 字节的存档必须被 atomic_load 拒载",
    "payload_flip_rejected": "载荷末字节翻一位的存档必须被 atomic_load 拒载",
    "fault_after_rebuild": "重建后注入异常 ⇒ RuntimeError，父 envelope 摘要不变，父文件仍可读回",
    "fault_before_commit": "提交前注入异常 ⇒ RuntimeError，父 envelope 摘要不变，父文件仍可读回",
    "rejection_keeps_parent": "准入拒绝 ⇒ status=rejected，父 envelope 摘要不变，父文件仍可读回",
    "commit_roundtrip": "一次参数迁移提交后 save→load，state/参数/moments/历史引用相等且 revision=1",
    "zero_optimizer_steps": "全程 Adam/SGD.step 被拦截，计数为 0",
}


def _pin(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _settled(root: Path) -> U2Runtime:
    model = U2Subject(U2Config.small()).double()
    runtime = U2Runtime(model, executor=ReferenceExecutor(root / "executor.bin"))
    fixture = synthetic_fixture(model.c, torch.float64)
    runtime.set_goal(fixture["goal"])
    runtime.set_records(fixture["records"])
    legal = torch.zeros(9, dtype=torch.bool)
    legal[0] = True
    receipt = runtime.issue(runtime.choose(legal), dt=0.05)
    runtime.execute_pending(checkpoint=root / "intent-0.bin")
    runtime.set_process_time(0.05)
    event = fixture["events"][0]
    runtime.receive(
        U2Packet(
            "precheck-packet-0",
            "sensor",
            "world",
            0.0,
            0.05,
            0.05,
            event["observations"],
            event["coordinates"],
            event["outcome"],
            receipt,
            feedback_record_ids=("red",),
        )
    )
    return runtime


def _attach_optimizer(runtime: U2Runtime) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(runtime.model.parameters(), lr=1e-3)
    for parameter in runtime.model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(2.0),
            "exp_avg": torch.full_like(parameter, 0.2),
            "exp_avg_sq": torch.full_like(parameter, 0.3),
        }
    runtime.optimizer = optimizer
    return optimizer


def _candidate(runtime: U2Runtime) -> tuple[U2Subject, torch.optim.Adam]:
    candidate, _ = U2Subject.from_bundle(runtime.model.export_bundle(runtime.state))
    with torch.no_grad():
        candidate.text_input.bias.add_(0.15)
        candidate.critic.last.bias.add_(0.1)
    candidate.mark_parameter_update()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-3)
    for parameter in candidate.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(2.0),
            "exp_avg": torch.full_like(parameter, 0.2),
            "exp_avg_sq": torch.full_like(parameter, 0.3),
        }
    return candidate, optimizer


def _fingerprint(runtime: U2Runtime) -> dict[str, str]:
    return {
        "state_sha256": value_digest(runtime.state),
        "parameters_sha256": value_digest(runtime.model.state_dict()),
        "optimizer_sha256": value_digest(runtime.optimizer.state_dict()),
        "history_reference": value_digest(runtime.history.reference()),
        "revision": runtime.model.parameter_revision,
    }


def _reload_matches(path: Path, expected: dict[str, str], executor: ReferenceExecutor) -> bool:
    restored = U2Runtime.load(path, executor=executor)
    actual = _fingerprint(restored)
    for name, digest in expected.items():
        assert_values_equal(actual[name], digest, f"reload.{name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    target = arguments.report.resolve()
    if target.parent != ROOT / "reports" or target.exists():
        parser.error("Report must be a new file directly inside reports/.")

    torch.set_num_threads(1)
    conditions = run_conditions(U2Subject(U2Config.small()).double())
    frozen = {
        "schema": "taiji-u2-22.3.4.3-checkpoint-precheck-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "verdict": "criteria_frozen_before_readings",
        "run_conditions": conditions,
        "criteria": CRITERIA,
        "source_pins": {name: _pin(name) for name in PINNED},
    }
    write_report(target, frozen)

    results: list[dict[str, object]] = []
    optimizer_steps = 0
    original_adam, original_sgd = torch.optim.Adam.step, torch.optim.SGD.step

    def forbidden_step(*args: object, **kwargs: object) -> None:
        nonlocal optimizer_steps
        optimizer_steps += 1
        raise AssertionError("optimizer.step is outside this precheck")

    def record(name: str, passed: bool, detail: str) -> None:
        results.append({"check": name, "passed": passed, "detail": detail})

    test_root = ROOT / ".seed_test_tmp"
    test_root.mkdir(exist_ok=True)
    try:
        torch.optim.Adam.step = forbidden_step
        torch.optim.SGD.step = forbidden_step
        with tempfile.TemporaryDirectory(prefix="u2-precheck-22-3-4-3-", dir=test_root) as tmp:
            root = Path(tmp)
            runtime = _settled(root)
            _attach_optimizer(runtime)
            expected = _fingerprint(runtime)

            parent_path = runtime.save(root / "parent.bin")
            record(
                "parent_roundtrip",
                _reload_matches(parent_path, expected, runtime.executor),
                f"parent.bin bytes={parent_path.stat().st_size}",
            )

            truncated = root / "truncated.bin"
            truncated.write_bytes(parent_path.read_bytes()[:-8])
            try:
                atomic_load(truncated)
                record("truncated_rejected", False, "truncated file loaded without error")
            except ValueError as error:
                record("truncated_rejected", True, str(error))

            flipped = root / "flipped.bin"
            raw = bytearray(parent_path.read_bytes())
            raw[-1] ^= 0x01
            flipped.write_bytes(bytes(raw))
            try:
                atomic_load(flipped)
                record(
                    "payload_flip_rejected",
                    False,
                    "corrupted file loaded without error",
                )
            except ValueError as error:
                record("payload_flip_rejected", True, str(error))

            before = value_digest(runtime.to_envelope())
            for name, fault in (
                ("fault_after_rebuild", "after_rebuild"),
                ("fault_before_commit", "before_commit"),
            ):
                candidate, optimizer = _candidate(runtime)
                try:
                    runtime.migrate_parameters(
                        candidate,
                        optimizer=optimizer,
                        acceptance=lambda model, state, opt: True,
                        fault=fault,
                    )
                    record(name, False, "injected fault did not raise")
                except RuntimeError as error:
                    unchanged = value_digest(runtime.to_envelope()) == before
                    reloaded = _reload_matches(parent_path, expected, runtime.executor)
                    record(
                        name,
                        unchanged and reloaded,
                        f"{error}; parent_unchanged={unchanged}; reload={reloaded}",
                    )

            candidate, optimizer = _candidate(runtime)
            rejection = runtime.migrate_parameters(
                candidate,
                optimizer=optimizer,
                acceptance=lambda model, state, opt: False,
            )
            unchanged = value_digest(runtime.to_envelope()) == before
            reloaded = _reload_matches(parent_path, expected, runtime.executor)
            record(
                "rejection_keeps_parent",
                rejection["status"] == "rejected" and unchanged and reloaded,
                f"status={rejection['status']}; parent_unchanged={unchanged}; reload={reloaded}",
            )

            candidate, optimizer = _candidate(runtime)
            commit = runtime.migrate_parameters(
                candidate,
                optimizer=optimizer,
                acceptance=lambda model, state, opt: True,
            )
            committed = _fingerprint(runtime)
            accepted_path = runtime.save(root / "accepted.bin")
            ok = (
                commit["status"] == "committed"
                and committed["revision"] == 1
                and _reload_matches(accepted_path, committed, runtime.executor)
            )
            record(
                "commit_roundtrip",
                ok,
                f"status={commit['status']}; revision={committed['revision']}; bytes={accepted_path.stat().st_size}",
            )
            record(
                "zero_optimizer_steps",
                optimizer_steps == 0,
                f"optimizer_steps={optimizer_steps}",
            )
    finally:
        torch.optim.Adam.step = original_adam
        torch.optim.SGD.step = original_sgd

    cpu, wall = time.process_time(), time.monotonic() - STARTED
    passed = all(row["passed"] for row in results)
    report = {
        "schema": "taiji-u2-22.3.4.3-checkpoint-precheck-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "verdict": (
            "checkpoint_persistence_verified" if passed else "checkpoint_persistence_failed"
        ),
        "run_conditions": conditions,
        "criteria": CRITERIA,
        "source_pins": {name: _pin(name) for name in PINNED},
        "results": results,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "gaps_before_22_3_4_3": [
            "外层不可回滚 attempt 总账未实现：resources.updates 随父回滚，不是独立写盘计费",
            "每组 step1 后的状态+真实 moment 的落盘目录/命名未冻结",
            "接受/拒绝/异常后的 S12 包保存与独立 worker 输入未实现",
            "六次更新执行器（F1/F2 各三步）尚不存在，当前只有单次迁移事务",
            "atomic_save 只 fsync 文件不 fsync 目录，掉电下 rename 持久性靠每次 save 后立即读回兜底",
        ],
    }
    write_report(target, report)
    print(json.dumps({"verdict": report["verdict"], "cpu": cpu, "wall": wall}, ensure_ascii=False))
    for row in results:
        print(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['check']}: {row['detail']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
