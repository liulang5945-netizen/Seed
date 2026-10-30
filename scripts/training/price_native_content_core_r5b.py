"""R5b pricing: candidate-only 50-update pilot plus the scheduled-path probe.

Phase A is 50 pure teacher-forced updates.  Phase B is a 10-update probe with
the self-prefix schedule forced on, so the frozen unit cost covers the
expensive branch.  Zero-step and both phase endpoints pass in-process and
fresh-process recovery gates before anything is frozen.  Pricing proves cost
and recovery only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.native_content_training import batch_schedule  # noqa: E402
from taiji.native_content_training_v2 import NativeContentV2Adapter, R5BRunTrainer  # noqa: E402

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
REPORT = ROOT / "reports/taiji_native_core_r5b_pricing_20261030.json"
CHECKPOINT_DIR = ROOT / "checkpoints/native_core_r5b/pricing"
UPDATES = 50
PROBE_UPDATES = 10
BATCH_SIZE = 4
PHASE_A = "teacher_forced_50"
PHASE_B = "self_prefix_10"


def _restore_probe(path: Path) -> int:
    started = time.perf_counter()
    trainer = R5BRunTrainer.load(Path(path))
    digest = trainer.probe_digest()
    payload = {
        "object": trainer.adapter.name,
        "steps": trainer.steps,
        "digest": digest,
        "restore_seconds": time.perf_counter() - started,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _fresh_probe(path: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore-probe", str(path)],
        cwd=str(ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError("fresh-process probe equality failed")
    payload: dict[str, Any] = json.loads(lines[-1])
    return payload


def _pilot(
    trainer: R5BRunTrainer,
    episodes: list[dict[str, Any]],
    *,
    updates: int,
    schedule_override: int | None,
) -> dict[str, Any]:
    """Run declared updates with the trainer-owned schedule position.

    ``schedule_override`` is the declared branch-probe mechanism: the trainer
    keeps every update on that exact position and records it, so the phase
    report proves which branch actually executed.
    """

    trainer.schedule_position_override = schedule_override
    losses: list[float] = []
    cpu_samples: list[float] = []
    wall_samples: list[float] = []
    positions: list[int] = []
    scheduled_episodes = 0
    supervised_bytes = 0
    for batch in batch_schedule(episodes, seed=0, updates=updates, batch_size=BATCH_SIZE):
        started_cpu = time.process_time()
        started_wall = time.perf_counter()
        metrics = trainer.update(batch.episodes)
        cpu_samples.append(time.process_time() - started_cpu)
        wall_samples.append(time.perf_counter() - started_wall)
        losses.append(float(metrics["loss_nats"]))
        positions.append(int(metrics["schedule_position"]))
        scheduled_episodes += int(metrics["episodes_scheduled"])
        supervised_bytes += int(metrics["scheduled_supervised_bytes"])
    return {
        "updates": updates,
        "loss_first5_median": statistics.median(losses[:5]),
        "loss_last5_median": statistics.median(losses[-5:]),
        "loss_all_finite": all(math.isfinite(value) for value in losses),
        "cpu_seconds_median": statistics.median(cpu_samples),
        "wall_seconds_median": statistics.median(wall_samples),
        "cpu_samples": cpu_samples,
        "wall_samples": wall_samples,
        "schedule_positions": positions,
        "scheduled_episodes": scheduled_episodes,
        "supervised_bytes": supervised_bytes,
    }


def _recovery(
    trainer: R5BRunTrainer,
    path: Path,
    digest: str,
    label: str,
) -> dict[str, Any]:
    trainer.save(path)
    in_process = R5BRunTrainer.load(path)
    fresh = _fresh_probe(path)
    return {
        "checkpoint": path.name,
        "label": label,
        "probe_digest": digest,
        "in_process_equal": in_process.probe_digest() == digest,
        "fresh_process_equal": fresh["digest"] == digest,
        "fresh_restore_seconds": fresh["restore_seconds"],
    }


def _price(*, approve: bool) -> int:
    if not approve:
        raise SystemExit("R5b pricing requires explicit user approval (--approve-r5b)")
    if REPORT.exists():
        raise SystemExit(f"R5b pricing report is write-once: {REPORT}")
    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    if payload.get("split") != "train" or not payload.get("training_read"):
        raise SystemExit("R5b pricing requires the declared readable train split")
    episodes = list(payload["episodes"])
    manifest_sha256 = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
    started_at = datetime.now(UTC)
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    failures: list[str] = []
    phase_a: dict[str, Any] | None = None
    phase_b: dict[str, Any] | None = None
    recovery_a: dict[str, Any] | None = None
    recovery_b: dict[str, Any] | None = None

    adapter = NativeContentV2Adapter()
    trainer = R5BRunTrainer.create(adapter)
    trainer.cursor.update({"run_id": "r5b-pricing", "seed": 0})
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    zero_digest = trainer.probe_digest()
    zero_recovery = _recovery(trainer, CHECKPOINT_DIR / "pricing-zero.pt", zero_digest, "zero")
    for gate in (zero_recovery,):
        if not (gate["in_process_equal"] and gate["fresh_process_equal"]):
            failures.append(f"recovery {gate['label']} differs")
    if not failures:
        phase_a = _pilot(trainer, episodes, updates=UPDATES, schedule_override=None)
        digest_a = trainer.probe_digest()
        recovery_a = _recovery(trainer, CHECKPOINT_DIR / "pricing-phase-a.pt", digest_a, PHASE_A)
        if not (recovery_a["in_process_equal"] and recovery_a["fresh_process_equal"]):
            failures.append(f"recovery {PHASE_A} differs")
        phase_b = _pilot(trainer, episodes, updates=PROBE_UPDATES, schedule_override=300)
        digest_b = trainer.probe_digest()
        recovery_b = _recovery(trainer, CHECKPOINT_DIR / "pricing-phase-b.pt", digest_b, PHASE_B)
        if not (recovery_b["in_process_equal"] and recovery_b["fresh_process_equal"]):
            failures.append(f"recovery {PHASE_B} differs")
        if not (phase_a["loss_all_finite"] and phase_b["loss_all_finite"]):
            failures.append("pilot loss is not finite")
        if phase_a["scheduled_episodes"] != 0:
            failures.append("phase A must stay on the pure teacher-forced branch")
        if phase_b["scheduled_episodes"] <= 0:
            failures.append("phase B branch probe executed no scheduled episode")

    unit: float | None = None
    unit_wall: float | None = None
    if phase_a is not None and phase_b is not None:
        unit = max(float(phase_a["cpu_seconds_median"]), float(phase_b["cpu_seconds_median"]))
        unit_wall = max(
            float(phase_a["wall_seconds_median"]), float(phase_b["wall_seconds_median"])
        )
    total_cpu = time.process_time() - started_cpu
    total_wall = time.perf_counter() - started_wall
    status = "r5b_priced" if not failures else "r5b_engineering_failure"
    pricing_cost = {"cpu_seconds": total_cpu, "wall_seconds": total_wall}
    report = {
        "format": "taiji-native-core-r5b-pricing-v1",
        "version": 1,
        "date": "2026-10-30",
        "contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md",
        "manifest": {
            "path": TRAIN.relative_to(ROOT).as_posix(),
            "sha256": manifest_sha256,
            "episodes": len(episodes),
            "split": payload["split"],
            "training_read": payload["training_read"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "protocol": {
            "batch_size": BATCH_SIZE,
            "updates_phase_a": UPDATES,
            "updates_phase_b": PROBE_UPDATES,
            "schedule": {
                "own_prefix_start": 257,
                "own_prefix_prob": 0.5,
                "prefix_fraction": "len(target)//2",
            },
            "approval": "explicit user approval (--approve-r5b)",
        },
        "phases": {"teacher_forced_50": phase_a, "self_prefix_10": phase_b},
        "recovery": {"zero": zero_recovery, "phase_a": recovery_a, "phase_b": recovery_b},
        "unit_cpu_seconds_per_update": unit,
        "unit_wall_seconds_per_update": unit_wall,
        "pricing_cost": pricing_cost,
        "status": status,
        "engineering_failures": failures,
        "read_boundaries": {
            "query_split_read": False,
            "holdout_split_read": False,
            "sealed_read": False,
            "optimizer_updates": trainer.steps,
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    report_path = REPORT
    if not report_path.exists():
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": status, "report": str(report_path)}, ensure_ascii=False))
    return 0 if status == "r5b_priced" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore-probe",
        type=Path,
        default=None,
        help="child mode: load one run checkpoint and print its probe digest",
    )
    parser.add_argument(
        "--approve-r5b",
        action="store_true",
        help="required: the user approved R5b execution",
    )
    args = parser.parse_args()
    if args.restore_probe is not None:
        return _restore_probe(args.restore_probe)
    return _price(approve=args.approve_r5b)


if __name__ == "__main__":
    raise SystemExit(main())
