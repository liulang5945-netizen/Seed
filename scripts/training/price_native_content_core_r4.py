"""R4 pricing runner: restore gates, unit cost, 50-update pilot, budget projection.

Execution is authorised only after the user approves R2/R4.  The runner reads
only ``plans/manifests/native_core_r1_train_v1.json``; it never evaluates the
query/holdout splits and never touches the sealed data.

Per object:

* zero-step save -> in-process and fresh-process restore -> probe digest equality;
* 5 warmup + 20 measured updates (median CPU/wall per update, 4 episodes each);
* 25-update checkpoint -> in-process and fresh-process restore;
* continue to 50 updates (loss first10 vs last10, NaN check, finite parameters);
* write-once report with the projected 15-run cost against 200,000 CPU s / 48 h.

Child mode ``--restore-probe <checkpoint>`` loads one pricing checkpoint in a
fresh process and prints its probe digest as one JSON line.
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

import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.native_content_training import (  # noqa: E402
    NativeRunTrainerV1,
    adapter_for,
    batch_schedule,
)

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
REPORT = ROOT / "reports/taiji_native_core_r4_pricing_20261030.json"
CHECKPOINT_DIR = ROOT / "checkpoints/native_core_r4_pricing"

OBJECTS = ("new_core_v1", "corrected_old_v1", "ordinary_v1")
SEED = 0
BATCH_SIZE = 4
UPDATES = 50
WARMUP = 5
MEASURED = 20
CHECKPOINT_STEP = 25
TARGET_UPDATES = 512
FALLBACK_UPDATES = 256
PROJECTED_RUNS = 15
SAFETY = 1.25
CPU_CAP_SECONDS = 200_000
WALL_CAP_HOURS = 48.0


# --- fresh-process probe ---------------------------------------------------


def _restore_probe(path: Path) -> int:
    started = time.perf_counter()
    trainer = NativeRunTrainerV1.load(Path(path))
    restore_seconds = time.perf_counter() - started
    payload = {
        "object": trainer.adapter.name,
        "steps": trainer.steps,
        "digest": trainer.probe_digest(),
        "restore_seconds": restore_seconds,
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
        raise RuntimeError("fresh-process probe printed no JSON line")
    payload: dict[str, Any] = json.loads(lines[-1])
    return payload


# --- per-object pricing ----------------------------------------------------


def _price_object(
    name: str,
    schedule: tuple[Any, ...],
    *,
    run_id: str,
    process: psutil.Process,
) -> dict[str, Any]:
    adapter = adapter_for(name)
    trainer = NativeRunTrainerV1.create(adapter)
    record: dict[str, Any] = {
        "object": name,
        "parameter_counts": trainer.parameter_counts(),
        "updates_executed": 0,
        "restore_ok": False,
    }

    zero_path = CHECKPOINT_DIR / f"{name}-zero-{run_id}.pt"
    zero_digest = trainer.probe_digest()
    started = time.perf_counter()
    trainer.save(zero_path)
    record["zero_step_restore"] = {
        "checkpoint": zero_path.relative_to(ROOT).as_posix(),
        "checkpoint_mb": zero_path.stat().st_size / 1e6,
        "save_seconds": time.perf_counter() - started,
    }
    in_process = NativeRunTrainerV1.load(zero_path)
    record["zero_step_restore"]["in_process_equal"] = in_process.probe_digest() == zero_digest
    fresh = _fresh_probe(zero_path)
    record["zero_step_restore"]["fresh_process_equal"] = fresh["digest"] == zero_digest
    record["zero_step_restore"]["fresh_restore_seconds"] = fresh["restore_seconds"]
    if not (
        record["zero_step_restore"]["in_process_equal"]
        and record["zero_step_restore"]["fresh_process_equal"]
    ):
        return record

    losses: list[float] = []
    warmup_cpu: list[float] = []
    warmup_wall: list[float] = []
    measured_cpu: list[float] = []
    measured_wall: list[float] = []
    measured_bytes: list[int] = []
    peak_rss_mb = process.memory_info().rss / 1e6
    first_gradient_groups: dict[str, float] = {}
    trainable_without_gradient: list[str] = []
    checkpoint_record: dict[str, Any] | None = None

    for batch in schedule:
        started_cpu = time.process_time()
        started_wall = time.perf_counter()
        metrics = trainer.update(batch.episodes)
        cpu_seconds = time.process_time() - started_cpu
        wall_seconds = time.perf_counter() - started_wall
        step = trainer.steps
        if not math.isfinite(float(metrics["loss_nats"])):
            raise ValueError(f"{name} step {step} produced a non-finite loss")
        losses.append(float(metrics["loss_nats"]))
        if step <= WARMUP:
            warmup_cpu.append(cpu_seconds)
            warmup_wall.append(wall_seconds)
        elif step <= CHECKPOINT_STEP:
            measured_cpu.append(cpu_seconds)
            measured_wall.append(wall_seconds)
            measured_bytes.append(int(metrics["target_bytes"]))
        if step == 1:
            first_gradient_groups = dict(metrics["gradient_group_norms"])
            trainable_without_gradient = list(metrics["trainable_without_gradient"])
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / 1e6)
        if step == CHECKPOINT_STEP:
            checkpoint_path = CHECKPOINT_DIR / f"{name}-step{CHECKPOINT_STEP}-{run_id}.pt"
            mid_digest = trainer.probe_digest()
            save_started = time.perf_counter()
            trainer.save(checkpoint_path)
            in_process_mid = NativeRunTrainerV1.load(checkpoint_path)
            fresh_mid = _fresh_probe(checkpoint_path)
            checkpoint_record = {
                "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
                "checkpoint_mb": checkpoint_path.stat().st_size / 1e6,
                "save_seconds": time.perf_counter() - save_started,
                "in_process_equal": in_process_mid.probe_digest() == mid_digest,
                "fresh_process_equal": fresh_mid["digest"] == mid_digest,
                "fresh_restore_seconds": fresh_mid["restore_seconds"],
            }
            if not (
                checkpoint_record["in_process_equal"] and checkpoint_record["fresh_process_equal"]
            ):
                record["checkpoint_step"] = checkpoint_record
                return record

    if len(measured_cpu) != MEASURED or len(warmup_cpu) != WARMUP:
        raise ValueError(f"{name} did not complete the declared timing windows")

    first10 = statistics.median(losses[:10])
    last10 = statistics.median(losses[-10:])
    record.update(
        {
            "updates_executed": trainer.steps,
            "target_byte_exposures": trainer.target_byte_exposures,
            "cpu_seconds_per_update_median": statistics.median(measured_cpu),
            "wall_seconds_per_update_median": statistics.median(measured_wall),
            "cpu_seconds_per_update_samples": measured_cpu,
            "wall_seconds_per_update_samples": measured_wall,
            "warmup_cpu_seconds": warmup_cpu,
            "warmup_wall_seconds": warmup_wall,
            "target_bytes_per_update": statistics.mean(measured_bytes),
            "loss_first10_median": first10,
            "loss_last10_median": last10,
            "loss_delta": first10 - last10,
            "loss_all_finite": all(math.isfinite(value) for value in losses),
            "parameters_finite": trainer.parameters_finite(),
            "first_update_gradient_group_norms": first_gradient_groups,
            "trainable_without_gradient": trainable_without_gradient,
            "peak_rss_mb": peak_rss_mb,
            "checkpoint_step": checkpoint_record,
        }
    )
    record["restore_ok"] = bool(
        checkpoint_record
        and checkpoint_record["in_process_equal"]
        and checkpoint_record["fresh_process_equal"]
        and record["loss_all_finite"]
        and record["parameters_finite"]
    )
    return record


# --- report ----------------------------------------------------------------


def _projection(cpu_per_update: float, wall_per_update: float, updates: int) -> dict[str, Any]:
    cpu_seconds = cpu_per_update * updates * PROJECTED_RUNS * SAFETY
    wall_seconds = wall_per_update * updates * PROJECTED_RUNS * SAFETY
    return {
        "updates_per_run": updates,
        "projected_runs": PROJECTED_RUNS,
        "safety_factor": SAFETY,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "wall_hours": wall_seconds / 3600.0,
    }


def _price(run_id: str) -> int:
    if REPORT.exists():
        raise SystemExit(f"pricing report is write-once: {REPORT}")
    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    if payload.get("split") != "train" or not payload.get("training_read"):
        raise SystemExit("pricing requires the declared readable train split")
    episodes = list(payload["episodes"])
    manifest_sha256 = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
    schedule = batch_schedule(episodes, seed=SEED, updates=UPDATES, batch_size=BATCH_SIZE)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    started_at = datetime.now(UTC)

    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for name in OBJECTS:
        try:
            results[name] = _price_object(name, schedule, run_id=run_id, process=process)
        except Exception as exc:  # noqa: BLE001 - recorded as an engineering failure
            failures.append(
                {
                    "object": name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break
        if not results[name]["restore_ok"]:
            failures.append({"object": name, "error": "restore or finite gate failed"})
            break

    per_object_projection = {
        name: _projection(
            record["cpu_seconds_per_update_median"],
            record["wall_seconds_per_update_median"],
            TARGET_UPDATES,
        )
        for name, record in results.items()
        if "cpu_seconds_per_update_median" in record
    }
    worst_cpu = max(
        (record["cpu_seconds_per_update_median"] for record in results.values() if record),
        default=None,
    )
    worst_wall = max(
        (record["wall_seconds_per_update_median"] for record in results.values() if record),
        default=None,
    )
    projection_512 = (
        None
        if worst_cpu is None or worst_wall is None
        else _projection(worst_cpu, worst_wall, TARGET_UPDATES)
    )
    projection_256 = (
        None
        if worst_cpu is None or worst_wall is None
        else _projection(worst_cpu, worst_wall, FALLBACK_UPDATES)
    )

    frozen_updates: int | None = None
    budget_note = "not evaluated: earlier engineering failure"
    if not failures and projection_512 is not None and projection_256 is not None:
        if (
            projection_512["cpu_seconds"] <= CPU_CAP_SECONDS
            and projection_512["wall_hours"] <= WALL_CAP_HOURS
        ):
            frozen_updates = TARGET_UPDATES
            budget_note = "512-update projection is within the declared caps"
        elif (
            projection_256["cpu_seconds"] <= CPU_CAP_SECONDS
            and projection_256["wall_hours"] <= WALL_CAP_HOURS
        ):
            frozen_updates = FALLBACK_UPDATES
            budget_note = (
                "512-update projection exceeds the caps; the single allowed reduction "
                "to 256 updates is within the caps"
            )
        else:
            budget_note = "even the 256-update projection exceeds the caps"

    if failures:
        status = "r4_engineering_failure"
    elif frozen_updates is None:
        status = "r4_budget_stop"
    else:
        status = "r4_priced"

    cost_gate_precondition = None
    if all(name in results for name in OBJECTS):
        new_core = results["new_core_v1"]["cpu_seconds_per_update_median"]
        baselines = [
            results["corrected_old_v1"]["cpu_seconds_per_update_median"],
            results["ordinary_v1"]["cpu_seconds_per_update_median"],
        ]
        cost_gate_precondition = {
            "new_core_over_best_baseline_cpu": new_core / max(baselines),
            "within_two_times_best_baseline": new_core <= 2.0 * max(baselines),
        }

    report = {
        "format": "taiji-native-core-r4-pricing-v1",
        "version": 1,
        "date": "2026-10-30",
        "contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md",
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
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "updates": UPDATES,
            "warmup_updates": WARMUP,
            "measured_updates": MEASURED,
            "checkpoint_step": CHECKPOINT_STEP,
            "target_updates_per_run": TARGET_UPDATES,
            "fallback_updates_per_run": FALLBACK_UPDATES,
            "projected_runs": PROJECTED_RUNS,
            "safety_factor": SAFETY,
            "cpu_cap_seconds": CPU_CAP_SECONDS,
            "wall_cap_hours": WALL_CAP_HOURS,
            "run_id": run_id,
        },
        "objects": results,
        "projection": {
            "basis": "max median CPU/wall per update across the priced objects",
            "per_object": per_object_projection,
            "updates_512": projection_512,
            "updates_256": projection_256,
            "frozen_updates_per_run": frozen_updates,
            "budget_note": budget_note,
        },
        "cost_gate_precondition": cost_gate_precondition,
        "status": status,
        "engineering_failures": failures,
        "read_boundaries": {
            "query_split_read": False,
            "holdout_split_read": False,
            "sealed_read": False,
            "optimizer_updates": sum(
                record.get("updates_executed", 0) for record in results.values()
            ),
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    if not REPORT.exists():
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(REPORT)}, ensure_ascii=False))
    return 0 if status != "r4_engineering_failure" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore-probe",
        type=Path,
        default=None,
        help="child mode: load one pricing checkpoint and print its probe digest",
    )
    parser.add_argument(
        "--run-id",
        default="20261030",
        help="checkpoint filename suffix; the report itself stays write-once",
    )
    args = parser.parse_args()
    if args.restore_probe is not None:
        return _restore_probe(args.restore_probe)
    return _price(args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
