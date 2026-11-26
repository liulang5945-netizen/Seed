"""R5c bounded training runner: Q1 3 runs + Q2 6 runs, 512 updates each.

Mirrors the approved R5b runner with the v3 candidate: zero-step and endpoint
restore gates (in-process and fresh-process), 512 deterministic updates with
the declared self-prefix schedule (per-episode coin from update 257, full
target supervision), per-update CPU/wall accounting, a unit-cost cap, and the
declared auxiliary pointer CE accounted per update.  Baseline endpoints are
not retrained; the R5b/Q2 semantics carry over with ``new_core_v3``.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.native_content_training import (  # noqa: E402
    batch_schedule,
    credit_on_configure,
    credit_stop_configure,
)
from taiji.native_content_training_v3 import NativeContentV3Adapter, R5CRunTrainer  # noqa: E402

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
PRICING_REPORT = ROOT / "reports/taiji_native_core_r5c_pricing_20261031.json"
REPORT = ROOT / "reports/taiji_native_core_r5c_training_20261031.json"
CHECKPOINT_DIR = ROOT / "checkpoints/native_core_r5c"
OBJECT = "new_core_v3"
SEEDS = (0, 1, 2)
BATCH_SIZE = 4
UPDATES = 512
UNIT_CAP_FACTOR = 1.5
CAP_CHECK_EVERY = 32
CPU_CAP_SECONDS = 60_000
WALL_CAP_HOURS = 48.0
_RUN_ID = "20261031"


@dataclass(frozen=True)
class RunSpecV1:
    name: str
    block: str
    object: str
    credit_mode: str
    seed: int

    def __post_init__(self) -> None:
        if self.block not in {"q1", "q2"}:
            raise ValueError("run block is outside the declared set")
        if self.credit_mode not in {"credit_on", "credit_stop"}:
            raise ValueError("run credit mode is outside the declared set")
        if self.seed not in SEEDS:
            raise ValueError("run seed is outside the declared set")


def run_specs() -> tuple[RunSpecV1, ...]:
    specs: list[RunSpecV1] = []
    for seed in SEEDS:
        specs.append(
            RunSpecV1(
                name=f"q1_{OBJECT}_seed{seed}",
                block="q1",
                object=OBJECT,
                credit_mode="credit_on",
                seed=seed,
            )
        )
    for seed in SEEDS:
        for mode in ("credit_on", "credit_stop"):
            specs.append(
                RunSpecV1(
                    name=f"q2_{mode}_seed{seed}",
                    block="q2",
                    object=OBJECT,
                    credit_mode=mode,
                    seed=seed,
                )
            )
    names = [spec.name for spec in specs]
    if len(specs) != 9 or len(set(names)) != len(names):
        raise ValueError("R5c run manifest must declare 9 unique runs")
    return tuple(specs)


def unit_cost() -> float:
    payload = json.loads(PRICING_REPORT.read_text(encoding="utf-8"))
    if payload.get("status") != "r5c_priced":
        raise ValueError("R5c training requires the frozen R5c pricing report")
    return float(payload["unit_cpu_seconds_per_update"])


def _restore_probe(path: Path) -> int:
    started = time.perf_counter()
    trainer = R5CRunTrainer.load(Path(path))
    print(
        json.dumps(
            {
                "object": trainer.adapter.name,
                "steps": trainer.steps,
                "digest": trainer.probe_digest(),
                "restore_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        )
    )
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


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _frozen_unchanged(before: Mapping[str, torch.Tensor], trainer: R5CRunTrainer) -> bool:
    parameters = dict(trainer.model.named_parameters())
    return all(torch.equal(value, parameters[name]) for name, value in before.items())


def train_run(
    spec: RunSpecV1, episodes: Sequence[Mapping[str, Any]], *, unit_cpu: float
) -> dict[str, Any]:
    adapter = NativeContentV3Adapter()
    trainer = R5CRunTrainer.create(
        adapter,
        configure=(
            credit_on_configure if spec.credit_mode == "credit_on" else credit_stop_configure
        ),
    )
    trainer.cursor.update(
        {
            "run_id": spec.name,
            "block": spec.block,
            "seed": spec.seed,
            "credit_mode": spec.credit_mode,
        }
    )
    schedule = batch_schedule(episodes, seed=spec.seed, updates=UPDATES, batch_size=BATCH_SIZE)
    checkpoint_dir = CHECKPOINT_DIR / _RUN_ID
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "run_id": spec.name,
        "block": spec.block,
        "object": spec.object,
        "credit_mode": spec.credit_mode,
        "seed": spec.seed,
        "updates_requested": UPDATES,
        "schedule_seed": spec.seed,
        "frozen_groups": list(trainer.frozen_groups),
        "parameter_counts": trainer.parameter_counts(),
        "steps_executed": 0,
        "restore_ok": False,
        "cap_stop": False,
    }
    frozen = set(trainer.frozen_groups)
    groups = trainer.model.credit_groups()
    frozen_names = tuple(name for group in frozen for name in groups.get(group, ()))
    parameters = dict(trainer.model.named_parameters())
    frozen_before = {name: parameters[name].detach().clone() for name in frozen_names}

    zero_digest = trainer.probe_digest()
    zero_path = checkpoint_dir / f"{spec.name}-zero.pt"
    trainer.save(zero_path)
    in_process = R5CRunTrainer.load(zero_path)
    fresh = _fresh_probe(zero_path)
    record["zero_restore"] = {
        "checkpoint": _display_path(zero_path),
        "probe_digest": zero_digest,
        "in_process_equal": in_process.probe_digest() == zero_digest,
        "fresh_process_equal": fresh["digest"] == zero_digest,
    }
    if not (
        record["zero_restore"]["in_process_equal"] and record["zero_restore"]["fresh_process_equal"]
    ):
        return record

    losses: list[float] = []
    cpu_samples: list[float] = []
    wall_samples: list[float] = []
    scheduled_samples: list[int] = []
    aux_samples: list[float] = []
    peak_rss_mb = psutil.Process().memory_info().rss / 1e6
    for batch in schedule:
        started_cpu = time.process_time()
        started_wall = time.perf_counter()
        metrics = trainer.update(batch.episodes)
        if adapter.schedule_is_active():
            scheduled_samples.append(trainer.steps)
        aux_samples.append(float(metrics.get("aux_pointer_loss", 0.0)))
        cpu_samples.append(time.process_time() - started_cpu)
        wall_samples.append(time.perf_counter() - started_wall)
        if not math.isfinite(float(metrics["loss_nats"])):
            raise ValueError(f"{spec.name} step {trainer.steps} produced a non-finite loss")
        losses.append(float(metrics["loss_nats"]))
        peak_rss_mb = max(peak_rss_mb, psutil.Process().memory_info().rss / 1e6)
        if trainer.steps % CAP_CHECK_EVERY == 0:
            median = statistics.median(cpu_samples)
            if median > UNIT_CAP_FACTOR * unit_cpu:
                record["cap_stop"] = True
                record["steps_executed"] = trainer.steps
                record["cpu_median_over_run"] = median
                return record

    endpoint_path = checkpoint_dir / f"{spec.name}-endpoint.pt"
    end_digest = trainer.probe_digest()
    trainer.save(endpoint_path)
    in_process_end = R5CRunTrainer.load(endpoint_path)
    fresh_end = _fresh_probe(endpoint_path)
    record["endpoint_restore"] = {
        "checkpoint": _display_path(endpoint_path),
        "probe_digest": end_digest,
        "in_process_equal": in_process_end.probe_digest() == end_digest,
        "fresh_process_equal": fresh_end["digest"] == end_digest,
    }
    record.update(
        {
            "steps_executed": trainer.steps,
            "target_byte_exposures": trainer.target_byte_exposures,
            "loss_first10_median": statistics.median(losses[:10]),
            "loss_last10_median": statistics.median(losses[-10:]),
            "loss_all_finite": all(math.isfinite(value) for value in losses),
            "scheduled_updates": len(scheduled_samples),
            "aux_pointer_loss_total": sum(aux_samples),
            "cpu_seconds_median": statistics.median(cpu_samples),
            "wall_seconds_median": statistics.median(wall_samples),
            "cpu_seconds_total": sum(cpu_samples),
            "wall_seconds_total": sum(wall_samples),
            "peak_rss_mb": peak_rss_mb,
            "frozen_parameters_unchanged": _frozen_unchanged(frozen_before, trainer),
            "parameters_finite": trainer.parameters_finite(),
            "parameter_revision": trainer.model.parameter_revision,
        }
    )
    record["restore_ok"] = bool(
        record["zero_restore"]["in_process_equal"]
        and record["zero_restore"]["fresh_process_equal"]
        and record["endpoint_restore"]["in_process_equal"]
        and record["endpoint_restore"]["fresh_process_equal"]
        and record["loss_all_finite"]
        and record["parameters_finite"]
    )
    return record


def _train(*, approve: bool) -> int:
    if not approve:
        raise SystemExit("R5c training requires explicit user approval (--approve-r5c)")
    if REPORT.exists():
        raise SystemExit(f"R5c training report is write-once: {REPORT}")
    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    if payload.get("split") != "train" or not payload.get("training_read"):
        raise SystemExit("R5c training requires the declared readable train split")
    episodes = list(payload["episodes"])
    manifest_sha256 = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
    pricing_sha256 = hashlib.sha256(PRICING_REPORT.read_bytes()).hexdigest()
    unit = unit_cost()
    specs = run_specs()
    started_at = datetime.now(UTC)
    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    budget_stop = False
    spent_cpu = 0.0
    for spec in specs:
        projected = unit * UPDATES * UNIT_CAP_FACTOR
        if spent_cpu + projected > CPU_CAP_SECONDS:
            budget_stop = True
            failures.append({"run": spec.name, "error": "CPU cap would be exceeded"})
            break
        try:
            record = train_run(spec, episodes, unit_cpu=unit)
        except Exception as exc:  # noqa: BLE001 - recorded as an engineering failure
            failures.append({"run": spec.name, "error": f"{type(exc).__name__}: {exc}"})
            break
        results[spec.name] = record
        spent_cpu += float(record.get("cpu_seconds_total", 0.0))
        if record.get("cap_stop"):
            failures.append({"run": spec.name, "error": "unit-cost cap exceeded"})
        elif not record.get("restore_ok"):
            failures.append({"run": spec.name, "error": "restore or finite gate failed"})
        if failures:
            break

    total_wall = time.perf_counter() - started_wall
    total_cpu = time.process_time() - started_cpu
    completed = len(results)
    reproducibility = {
        f"seed{seed}": (
            results.get(f"q1_{OBJECT}_seed{seed}", {})
            .get("endpoint_restore", {})
            .get("probe_digest")
            == results.get(f"q2_credit_on_seed{seed}", {})
            .get("endpoint_restore", {})
            .get("probe_digest")
        )
        for seed in SEEDS
        if f"q1_{OBJECT}_seed{seed}" in results and f"q2_credit_on_seed{seed}" in results
    }
    if failures:
        status = "r5c_budget_stop" if budget_stop else "r5c_engineering_failure"
    elif completed == len(specs) and total_wall <= WALL_CAP_HOURS * 3600:
        status = "r5c_training_complete"
    elif total_wall > WALL_CAP_HOURS * 3600:
        status = "r5c_budget_stop"
    else:
        status = "r5c_engineering_failure"

    report = {
        "format": "taiji-native-core-r5c-training-v1",
        "version": 1,
        "date": "2026-10-31",
        "contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R5C_COPYEXPR_CANDIDATE_20261031.md",
        "manifest": {
            "path": TRAIN.relative_to(ROOT).as_posix(),
            "sha256": manifest_sha256,
            "episodes": len(episodes),
            "split": payload["split"],
            "training_read": payload["training_read"],
        },
        "pricing_report": {
            "path": PRICING_REPORT.relative_to(ROOT).as_posix(),
            "sha256": pricing_sha256,
            "unit_cpu_seconds_per_update": unit,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "batch_size": BATCH_SIZE,
            "updates_per_run": UPDATES,
            "runs": len(specs),
            "unit_cap_factor": UNIT_CAP_FACTOR,
            "cpu_cap_seconds": CPU_CAP_SECONDS,
            "wall_cap_hours": WALL_CAP_HOURS,
            "run_id": _RUN_ID,
            "credit_stop_groups": ["reader", "decoder"],
            "schedule": {"own_prefix_start": 257, "own_prefix_prob": 0.5},
            "pointer_ce_weight": 0.3,
            "approval": "explicit user approval (--approve-r5c)",
        },
        "runs": results,
        "reproducibility_q1_vs_q2_credit_on": reproducibility,
        "summary": {
            "runs_completed": completed,
            "runs_declared": len(specs),
            "total_cpu_seconds": total_cpu,
            "total_wall_seconds": total_wall,
            "sum_run_cpu_seconds": spent_cpu,
        },
        "status": status,
        "engineering_failures": failures,
        "read_boundaries": {
            "query_split_read": False,
            "holdout_split_read": False,
            "sealed_read": False,
            "optimizer_updates": sum(int(r.get("steps_executed", 0)) for r in results.values()),
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(REPORT)}, ensure_ascii=False))
    return 0 if status != "r5c_engineering_failure" else 1


def main() -> int:
    global _RUN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore-probe",
        type=Path,
        default=None,
        help="child mode: load one checkpoint and print its probe digest",
    )
    parser.add_argument("--run-id", default=_RUN_ID, help="checkpoint directory suffix")
    parser.add_argument("--approve-r5c", action="store_true", help="required: user approval")
    args = parser.parse_args()
    if args.restore_probe is not None:
        return _restore_probe(args.restore_probe)
    _RUN_ID = str(args.run_id)
    return _train(approve=args.approve_r5c)


if __name__ == "__main__":
    raise SystemExit(main())
