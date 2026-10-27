"""Run the fixed N177.6 T1 arms with isolated, recoverable NativeCheckpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from taiji.adapter import TSKV8Adapter
from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4
from taiji.config import TaijiConfig
from taiji.internalization import content_digest
from taiji.joint_input_producer import JointInputProducerV1
from taiji.joint_t1_course import (
    JOINT_T1_F0_ONLY_UPDATES,
    JOINT_T1_UPDATE_COUNT,
    build_joint_t1_split,
    build_joint_t1_update_order,
    joint_t1_course_contract,
    joint_t1_split_digest,
    joint_t1_update_order_sha256,
)
from taiji.joint_t1_trace_cache import JointT1TraceCacheV1
from taiji.joint_t1_trainer import JOINT_T1_MODES, JointT1TrainerV1

DEFAULT_PARENT = (
    REPOSITORY_ROOT
    / "data"
    / "checkpoints"
    / "n1_plan176_foundation_pilot_20261026"
    / "curriculum_v4"
    / "batch_02560.pt"
)
DEFAULT_TRACE_CACHE = (
    REPOSITORY_ROOT
    / "data"
    / "checkpoints"
    / "n1_plan177_5a_preflight_20261027"
    / "train_trace_cache_v1.pt"
)
DEFAULT_ZERO_CHECKPOINT = (
    REPOSITORY_ROOT
    / "data"
    / "checkpoints"
    / "n1_plan177_5a_preflight_20261027"
    / "correct_seed_0_zero_cached.pt"
)
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "data" / "checkpoints" / "n1_plan177_6_t1_pilot_20261027"
DEFAULT_REPORT = REPOSITORY_ROOT / "reports" / "taiji_n1_plan177_6_t1_pilot_20261027.json"
PREFLIGHT_SCRIPT = (
    REPOSITORY_ROOT / "scripts" / "training" / "preflight_taiji_n1_plan177_5a_checkpoint.py"
)

CPU_CAP_SECONDS = 2_400_000
WALL_CAP_SECONDS = 220_000
ARTIFACT_CAP_BYTES = 8 * 1024**3
CHECKPOINT_INTERVAL = 256
CANARY_UPDATES = 2
ARTIFACT_RESERVE_BYTES = 32 * 1024**2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trainer_signature(trainer: JointT1TrainerV1) -> dict[str, Any]:
    return {
        "mode": trainer.mode,
        "seed": trainer.seed,
        "update_cursor": trainer.update_cursor,
        "completed_updates": trainer.completed_updates,
        "course_sha256": trainer.course_sha256,
        "train_split_sha256": trainer.train_split_sha256,
        "update_order_sha256": trainer.update_order_sha256,
        "input_trace_cache_sha256": (
            None if trainer.input_trace_cache is None else trainer.input_trace_cache.cache_sha256
        ),
        "v4_parameters_sha256": content_digest(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in trainer.model.state_dict().items()
            }
        ),
        "producer_parameters_sha256": content_digest(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in trainer.producer.state_dict().items()
            }
        ),
        "v4_optimizer_sha256": content_digest(trainer.language_trainer.optimizer.state_dict()),
        "producer_optimizer_sha256": content_digest(trainer.producer_optimizer.state_dict()),
        "next_episode_ids": (
            None
            if trainer.update_cursor >= len(trainer.update_order)
            else list(trainer.update_order[trainer.update_cursor])
        ),
    }


def _arm_id(mode: str, seed: int) -> str:
    return f"{mode}__seed_{seed}"


def _arm_order() -> tuple[tuple[str, int], ...]:
    return tuple((mode, seed) for seed in range(3) for mode in JOINT_T1_MODES)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _artifact_bytes(run_root: Path, trace_cache_path: Path, report_path: Path) -> int:
    preflight_root = trace_cache_path.parent
    return (
        _tree_bytes(run_root)
        + _tree_bytes(preflight_root)
        + (report_path.stat().st_size if report_path.exists() else 0)
        + _tree_bytes(report_path.parent / "taiji_n1_plan177_5a_preflight_20261027_v2.json")
    )


def _preflight_overhead(report_path: Path) -> tuple[float, float]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    projection = report["resource_projection"]
    cache_profile = report["train_input_trace_cache"]
    zero_profile = report["latest_zero_checkpoint_preflight"]
    backward_profiles = report["backward_only_credit_profile"]["modes"]
    cached_forward = report["cached_forward_only_profile"]
    preflight_cpu = (
        projection["cache_build_cpu_upper_seconds"]
        + cache_profile["fresh_process_load_and_verify_cpu_seconds"]
        + zero_profile["parent_process_cpu_seconds"]
        + zero_profile["fresh_process_cpu_seconds"]
        + sum(item["process_cpu_seconds"] for item in backward_profiles)
        + cached_forward["f0_only_batch"]["process_cpu_seconds"]
        + cached_forward["mixed_batch"]["process_cpu_seconds"]
    )
    preflight_wall = (
        projection["cache_build_wall_upper_seconds"]
        + cache_profile["fresh_process_load_and_verify_wall_seconds"]
        + zero_profile["wall_seconds"]
        + sum(item["wall_seconds"] for item in backward_profiles)
        + cached_forward["f0_only_batch"]["wall_seconds"]
        + cached_forward["mixed_batch"]["wall_seconds"]
    )
    return preflight_cpu, preflight_wall


def _fresh_process_restore(
    checkpoint_path: Path,
    trainer: JointT1TrainerV1,
    trace_cache_path: Path,
    timeout_seconds: int,
) -> tuple[float, dict[str, Any]]:
    command = [
        sys.executable,
        str(PREFLIGHT_SCRIPT),
        "--restore-only",
        str(checkpoint_path),
        "--trace-cache",
        str(trace_cache_path),
    ]
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "fresh-process training checkpoint restore failed: " + result.stderr[-4000:]
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("fresh-process restore returned no structured signature")
    payload = json.loads(lines[-1])
    if payload.get("signature") != _trainer_signature(trainer):
        raise RuntimeError("fresh-process restored parameters/optimizers/cursor differ")
    return float(payload["process_cpu_seconds"]), payload


def _make_trainer(
    *,
    parent_payload: dict[str, Any],
    parent_sha256: str,
    adapter_checkpoint: dict[str, Any],
    train_episodes,
    trace_cache: JointT1TraceCacheV1,
    mode: str,
    seed: int,
) -> JointT1TrainerV1:
    language_trainer, _, _ = ConditionalLanguageTrainerV4.from_checkpoint(
        copy.deepcopy(parent_payload)
    )
    torch.manual_seed(seed)
    adapter = TSKV8Adapter.from_native_checkpoint(copy.deepcopy(adapter_checkpoint))
    producer = JointInputProducerV1(
        percept_feature_dim=adapter.perception.feature_dim,
        bundle_id=f"joint-input-producer:n177-t1:seed-{seed}",
    )
    return JointT1TrainerV1(
        language_trainer=language_trainer,
        producer=producer,
        base_adapter_checkpoint=adapter_checkpoint,
        train_episodes=train_episodes,
        mode=mode,
        seed=seed,
        parent_v4_sha256=parent_sha256,
        input_trace_cache=trace_cache,
    )


def _run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--trace-cache", type=Path, default=DEFAULT_TRACE_CACHE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restore-timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    state_path = args.run_root / "run_state.json"
    manifest_path = args.run_root / "run_manifest.json"
    metrics_path = args.run_root / "metrics.jsonl"
    has_existing_run = args.run_root.exists() and any(args.run_root.iterdir())
    if has_existing_run and not args.resume:
        raise FileExistsError(
            "N177.6 run root is write-once; use --resume to continue its checkpoints"
        )
    if args.resume and not state_path.is_file():
        raise FileNotFoundError("N177.6 resume requires its existing run_state.json")
    if not args.parent.is_file() or not args.trace_cache.is_file():
        raise FileNotFoundError("N177.6 requires the fixed V4 parent and verified train cache")

    args.run_root.mkdir(parents=True, exist_ok=True)
    session_cpu_start = time.process_time()
    session_wall_start = time.perf_counter()
    preflight_cpu, preflight_wall = _preflight_overhead(args.report)
    external_cpu_seconds = 0.0

    parent_sha256 = _sha256_file(args.parent)
    parent_payload = torch.load(args.parent, map_location="cpu", weights_only=False)
    base_adapter = TSKV8Adapter(TaijiConfig(seed=17751027), episode_id="n177-t1:adapter-baseline")
    adapter_checkpoint = base_adapter.native_checkpoint()
    train_episodes = build_joint_t1_split("train")
    trace_cache = JointT1TraceCacheV1.from_payload(
        torch.load(args.trace_cache, map_location="cpu", weights_only=False)
    )
    trace_cache.validate_for(
        base_adapter_checkpoint=adapter_checkpoint,
        train_episodes=train_episodes,
    )
    run_manifest = {
        "format": "taiji-n177-6-t1-run-v1",
        "parent_v4_sha256": parent_sha256,
        "course_sha256": str(joint_t1_course_contract()["course_sha256"]),
        "train_split_sha256": joint_t1_split_digest(train_episodes),
        "adapter_checkpoint_sha256": content_digest(adapter_checkpoint),
        "update_order_sha256": joint_t1_update_order_sha256(
            build_joint_t1_update_order(train_episodes)
        ),
        "input_trace_cache_sha256": trace_cache.cache_sha256,
        "input_trace_cache_bytes": args.trace_cache.stat().st_size,
        "arm_order": [{"mode": mode, "seed": seed} for seed in range(3) for mode in JOINT_T1_MODES],
        "updates_per_arm": JOINT_T1_UPDATE_COUNT,
        "batch_size": 4,
        "f0_only_updates": JOINT_T1_F0_ONLY_UPDATES,
        "hard_caps": {
            "process_cpu_seconds": CPU_CAP_SECONDS,
            "wall_seconds": WALL_CAP_SECONDS,
            "artifact_bytes": ARTIFACT_CAP_BYTES,
        },
        "stop_rule": "stop on first cap or integrity failure; keep latest checkpoint; no automatic retry/top-up",
    }
    if args.resume:
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if content_digest(stored_manifest) != content_digest(run_manifest):
            raise ValueError("N177.6 resume manifest differs from the fixed run contract")
        progress = json.loads(state_path.read_text(encoding="utf-8"))
        carried_cpu = float(progress.get("process_cpu_seconds", 0.0))
        carried_wall = float(progress.get("wall_seconds", 0.0))
        if metrics_path.is_file():
            with metrics_path.open("r", encoding="utf-8") as handle:
                last_line = None
                for line in handle:
                    if line.strip():
                        last_line = line
            if last_line is not None:
                last_metric = json.loads(last_line)
                carried_cpu = max(
                    carried_cpu,
                    float(last_metric.get("cumulative_process_cpu_seconds", 0.0)),
                )
                carried_wall = max(
                    carried_wall,
                    float(last_metric.get("cumulative_wall_seconds", 0.0)),
                )
        progress["process_cpu_seconds"] = carried_cpu
        progress["wall_seconds"] = carried_wall
        progress["status"] = "running"
        progress["resumed_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _atomic_json(state_path, progress)
    else:
        _atomic_json(manifest_path, run_manifest)
        progress = {
            "format": "taiji-n177-6-t1-progress-v1",
            "status": "running",
            "manifest_sha256": content_digest(run_manifest),
            "completed_arms": [],
            "latest_checkpoint_by_arm": {},
            "process_cpu_seconds": preflight_cpu,
            "wall_seconds": preflight_wall,
            "artifact_bytes": _artifact_bytes(args.run_root, args.trace_cache, args.report),
            "optimizer_updates": 0,
            "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _atomic_json(state_path, progress)

    cpu_carry = float(progress["process_cpu_seconds"])
    wall_carry = float(progress["wall_seconds"])

    def current_usage() -> tuple[float, float]:
        return (
            cpu_carry + time.process_time() - session_cpu_start + external_cpu_seconds,
            wall_carry + time.perf_counter() - session_wall_start,
        )

    def save_checkpoint(trainer: JointT1TrainerV1, name: str, arm_id: str) -> Path:
        target = args.run_root / arm_id / name
        if target.exists():
            raise FileExistsError(f"N177.6 checkpoint path is write-once: {target}")
        projected_artifacts = (
            _artifact_bytes(args.run_root, args.trace_cache, args.report) + 24 * 1024**2
        )
        if projected_artifacts > ARTIFACT_CAP_BYTES:
            raise RuntimeError("N177.6 artifact cap would be exceeded by the next checkpoint")
        trainer.save_checkpoint(target)
        return target

    def checkpoint_progress(
        trainer: JointT1TrainerV1,
        arm_id: str,
        checkpoint_path: Path,
        *,
        completed: bool = False,
    ) -> None:
        cpu_used, wall_used = current_usage()
        progress["process_cpu_seconds"] = cpu_used
        progress["wall_seconds"] = wall_used
        progress["artifact_bytes"] = _artifact_bytes(args.run_root, args.trace_cache, args.report)
        progress["optimizer_updates"] = sum(
            int(item.get("completed_updates", 0))
            for item in progress["latest_checkpoint_by_arm"].values()
        )
        progress["latest_checkpoint_by_arm"][arm_id] = {
            "path": str(checkpoint_path),
            "update_cursor": trainer.update_cursor,
            "completed_updates": trainer.completed_updates,
        }
        if completed and arm_id not in progress["completed_arms"]:
            progress["completed_arms"].append(arm_id)
        _atomic_json(state_path, progress)

    for seed in range(3):
        for mode in JOINT_T1_MODES:
            arm_id = _arm_id(mode, seed)
            if arm_id in progress.get("completed_arms", []):
                continue
            arm_directory = args.run_root / arm_id
            arm_directory.mkdir(parents=True, exist_ok=True)
            latest = progress.get("latest_checkpoint_by_arm", {}).get(arm_id)
            latest_path = (
                Path(latest["path"])
                if latest is not None and Path(latest["path"]).is_file()
                else None
            )
            if latest_path is None:
                candidates = tuple(arm_directory.glob("*.pt"))
                if candidates:
                    latest_path = max(candidates, key=lambda item: item.stat().st_mtime)
            if latest_path is not None:
                zero_path = latest_path
                zero_payload = torch.load(zero_path, map_location="cpu", weights_only=False)
                trainer = JointT1TrainerV1.from_native_checkpoint(
                    zero_payload,
                    input_trace_cache=trace_cache,
                )
                if (trainer.mode, trainer.seed) != (mode, seed):
                    raise ValueError("resume checkpoint belongs to another N177.6 arm")
            elif mode == "correct" and seed == 0:
                zero_path = DEFAULT_ZERO_CHECKPOINT
                zero_payload = torch.load(zero_path, map_location="cpu", weights_only=False)
                trainer = JointT1TrainerV1.from_native_checkpoint(
                    zero_payload,
                    input_trace_cache=trace_cache,
                )
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "zero_saved",
                        "arm_id": arm_id,
                        "mode": mode,
                        "seed": seed,
                        "checkpoint": str(zero_path),
                        "signature": _trainer_signature(trainer),
                    },
                )
                external_cpu_seconds += _fresh_process_restore(
                    zero_path,
                    trainer,
                    args.trace_cache,
                    args.restore_timeout_seconds,
                )[0]
            else:
                trainer = _make_trainer(
                    parent_payload=parent_payload,
                    parent_sha256=parent_sha256,
                    adapter_checkpoint=adapter_checkpoint,
                    train_episodes=train_episodes,
                    trace_cache=trace_cache,
                    mode=mode,
                    seed=seed,
                )
                zero_path = save_checkpoint(trainer, "update_000000_zero.pt", arm_id)
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "zero_saved",
                        "arm_id": arm_id,
                        "mode": mode,
                        "seed": seed,
                        "checkpoint": str(zero_path),
                        "signature": _trainer_signature(trainer),
                    },
                )
                external_cpu_seconds += _fresh_process_restore(
                    zero_path,
                    trainer,
                    args.trace_cache,
                    args.restore_timeout_seconds,
                )[0]
                checkpoint_payload = torch.load(zero_path, map_location="cpu", weights_only=False)
                trainer = JointT1TrainerV1.from_native_checkpoint(
                    checkpoint_payload,
                    input_trace_cache=trace_cache,
                )
            checkpoint_progress(trainer, arm_id, zero_path)

            if mode == "correct" and seed == 0 and trainer.update_cursor < CANARY_UPDATES:
                for _ in range(trainer.update_cursor, CANARY_UPDATES):
                    cpu_used, wall_used = current_usage()
                    if (
                        cpu_used >= CPU_CAP_SECONDS
                        or wall_used >= WALL_CAP_SECONDS
                        or _artifact_bytes(args.run_root, args.trace_cache, args.report)
                        >= ARTIFACT_CAP_BYTES
                    ):
                        progress["status"] = "budget_stop"
                        progress["stop_reason"] = "hard cap reached before canary completion"
                        _atomic_json(state_path, progress)
                        return 2
                    receipt = trainer.update()
                    _append_jsonl(
                        metrics_path,
                        {
                            "event": "update",
                            "arm_id": arm_id,
                            "cumulative_process_cpu_seconds": current_usage()[0],
                            "cumulative_wall_seconds": current_usage()[1],
                            **receipt,
                        },
                    )
                canary_path = save_checkpoint(trainer, "canary_update_000002.pt", arm_id)
                external_cpu_seconds += _fresh_process_restore(
                    canary_path,
                    trainer,
                    args.trace_cache,
                    args.restore_timeout_seconds,
                )[0]
                canary_payload = torch.load(canary_path, map_location="cpu", weights_only=False)
                trainer = JointT1TrainerV1.from_native_checkpoint(
                    canary_payload,
                    input_trace_cache=trace_cache,
                )
                checkpoint_progress(trainer, arm_id, canary_path)

            status = "completed"
            while trainer.update_cursor < JOINT_T1_UPDATE_COUNT:
                cpu_used, wall_used = current_usage()
                artifact_used = _artifact_bytes(args.run_root, args.trace_cache, args.report)
                if (
                    cpu_used >= CPU_CAP_SECONDS
                    or wall_used >= WALL_CAP_SECONDS
                    or artifact_used >= ARTIFACT_CAP_BYTES
                ):
                    status = "budget_stop"
                    break
                receipt = trainer.update()
                cpu_used, wall_used = current_usage()
                _append_jsonl(
                    metrics_path,
                    {
                        "event": "update",
                        "arm_id": arm_id,
                        "cumulative_process_cpu_seconds": cpu_used,
                        "cumulative_wall_seconds": wall_used,
                        **receipt,
                    },
                )
                checkpoint_due = (
                    trainer.update_cursor % CHECKPOINT_INTERVAL == 0
                    or trainer.update_cursor == JOINT_T1_F0_ONLY_UPDATES
                    or trainer.update_cursor == JOINT_T1_UPDATE_COUNT
                )
                if checkpoint_due:
                    suffix = (
                        "_f0_f1_boundary"
                        if trainer.update_cursor == JOINT_T1_F0_ONLY_UPDATES
                        else ""
                    )
                    checkpoint_path = save_checkpoint(
                        trainer,
                        f"update_{trainer.update_cursor:06d}{suffix}.pt",
                        arm_id,
                    )
                    checkpoint_progress(trainer, arm_id, checkpoint_path)
                if (
                    cpu_used >= CPU_CAP_SECONDS
                    or wall_used >= WALL_CAP_SECONDS
                    or _artifact_bytes(args.run_root, args.trace_cache, args.report)
                    >= ARTIFACT_CAP_BYTES
                ):
                    status = "budget_stop"
                    if not checkpoint_due:
                        emergency = save_checkpoint(
                            trainer,
                            f"budget_stop_update_{trainer.update_cursor:06d}.pt",
                            arm_id,
                        )
                        checkpoint_progress(trainer, arm_id, emergency)
                    break

            if status == "completed":
                final_path = args.run_root / arm_id / f"update_{JOINT_T1_UPDATE_COUNT:06d}.pt"
                checkpoint_progress(trainer, arm_id, final_path, completed=True)
            else:
                progress["status"] = "budget_stop"
                progress["stop_reason"] = "one or more fixed N177.6 hard caps reached"
                checkpoint_path = progress["latest_checkpoint_by_arm"].get(arm_id, {}).get("path")
                if checkpoint_path is None:
                    emergency = save_checkpoint(
                        trainer,
                        f"budget_stop_update_{trainer.update_cursor:06d}.pt",
                        arm_id,
                    )
                    checkpoint_path = str(emergency)
                checkpoint_progress(trainer, arm_id, Path(checkpoint_path))
                _atomic_json(state_path, progress)
                return 2

    progress["status"] = "training_complete_evaluation_pending"
    progress["process_cpu_seconds"], progress["wall_seconds"] = current_usage()
    progress["artifact_bytes"] = _artifact_bytes(args.run_root, args.trace_cache, args.report)
    progress["ended_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_json(state_path, progress)
    report = {
        "format": "taiji-n177-6-t1-run-report-v1",
        "status": progress["status"],
        "run_manifest": run_manifest,
        "resources": {
            "process_cpu_seconds": progress["process_cpu_seconds"],
            "wall_seconds": progress["wall_seconds"],
            "artifact_bytes": progress["artifact_bytes"],
            "caps": run_manifest["hard_caps"],
        },
        "arms_completed": len(progress["completed_arms"]),
        "completed_arm_ids": progress["completed_arms"],
        "evaluation": "pending separate read-only scorer",
        "sealed_reads": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.report, report)
    return 0


if __name__ == "__main__":
    run_root = DEFAULT_RUN_ROOT
    if "--run-root" in sys.argv:
        index = sys.argv.index("--run-root")
        if index + 1 < len(sys.argv):
            run_root = Path(sys.argv[index + 1])
    try:
        raise SystemExit(_run())
    except Exception as error:
        state_path = run_root / "run_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = "engineering_failure"
            state["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "latest_checkpoint_by_arm": state.get("latest_checkpoint_by_arm", {}),
            }
            _atomic_json(state_path, state)
        raise
