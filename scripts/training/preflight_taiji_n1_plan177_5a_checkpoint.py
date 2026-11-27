"""Zero-update NativeCheckpoint preflight for the N177.5A joint T1 trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from taiji.adapter import TSKV8Adapter
from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4
from taiji.config import TaijiConfig
from taiji.contracts import NativeCheckpoint
from taiji.internalization import content_digest
from taiji.joint_input_producer import JointInputProducerV1
from taiji.joint_t1_course import (
    JOINT_T1_F0_ONLY_UPDATES,
    JOINT_T1_UPDATE_COUNT,
    build_joint_t1_split,
)
from taiji.joint_t1_trace_cache import (
    JointT1TraceCacheV1,
    build_joint_t1_trace_cache,
    verify_joint_t1_trace_cache_replay,
)
from taiji.joint_t1_trainer import JOINT_T1_MODES, JointT1TrainerV1

DEFAULT_PARENT = (
    REPOSITORY_ROOT
    / "data"
    / "checkpoints"
    / "n1_plan176_foundation_pilot_20261026"
    / "curriculum_v4"
    / "batch_02560.pt"
)
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "data"
    / "checkpoints"
    / "n1_plan177_5a_preflight_20261027"
    / "correct_seed_0_zero_cached.pt"
)
DEFAULT_REPORT = REPOSITORY_ROOT / "reports" / "taiji_n1_plan177_5a_preflight_20261027_v2.json"
DEFAULT_TRACE_CACHE = (
    REPOSITORY_ROOT
    / "data"
    / "checkpoints"
    / "n1_plan177_5a_preflight_20261027"
    / "train_trace_cache_v1.pt"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(trainer: JointT1TrainerV1) -> dict[str, Any]:
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


def _restore_only(checkpoint: Path, trace_cache_path: Path | None) -> dict[str, Any]:
    start_cpu = time.process_time()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cache = None
    if trace_cache_path is not None:
        cache = JointT1TraceCacheV1.from_payload(
            torch.load(trace_cache_path, map_location="cpu", weights_only=False)
        )
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=cache)
    signature = _signature(trainer)
    envelope = NativeCheckpoint.from_payload(payload)
    return {
        "fresh_process_restore": True,
        "process_cpu_seconds": time.process_time() - start_cpu,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "native_component_keys": sorted(envelope.components),
        "signature": signature,
    }


def _profile_forward_batch(
    trainer: JointT1TrainerV1,
    batch_index: int,
) -> dict[str, Any]:
    if not 0 <= batch_index < len(trainer.update_order):
        raise ValueError("requested forward-only batch is outside the fixed schedule")
    batch_ids = trainer.update_order[batch_index]
    episode_ids = tuple(batch_ids)
    turns = 0
    input_bytes = 0
    answer_bytes = 0
    parser_losses = []
    answer_losses = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for episode_id in episode_ids:
        episode = trainer.train_episodes[episode_id]
        turns += len(episode.turns)
        input_bytes += sum(len(bytes(turn.frame.payload)) for turn in episode.turns)
        parser_loss, answer_loss, exposure, _ = trainer._run_episode(episode)
        parser_losses.append(float(parser_loss.detach()))
        answer_losses.append(float(answer_loss.detach()))
        answer_bytes += exposure
        del parser_loss, answer_loss
    return {
        "batch_index": batch_index,
        "episode_ids": list(episode_ids),
        "turns": turns,
        "input_bytes": input_bytes,
        "answer_target_bytes": answer_bytes,
        "parser_loss_nats_mean": sum(parser_losses) / len(parser_losses),
        "answer_loss_nats_per_byte_mean": sum(answer_losses) / len(answer_losses),
        "process_cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.perf_counter() - wall_start,
        "backward_passes": 0,
        "optimizer_updates": 0,
    }


def _run_forward_only(
    checkpoint: Path,
    report_path: Path,
    trace_cache_path: Path | None,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cache = None
    if trace_cache_path is not None:
        cache_payload = torch.load(trace_cache_path, map_location="cpu", weights_only=False)
        cache = JointT1TraceCacheV1.from_payload(cache_payload)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=cache)
    before = _signature(trainer)
    f0 = _profile_forward_batch(trainer, 0)
    mixed = _profile_forward_batch(trainer, JOINT_T1_F0_ONLY_UPDATES)
    after = _signature(trainer)
    if before != after:
        raise RuntimeError("forward-only preflight changed model, optimizer or course state")
    report = {
        "format": "taiji-n177-5a-forward-only-profile-v1",
        "status": "pass",
        "checkpoint_path": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "trace_cache_path": None if trace_cache_path is None else str(trace_cache_path),
        "trace_cache_bytes": (
            None if trace_cache_path is None else trace_cache_path.stat().st_size
        ),
        "course_sha256": trainer.course_sha256,
        "train_split_sha256": trainer.train_split_sha256,
        "update_order_sha256": trainer.update_order_sha256,
        "parameters_and_optimizers_unchanged": True,
        "f0_only_batch": f0,
        "mixed_batch": mixed,
        "no_backward_or_optimizer_step": True,
    }
    if report_path.is_file():
        base_report = json.loads(report_path.read_text(encoding="utf-8"))
        key = (
            "uncached_forward_only_profile"
            if trace_cache_path is None
            else "cached_forward_only_profile"
        )
        base_report[key] = report
        report_path.write_text(
            json.dumps(base_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _build_trace_cache(
    *,
    parent_path: Path,
    cache_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if not parent_path.is_file():
        raise FileNotFoundError(f"fixed N176 V4 parent does not exist: {parent_path}")
    if cache_path.exists():
        return _verify_existing_trace_cache(
            parent_path=parent_path,
            cache_path=cache_path,
            report_path=report_path,
        )
    adapter = TSKV8Adapter(TaijiConfig(seed=17751027), episode_id="n177-t1:adapter-baseline")
    adapter_checkpoint = adapter.native_checkpoint()
    episodes = build_joint_t1_split("train")
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    cache = build_joint_t1_trace_cache(
        base_adapter_checkpoint=adapter_checkpoint,
        train_episodes=episodes,
    )
    build_cpu = time.process_time() - cpu_start
    build_wall = time.perf_counter() - wall_start
    f0_episode = next(item for item in episodes if item.level == "F0")
    f1_episode = next(item for item in episodes if item.level == "F1")
    replay = verify_joint_t1_trace_cache_replay(
        cache,
        base_adapter_checkpoint=adapter_checkpoint,
        train_episodes=episodes,
        episode_ids=(f0_episode.episode_id, f1_episode.episode_id),
    )
    cache.save(cache_path)
    profile = {
        "format": "taiji-n177-5a-train-input-cache-preflight-v1",
        "status": "pass",
        "train_only": True,
        "query_or_holdout_materialized": False,
        "optimizer_updates": 0,
        "parent_checkpoint_sha256": _sha256_file(parent_path),
        "adapter_checkpoint_sha256": cache.adapter_checkpoint_sha256,
        "course_sha256": cache.course_sha256,
        "train_split_sha256": cache.train_split_sha256,
        "cache_sha256": cache.cache_sha256,
        "episodes": len(episodes),
        "turns": cache.turn_count,
        "input_bytes": cache.input_byte_count,
        "cache_path": str(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "build_process_cpu_seconds": build_cpu,
        "build_wall_seconds": build_wall,
        "fresh_adapter_replay_audit": replay,
        "limitations": [
            "Only frozen Adapter traces and their per-turn CognitiveState snapshots are cached.",
            "V4 TaskState, semantic parses and producer/world candidates are rebuilt on every update.",
            "No backward pass or optimizer step was executed.",
        ],
    }
    if report_path.is_file():
        base_report = json.loads(report_path.read_text(encoding="utf-8"))
        base_report["train_input_trace_cache"] = profile
        base_report["formal_data_files_written"] = False
        base_report["train_only_trace_cache_written"] = True
        report_path.write_text(
            json.dumps(base_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return profile


def _verify_existing_trace_cache(
    *,
    parent_path: Path,
    cache_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"T1 train trace cache does not exist: {cache_path}")
    adapter = TSKV8Adapter(TaijiConfig(seed=17751027), episode_id="n177-t1:adapter-baseline")
    adapter_checkpoint = adapter.native_checkpoint()
    episodes = build_joint_t1_split("train")
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache = JointT1TraceCacheV1.from_payload(payload)
    cache.validate_for(
        base_adapter_checkpoint=adapter_checkpoint,
        train_episodes=episodes,
    )
    f0_episode = next(item for item in episodes if item.level == "F0")
    f1_episode = next(item for item in episodes if item.level == "F1")
    replay = verify_joint_t1_trace_cache_replay(
        cache,
        base_adapter_checkpoint=adapter_checkpoint,
        train_episodes=episodes,
        episode_ids=(f0_episode.episode_id, f1_episode.episode_id),
    )
    verify_cpu = time.process_time() - cpu_start
    verify_wall = time.perf_counter() - wall_start
    profile = {
        "format": "taiji-n177-5a-train-input-cache-preflight-v1",
        "status": "pass",
        "cache_preexisted_at_verification": True,
        "cache_build_cost_captured": False,
        "train_only": True,
        "query_or_holdout_materialized": False,
        "optimizer_updates": 0,
        "parent_checkpoint_sha256": _sha256_file(parent_path),
        "adapter_checkpoint_sha256": cache.adapter_checkpoint_sha256,
        "course_sha256": cache.course_sha256,
        "train_split_sha256": cache.train_split_sha256,
        "cache_sha256": cache.cache_sha256,
        "episodes": len(episodes),
        "turns": cache.turn_count,
        "input_bytes": cache.input_byte_count,
        "cache_path": str(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "fresh_process_load_and_verify_cpu_seconds": verify_cpu,
        "fresh_process_load_and_verify_wall_seconds": verify_wall,
        "fresh_adapter_replay_audit": replay,
        "limitations": [
            "The source artifact existed before this verification command, so its build CPU is not reported.",
            "Only frozen Adapter traces and per-turn CognitiveState snapshots are cached.",
            "V4 TaskState, semantic parses and producer/world candidates remain uncached.",
            "No backward pass or optimizer step was executed.",
        ],
    }
    if report_path.is_file():
        base_report = json.loads(report_path.read_text(encoding="utf-8"))
        base_report["train_input_trace_cache"] = profile
        base_report["formal_data_files_written"] = False
        base_report["train_only_trace_cache_written"] = True
        report_path.write_text(
            json.dumps(base_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return profile


def _run_credit_profile(
    *,
    parent_path: Path,
    cache_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if not parent_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("backward-only profile requires the fixed parent and trace cache")
    adapter = TSKV8Adapter(TaijiConfig(seed=17751027), episode_id="n177-t1:adapter-baseline")
    adapter_checkpoint = adapter.native_checkpoint()
    episodes = build_joint_t1_split("train")
    cache = JointT1TraceCacheV1.from_payload(
        torch.load(cache_path, map_location="cpu", weights_only=False)
    )
    cache.validate_for(base_adapter_checkpoint=adapter_checkpoint, train_episodes=episodes)
    parent_sha = _sha256_file(parent_path)
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    profiles = []
    for mode in JOINT_T1_MODES:
        language_trainer, _, _ = ConditionalLanguageTrainerV4.from_checkpoint(parent_payload)
        torch.manual_seed(0)
        producer = JointInputProducerV1(
            percept_feature_dim=adapter.perception.feature_dim,
            bundle_id="joint-input-producer:n177-t1:seed-0",
        )
        trainer = JointT1TrainerV1(
            language_trainer=language_trainer,
            producer=producer,
            base_adapter_checkpoint=adapter_checkpoint,
            train_episodes=episodes,
            mode=mode,
            seed=0,
            parent_v4_sha256=parent_sha,
            input_trace_cache=cache,
        )
        profiles.append(trainer.profile_backward_only(batch_index=640))
        del trainer, producer, language_trainer
    profile = {
        "format": "taiji-n177-5a-backward-only-credit-profile-v1",
        "status": "pass",
        "parent_checkpoint_sha256": parent_sha,
        "course_sha256": cache.course_sha256,
        "train_split_sha256": cache.train_split_sha256,
        "input_trace_cache_sha256": cache.cache_sha256,
        "input_trace_cache_bytes": cache_path.stat().st_size,
        "modes": profiles,
        "optimizer_updates": 0,
        "sealed_reads": 0,
        "parameters_and_optimizers_unchanged": all(
            item["parameters_optimizers_and_cursor_unchanged"] for item in profiles
        ),
        "limitations": [
            "Backward/gradient routing was measured on one mixed train-only batch per mode.",
            "No optimizer step, formal update, evaluation target or sealed item was consumed.",
            "This is not an estimate of total all-arm cost; the fixed schedule still needs projection.",
        ],
    }
    if not profile["parameters_and_optimizers_unchanged"]:
        raise RuntimeError("backward-only credit profile changed parameters or optimizer state")
    if report_path.is_file():
        base_report = json.loads(report_path.read_text(encoding="utf-8"))
        uncached_report_path = (
            REPOSITORY_ROOT
            / "reports"
            / "taiji_n1_plan177_5a_zero_checkpoint_preflight_20261027.json"
        )
        uncached_report = json.loads(uncached_report_path.read_text(encoding="utf-8"))
        uncached = uncached_report["forward_only_profile"]
        cached = base_report["cached_forward_only_profile"]
        cached_f0 = cached["f0_only_batch"]
        cached_mixed = cached["mixed_batch"]
        uncached_f0 = uncached["f0_only_batch"]
        uncached_mixed = uncached["mixed_batch"]
        arms_by_mode = {item["mode"]: item for item in profiles}
        training_cpu = training_wall = 0.0
        mode_projections = []
        for mode in JOINT_T1_MODES:
            item = arms_by_mode[mode]
            cpu_scale = item["process_cpu_seconds"] / cached_mixed["process_cpu_seconds"]
            wall_scale = item["wall_seconds"] / cached_mixed["wall_seconds"]
            f0_cpu = cached_f0["process_cpu_seconds"] * cpu_scale
            f0_wall = cached_f0["wall_seconds"] * wall_scale
            per_arm_cpu = (
                JOINT_T1_F0_ONLY_UPDATES * f0_cpu
                + (JOINT_T1_UPDATE_COUNT - JOINT_T1_F0_ONLY_UPDATES) * item["process_cpu_seconds"]
            )
            per_arm_wall = (
                JOINT_T1_F0_ONLY_UPDATES * f0_wall
                + (JOINT_T1_UPDATE_COUNT - JOINT_T1_F0_ONLY_UPDATES) * item["wall_seconds"]
            )
            mode_cpu = 3 * per_arm_cpu
            mode_wall = 3 * per_arm_wall
            training_cpu += mode_cpu
            training_wall += mode_wall
            mode_projections.append(
                {
                    "mode": mode,
                    "estimated_cpu_seconds_all_three_seeds": mode_cpu,
                    "estimated_wall_seconds_all_three_seeds": mode_wall,
                }
            )

        unique_train_turns = 64 * 5 * 5 + 64 * 5 * 8
        cache_cpu_upper = (
            max(
                uncached_f0["process_cpu_seconds"] / uncached_f0["turns"],
                uncached_mixed["process_cpu_seconds"] / uncached_mixed["turns"],
            )
            * unique_train_turns
        )
        cache_wall_upper = (
            max(
                uncached_f0["wall_seconds"] / uncached_f0["turns"],
                uncached_mixed["wall_seconds"] / uncached_mixed["turns"],
            )
            * unique_train_turns
        )
        evaluation_turns_per_arm = 2 * (32 * 4 * 5 + 32 * 4 * 8)
        evaluation_cpu = (
            cached_mixed["process_cpu_seconds"]
            / cached_mixed["turns"]
            * evaluation_turns_per_arm
            * 15
            * 1.5
        )
        evaluation_wall = (
            cached_mixed["wall_seconds"]
            / cached_mixed["turns"]
            * evaluation_turns_per_arm
            * 15
            * 1.5
        )
        checkpoint_slots = 15 * 12
        checkpoint_bytes_with_optimizer_allowance = (
            base_report["latest_zero_checkpoint_preflight"]["zero_checkpoint_bytes"] + 1024 * 1024
        )
        artifact_base = (
            checkpoint_slots * checkpoint_bytes_with_optimizer_allowance
            + cache_path.stat().st_size
            + 512 * 1024 * 1024
        )
        optimizer_step_overhead_cpu = training_cpu * 0.10
        optimizer_step_overhead_wall = training_wall * 0.10
        total_cpu_before_reserve = (
            training_cpu
            + optimizer_step_overhead_cpu
            + cache_cpu_upper
            + evaluation_cpu
            + 15 * 102.90625
            + checkpoint_slots
        )
        total_wall_before_reserve = (
            training_wall
            + optimizer_step_overhead_wall
            + cache_wall_upper
            + evaluation_wall
            + 15 * 172.94031709991395
            + checkpoint_slots
        )
        resource_projection = {
            "basis": "one mixed train-only backward profile per mode; F0 scaled from cached forward profile; cache build bounded by uncached full-forward cost per unique turn; evaluation allowance is 1.5x cached per-turn forward",
            "mode_training_estimates": mode_projections,
            "training_cpu_seconds_before_optimizer_overhead": training_cpu,
            "training_wall_seconds_before_optimizer_overhead": training_wall,
            "optimizer_step_overhead_fraction": 0.10,
            "trace_cache_build_cpu_upper_seconds": cache_cpu_upper,
            "trace_cache_build_wall_upper_seconds": cache_wall_upper,
            "evaluation_turns_per_arm": evaluation_turns_per_arm,
            "evaluation_cpu_upper_seconds": evaluation_cpu,
            "evaluation_wall_upper_seconds": evaluation_wall,
            "zero_restore_cpu_upper_seconds": 15 * 102.90625,
            "zero_restore_wall_upper_seconds": 15 * 172.94031709991395,
            "checkpoint_slots": checkpoint_slots,
            "one_checkpoint_optimizer_growth_allowance_bytes": 1024 * 1024,
            "artifact_base_bytes_including_512_mib_outputs": artifact_base,
            "total_cpu_before_30_percent_reserve": total_cpu_before_reserve,
            "total_wall_before_30_percent_reserve": total_wall_before_reserve,
            "projected_cpu_with_30_percent_reserve": total_cpu_before_reserve * 1.3,
            "projected_wall_with_30_percent_reserve": total_wall_before_reserve * 1.3,
            "projected_artifacts_with_30_percent_reserve_bytes": artifact_base * 1.3,
            "proposed_hard_caps": {
                "process_cpu_seconds": 2200000,
                "wall_seconds": 200000,
                "artifact_bytes": 8 * 1024**3,
            },
            "limitations": [
                "Cache build duration itself was not captured; the upper estimate uses a larger uncached full-forward cost per unique input turn.",
                "F0 backward and full evaluation are projected from the measured mixed-batch profiles with stated multipliers.",
                "The first two optimizer updates remain a canary inside the fixed run caps.",
            ],
        }
        base_report["backward_only_credit_profile"] = profile
        base_report["resource_projection"] = resource_projection
        report_path.write_text(
            json.dumps(base_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return profile


def _fresh_restore(
    checkpoint: Path,
    expected_signature: dict[str, Any],
    *,
    trace_cache_path: Path | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--restore-only", str(checkpoint)]
    if trace_cache_path is not None:
        command.extend(("--trace-cache", str(trace_cache_path)))
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh-process NativeCheckpoint restore failed: " + completed.stderr[-4000:]
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("fresh-process restore returned no structured result")
    restored = json.loads(lines[-1])
    if restored.get("signature") != expected_signature:
        raise RuntimeError("fresh-process model/optimizer/course cursor differs from zero state")
    return restored


def _build_zero_trainer(
    parent_path: Path,
    trace_cache_path: Path | None,
) -> tuple[JointT1TrainerV1, dict[str, Any]]:
    adapter_config = TaijiConfig(seed=17751027)
    adapter = TSKV8Adapter(adapter_config, episode_id="n177-t1:adapter-baseline")
    adapter_checkpoint = adapter.native_checkpoint()
    parent_sha256 = _sha256_file(parent_path)
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    language_trainer, _, _ = ConditionalLanguageTrainerV4.from_checkpoint(parent_payload)
    torch.manual_seed(0)
    producer = JointInputProducerV1(
        percept_feature_dim=adapter.perception.feature_dim,
        bundle_id="joint-input-producer:n177-t1:seed-0",
    )
    episodes = build_joint_t1_split("train")
    trace_cache = None
    if trace_cache_path is not None:
        trace_cache = JointT1TraceCacheV1.from_payload(
            torch.load(trace_cache_path, map_location="cpu", weights_only=False)
        )
    trainer = JointT1TrainerV1(
        language_trainer=language_trainer,
        producer=producer,
        base_adapter_checkpoint=adapter_checkpoint,
        train_episodes=episodes,
        mode="correct",
        seed=0,
        parent_v4_sha256=parent_sha256,
        input_trace_cache=trace_cache,
    )
    return trainer, {
        "parent_checkpoint_sha256": parent_sha256,
        "parent_checkpoint_bytes": parent_path.stat().st_size,
        "adapter_config": adapter_config.to_dict(),
        "adapter_checkpoint_sha256": content_digest(adapter_checkpoint),
        "percept_feature_dim": adapter.perception.feature_dim,
    }


def _run_preflight(
    *,
    parent_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    trace_cache_path: Path | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not parent_path.is_file():
        raise FileNotFoundError(f"fixed N176 V4 parent does not exist: {parent_path}")
    if checkpoint_path.exists():
        raise FileExistsError(f"zero checkpoint path is write-once: {checkpoint_path}")
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    trainer, provenance = _build_zero_trainer(parent_path, trace_cache_path)
    expected = _signature(trainer)
    trainer.save_checkpoint(checkpoint_path)
    restored = _fresh_restore(
        checkpoint_path,
        expected,
        trace_cache_path=trace_cache_path,
        timeout_seconds=timeout_seconds,
    )
    elapsed_wall = time.perf_counter() - start_wall
    elapsed_cpu = time.process_time() - start_cpu
    report = {
        "format": "taiji-n177-5a-zero-checkpoint-preflight-v1",
        "status": "pass",
        "date": "2026-10-27",
        "mode": "correct",
        "seed": 0,
        "training_updates": 0,
        "optimizer_steps": 0,
        "sealed_reads": 0,
        "formal_data_files_written": False,
        "parent": provenance,
        "course_sha256": trainer.course_sha256,
        "train_split_sha256": trainer.train_split_sha256,
        "update_order_sha256": trainer.update_order_sha256,
        "update_count": len(trainer.update_order),
        "episodes_in_train_split": len(trainer.train_episodes),
        "zero_checkpoint_path": str(checkpoint_path),
        "zero_checkpoint_bytes": checkpoint_path.stat().st_size,
        "input_trace_cache_sha256": (
            None if trainer.input_trace_cache is None else trainer.input_trace_cache.cache_sha256
        ),
        "input_trace_cache_bytes": (
            None if trace_cache_path is None else trace_cache_path.stat().st_size
        ),
        "fresh_process_restore": restored["fresh_process_restore"],
        "fresh_process_cpu_seconds": restored["process_cpu_seconds"],
        "native_component_keys": restored["native_component_keys"],
        "signature": expected,
        "parent_process_cpu_seconds": elapsed_cpu,
        "wall_seconds": elapsed_wall,
        "limitations": [
            "No optimizer update or backward pass was executed.",
            "One zero-state checkpoint does not estimate post-update Adam moment size.",
            "No model behavior or capability claim is made by this preflight.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stored = {}
    if report_path.is_file():
        stored = json.loads(report_path.read_text(encoding="utf-8"))
    stored["latest_zero_checkpoint_preflight"] = report
    report_path.write_text(
        json.dumps(stored, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-only", type=Path)
    parser.add_argument("--forward-only", type=Path)
    parser.add_argument("--build-trace-cache", action="store_true")
    parser.add_argument("--profile-backward-only", action="store_true")
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--trace-cache", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.restore_only is not None:
        print(
            json.dumps(
                _restore_only(args.restore_only, args.trace_cache),
                sort_keys=True,
            )
        )
        return 0
    if args.forward_only is not None:
        profile = _run_forward_only(args.forward_only, args.report, args.trace_cache)
        print(json.dumps(profile, ensure_ascii=False, sort_keys=True))
        return 0
    if args.build_trace_cache:
        profile = _build_trace_cache(
            parent_path=args.parent,
            cache_path=DEFAULT_TRACE_CACHE,
            report_path=args.report,
        )
        print(json.dumps(profile, ensure_ascii=False, sort_keys=True))
        return 0
    if args.profile_backward_only:
        profile = _run_credit_profile(
            parent_path=args.parent,
            cache_path=DEFAULT_TRACE_CACHE,
            report_path=args.report,
        )
        print(json.dumps(profile, ensure_ascii=False, sort_keys=True))
        return 0
    report = _run_preflight(
        parent_path=args.parent,
        checkpoint_path=args.checkpoint,
        report_path=args.report,
        trace_cache_path=args.trace_cache,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
