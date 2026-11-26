"""Reproducible N177.5A resource projection from measured preflight profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "taiji_n1_plan177_5a_preflight_20261027_v2.json"
UNCACHED_REPORT = ROOT / "reports" / "taiji_n1_plan177_5a_zero_checkpoint_preflight_20261027.json"
TRACE_CACHE = (
    ROOT / "data" / "checkpoints" / "n1_plan177_5a_preflight_20261027" / "train_trace_cache_v1.pt"
)

F0_ONLY_UPDATES = 640
UPDATES_PER_ARM = 2560
SEEDS = 3
CHECKPOINTS_PER_ARM = 12
ARM_COUNT = 15
RESERVE = 1.30
CPU_CAP = 2_400_000
WALL_CAP = 220_000
ARTIFACT_CAP = 8 * 1024**3


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project() -> dict[str, Any]:
    report = _read(REPORT)
    uncached = _read(UNCACHED_REPORT)["forward_only_profile"]
    cached = report["cached_forward_only_profile"]
    gradient = report["backward_only_credit_profile"]
    cache = report["train_input_trace_cache"]
    cached_f0 = cached["f0_only_batch"]
    cached_mixed = cached["mixed_batch"]
    profiles = {item["mode"]: item for item in gradient["modes"]}

    mode_estimates = []
    training_cpu = training_wall = 0.0
    for mode, item in profiles.items():
        f0_cpu = (
            cached_f0["process_cpu_seconds"]
            * item["process_cpu_seconds"]
            / cached_mixed["process_cpu_seconds"]
        )
        f0_wall = cached_f0["wall_seconds"] * item["wall_seconds"] / cached_mixed["wall_seconds"]
        arm_cpu = SEEDS * (
            F0_ONLY_UPDATES * f0_cpu
            + (UPDATES_PER_ARM - F0_ONLY_UPDATES) * item["process_cpu_seconds"]
        )
        arm_wall = SEEDS * (
            F0_ONLY_UPDATES * f0_wall + (UPDATES_PER_ARM - F0_ONLY_UPDATES) * item["wall_seconds"]
        )
        training_cpu += arm_cpu
        training_wall += arm_wall
        mode_estimates.append(
            {
                "mode": mode,
                "cpu_seconds_all_seeds": arm_cpu,
                "wall_seconds_all_seeds": arm_wall,
            }
        )

    unique_train_turns = 64 * 5 * 5 + 64 * 5 * 8
    uncached_f0 = uncached["f0_only_batch"]
    uncached_mixed = uncached["mixed_batch"]
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
        * ARM_COUNT
        * 1.5
    )
    evaluation_wall = (
        cached_mixed["wall_seconds"]
        / cached_mixed["turns"]
        * evaluation_turns_per_arm
        * ARM_COUNT
        * 1.5
    )
    cache_load_cpu = 15 * cache["fresh_process_load_and_verify_cpu_seconds"]
    cache_load_wall = 15 * cache["fresh_process_load_and_verify_wall_seconds"]
    optimizer_overhead_cpu = training_cpu * 0.10
    optimizer_overhead_wall = training_wall * 0.10
    checkpoint_slots = ARM_COUNT * CHECKPOINTS_PER_ARM
    checkpoint_bytes_with_moments = (
        report["latest_zero_checkpoint_preflight"]["zero_checkpoint_bytes"] + 1024**2
    )
    artifact_base = (
        checkpoint_slots * checkpoint_bytes_with_moments + cache["cache_bytes"] + 512 * 1024**2
    )
    cpu_base = (
        training_cpu
        + optimizer_overhead_cpu
        + cache_cpu_upper
        + evaluation_cpu
        + cache_load_cpu
        + checkpoint_slots
    )
    wall_base = (
        training_wall
        + optimizer_overhead_wall
        + cache_wall_upper
        + evaluation_wall
        + cache_load_wall
        + checkpoint_slots
    )
    projection = {
        "format": "taiji-n177-5a-resource-projection-v1",
        "status": "within_caps",
        "modes": mode_estimates,
        "formal_arms": ARM_COUNT,
        "updates_per_arm": UPDATES_PER_ARM,
        "optimizer_updates_total": ARM_COUNT * UPDATES_PER_ARM,
        "cache_unique_train_turns": unique_train_turns,
        "cache_build_cpu_upper_seconds": cache_cpu_upper,
        "cache_build_wall_upper_seconds": cache_wall_upper,
        "cache_build_measurement": "not captured; bounded with the larger measured uncached full-forward cost per unique train turn",
        "evaluation_turns_per_arm": evaluation_turns_per_arm,
        "evaluation_cpu_upper_seconds": evaluation_cpu,
        "evaluation_wall_upper_seconds": evaluation_wall,
        "optimizer_step_overhead_fraction": 0.10,
        "checkpoint_slots": checkpoint_slots,
        "checkpoint_bytes_with_optimizer_allowance": checkpoint_bytes_with_moments,
        "cache_bytes": cache["cache_bytes"],
        "evaluation_and_logs_allowance_bytes": 512 * 1024**2,
        "total_cpu_before_reserve": cpu_base,
        "total_wall_before_reserve": wall_base,
        "reserve_fraction": RESERVE - 1.0,
        "projected_cpu_with_reserve": cpu_base * RESERVE,
        "projected_wall_with_reserve": wall_base * RESERVE,
        "projected_artifact_bytes_with_reserve": artifact_base * RESERVE,
        "hard_caps": {
            "process_cpu_seconds": CPU_CAP,
            "wall_seconds": WALL_CAP,
            "artifact_bytes": ARTIFACT_CAP,
        },
        "authorization_basis": "user-provided AGENTS.md grants reversible project work; the parent checkpoint remains immutable and every arm writes isolated checkpoints",
        "stop_rule": "stop at the first hard cap, preserve the latest checkpoint, and do not retry or raise the cap automatically",
    }
    if (
        projection["projected_cpu_with_reserve"] > CPU_CAP
        or projection["projected_wall_with_reserve"] > WALL_CAP
        or projection["projected_artifact_bytes_with_reserve"] > ARTIFACT_CAP
    ):
        projection["status"] = "exceeds_caps"
    report["resource_projection"] = projection
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return projection


if __name__ == "__main__":
    print(json.dumps(project(), ensure_ascii=False, sort_keys=True))
