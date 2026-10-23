"""Versioned single-arm resume with a warm-start-only 7 GiB memory floor."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as r
from scripts.training import run_taiji_r2_fast_ledger_resume as r2
from scripts.training.run_taiji_r2_checkpoint_resume_adapter import validate_prepared
from scripts.training.run_taiji_r2_fast_ledger_resume import (
    SOURCE_REVISION_SHA256,
    selected_arm_projection_feasible,
    sha,
)
from scripts.training.run_taiji_r2_fast_ledger_resume import (
    SOURCES as V1_SOURCES,
)
from scripts.training.taiji_r2_resume_ledger_anchor import (
    BASELINE,
    current_fast_charges,
    current_fast_reservations,
    verified_snapshot,
)
from scripts.training.taiji_r2_target_update_projection import throughput_projection

SOURCES = (
    "scripts/training/run_taiji_r2_fast_ledger_resume_v2.py",
    *V1_SOURCES,
    "tests/taiji_native/test_r2_fast_ledger_resume_v2_contract.py",
)
WARM_MEMORY_FLOOR_GIB = 7.0
CPU_LIMIT_PERCENT = 85.0


def sample_resources(psutil_module, *, arm):
    floor = WARM_MEMORY_FLOOR_GIB if arm == "warm_start" else 8.0
    samples = [psutil_module.cpu_percent(interval=1.0), psutil_module.cpu_percent(interval=1.0)]
    available = psutil_module.virtual_memory().available / 1024**3
    if available < floor:
        raise r.Refuse(f"available memory below selected-arm floor {floor:g} GiB: {available:.2f}")
    if any(sample > CPU_LIMIT_PERCENT for sample in samples):
        raise r.Refuse(f"system CPU start samples exceed {CPU_LIMIT_PERCENT:g}%: {samples}")
    return {
        "cpu_percent_samples": samples,
        "available_memory_gib": round(available, 3),
        "memory_floor_gib": floor,
        "memory_floor_scope": "warm_start_only" if arm == "warm_start" else "default",
    }


def build_manifest_v2(
    arm,
    *,
    source_hashes,
    memory_policy,
    **manifest_args,
):
    if arm != "warm_start" or memory_policy != {
        "memory_floor_gib": WARM_MEMORY_FLOOR_GIB,
        "scope": "warm_start_only",
    }:
        raise r.Refuse("v2 memory floor is approved only for the selected warm_start arm")
    base = r2.build_manifest(
        arm,
        source_hashes={
            path: digest for path, digest in source_hashes.items() if path in V1_SOURCES
        },
        **manifest_args,
    )
    if set(source_hashes) != set(SOURCES) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for value in source_hashes.values()
    ):
        raise r.Refuse("v2 resume source set or digest is malformed")
    base["schema"] = "taiji-r2-fast-ledger-resume-v2"
    base["sources"] = {
        str((ROOT / relative).resolve()): digest for relative, digest in source_hashes.items()
    }
    base["resource_policy"] = dict(memory_policy)
    return base


def _selected_projection(current_arm, contexts, *, current_stage, current_episodes):
    projection = throughput_projection(
        current_arm,
        contexts,
        current_stage=current_stage,
        current_episodes=current_episodes,
    )
    projection["global_feasible"] = projection["feasible"]
    projection["selected_arm_feasible"] = selected_arm_projection_feasible(current_arm, projection)
    projection["projection_scope"] = "selected_arm"
    projection["deferred_arms"] = [name for name in r.ARMS if name != current_arm]
    projection["feasible"] = projection["selected_arm_feasible"]
    return projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=r.ARMS, default="warm_start")
    args = parser.parse_args()
    r.torch.set_num_threads(1)
    snapshot = verified_snapshot()
    prepared = validate_prepared(
        allow_terminal_artifacts=True,
        ledger_snapshot=snapshot["effective_charges"],
    )
    progress = r.read_jsonl(r.paths(args.arm)["extension_progress"])
    commit = next(
        event for event in reversed(progress) if event.get("event") == "durable_training_commit"
    )
    projection = _selected_projection(
        args.arm,
        prepared["parents"],
        current_stage=commit["stage"],
        current_episodes=commit["episodes"],
    )
    if not selected_arm_projection_feasible(args.arm, projection):
        raise r.Refuse(f"selected arm exceeds current CPU headroom: {projection}")
    policy = {"memory_floor_gib": WARM_MEMORY_FLOOR_GIB, "scope": "warm_start_only"}
    manifest = build_manifest_v2(
        args.arm,
        source_hashes={relative: sha(ROOT / relative) for relative in SOURCES},
        memory_policy=policy,
        source_revision_sha256=SOURCE_REVISION_SHA256,
        extension_attempt_sha256=sha(r.paths(args.arm)["extension_attempt"]),
        latest_commit=commit["receipt"],
        forecast=projection,
        snapshot={
            "baseline_sequence": BASELINE["sequence"],
            "baseline_event_sha256": BASELINE["event_sha256"],
            **snapshot,
        },
        approved_budget={"per_arm": r.PER_ARM_CPU_LIMIT, "aggregate": r.AGGREGATE_CPU_LIMIT},
    )
    manifest_path = (
        r.REPORTS_DIR / f"taiji_r2_fast_ledger_resume_v2_{args.arm}_{uuid.uuid4().hex}.json"
    )
    r._write_json_once(manifest_path, manifest)

    r._validate_prepared = lambda **_kwargs: prepared
    r.latest_charges = current_fast_charges
    r.latest_reservations = current_fast_reservations
    r._throughput_projection = _selected_projection
    r.resource_snapshot = lambda: sample_resources(r.psutil, arm=args.arm)
    print(r.execute_arm(args.arm, resume=True)["status"])


if __name__ == "__main__":
    main()
