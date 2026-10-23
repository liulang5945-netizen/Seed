"""Resume any continuation arm under the D1-approved v1.5 cap."""

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
    sha,
)
from scripts.training.run_taiji_r2_fast_ledger_resume_v2 import (
    _selected_projection,
    sample_resources,
)
from scripts.training.run_taiji_r2_fast_ledger_resume_v3 import (
    SOURCES as V3_SOURCES,
)
from scripts.training.taiji_r2_continuation_budget_amendment_v1_5 import (
    AGGREGATE_CPU_LIMIT,
    PER_ARM_CPU_LIMIT,
    activate_budget_amendment,
)
from scripts.training.taiji_r2_continuation_budget_amendment_v1_5 import (
    SOURCES as AMENDMENT_SOURCES,
)
from scripts.training.taiji_r2_resume_ledger_anchor import (
    BASELINE,
    current_fast_charges,
    current_fast_reservations,
    verified_snapshot,
)

SOURCES = (
    "scripts/training/run_taiji_r2_fast_ledger_resume_v4.py",
    *V3_SOURCES,
    *AMENDMENT_SOURCES,
    "tests/taiji_native/test_r2_fast_ledger_resume_v4_contract.py",
)
APPROVED_BUDGET = {"per_arm": PER_ARM_CPU_LIMIT, "aggregate": AGGREGATE_CPU_LIMIT}


def memory_policy(arm):
    if arm == "warm_start":
        return {"memory_floor_gib": 7.0, "scope": "warm_start_only"}
    return {"memory_floor_gib": 8.0, "scope": "default"}


def build_manifest_v4(arm, *, source_hashes, budget_amendment, **manifest_args):
    if arm not in r.ARMS or set(source_hashes) != set(SOURCES):
        raise r.Refuse("v1.5 manifest arm or source set differs")
    if not isinstance(budget_amendment, dict):
        raise r.Refuse("v1.5 manifest budget amendment is missing")
    amendment_sha = budget_amendment.get("amendment_sha256")
    if (
        budget_amendment.get("schema") != "taiji-r2-continuation-budget-amendment-v1.5"
        or not isinstance(amendment_sha, str)
        or len(amendment_sha) != 64
        or any(character not in "0123456789abcdef" for character in amendment_sha)
        or manifest_args.get("approved_budget") != APPROVED_BUDGET
    ):
        raise r.Refuse("v1.5 manifest amendment digest or approved budget differs")
    previous_runtime_budget = (r.PER_ARM_CPU_LIMIT, r.AGGREGATE_CPU_LIMIT)
    r.PER_ARM_CPU_LIMIT, r.AGGREGATE_CPU_LIMIT = (
        APPROVED_BUDGET["per_arm"],
        APPROVED_BUDGET["aggregate"],
    )
    try:
        manifest = r2.build_manifest(
            arm,
            source_hashes={path: source_hashes[path] for path in r2.SOURCES},
            **manifest_args,
        )
    finally:
        r.PER_ARM_CPU_LIMIT, r.AGGREGATE_CPU_LIMIT = previous_runtime_budget
    manifest["schema"] = "taiji-r2-fast-ledger-resume-v4"
    manifest["sources"] = {
        str((ROOT / path).resolve()): digest for path, digest in source_hashes.items()
    }
    manifest["approved_budget"] = dict(APPROVED_BUDGET)
    manifest["budget_amendment_sha256"] = amendment_sha
    manifest["resource_policy"] = memory_policy(arm)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=r.ARMS, required=True)
    args = parser.parse_args()
    amendment = activate_budget_amendment()
    r.PER_ARM_CPU_LIMIT = PER_ARM_CPU_LIMIT
    r.AGGREGATE_CPU_LIMIT = AGGREGATE_CPU_LIMIT
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
    forecast = _selected_projection(
        args.arm,
        prepared["parents"],
        current_stage=commit["stage"],
        current_episodes=commit["episodes"],
    )
    if not forecast["selected_arm_feasible"]:
        raise r.Refuse(f"{args.arm} projection exceeds v1.5 arm or aggregate cap: {forecast}")
    manifest = build_manifest_v4(
        args.arm,
        source_hashes={path: sha(ROOT / path) for path in SOURCES},
        budget_amendment=amendment,
        source_revision_sha256=SOURCE_REVISION_SHA256,
        extension_attempt_sha256=sha(r.paths(args.arm)["extension_attempt"]),
        latest_commit=commit["receipt"],
        forecast=forecast,
        snapshot={
            "baseline_sequence": BASELINE["sequence"],
            "baseline_event_sha256": BASELINE["event_sha256"],
            **snapshot,
        },
        approved_budget=dict(APPROVED_BUDGET),
    )
    manifest_path = (
        r.REPORTS_DIR / f"taiji_r2_fast_ledger_resume_v4_{args.arm}_{uuid.uuid4().hex}.json"
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
