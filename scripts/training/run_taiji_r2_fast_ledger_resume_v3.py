"""Resume warm_start with an additive v1.4 cap and selected-arm resource floor."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as r
from scripts.training.run_taiji_r2_checkpoint_resume_adapter import validate_prepared
from scripts.training.run_taiji_r2_fast_ledger_resume import SOURCE_REVISION_SHA256, sha
from scripts.training.run_taiji_r2_fast_ledger_resume_v2 import (
    SOURCES as V2_SOURCES,
)
from scripts.training.run_taiji_r2_fast_ledger_resume_v2 import (
    WARM_MEMORY_FLOOR_GIB,
    _selected_projection,
    build_manifest_v2,
    sample_resources,
)
from scripts.training.taiji_r2_continuation_budget_amendment_v1_4 import (
    AGGREGATE_CPU_LIMIT,
    PER_ARM_CPU_LIMIT,
    activate_budget_amendment,
)
from scripts.training.taiji_r2_continuation_budget_amendment_v1_4 import (
    SOURCES as AMENDMENT_SOURCES,
)
from scripts.training.taiji_r2_resume_ledger_anchor import (
    BASELINE,
    current_fast_charges,
    current_fast_reservations,
    verified_snapshot,
)

SOURCES = (
    "scripts/training/run_taiji_r2_fast_ledger_resume_v3.py",
    *V2_SOURCES,
    *AMENDMENT_SOURCES,
    "tests/taiji_native/test_r2_fast_ledger_resume_v3_contract.py",
)


def build_manifest_v3(arm, *, source_hashes, budget_amendment, **manifest_args):
    if arm != "warm_start" or set(source_hashes) != set(SOURCES):
        raise r.Refuse("v1.4 cap and 7 GiB floor are scoped to warm_start only")
    expected_budget = {"per_arm": PER_ARM_CPU_LIMIT, "aggregate": AGGREGATE_CPU_LIMIT}
    amendment_sha = (
        budget_amendment.get("amendment_sha256") if isinstance(budget_amendment, dict) else None
    )
    if (
        manifest_args.get("approved_budget") != expected_budget
        or not isinstance(amendment_sha, str)
        or len(amendment_sha) != 64
        or any(character not in "0123456789abcdef" for character in amendment_sha)
    ):
        raise r.Refuse("v1.4 manifest amendment digest or approved budget differs")
    previous_runtime_budget = (r.PER_ARM_CPU_LIMIT, r.AGGREGATE_CPU_LIMIT)
    r.PER_ARM_CPU_LIMIT, r.AGGREGATE_CPU_LIMIT = (
        expected_budget["per_arm"],
        expected_budget["aggregate"],
    )
    try:
        v2_manifest_args = dict(manifest_args)
        v2_manifest_args["approved_budget"] = expected_budget
        manifest = build_manifest_v2(
            arm,
            source_hashes={path: source_hashes[path] for path in V2_SOURCES},
            memory_policy={"memory_floor_gib": WARM_MEMORY_FLOOR_GIB, "scope": "warm_start_only"},
            **v2_manifest_args,
        )
    finally:
        r.PER_ARM_CPU_LIMIT, r.AGGREGATE_CPU_LIMIT = previous_runtime_budget
    manifest["schema"] = "taiji-r2-fast-ledger-resume-v3"
    manifest["sources"] = {
        str((ROOT / path).resolve()): digest for path, digest in source_hashes.items()
    }
    manifest["approved_budget"] = expected_budget
    manifest["budget_amendment_sha256"] = amendment_sha
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=r.ARMS, default="warm_start")
    args = parser.parse_args()
    if args.arm != "warm_start":
        raise r.Refuse("v1.4 budget and 7 GiB floor are scoped to warm_start only")
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
        raise r.Refuse(f"warm_start projection exceeds amended arm or aggregate cap: {forecast}")
    manifest = build_manifest_v3(
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
        approved_budget={"per_arm": PER_ARM_CPU_LIMIT, "aggregate": AGGREGATE_CPU_LIMIT},
    )
    manifest_path = (
        r.REPORTS_DIR / f"taiji_r2_fast_ledger_resume_v3_warm_start_{uuid.uuid4().hex}.json"
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
