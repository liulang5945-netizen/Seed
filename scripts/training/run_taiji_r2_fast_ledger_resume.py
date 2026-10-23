"""Resume one arm using a pinned full-audit anchor plus a verified ledger tail."""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as r
from scripts.training.run_taiji_r2_checkpoint_resume_adapter import validate_prepared
from scripts.training.taiji_r2_resume_ledger_anchor import (
    BASELINE,
    current_fast_charges,
    current_fast_reservations,
    verified_snapshot,
)
from scripts.training.taiji_r2_target_update_projection import throughput_projection

SOURCE_REVISION_SHA256 = "2f6938afb910a7bca2c9d5bd5c3d4f79e5a67aa2cd19da08c0a9577589640bf7"
SOURCES = (
    "scripts/training/run_taiji_r2_fast_ledger_resume.py",
    "scripts/training/taiji_r2_resume_ledger_anchor.py",
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py",
    "scripts/training/taiji_r2_target_update_projection.py",
    "tests/taiji_native/test_r2_fast_ledger_resume_contract.py",
    "tests/taiji_native/test_r2_resume_ledger_anchor_contract.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha(path: Path) -> str:
    return r.sha256_file(path)


def build_manifest(
    arm,
    *,
    source_revision_sha256,
    extension_attempt_sha256,
    source_hashes,
    latest_commit,
    forecast,
    snapshot,
    approved_budget,
):
    if source_revision_sha256 != SOURCE_REVISION_SHA256:
        raise r.Refuse("fast ledger resume only accepts pinned source revision v1.3")
    if arm not in r.ARMS or set(source_hashes) != set(SOURCES):
        raise r.Refuse("fast ledger resume identity or source set differs")
    if (
        not isinstance(extension_attempt_sha256, str)
        or not _SHA256.fullmatch(extension_attempt_sha256)
        or any(not _SHA256.fullmatch(value) for value in source_hashes.values())
    ):
        raise r.Refuse("fast ledger resume source identity is malformed")
    if (
        not isinstance(forecast, dict)
        or forecast.get("arm") != arm
        or forecast.get("feasible") is not True
    ):
        raise r.Refuse(f"target-update forecast exceeds approved headroom: {forecast}")
    if (
        snapshot.get("baseline_sequence") != BASELINE["sequence"]
        or snapshot.get("baseline_event_sha256") != BASELINE["event_sha256"]
        or not isinstance(snapshot.get("sequence"), int)
        or isinstance(snapshot.get("sequence"), bool)
        or not isinstance(snapshot.get("event_sha256"), str)
        or not _SHA256.fullmatch(snapshot["event_sha256"])
        or not isinstance(snapshot.get("charges"), dict)
        or not isinstance(snapshot.get("reservations"), dict)
        or not isinstance(latest_commit, dict)
        or approved_budget != {"per_arm": r.PER_ARM_CPU_LIMIT, "aggregate": r.AGGREGATE_CPU_LIMIT}
    ):
        raise r.Refuse("fast ledger resume anchor, checkpoint or budget differs")
    return {
        "schema": "taiji-r2-fast-ledger-resume-v1",
        "arm": arm,
        "sources": {
            str((ROOT / relative).resolve()): digest for relative, digest in source_hashes.items()
        },
        "source_revision_sha256": SOURCE_REVISION_SHA256,
        "extension_attempt_sha256": extension_attempt_sha256,
        "latest_commit": latest_commit,
        "forecast": forecast,
        "approved_budget": approved_budget,
        "ledger_anchor": {
            key: snapshot[key]
            for key in (
                "baseline_sequence",
                "baseline_event_sha256",
                "sequence",
                "event_sha256",
                "charges",
                "reservations",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=r.ARMS, required=True)
    args = parser.parse_args()
    r.torch.set_num_threads(1)

    # Replay only events after the checkpointed full-history audit. The training
    # ledger writer validates each append and the final endpoint gate replays all
    # historical events again before evaluation is admitted.
    snapshot = verified_snapshot()
    prepared = validate_prepared(
        allow_terminal_artifacts=True,
        ledger_snapshot=snapshot["effective_charges"],
    )
    progress = r.read_jsonl(r.paths(args.arm)["extension_progress"])
    commit = next(e for e in reversed(progress) if e.get("event") == "durable_training_commit")
    forecast = throughput_projection(
        args.arm,
        prepared["parents"],
        current_stage=commit["stage"],
        current_episodes=commit["episodes"],
    )
    progress = r.read_jsonl(r.paths(args.arm)["extension_progress"])
    commit = next(e for e in reversed(progress) if e.get("event") == "durable_training_commit")
    manifest = build_manifest(
        args.arm,
        source_revision_sha256=SOURCE_REVISION_SHA256,
        extension_attempt_sha256=sha(r.paths(args.arm)["extension_attempt"]),
        source_hashes={relative: sha(ROOT / relative) for relative in SOURCES},
        latest_commit=commit["receipt"],
        forecast=forecast,
        snapshot={
            "baseline_sequence": BASELINE["sequence"],
            "baseline_event_sha256": BASELINE["event_sha256"],
            **snapshot,
        },
        approved_budget={"per_arm": r.PER_ARM_CPU_LIMIT, "aggregate": r.AGGREGATE_CPU_LIMIT},
    )
    path = r.REPORTS_DIR / f"taiji_r2_fast_ledger_resume_{args.arm}_{uuid.uuid4().hex}.json"
    r._write_json_once(path, manifest)

    # Keep all parent/source/restore checks from validate_prepared, while the
    # runner's start and per-work-unit guards read the checksummed current ledger
    # summary. Each resume invocation independently verifies the chain tail.
    r._validate_prepared = lambda **_kwargs: prepared
    r.latest_charges = current_fast_charges
    r.latest_reservations = current_fast_reservations
    r._throughput_projection = throughput_projection
    print(r.execute_arm(args.arm, resume=True)["status"])


if __name__ == "__main__":
    main()
