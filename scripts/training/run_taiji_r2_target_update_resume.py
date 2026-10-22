"""Manifest-bound resume with receipt ancestry and target-update CPU forecasting."""

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
from scripts.training.taiji_r2_target_update_projection import throughput_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=r.ARMS, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    r.torch.set_num_threads(1)
    prepared = validate_prepared(allow_terminal_artifacts=True)
    progress = r.read_jsonl(r.paths(args.arm)["extension_progress"])
    commit = next(e for e in reversed(progress) if e.get("event") == "durable_training_commit")
    forecast = throughput_projection(
        args.arm,
        prepared["parents"],
        current_stage=commit["stage"],
        current_episodes=commit["episodes"],
    )
    if args.check_only:
        print(forecast)
        return
    if not forecast["feasible"]:
        raise r.Refuse(f"target-update forecast exceeds approved headroom: {forecast}")
    sources = {
        str((ROOT / relative).resolve()): r.sha256_file(ROOT / relative)
        for relative in (
            "scripts/training/run_taiji_r2_target_update_resume.py",
            "scripts/training/taiji_r2_target_update_projection.py",
            "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py",
            "tests/taiji_native/test_r2_target_update_projection_contract.py",
        )
    }
    r._write_json_once(
        r.REPORTS_DIR / f"taiji_r2_target_update_resume_{args.arm}_{uuid.uuid4().hex}.json",
        {
            "schema": "taiji-r2-target-update-resume-v1",
            "arm": args.arm,
            "sources": sources,
            "source_revision_sha256": r.sha256_file(r._source_revision_path()),
            "extension_attempt_sha256": r.sha256_file(r.paths(args.arm)["extension_attempt"]),
            "latest_commit": commit["receipt"],
            "forecast": forecast,
            "utc": r.utc_now(),
            "approved_budget": {"per_arm": r.PER_ARM_CPU_LIMIT, "aggregate": r.AGGREGATE_CPU_LIMIT},
        },
    )
    r._validate_prepared = validate_prepared
    r._throughput_projection = throughput_projection
    print(r.execute_arm(args.arm, resume=True)["status"])


if __name__ == "__main__":
    main()
