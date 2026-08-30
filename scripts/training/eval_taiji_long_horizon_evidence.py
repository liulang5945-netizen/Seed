"""Run the deterministic R5C-S0 long-horizon evidence canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.structural_evidence import (  # noqa: E402
    StructuralEvidenceLedger,
)
from taiji.structural_growth import StructuralRuntimeObservation  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s0-long-horizon-evidence-v1"


def _observation(tick: int, *, usage: float = 0.4) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id="network:canary",
        region_id="region:canary",
        tick=tick,
        usage=usage,
        resource_pressure=0.2,
        prediction_error=0.7,
        learning_gain=0.1,
        holdout_transfer=0.8,
        evidence_id=f"canary:evidence:{tick}",
    )


def evaluate() -> dict[str, object]:
    ledger = StructuralEvidenceLedger(window_capacity=2)
    first = _observation(1)
    second = _observation(2)
    accepted = ledger.append(first)
    duplicate = ledger.append(first)
    duplicate_idempotent = (
        duplicate.status == "duplicate" and ledger.observed_count == 1
    )
    conflict_rejected = False
    try:
        ledger.append(_observation(1, usage=0.9))
    except ValueError:
        conflict_rejected = True
    sealed = ledger.append(second)
    summary = ledger.sealed_summaries[0]
    restored = StructuralEvidenceLedger.from_payload(ledger.to_payload())

    bounded = StructuralEvidenceLedger(window_capacity=4, max_evidence_index=1)
    bounded.append(_observation(1))
    before_capacity_failure = bounded.to_payload()
    capacity_failure_is_atomic = False
    try:
        bounded.append(_observation(2))
    except OverflowError:
        capacity_failure_is_atomic = bounded.to_payload() == before_capacity_failure

    metrics = {
        "accepted_first": accepted.status == "accepted",
        "duplicate_idempotent": duplicate_idempotent,
        "conflicting_reuse_rejected": conflict_rejected,
        "window_sealed_at_capacity": sealed.status == "window_sealed",
        "summary_observation_count": summary.observation_count,
        "summary_digest_bound": summary.window_digest == sealed.sealed_window_digest,
        "checkpoint_roundtrip": restored.digest == ledger.digest,
        "capacity_failure_atomic": capacity_failure_is_atomic,
    }
    passed = all(
        [
            metrics["accepted_first"],
            metrics["duplicate_idempotent"],
            metrics["conflicting_reuse_rejected"],
            metrics["window_sealed_at_capacity"],
            metrics["summary_observation_count"] == 2,
            metrics["summary_digest_bound"],
            metrics["checkpoint_roundtrip"],
            metrics["capacity_failure_atomic"],
        ]
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": metrics,
        "gate": {
            "passed": passed,
            "criterion": (
                "long-horizon structural evidence must be bounded, content-addressed, "
                "deduplicated, monotonic, checkpointable, and atomic on capacity failure; "
                "the window must not itself mutate topology"
            ),
        },
        "boundary": (
            "This canary proves evidence capture and window integrity only. It does not "
            "trigger growth, admit a structural candidate, or claim open-domain autonomy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s0_long_horizon_evidence_20260830.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
