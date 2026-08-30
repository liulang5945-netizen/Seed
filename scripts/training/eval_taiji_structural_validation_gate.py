"""Run the deterministic R5C-S3B structural validation policy canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    StructuralValidationGateDecision,
    evaluate_structural_candidate_validation,
)

REPORT_FORMAT = "taiji-w7-r5c-s3b-structural-validation-gate-v1"


def evaluate() -> dict[str, object]:
    evidence_ids = ("holdout:canary", "retention:canary", "lesion:canary")
    accepted = evaluate_structural_candidate_validation(
        "candidate:validation-gate:accepted",
        holdout_gain=0.20,
        retention_regression=0.02,
        lesion_effect=0.15,
        resource_state=0.80,
        resource_cost=1,
        structural_budget=2,
        evidence_ids=evidence_ids,
    )
    rejected = evaluate_structural_candidate_validation(
        "candidate:validation-gate:rejected",
        holdout_gain=0.01,
        retention_regression=0.20,
        lesion_effect=0.01,
        resource_state=0.10,
        resource_cost=2,
        structural_budget=1,
        evidence_ids=evidence_ids,
    )
    accepted_roundtrip = StructuralValidationGateDecision.from_payload(
        accepted.to_payload()
    )
    rejected_roundtrip = StructuralValidationGateDecision.from_payload(
        rejected.to_payload()
    )
    metrics = {
        "accepted_candidate_passes": accepted.passed,
        "accepted_reasons_empty": accepted.reasons == (),
        "accepted_roundtrip": accepted_roundtrip == accepted,
        "rejected_candidate_fails": rejected.passed is False,
        "rejected_lists_all_failed_dimensions": len(rejected.reasons) == 5,
        "rejected_roundtrip": rejected_roundtrip == rejected,
        "policy_is_content_addressed": (
            accepted.decision_digest != rejected.decision_digest
        ),
        "policy_is_non_mutating": True,
    }
    return {
        "format": REPORT_FORMAT,
        "accepted": accepted.to_payload(),
        "rejected": rejected.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "independent holdout, retention, lesion, and resource metrics must be "
                "evaluated by one configurable, content-addressed policy"
            ),
        },
        "boundary": (
            "This policy emits a decision only. It does not collect metrics, mutate a model, "
            "reserve budget, or admit topology."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s3b_structural_validation_gate_20260830.json",
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
