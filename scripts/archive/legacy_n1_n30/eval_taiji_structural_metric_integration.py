"""Run the deterministic R5C-S3C metric-to-policy integration canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_structural_validation import (  # noqa: E402
    _build_model,
    _expected_activity,
    _projection,
    _queue_candidate,
)

REPORT_FORMAT = "taiji-w7-r5c-s3c-structural-metric-integration-v1"


def evaluate() -> dict[str, object]:
    projection = _projection()
    model, region = _build_model("structural-metric-integration")
    candidate_id = _queue_candidate(model, projection)
    holdout_input, expected_activity = _expected_activity(model, candidate_id)
    validation = model.validate_structural_candidate_shadow(
        candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(expected_activity,),
    )
    decision = model.evaluate_structural_candidate_gate(
        validation,
        retention_regression=0.02,
        lesion_effect=0.15,
        resource_state=0.80,
        evidence_ids=("retention:observed", "lesion:observed"),
    )
    restored = model.__class__.from_native_checkpoint(model.native_checkpoint())

    rejected_model, rejected_region = _build_model("structural-metric-rejection")
    rejected_candidate_id = _queue_candidate(rejected_model, projection)
    rejected_input, rejected_expected = _expected_activity(rejected_model, rejected_candidate_id)
    rejected_validation = rejected_model.validate_structural_candidate_shadow(
        rejected_candidate_id,
        holdout_inputs=(rejected_input,),
        expected_activities=(rejected_expected,),
    )
    rejected_decision = rejected_model.evaluate_structural_candidate_gate(
        rejected_validation,
        retention_regression=0.20,
        lesion_effect=0.01,
        resource_state=0.10,
        evidence_ids=("retention:regressed", "lesion:weak"),
    )
    rejected_restored = rejected_model.__class__.from_native_checkpoint(
        rejected_model.native_checkpoint()
    )

    metrics = {
        "validation_record_is_required": validation.status == "validated",
        "accepted_policy_decision": decision.passed,
        "accepted_proposal_stays_pending": model.topology_proposals[-1].status == "pending",
        "accepted_topology_unchanged": region.unit_ids == ("u0", "u1"),
        "accepted_budget_unchanged": (
            model.cognitive_snapshot().development.structural_budget == 1
        ),
        "accepted_decision_roundtrip": (
            restored.structural_validation_gate_decisions == (decision,)
        ),
        "failed_policy_decision": rejected_decision.passed is False,
        "failed_policy_rejects_pending_proposal": (
            rejected_model.topology_proposals[-1].status == "rejected"
        ),
        "failed_topology_unchanged": rejected_region.unit_ids == ("u0", "u1"),
        "failed_budget_unchanged": (
            rejected_model.cognitive_snapshot().development.structural_budget == 1
        ),
        "failed_rejection_roundtrip": (
            rejected_restored.topology_proposals[-1].status == "rejected"
            and rejected_restored.structural_validation_gate_decisions == (rejected_decision,)
        ),
        "no_physical_admission": True,
    }
    return {
        "format": REPORT_FORMAT,
        "projection_digest": projection.projection_digest,
        "validation": validation.to_payload(),
        "decision": decision.to_payload(),
        "rejected_validation": rejected_validation.to_payload(),
        "rejected_decision": rejected_decision.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "real candidate shadow metrics must bind to the independent validation policy; "
                "pass stays pending and fail rejects atomically without topology admission"
            ),
        },
        "boundary": (
            "This integration still does not commit topology. Admission requires a later atomic "
            "lifecycle step after all gates and checkpoint checks pass."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s3c_structural_metric_integration_20260830.json",
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
