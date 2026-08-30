"""Run the deterministic R5C-S3D atomic structural admission canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s3d-structural-admission-v1"


def evaluate() -> dict[str, object]:
    projection = _projection()
    model, region = _build_model("structural-admission")
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
        evidence_ids=("retention:admission", "lesion:admission"),
    )
    before_units = region.unit_ids
    before_budget = model.cognitive_snapshot().development.structural_budget
    admission = model.admit_structural_candidate(validation, decision)
    repeated = model.admit_structural_candidate(validation, decision)
    restored = model.__class__.from_native_checkpoint(model.native_checkpoint())

    metrics = {
        "policy_passed_before_admission": decision.passed,
        "admission_committed": admission.status == "admitted",
        "admission_is_idempotent": repeated == admission,
        "topology_grew_once": (
            before_units == ("u0", "u1") and region.unit_ids == ("u0", "u1", "u2")
        ),
        "budget_decreased_once": (
            before_budget == 1
            and model.cognitive_snapshot().development.structural_budget == 0
        ),
        "topology_digest_changed": (
            admission.topology_before_digest != admission.topology_after_digest
        ),
        "budget_lineage_recorded": (
            admission.structural_budget_after
            == admission.structural_budget_before - 1
        ),
        "admission_roundtrip": restored.structural_admission_results == (admission,),
        "restored_topology_is_admitted": restored.neuron_regions[0].unit_ids == region.unit_ids,
        "restored_budget_is_admitted": (
            restored.cognitive_snapshot().development.structural_budget == 0
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "projection_digest": projection.projection_digest,
        "validation": validation.to_payload(),
        "decision": decision.to_payload(),
        "admission": admission.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "only a bound and policy-approved candidate may commit one topology change; "
                "the commit and budget lineage must survive checkpoint restore"
            ),
        },
        "boundary": (
            "This canary admits one bounded neuron candidate. It does not enable unbounded growth, "
            "full retraining, or automatic multi-step structural expansion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s3d_structural_admission_20260830.json",
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
