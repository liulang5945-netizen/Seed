from __future__ import annotations

import pytest

from taiji import (
    StructuralValidationGateDecision,
    evaluate_structural_candidate_validation,
)


def test_structural_validation_gate_accepts_causal_candidate() -> None:
    decision = evaluate_structural_candidate_validation(
        "candidate:good",
        holdout_gain=0.20,
        retention_regression=0.02,
        lesion_effect=0.15,
        resource_state=0.80,
        resource_cost=1,
        structural_budget=2,
        evidence_ids=("holdout:1", "retention:1", "lesion:1"),
    )

    assert decision.passed is True
    assert decision.reasons == ()
    assert StructuralValidationGateDecision.from_payload(decision.to_payload()) == decision


def test_structural_validation_gate_rejects_any_failed_dimension() -> None:
    decision = evaluate_structural_candidate_validation(
        "candidate:bad",
        holdout_gain=0.01,
        retention_regression=0.20,
        lesion_effect=0.01,
        resource_state=0.10,
        resource_cost=2,
        structural_budget=1,
        evidence_ids=("holdout:2", "retention:2", "lesion:2"),
    )

    assert decision.passed is False
    assert decision.reasons == (
        "holdout_gain_below_threshold",
        "retention_regression_above_threshold",
        "lesion_effect_below_threshold",
        "resource_state_below_threshold",
        "structural_budget_insufficient",
    )


def test_structural_validation_gate_rejects_tampered_digest_and_duplicate_evidence() -> None:
    decision = evaluate_structural_candidate_validation(
        "candidate:digest",
        holdout_gain=0.20,
        retention_regression=0.02,
        lesion_effect=0.15,
        resource_state=0.80,
        resource_cost=1,
        structural_budget=1,
        evidence_ids=("holdout:3",),
    )
    payload = decision.to_payload()
    payload["holdout_gain"] = 0.21
    with pytest.raises(ValueError, match="digest mismatch"):
        StructuralValidationGateDecision.from_payload(payload)
    with pytest.raises(ValueError, match="must be unique"):
        evaluate_structural_candidate_validation(
            "candidate:duplicate",
            holdout_gain=0.20,
            retention_regression=0.02,
            lesion_effect=0.15,
            resource_state=0.80,
            resource_cost=1,
            structural_budget=1,
            evidence_ids=("same", "same"),
        )
