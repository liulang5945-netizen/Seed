from __future__ import annotations

import copy

import pytest

from taiji.internalization import content_digest
from taiji.joint_learning import JointCreditComparisonSpecV1, JointCreditRouteV1


def make_route() -> JointCreditRouteV1:
    return JointCreditRouteV1(
        joint_bundle_id="joint-v1:learning-contract",
        owner_id="user-1",
        scope_id="scope-1",
        session_id="session-1",
        task_id="task-1",
        joint_revision=7,
        intent_id="intent-1",
        execution_id="executor-1",
        outcome_sha256=content_digest("settled-outcome"),
        outcome_trust="caller_asserted_experienced_not_authenticated",
        state_owner="CognitiveState",
        eligible_receivers=("world_model", "executive", "language_candidate"),
        world_model_sha256=content_digest("world-model"),
        world_prediction_sha256=content_digest("selected-prediction"),
        executive_sha256=content_digest("executive"),
        decision_sha256=content_digest("selected-decision"),
        language_feedback_id="feedback-1",
        language_feedback_sha256=content_digest("correct-target"),
        language_parameter_revision=2,
        apply_updates=False,
    )


def test_six_credit_controls_are_distinct_and_require_restorable_checkpoints():
    route = make_route()
    correct = JointCreditComparisonSpecV1.for_route(route, "correct_feedback")
    permuted = JointCreditComparisonSpecV1.for_route(
        route,
        "permuted_feedback",
        alternate_feedback_sha256=content_digest("permuted-target"),
    )
    no_update = JointCreditComparisonSpecV1.for_route(route, "no_update")
    readout = JointCreditComparisonSpecV1.for_route(route, "readout_only")
    owner_frozen = JointCreditComparisonSpecV1.for_route(route, "responsible_owner_frozen")
    state_only = JointCreditComparisonSpecV1.for_route(route, "state_only_change")

    assert correct.parameter_update_owners == route.eligible_receivers
    assert correct.state_owner == "CognitiveState" and correct.state_write_allowed
    assert permuted.feedback_sha256 != correct.feedback_sha256
    assert no_update.state_owner is None and not no_update.state_write_allowed
    assert no_update.frozen_parameter_owners == ("all_parameters",)
    assert readout.parameter_update_owners == ("language_candidate.output",)
    assert "language_candidate.non_readout" in readout.frozen_parameter_owners
    assert set(owner_frozen.frozen_parameter_owners) == set(route.eligible_receivers)
    assert owner_frozen.unrelated_parameter_updates_allowed
    assert state_only.state_owner == "CognitiveState" and state_only.state_write_allowed
    assert state_only.frozen_parameter_owners == ("all_parameters",)
    assert (
        len(
            {
                content_digest(spec.to_payload())
                for spec in (correct, permuted, no_update, readout, owner_frozen, state_only)
            }
        )
        == 6
    )
    for spec in (correct, permuted, no_update, readout, owner_frozen, state_only):
        assert spec.pre_update_checkpoint_required
        assert spec.post_update_checkpoint_required
        assert spec.same_native_checkpoint_required
        assert spec.fresh_process_restore_required
        assert spec.apply_updates is False
        restored = JointCreditComparisonSpecV1.from_payload(spec.to_payload())
        assert content_digest(restored.to_payload()) == content_digest(spec.to_payload())


def test_control_spec_rejects_permuted_target_reuse_or_update_application():
    route = make_route()
    with pytest.raises(ValueError, match="differ from the correct target"):
        JointCreditComparisonSpecV1.for_route(
            route,
            "permuted_feedback",
            alternate_feedback_sha256=route.language_feedback_sha256,
        )
    payload = JointCreditComparisonSpecV1.for_route(route, "state_only_change").to_payload()
    tampered = copy.deepcopy(payload)
    tampered["apply_updates"] = True
    tampered["spec_sha256"] = content_digest(
        {key: value for key, value in tampered.items() if key != "spec_sha256"}
    )
    with pytest.raises(ValueError, match="cannot apply updates"):
        JointCreditComparisonSpecV1.from_payload(tampered)
