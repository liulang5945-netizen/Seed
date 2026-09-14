from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from taiji import (
    ContentPlan,
    Goal,
    GSelectionCandidate,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)
from taiji.g_selection_residual import (
    RESIDUAL_G_LEARNER_FORMAT,
    ResidualGSelectionLearner,
    _apply_selection_rule,
)
from taiji.local_learning import apply_linear_delta

PARENT_MANIFEST = "a" * 64
K_DIGESTS = {"k1": "b" * 64, "k2": "c" * 64}


def _proposal(
    candidate_id: str, goal_id: str, content_id: str, *, score: float, confidence: float = 0.9
) -> GSelectionCandidate:
    goal = Goal(goal_id, f"goal {goal_id}", priority=0.8)
    content = ContentPlan(
        content_id=content_id,
        intent_id=f"intent:{content_id}",
        intent_kind="report",
        source_goal_id=goal.goal_id,
        expected_outcome="a test report",
    )
    return GSelectionCandidate.create(
        candidate_id=candidate_id,
        source="k.test",
        candidate_role="proposal",
        status="resolved",
        goal=goal,
        content_plan=content,
        goal_score=score,
        content_score=score,
        confidence=confidence,
        ambiguity=0.1,
    )


def _safe(candidate_id: str = "abstain", *, role: str = "abstain") -> GSelectionCandidate:
    return GSelectionCandidate.create(
        candidate_id=candidate_id,
        source=f"runtime.{role}",
        candidate_role=role,
        status="abstained" if role == "abstain" else "ambiguous",
        goal=None,
        content_plan=None,
        goal_score=0.0,
        content_score=0.0,
        confidence=0.0,
        ambiguity=1.0,
    )


def _set(*, split: str = "train", target: str = "good") -> GSelectionCandidateSet:
    good = _proposal("good", "goal:good", "content:good", score=0.9)
    weak = _proposal("weak", "goal:weak", "content:weak", score=0.25)
    abstain = _safe()
    return GSelectionCandidateSet.create(
        example_id=f"example-{split}-{target}",
        family_id=f"family-{split}",
        split=split,
        project_id=f"project-{split}",
        path=f"{split}.py",
        input_digest="d" * 64,
        candidates=(good, weak, abstain),
        target_candidate_id=target,
        target_kind="pair" if target != "abstain" else "abstain",
    )


def _unsafe_set() -> GSelectionCandidateSet:
    unsafe = _proposal("unsafe", "goal:unsafe", "content:unsafe", score=0.9, confidence=0.2)
    return GSelectionCandidateSet.create(
        example_id="example-unsafe",
        family_id="family-unsafe",
        split="validation",
        project_id="project-unsafe",
        path="unsafe.py",
        input_digest="f" * 64,
        candidates=(unsafe, _safe()),
        target_candidate_id="abstain",
        target_kind="abstain",
    )


def _trained_parent() -> GSelectionLearner:
    parent = GSelectionLearner(
        parent_manifest_digest=PARENT_MANIFEST,
        k_checkpoint_digests=K_DIGESTS,
    )
    parent.fit((_set(),), epochs=2, learning_rate=0.1)
    return parent


def test_birth_equivalence_is_exact_and_frozen_head_matches_parent() -> None:
    parent = _trained_parent()
    child = ResidualGSelectionLearner.from_parent_learner(parent)

    assert child.parameter_count == 26
    assert child.trainable_parameter_count == 13
    assert child.frozen_parameter_count == 13
    assert child.parent_head_state_digest == content_digest(
        {
            "weight": parent.model.weight.detach().cpu(),
            "bias": parent.model.bias.detach().cpu(),
        }
    )
    delta_weight = child.delta_head.weight.detach()
    assert torch.count_nonzero(delta_weight).item() == 0
    assert torch.count_nonzero(child.delta_head.bias.detach()).item() == 0

    for _ in range(3):
        candidate_set = _set()
        parent_decision = parent.select(candidate_set)
        child_decision = child.select(candidate_set)
        assert child_decision.selection_status == parent_decision.selection_status
        assert child_decision.selected_candidate_id == parent_decision.selected_candidate_id
        for candidate in candidate_set.candidates:
            assert child.total_score(candidate) == parent.score(candidate)


def test_hinge_loss_is_zero_at_birth_by_construction() -> None:
    parent = _trained_parent()
    child = ResidualGSelectionLearner.from_parent_learner(parent)

    for candidate_set in (_set(), _unsafe_set()):
        loss, error = child.invariant_hinge(candidate_set)
        assert loss == 0.0
        assert torch.count_nonzero(error).item() == 0


def test_hinge_activates_on_margin_erosion_and_pushes_back() -> None:
    parent = _trained_parent()
    child = ResidualGSelectionLearner.from_parent_learner(parent)
    candidate_set = _set()
    good = next(c for c in candidate_set.candidates if c.candidate_id == "good")
    weak = next(c for c in candidate_set.candidates if c.candidate_id == "weak")
    abstain = next(c for c in candidate_set.candidates if c.candidate_id == "abstain")

    # Erode the parent's argmax margin: a negative joint-score delta weight
    # boosts the weak (low-joint) proposal relative to good.  Both decision
    # margins (argmax and safe-clear) erode, so both hinge terms activate.
    with torch.no_grad():
        child.delta_head.weight[0, 2] = -0.5
    parent_gap = parent.score(good) - parent.score(weak)
    parent_safe_gap = parent.score(good) - parent.score(abstain)
    eroded_gap = child.total_score(good) - child.total_score(weak)
    eroded_safe_gap = child.total_score(good) - child.total_score(abstain)
    assert eroded_gap < parent_gap
    assert eroded_safe_gap < parent_safe_gap

    loss, error = child.invariant_hinge(candidate_set)
    assert loss == pytest.approx(
        (parent_gap - eroded_gap) + (parent_safe_gap - eroded_safe_gap), abs=1e-6
    )
    error_by_id = {
        candidate.candidate_id: float(error[index])
        for index, candidate in enumerate(candidate_set.candidates)
    }
    assert error_by_id["good"] == -2.0
    assert error_by_id["weak"] == 1.0
    assert error_by_id["abstain"] == 1.0

    # One hinge step must recover part of the eroded margin (finite support
    # direction: the violation decreases).
    inputs = torch.tensor(
        [candidate.feature_vector for candidate in candidate_set.candidates],
        dtype=torch.float32,
    )
    apply_linear_delta(child.delta_head, inputs, error, 0.1)
    recovered_gap = child.total_score(good) - child.total_score(weak)
    assert recovered_gap > eroded_gap


def test_safe_boundary_hinge_guards_rejection_margin() -> None:
    parent = _trained_parent()
    child = ResidualGSelectionLearner.from_parent_learner(parent)
    candidate_set = _unsafe_set()
    unsafe = next(c for c in candidate_set.candidates if c.candidate_id == "unsafe")
    abstain = next(c for c in candidate_set.candidates if c.candidate_id == "abstain")

    assert parent.select(candidate_set).selection_status == "abstained"
    assert child.invariant_hinge(candidate_set)[0] == 0.0

    # A positive joint-score delta weight boosts the high-joint unsafe
    # proposal relative to the abstain candidate beyond the parent's own
    # encroachment.
    with torch.no_grad():
        child.delta_head.weight[0, 2] = 1.0
    loss, error = child.invariant_hinge(candidate_set)
    assert loss > 0.0
    error_by_id = {
        candidate.candidate_id: float(error[index])
        for index, candidate in enumerate(candidate_set.candidates)
    }
    assert error_by_id["unsafe"] == 1.0
    assert error_by_id["abstain"] == -1.0
    assert child.total_score(unsafe) - child.total_score(abstain) > parent.score(
        unsafe
    ) - parent.score(abstain)


def test_invariant_fit_updates_only_delta_head() -> None:
    parent = _trained_parent()
    child = ResidualGSelectionLearner.from_parent_learner(parent)
    parent_digest_before = child.parent_head_state_digest

    result = child.invariant_fit(
        ({"candidate_set": _set()},),
        (_unsafe_set(),),
        epochs=2,
        learning_rate=0.1,
        order_seed=3,
        constraint_digest="e" * 64,
    )

    assert result["training_steps"] == 4
    assert result["constraint_steps"] == 2
    assert result["revision"] == 1
    assert child.parent_head_state_digest == parent_digest_before
    assert torch.count_nonzero(child.delta_head.weight.detach()).item() > 0
    assert child.last_train_digest == result["dataset_digest"]
    assert all(parameter.requires_grad is False for parameter in child.parent_head.parameters())
    # The frozen parent learner must be untouched by the child's fit.
    assert parent.training_steps == 2


def test_checkpoint_roundtrip_and_dual_tamper_are_fail_closed() -> None:
    parent = _trained_parent()
    child = ResidualGSelectionLearner.from_parent_learner(parent)
    with torch.no_grad():
        child.delta_head.weight[0, 0] = 0.3
    payload = child.checkpoint()

    assert payload["format"] == RESIDUAL_G_LEARNER_FORMAT
    assert payload["parameter_count"] == 26
    restored = ResidualGSelectionLearner.from_checkpoint(payload)
    assert content_digest(restored.checkpoint()) == content_digest(payload)
    restored.assert_lineage(parent_manifest_digest=PARENT_MANIFEST, k_checkpoint_digests=K_DIGESTS)
    assert restored.parent_head_state_digest == child.parent_head_state_digest

    tampered_revision = deepcopy(dict(payload))
    tampered_revision["revision"] = int(tampered_revision["revision"]) + 1
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        ResidualGSelectionLearner.from_checkpoint(tampered_revision)

    # Layered-bypass tamper: re-sign the outer and model-state digests so
    # only the frozen-parent-head integrity check can catch the mutation.
    tampered_parent_head = deepcopy(dict(payload))
    tampered_parent_head["parent_weight"] = tampered_parent_head["parent_weight"].clone()
    tampered_parent_head["parent_weight"][0, 0] += 1.0
    tampered_parent_head.pop("checkpoint_digest")
    tampered_parent_head["model_state_digest"] = content_digest(
        {
            "parent_weight": tampered_parent_head["parent_weight"],
            "parent_bias": tampered_parent_head["parent_bias"],
            "delta_weight": tampered_parent_head["delta_weight"],
            "delta_bias": tampered_parent_head["delta_bias"],
        }
    )
    tampered_parent_head["checkpoint_digest"] = content_digest(tampered_parent_head)
    with pytest.raises(ValueError, match="frozen parent head digest mismatch"):
        ResidualGSelectionLearner.from_checkpoint(tampered_parent_head)

    with pytest.raises(ValueError, match="lineage mismatch"):
        restored.assert_lineage(parent_manifest_digest="e" * 64, k_checkpoint_digests=K_DIGESTS)


def test_from_parent_learner_and_selection_rule_guards() -> None:
    child = ResidualGSelectionLearner.from_parent_learner(_trained_parent())
    with pytest.raises(ValueError, match="13-parameter"):
        ResidualGSelectionLearner.from_parent_learner(child)

    # The shared selection-rule replication must agree with the frozen
    # learner's own select on the same scores.
    parent = _trained_parent()
    candidate_set = _set()
    scored = [(candidate, parent.score(candidate)) for candidate in candidate_set.candidates]
    selected, status = _apply_selection_rule(
        scored,
        confidence_floor=parent.confidence_floor,
        selection_margin=parent.selection_margin,
    )
    decision = parent.select(candidate_set)
    assert selected.candidate_id == decision.selected_candidate_id
    assert status == decision.selection_status
