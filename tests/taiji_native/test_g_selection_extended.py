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
from taiji.g_selection_extended import (
    EXTENDED_G_LEARNER_FORMAT,
    ExtendedGSelectionLearner,
)

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


def test_birth_equivalence_is_exact_and_relative_dims_start_zero() -> None:
    parent = _trained_parent()
    child = ExtendedGSelectionLearner.from_parent_learner(parent)

    assert child.parameter_count == 17
    assert child.parent_manifest_digest == PARENT_MANIFEST
    weight = child.head.weight.detach()
    assert torch.count_nonzero(weight[0, 12:]).item() == 0
    assert torch.equal(weight[0, :12].cpu(), parent.model.weight.detach()[0].cpu())
    assert float(child.head.bias.detach().reshape(())) == float(
        parent.model.bias.detach().reshape(())
    )
    assert child.feature_source_state_digest == content_digest(
        {
            "weight": parent.model.weight.detach().cpu(),
            "bias": parent.model.bias.detach().cpu(),
        }
    )

    for candidate_set in (_set(), _unsafe_set()):
        parent_decision = parent.select(candidate_set)
        child_decision = child.select(candidate_set)
        assert child_decision.selection_status == parent_decision.selection_status
        assert child_decision.selected_candidate_id == parent_decision.selected_candidate_id
        child_scores = child.total_scores(candidate_set)
        for candidate in candidate_set.candidates:
            assert child_scores[candidate.candidate_id] == parent.score(candidate)


def test_relative_features_are_non_drifting_environment_constants() -> None:
    parent = _trained_parent()
    child = ExtendedGSelectionLearner.from_parent_learner(parent)
    candidate_set = _set()
    before = child.relative_features(candidate_set)
    digest_before = child.feature_source_state_digest

    result = child.invariant_fit(
        ({"candidate_set": _set(split="train", target="weak")},),
        (_unsafe_set(),),
        epochs=2,
        learning_rate=0.1,
        order_seed=1,
        constraint_digest="e" * 64,
    )

    assert child.relative_features(candidate_set) == before
    assert child.feature_source_state_digest == digest_before
    assert result["training_steps"] == 4
    assert result["revision"] == 1
    assert torch.count_nonzero(child.head.weight.detach()[0, 12:]).item() > 0


def test_hinge_delegates_to_canonical_margin_preservation() -> None:
    parent = _trained_parent()
    child = ExtendedGSelectionLearner.from_parent_learner(parent)
    candidate_set = _set()

    # Birth: current scores equal the frozen reference scores, so the
    # margin-preservation hinge must be exactly zero.
    loss, error_by_id = child.invariant_hinge(candidate_set)
    assert loss == 0.0
    assert all(value == 0.0 for value in error_by_id.values())

    # Erode the parent's argmax margin through the base-12 weights; the
    # canonical hinge must activate with the frozen parent's margins as
    # reference.
    with torch.no_grad():
        child.head.weight[0, 2] = -0.5
    eroded_loss, eroded_error = child.invariant_hinge(candidate_set)
    assert eroded_loss > 0.0
    assert eroded_error["good"] < 0.0
    assert max(eroded_error["weak"], eroded_error["abstain"]) > 0.0


def test_checkpoint_roundtrip_and_tamper_are_fail_closed() -> None:
    parent = _trained_parent()
    child = ExtendedGSelectionLearner.from_parent_learner(parent)
    with torch.no_grad():
        child.head.weight[0, 13] = 0.4
    payload = child.checkpoint()

    assert payload["format"] == EXTENDED_G_LEARNER_FORMAT
    assert payload["parameter_count"] == 17
    restored = ExtendedGSelectionLearner.from_checkpoint(payload)
    assert content_digest(restored.checkpoint()) == content_digest(payload)
    restored.assert_lineage(parent_manifest_digest=PARENT_MANIFEST, k_checkpoint_digests=K_DIGESTS)
    assert restored.feature_source_state_digest == child.feature_source_state_digest

    tampered = deepcopy(dict(payload))
    tampered["revision"] = int(tampered["revision"]) + 1
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        ExtendedGSelectionLearner.from_checkpoint(tampered)

    # Layered bypass: re-sign the outer digest and mutate the stored
    # feature-source tensors, so only the frozen feature-source integrity
    # check can catch the drift of the non-drifting parent-relative
    # features.
    bypass = deepcopy(dict(payload))
    bypass["feature_source_weight"] = bypass["feature_source_weight"].clone()
    bypass["feature_source_weight"][0, 0] += 1.0
    bypass.pop("checkpoint_digest")
    bypass["checkpoint_digest"] = content_digest(bypass)
    with pytest.raises(ValueError, match="frozen feature source digest mismatch"):
        ExtendedGSelectionLearner.from_checkpoint(bypass)

    with pytest.raises(ValueError, match="lineage mismatch"):
        restored.assert_lineage(parent_manifest_digest="e" * 64, k_checkpoint_digests=K_DIGESTS)


def test_from_parent_learner_rejects_wrong_parent_capacity() -> None:
    child = ExtendedGSelectionLearner.from_parent_learner(_trained_parent())
    with pytest.raises(ValueError, match="13-parameter"):
        ExtendedGSelectionLearner.from_parent_learner(child)
