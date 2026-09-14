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
from taiji.g_selection_context import (
    CONTEXT_FEATURE_NAMES,
    CONTEXT_G_LEARNER_FORMAT,
    ContextGSelectionLearner,
    context_features,
)

PARENT_MANIFEST = "a" * 64
K_DIGESTS = {"k1": "b" * 64, "k2": "c" * 64}


def _proposal(
    candidate_id: str, goal_id: str, content_id: str, *, score: float
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
        confidence=0.9,
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


def _trained_parent() -> GSelectionLearner:
    parent = GSelectionLearner(
        parent_manifest_digest=PARENT_MANIFEST,
        k_checkpoint_digests=K_DIGESTS,
    )
    parent.fit((_set(),), epochs=2, learning_rate=0.1)
    return parent


def test_birth_equivalence_matches_parent_selections_exactly() -> None:
    parent = _trained_parent()
    child = ContextGSelectionLearner.from_parent_learner(parent)

    assert child.parameter_count == 22
    assert child.parent_manifest_digest == PARENT_MANIFEST
    assert child.k_checkpoint_digests == K_DIGESTS
    weight = child.model.weight.detach()
    assert torch.count_nonzero(weight[0, len(weight[0]) - 9 :]).item() == 0
    assert torch.equal(weight[0, :12].cpu(), parent.model.weight.detach()[0].cpu())
    assert float(child.model.bias.detach().reshape(())) == float(
        parent.model.bias.detach().reshape(())
    )

    for _ in range(3):
        candidate_set = _set()
        parent_decision = parent.select(candidate_set)
        child_decision = child.select(candidate_set)
        assert child_decision.selection_status == parent_decision.selection_status
        assert child_decision.selected_candidate_id == parent_decision.selected_candidate_id
        context = context_features(candidate_set)
        for candidate in candidate_set.candidates:
            parent_score = parent.score(candidate)
            child_score = child.score(candidate, context[candidate.candidate_id])
            assert abs(child_score - parent_score) <= 1e-6


def test_context_weights_change_selection_when_scaled() -> None:
    parent = _trained_parent()
    neutral = ContextGSelectionLearner.from_parent_learner(parent)
    candidate_set = _set()
    context = context_features(candidate_set)

    # candidate_joint_minus_mean is the last (9th) context feature.
    joint_minus_mean = {candidate_id: values[-1] for candidate_id, values in context.items()}
    assert max(joint_minus_mean.values()) == pytest.approx(joint_minus_mean["good"], abs=1e-9)

    # A large negative weight on joint-minus-mean (context index 8,
    # absolute position 20) pushes the above-mean proposal below the safe
    # candidate, flipping selection to abstention.
    penalised = ContextGSelectionLearner.from_parent_learner(parent)
    with torch.no_grad():
        penalised.model.weight[0, 20] = -10.0
    assert neutral.select(candidate_set).selection_status == "selected"
    assert penalised.select(candidate_set).selection_status == "abstained"


def test_manual_score_equals_explicit_linear_combination() -> None:
    parent = _trained_parent()
    child = ContextGSelectionLearner.from_parent_learner(parent)
    candidate_set = _set()
    context = context_features(candidate_set)
    candidate = next(c for c in candidate_set.candidates if c.candidate_id == "good")

    with torch.no_grad():
        child.model.weight[0, 12] = 0.5
        child.model.weight[0, 20] = -2.0
        child.model.bias.fill_(0.25)

    expected = (
        sum(
            float(w) * float(f)
            for w, f in zip(
                child.model.weight.detach()[0][:12], candidate.feature_vector, strict=True
            )
        )
        + 0.5 * context["good"][0]
        - 2.0 * context["good"][8]
        + 0.25
    )
    assert child.score(candidate, context["good"]) == pytest.approx(expected, abs=1e-6)


def test_checkpoint_roundtrip_and_tamper_are_fail_closed() -> None:
    parent = _trained_parent()
    child = ContextGSelectionLearner.from_parent_learner(parent)
    payload = child.checkpoint()

    assert payload["format"] == CONTEXT_G_LEARNER_FORMAT
    assert payload["parameter_count"] == 22
    restored = ContextGSelectionLearner.from_checkpoint(payload)
    assert content_digest(restored.checkpoint()) == content_digest(payload)
    restored.assert_lineage(parent_manifest_digest=PARENT_MANIFEST, k_checkpoint_digests=K_DIGESTS)

    tampered = deepcopy(dict(payload))
    tampered["revision"] = int(tampered["revision"]) + 1
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        ContextGSelectionLearner.from_checkpoint(tampered)

    with pytest.raises(ValueError, match="lineage mismatch"):
        restored.assert_lineage(parent_manifest_digest="e" * 64, k_checkpoint_digests=K_DIGESTS)


def test_functional_fit_updates_only_context_learner_state() -> None:
    parent = _trained_parent()
    child = ContextGSelectionLearner.from_parent_learner(parent)
    record = {"candidate_set": _set()}
    constraint = [
        tuple(
            (*candidate.feature_vector, *(0.0,) * len(CONTEXT_FEATURE_NAMES))
            for candidate in _set().candidates
        )
    ]

    result = child.functional_fit(
        parent,
        (record,),
        constraint,
        epochs=2,
        learning_rate=0.1,
        order_seed=3,
        functional_weight=1.0,
        constraint_digest="f" * 64,
    )

    assert result["training_steps"] == 4
    assert result["revision"] == 1
    assert child.training_steps == 4
    assert child.revision == 1
    assert child.last_train_digest == result["dataset_digest"]
    assert all(parameter.requires_grad is False for parameter in child.model.parameters())
    # Parent must be untouched by the child's fit.
    assert parent.training_steps == 2


def test_from_parent_learner_rejects_wrong_parent_capacity() -> None:
    child = ContextGSelectionLearner.from_parent_learner(_trained_parent())
    with pytest.raises(ValueError, match="13-parameter"):
        ContextGSelectionLearner.from_parent_learner(child)
