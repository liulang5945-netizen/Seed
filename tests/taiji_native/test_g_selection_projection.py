from __future__ import annotations

import pytest

from taiji import (
    ContentPlan,
    Goal,
    GSelectionCandidate,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)
from taiji.g_selection_extended import ExtendedGSelectionLearner
from taiji.g_selection_projection import project_to_joint_feasible_region

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


def test_projection_converges_on_simple_feasible_system() -> None:
    result = project_to_joint_feasible_region([((1.0, 0.0), 1.0, "floor")], [0.0, 0.0])

    assert result["converged"] is True
    # The frozen criterion is violation-side: the projection must land on
    # the feasible side of the boundary (finite rho/lr leaves a small
    # feasible-side overshoot that the distance audit records honestly).
    assert result["weights"][0] >= 1.0 - 1e-6
    assert result["weights"][0] < 1.2
    assert abs(result["weights"][1]) < 1e-2
    assert result["max_violation"] <= 1e-6


def test_projection_is_deterministic() -> None:
    constraints = [((1.0, 0.0), 1.0, "floor"), ((0.0, 1.0), 0.5, "ceil")]
    first = project_to_joint_feasible_region(constraints, [0.0, 0.0])
    second = project_to_joint_feasible_region(constraints, [0.0, 0.0])

    assert first["weights"] == second["weights"]
    assert first["distance"] == second["distance"]


def test_anchor_already_feasible_stays_put() -> None:
    result = project_to_joint_feasible_region([((1.0, 0.0), 0.5, "floor")], [1.0, 0.25])

    assert result["converged"] is True
    assert result["distance"]["l2"] < 1e-2
    assert result["weights"][0] == pytest.approx(1.0, abs=1e-2)
    assert result["weights"][1] == pytest.approx(0.25, abs=1e-2)


def test_infeasible_system_reports_honest_failure() -> None:
    result = project_to_joint_feasible_region(
        [((1.0, 0.0), 1.0, "lower"), ((-1.0, 0.0), 1.0, "upper")], [0.0, 0.0]
    )

    assert result["converged"] is False
    assert result["total_violation"] > 1e-5
    assert set(result["violation_per_constraint_family"]) == {"lower", "upper"}


def test_extended_learner_applies_projection_without_touching_feature_source() -> None:
    parent = _trained_parent()
    learner = ExtendedGSelectionLearner.from_parent_learner(parent)
    learner.invariant_fit(
        ({"candidate_set": _set()},),
        (_unsafe_set(),),
        epochs=2,
        learning_rate=0.1,
        order_seed=0,
        constraint_digest="e" * 64,
    )
    bias_before = float(learner.head.bias.detach().reshape(()))
    feature_digest_before = learner.feature_source_state_digest

    result = project_to_joint_feasible_region(
        [((1.0,) * 16, 0.1, "synthetic")],
        [float(v) for v in learner.head.weight.detach().reshape(-1)],
    )
    learner.apply_projected_weights(result["weights"], projection_digest="p" * 64)

    assert learner.revision == 2
    assert learner.last_train_digest == "p" * 64
    assert float(learner.head.bias.detach().reshape(())) == bias_before
    assert learner.feature_source_state_digest == feature_digest_before
    assert learner.select(_set(split="validation")).selection_status in {
        "selected",
        "abstained",
        "reobserve",
    }
    restored = ExtendedGSelectionLearner.from_checkpoint(learner.checkpoint())
    assert content_digest(restored.checkpoint()) == content_digest(learner.checkpoint())
