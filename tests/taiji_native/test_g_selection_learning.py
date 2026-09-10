from __future__ import annotations

from copy import deepcopy

import pytest

from taiji import (
    ContentPlan,
    Goal,
    GSelectionCandidate,
    GSelectionCandidateSet,
    GSelectionDecision,
    GSelectionLearner,
    content_digest,
)

PARENT_MANIFEST = "a" * 64
K_DIGESTS = {"k1": "b" * 64, "k2": "c" * 64}


def _proposal(candidate_id: str, goal_id: str, content_id: str, *, score: float) -> GSelectionCandidate:
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


def _learner() -> GSelectionLearner:
    return GSelectionLearner(
        parent_manifest_digest=PARENT_MANIFEST,
        k_checkpoint_digests=K_DIGESTS,
    )


def test_zero_step_g_selects_high_signal_proposal_and_roundtrips() -> None:
    learner = _learner()
    candidate_set = _set()

    decision = learner.select(candidate_set)

    assert decision.selection_status == "selected"
    assert decision.selected_candidate_id == "good"
    assert decision.selected_goal_id == "goal:good"
    assert decision.external_target_used is False
    assert GSelectionDecision.from_payload(decision.to_payload()) == decision


def test_g_fit_changes_only_g_state_and_rejects_non_train_inputs() -> None:
    learner = _learner()
    before = learner.checkpoint()
    result = learner.fit((_set(),), epochs=2, learning_rate=0.1)

    assert result["candidate_sets"] == 1
    assert learner.training_steps == 2
    assert learner.revision == 1
    assert learner.checkpoint()["checkpoint_digest"] != before["checkpoint_digest"]
    assert learner.parent_manifest_digest == PARENT_MANIFEST
    assert learner.k_checkpoint_digests == K_DIGESTS
    assert learner.parameter_count == 13
    assert all(parameter.requires_grad is False for parameter in learner.model.parameters())

    with pytest.raises(ValueError, match="train candidate sets only"):
        learner.fit((_set(split="validation"),))


def test_g_checkpoint_digest_and_lineage_are_fail_closed() -> None:
    learner = _learner()
    learner.fit((_set(),), epochs=1)
    checkpoint = learner.checkpoint()
    restored = GSelectionLearner.from_checkpoint(checkpoint)

    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)
    restored.assert_lineage(parent_manifest_digest=PARENT_MANIFEST, k_checkpoint_digests=K_DIGESTS)

    with pytest.raises(ValueError, match="lineage mismatch"):
        restored.assert_lineage(parent_manifest_digest="e" * 64, k_checkpoint_digests=K_DIGESTS)

    tampered = deepcopy(checkpoint)
    tampered["revision"] = 99
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        GSelectionLearner.from_checkpoint(tampered)


def test_g_abstains_when_proposal_is_not_safe() -> None:
    learner = _learner()
    unsafe = _proposal("unsafe", "goal:unsafe", "content:unsafe", score=0.9)
    unsafe = GSelectionCandidate.create(
        candidate_id=unsafe.candidate_id,
        source=unsafe.source,
        candidate_role=unsafe.candidate_role,
        status=unsafe.status,
        goal=unsafe.goal,
        content_plan=unsafe.content_plan,
        goal_score=unsafe.goal_score,
        content_score=unsafe.content_score,
        confidence=0.2,
        ambiguity=unsafe.ambiguity,
    )
    candidate_set = GSelectionCandidateSet.create(
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

    decision = learner.select(candidate_set)

    assert decision.selection_status == "abstained"
    assert decision.selected_candidate_id == "abstain"
    assert decision.selected_goal_id is None
    assert decision.selected_content_id is None
