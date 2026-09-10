from __future__ import annotations

import pytest

from taiji import (
    ContentPlan,
    Goal,
    GSelectionCandidate,
    GSelectionCandidateSet,
    content_digest,
)


def _candidate(*, candidate_id: str, goal: Goal | None, content: ContentPlan | None) -> GSelectionCandidate:
    return GSelectionCandidate.create(
        candidate_id=candidate_id,
        source="k.test",
        status="resolved" if goal is not None and content is not None else "abstained",
        goal=goal,
        content_plan=content,
        goal_score=0.9 if goal is not None else 0.0,
        content_score=0.8 if content is not None else 0.0,
        confidence=0.8 if content is not None else 0.0,
        ambiguity=0.1 if content is not None else 1.0,
    )


def _set() -> GSelectionCandidateSet:
    goal = Goal("goal:test", "inspect the test input", priority=0.9)
    content = ContentPlan(
        content_id="content:test",
        intent_id="intent:test",
        intent_kind="report",
        source_goal_id=goal.goal_id,
        expected_outcome="a report",
    )
    selected = _candidate(candidate_id="k1", goal=goal, content=content)
    abstain = _candidate(candidate_id="abstain", goal=None, content=None)
    return GSelectionCandidateSet.create(
        example_id="example-1",
        family_id="family-1",
        split="train",
        project_id="project-1",
        path="input.ts",
        input_digest="a" * 64,
        candidates=(selected, abstain),
        target_candidate_id=selected.candidate_id,
        target_kind="pair",
    )


def test_candidate_set_round_trip_and_runtime_payload_excludes_target() -> None:
    item = _set()
    restored = GSelectionCandidateSet.from_payload(item.to_payload())

    assert restored == item
    assert restored.target_candidate().candidate_id == "k1"
    assert restored.target_available
    assert "target_candidate_id" not in restored.to_inference_payload()
    assert "target_kind" not in restored.to_inference_payload()
    assert restored.inference_digest == content_digest(
        {
            key: value
            for key, value in restored.to_inference_payload().items()
            if key != "inference_digest"
        }
    )


def test_candidate_features_and_margin_are_deterministic() -> None:
    item = _set()
    selected = item.target_candidate()

    assert len(selected.feature_vector) == 10
    assert selected.feature_vector[2] == pytest.approx(0.8)
    assert item.score_margin == pytest.approx(0.8)


def test_candidate_set_rejects_duplicate_ids_and_tampering() -> None:
    item = _set()
    with pytest.raises(ValueError, match="candidate ids"):
        GSelectionCandidateSet.create(
            example_id=item.example_id,
            family_id=item.family_id,
            split=item.split,
            project_id=item.project_id,
            path=item.path,
            input_digest=item.input_digest,
            candidates=(item.candidates[0], item.candidates[0]),
            target_candidate_id=item.target_candidate_id,
            target_kind=item.target_kind,
        )

    payload = item.to_payload()
    payload["candidates"][0]["confidence"] = 0.1
    with pytest.raises(ValueError, match="candidate digest"):
        GSelectionCandidateSet.from_payload(payload)
