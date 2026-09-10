from __future__ import annotations

import pytest

from taiji import (
    GSelectionBehaviorOutcome,
    GSelectionBehaviorSet,
    GSelectionCandidate,
)


def _candidate(candidate_id: str, role: str) -> GSelectionCandidate:
    return GSelectionCandidate.create(
        candidate_id=candidate_id,
        source="test",
        candidate_role=role,
        status="resolved" if role == "proposal" else "abstained",
        goal=None,
        content_plan=None,
        goal_score=0.0,
        content_score=0.0,
        confidence=0.0,
        ambiguity=1.0,
    )


def test_behavior_outcome_round_trip_and_frozen_utility() -> None:
    candidate = _candidate("abstain", "abstain")
    outcome = GSelectionBehaviorOutcome.create(
        candidate_id=candidate.candidate_id,
        candidate_digest=candidate.candidate_digest,
        candidate_role=candidate.candidate_role,
        snapshot_match=True,
        planner_accepted=False,
        route_valid=False,
        parameter_valid=False,
        world_consistent=False,
        execution_success=False,
        safe_exit_valid=True,
        safe_exit_progress=False,
    )

    assert outcome.utility == pytest.approx(0.55)
    assert GSelectionBehaviorOutcome.from_payload(outcome.to_payload()) == outcome


def test_behavior_set_selects_highest_utility_and_round_trips() -> None:
    abstain = _candidate("abstain", "abstain")
    reobserve = _candidate("reobserve", "reobserve")
    first = GSelectionBehaviorOutcome.create(
        candidate_id=abstain.candidate_id,
        candidate_digest=abstain.candidate_digest,
        candidate_role=abstain.candidate_role,
        snapshot_match=True,
        planner_accepted=False,
        route_valid=False,
        parameter_valid=False,
        world_consistent=False,
        execution_success=False,
        safe_exit_valid=True,
        safe_exit_progress=False,
    )
    second = GSelectionBehaviorOutcome.create(
        candidate_id=reobserve.candidate_id,
        candidate_digest=reobserve.candidate_digest,
        candidate_role=reobserve.candidate_role,
        snapshot_match=True,
        planner_accepted=False,
        route_valid=False,
        parameter_valid=False,
        world_consistent=False,
        execution_success=False,
        safe_exit_valid=True,
        safe_exit_progress=True,
    )
    item = GSelectionBehaviorSet.create(
        candidate_set_digest="a" * 64,
        inference_digest="b" * 64,
        split="train",
        project_id="project",
        path="missing.txt",
        outcomes=(first, second),
    )

    assert item.behavior_target_candidate_id == "reobserve"
    assert item.utility_margin == pytest.approx(0.45)
    assert GSelectionBehaviorSet.from_payload(item.to_payload()) == item


def test_behavior_outcome_rejects_tampered_utility() -> None:
    candidate = _candidate("abstain", "abstain")
    outcome = GSelectionBehaviorOutcome.create(
        candidate_id=candidate.candidate_id,
        candidate_digest=candidate.candidate_digest,
        candidate_role=candidate.candidate_role,
        snapshot_match=True,
        planner_accepted=False,
        route_valid=False,
        parameter_valid=False,
        world_consistent=False,
        execution_success=False,
        safe_exit_valid=True,
        safe_exit_progress=False,
    )
    payload = outcome.to_payload()
    payload["utility"] = 1.0
    with pytest.raises(ValueError, match="utility"):
        GSelectionBehaviorOutcome.from_payload(payload)
