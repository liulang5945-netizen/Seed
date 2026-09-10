from __future__ import annotations

import pytest

from seed_platform.workbench import CapabilitySnapshot
from taiji import (
    GSelectionCandidate,
    GSelectionCandidateSet,
    GSelectionDecision,
    NativeReadOnlyIntentPlanner,
    ReadOnlyAbstention,
    ReadOnlyIntentPolicy,
    WorkbenchObservation,
    WorkbenchObservationSchema,
    project_g_decision,
)


def _candidate(candidate_id: str, role: str) -> GSelectionCandidate:
    return GSelectionCandidate.create(
        candidate_id=candidate_id,
        source=f"runtime.{role}",
        candidate_role=role,
        status="ambiguous" if role == "reobserve" else "abstained",
        goal=None,
        content_plan=None,
        goal_score=0.0,
        content_score=0.0,
        confidence=0.0,
        ambiguity=1.0,
    )


def _candidate_set() -> GSelectionCandidateSet:
    reobserve = _candidate("reobserve", "reobserve")
    abstain = _candidate("abstain", "abstain")
    return GSelectionCandidateSet.create(
        example_id="projection-example",
        family_id="projection-family",
        split="validation",
        project_id="projection-project",
        path="missing.py",
        input_digest="d" * 64,
        candidates=(reobserve, abstain),
        target_candidate_id="reobserve",
        target_kind="reobserve",
    )


def _observation(snapshot: CapabilitySnapshot) -> WorkbenchObservation:
    schema = WorkbenchObservationSchema(
        language_ids=("unknown", "python"),
        selection_states=("unknown", "resolved"),
        task_kinds=("inspect",),
        extensions=("<none>", ".py"),
    )
    return WorkbenchObservation.from_workbench_evidence(
        observation_id="projection-observation",
        project_id="projection-project",
        task_id="projection-task",
        path="missing.py",
        capability_snapshot_id=snapshot.snapshot_id,
        capability_revision=snapshot.revision,
        read_result={"success": False, "byte_length": 0},
        language_result=None,
        task_kind="inspect",
        schema=schema,
    )


def test_reobserve_projects_to_non_executable_root_list_and_roundtrips() -> None:
    candidate_set = _candidate_set()
    selected = next(item for item in candidate_set.candidates if item.candidate_role == "reobserve")
    decision = GSelectionDecision.create(
        candidate_set=candidate_set,
        selected=selected,
        selection_status="reobserve",
        candidate_scores=(("abstain", 0.2), ("reobserve", 0.8)),
    )
    snapshot = CapabilitySnapshot.default()
    projected = project_g_decision(
        decision,
        candidate_set,
        planner=NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=())),
        observation=_observation(snapshot),
        capability_snapshot=snapshot,
    )

    assert isinstance(projected, ReadOnlyAbstention)
    assert projected.next_step == "workspace.list"
    assert projected.reason_code == "g_reobserve_requested"
    assert projected.to_payload()["action_intent"] is None
    assert ReadOnlyAbstention.from_payload(projected.to_payload()) == projected


def test_projection_rejects_crossed_candidate_lineage() -> None:
    candidate_set = _candidate_set()
    selected = next(item for item in candidate_set.candidates if item.candidate_role == "reobserve")
    decision = GSelectionDecision.create(
        candidate_set=candidate_set,
        selected=selected,
        selection_status="reobserve",
        candidate_scores=(("abstain", 0.2), ("reobserve", 0.8)),
    )
    other = GSelectionCandidateSet.create(
        example_id="other-example",
        family_id="projection-family",
        split="validation",
        project_id="projection-project",
        path="other.py",
        input_digest="e" * 64,
        candidates=candidate_set.candidates,
        target_candidate_id="reobserve",
        target_kind="reobserve",
    )
    snapshot = CapabilitySnapshot.default()
    with pytest.raises(ValueError, match="lineage mismatch"):
        project_g_decision(
            decision,
            other,
            planner=NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=())),
            observation=_observation(snapshot),
            capability_snapshot=snapshot,
        )
