"""Project a label-free G decision into the native read-only action boundary.

G owns candidate selection, while the host owns capability execution.  This
module keeps that boundary explicit: ``reobserve`` can only become a typed,
non-executable ``workspace.list`` abstention; it cannot carry a goal/content
pair or execute a Workbench action.
"""

from __future__ import annotations

from typing import Any

from .g_selection import GSelectionCandidateSet
from .g_selection_learning import GSelectionDecision
from .read_only_intent import (
    NativeReadOnlyIntentPlanner,
    ReadOnlyAbstention,
    ReadOnlyIntentDecision,
)
from .workbench_observation import WorkbenchObservation

TAIJI_G_ACTION_PROJECTION_FORMAT = "taiji-g-selection-action-projection-v1"
TAIJI_G_ACTION_PROJECTION_VERSION = 1


def project_g_decision(
    decision: GSelectionDecision,
    candidate_set: GSelectionCandidateSet,
    *,
    planner: NativeReadOnlyIntentPlanner,
    observation: WorkbenchObservation,
    capability_snapshot: Any,
    world: Any = None,
    tick: int = 1,
) -> ReadOnlyAbstention | ReadOnlyIntentDecision:
    """Project one G decision without executing a capability.

    A ``reobserve`` decision is deliberately projected to ``workspace.list``
    with no ``ActionIntent``.  Proposal decisions are only rendered through
    the read-only planner and require a world; the caller remains responsible
    for any later Workbench execution and outcome accounting.
    """

    if not isinstance(decision, GSelectionDecision):
        raise TypeError("G action projection requires a GSelectionDecision")
    if not isinstance(candidate_set, GSelectionCandidateSet):
        raise TypeError("G action projection requires a GSelectionCandidateSet")
    if not isinstance(planner, NativeReadOnlyIntentPlanner):
        raise TypeError("G action projection requires a native read-only planner")
    if not isinstance(observation, WorkbenchObservation):
        raise TypeError("G action projection requires a WorkbenchObservation")
    if decision.candidate_set_digest != candidate_set.candidate_set_digest:
        raise ValueError("G action decision and candidate-set lineage mismatch")
    selected = next(
        (
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_id == decision.selected_candidate_id
        ),
        None,
    )
    if selected is None:
        raise ValueError("G action decision selected candidate is missing")
    if selected.candidate_digest != decision.selected_candidate_digest:
        raise ValueError("G action decision selected candidate digest mismatch")

    if decision.selection_status == "reobserve":
        if selected.candidate_role != "reobserve":
            raise ValueError("reobserve decision must select a reobserve candidate")
        return planner.abstain(
            observation=observation,
            capability_snapshot=capability_snapshot,
            reason_code="g_reobserve_requested",
            next_step="workspace.list",
            confidence=decision.confidence,
        )

    if decision.selection_status == "abstained":
        if selected.candidate_role != "abstain":
            raise ValueError("abstained decision must select an abstain candidate")
        return planner.abstain(
            observation=observation,
            capability_snapshot=capability_snapshot,
            reason_code="g_safe_abstention",
            next_step="none",
            confidence=decision.confidence,
        )

    if decision.selection_status != "selected":
        raise ValueError("unsupported G action selection status")
    if selected.candidate_role != "proposal" or selected.goal is None or selected.content_plan is None:
        raise ValueError("selected G action must contain a complete proposal")
    if world is None:
        raise ValueError("proposal action projection requires a verified world")
    return planner.propose(
        observation=observation,
        world=world,
        goal=selected.goal,
        content=selected.content_plan,
        capability_snapshot=capability_snapshot,
        tick=int(tick),
    )


__all__ = [
    "TAIJI_G_ACTION_PROJECTION_FORMAT",
    "TAIJI_G_ACTION_PROJECTION_VERSION",
    "project_g_decision",
]
