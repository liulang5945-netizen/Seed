"""Evaluate variable-horizon branch competition and online trace plasticity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ActionIntent,
    ConceptFormationOrgan,
    EpisodicMemoryRecord,
    Goal,
    GoalPlanner,
    GoalState,
    ImaginedRollout,
    Outcome,
    PlanningCandidate,
    PlanningConfig,
    WorldAction,
    WorldObject,
    WorldState,
    WorldTransition,
)

MANIFEST_FORMAT = "taiji-concept-branch-manifest-v1"
REPORT_FORMAT = "taiji-concept-branch-v1"
GOOD_SEQUENCE = ("approach", "confirm", "archive")
ALT_SEQUENCE = ("approach", "inspect", "wait")


def _state(tick: int, latent: torch.Tensor, object_id: str) -> WorldState:
    return WorldState(
        tick=tick,
        latent=latent.detach().clone(),
        objects=(WorldObject(object_id, attributes={"stage": tick}),),
        relations=(("agent", "sequence", object_id),),
    )


def build_branch_records(
    cue: torch.Tensor, *, episode_count: int = 2
) -> tuple[EpisodicMemoryRecord, ...]:
    """Build two branches sharing their first transition and initial state."""

    if episode_count < 2:
        raise ValueError("branch formation requires at least two episodes per branch")
    dimension = cue.numel()
    start = torch.zeros(dimension)
    start[0] = 1.0
    common_after = torch.zeros(dimension)
    common_after[1] = 1.0
    good_after = torch.zeros(dimension)
    good_after[2] = 1.0
    good_final = torch.zeros(dimension)
    good_final[3 % dimension] = 1.0
    alt_after = torch.zeros(dimension)
    alt_after[2] = -1.0
    alt_final = torch.zeros(dimension)
    alt_final[3 % dimension] = -1.0
    branches = (
        (GOOD_SEQUENCE, (good_after, good_final), (0.40, 0.80, 1.00), (0.0, 0.05, 0.10)),
        (ALT_SEQUENCE, (alt_after, alt_final), (0.30, 0.60, 0.80), (0.20, 0.25, 0.30)),
    )
    records: list[EpisodicMemoryRecord] = []
    index = 0
    for branch_index, (actions, branch_states, rewards, errors) in enumerate(branches):
        for episode_index in range(episode_count):
            object_id = f"branch-object-{branch_index}-{episode_index}"
            states = (
                _state(0, start, object_id),
                _state(1, common_after, object_id),
                _state(2, branch_states[0], object_id),
                _state(3, branch_states[1], object_id),
            )
            for tick, action_kind in enumerate(actions, start=1):
                intent_id = f"branch-intent-{index}"
                action = WorldAction(intent_id, action_kind, tick - 1, target_id=object_id)
                outcome = Outcome(intent_id, rewards[tick - 1], success=True, tick=tick)
                transition = WorldTransition(states[tick - 1], action, states[tick], outcome)
                records.append(
                    EpisodicMemoryRecord(
                        memory_id=f"branch-memory-{index}",
                        episode_id=f"branch-episode-{branch_index}-{episode_index}",
                        tick=tick,
                        cue=cue.detach().clone(),
                        action_intent=ActionIntent(intent_id, action_kind, tick=tick - 1),
                        outcome=outcome,
                        world_transition=transition,
                        prediction_error=errors[tick - 1],
                        event_ids=(f"branch-event-{index}",),
                        assembly_ids=(f"branch-assembly-{index}",),
                        object_ids=(object_id,),
                        relation_ids=(f"agent:sequence:{object_id}",),
                    )
                )
                index += 1
    return tuple(records)


def _rollout(
    rollout_id: str, action_kinds: tuple[str, ...], *, tick: int, prior: float
) -> ImaginedRollout:
    steps = tuple(
        PlanningCandidate(
            candidate_id=f"{rollout_id}-step-{index}",
            action=WorldAction(
                action_id=f"{rollout_id}-action-{index}",
                kind=kind,
                tick=tick + index,
                target_id="branch-holdout",
                parameters={"action_symbol": 10},
                provenance="imagined",
            ),
            predicted_reward=0.05 if rollout_id == "branch-good" else 0.10,
            success_probability=0.50 if rollout_id == "branch-good" else 0.55,
            expected_progress=(index + 1) / len(action_kinds),
        )
        for index, kind in enumerate(action_kinds)
    )
    return ImaginedRollout(
        rollout_id=rollout_id,
        goal_id="branch-goal",
        steps=steps,
        confidence=0.90,
        concept_sequence_affinity=prior,
    )


def evaluate() -> dict[str, object]:
    cue = torch.tensor([1.0, 0.0, 0.0, 0.0])
    records = build_branch_records(cue)
    organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
    concepts = organ.consolidate(records, tick=3)
    if len(concepts) != 1:
        raise RuntimeError(f"expected one shared branch concept, got {len(concepts)}")
    concept = concepts[0]
    initial = records[0].world_transition.before
    common_after = records[0].world_transition.after
    good_suffix = organ.suffix_sequence_affinity(
        concept, GOOD_SEQUENCE[1:], current_state=common_after
    )
    alt_suffix = organ.suffix_sequence_affinity(
        concept, ALT_SEQUENCE[1:], current_state=common_after
    )
    horizon_affinity = {
        str(horizon): organ.suffix_sequence_affinity(
            concept, GOOD_SEQUENCE[:horizon], current_state=initial
        )
        for horizon in (1, 2, 3)
    }
    reordered = organ.suffix_sequence_affinity(
        concept, ("archive", "approach", "confirm"), current_state=initial
    )
    direct = GoalPlanner(PlanningConfig(concept_sequence_weight=2.0)).plan_rollouts(
        GoalState(tick=0, goals=(Goal("branch-goal", "choose the better branch", 1.0),)),
        (
            _rollout("branch-good", GOOD_SEQUENCE[1:], tick=1, prior=good_suffix),
            _rollout("branch-alt", ALT_SEQUENCE[1:], tick=1, prior=alt_suffix),
        ),
        tick=1,
    )
    good_trace = next(
        trace for trace in concept.sequence_traces if trace.action_kinds == GOOD_SEQUENCE
    )
    alt_trace = next(
        trace for trace in concept.sequence_traces if trace.action_kinds == ALT_SEQUENCE
    )
    old_visits = good_trace.visits
    old_credit = good_trace.step_credit[1]
    updates = organ.update_sequence_trace(
        "confirm",
        before_state=common_after,
        after_state=records[1].world_transition.after,
        outcome=Outcome("online-feedback", 2.0, success=True, tick=1),
        prediction_error=0.0,
    )
    updated_trace = next(
        trace for trace in organ.concepts[0].sequence_traces if trace.action_kinds == GOOD_SEQUENCE
    )
    unchanged_alt_trace = next(
        trace for trace in organ.concepts[0].sequence_traces if trace.action_kinds == ALT_SEQUENCE
    )
    lesion = ConceptFormationOrgan.from_checkpoint(organ.checkpoint())
    lesion_ids = lesion.lesion_sequence_traces((concept.concept_id,))
    lesioned_affinity = lesion.suffix_sequence_affinity(
        lesion.concepts[0], GOOD_SEQUENCE[1:], current_state=common_after
    )
    checkpoint = ConceptFormationOrgan.from_checkpoint(organ.checkpoint())
    checkpoint_trace = checkpoint.concepts[0].sequence_traces[0]
    checkpoint_recovery = bool(
        checkpoint_trace.visits == updated_trace.visits
        and checkpoint_trace.step_credit == updated_trace.step_credit
    )
    gate_passed = bool(
        good_suffix > alt_suffix
        and all(float(value) > 0.0 for value in horizon_affinity.values())
        and reordered == 0.0
        and direct.selected.rollout_id == "branch-good"
        and updates == 1
        and updated_trace.visits > old_visits
        and updated_trace.step_credit[1] > old_credit
        and unchanged_alt_trace.visits == alt_trace.visits
        and lesion_ids == (concept.concept_id,)
        and lesioned_affinity == 0.0
        and checkpoint_recovery
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "concept_count": len(concepts),
            "trace_count": len(concept.sequence_traces),
            "good_suffix_affinity": good_suffix,
            "alt_suffix_affinity": alt_suffix,
            "horizon_affinity": horizon_affinity,
            "reordered_affinity": reordered,
            "selected_branch": direct.selected.rollout_id,
            "trace_updates": updates,
            "trace_visits_before": old_visits,
            "trace_visits_after": updated_trace.visits,
            "step_credit_before": old_credit,
            "step_credit_after": updated_trace.step_credit[1],
            "alternative_trace_visits": unchanged_alt_trace.visits,
            "lesioned_affinity": lesioned_affinity,
            "checkpoint_recovery": checkpoint_recovery,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "shared-prefix branches must compete across variable horizons, the better suffix must win from learned after-state and credit, trace lesion must remove the sequence prior, and experienced outcome/error feedback must update and checkpoint the existing trace",
        },
        "boundary": "This is a closed-world branch-memory gate; it does not claim open-domain planning or general intelligence.",
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "variable-horizon shared-prefix branch competition and trace plasticity",
        "branches": {"good": list(GOOD_SEQUENCE), "alternative": list(ALT_SEQUENCE)},
        "feedback": "one experienced branch-specific confirm transition updates only the matching trace",
        "lesion": "sequence_traces disabled while concept identity remains",
        "checkpoint": "ConceptFormationOrgan checkpoint after online update",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_branch_manifest_20260826.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_branch_20260826.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
