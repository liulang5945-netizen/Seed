"""Evaluate state-conditioned suffix retrieval and sequence credit assignment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ActionIntent,
    ConceptFormationOrgan,
    EnvironmentOutcome,
    EpisodicMemoryRecord,
    Goal,
    GoalPlanner,
    ImaginedRollout,
    Observation,
    Outcome,
    PlanningCandidate,
    PlanningConfig,
    TaijiConfig,
    TSKV8Adapter,
    WorldAction,
    WorldObject,
    WorldState,
    WorldTransition,
)

MANIFEST_FORMAT = "taiji-concept-suffix-manifest-v1"
REPORT_FORMAT = "taiji-concept-suffix-v1"
ACTION_SEQUENCE = ("approach", "confirm", "archive")
PREDICTION_ERRORS = (0.0, 0.20, 0.40)


def _runtime_config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(16, 12),
        synapse_fan_in=4,
        motor_fan_in=6,
        memory_units=16,
        memory_fan_in=4,
        memory_readout_fan_in=6,
        memory_meta_dim=6,
        memory_time_dim=4,
        memory_episode_dim=4,
        lateral_fan_in=4,
        concept_capacity=4,
    )


def _state(tick: int, latent: torch.Tensor, object_id: str) -> WorldState:
    return WorldState(
        tick=tick,
        latent=latent.detach().clone(),
        objects=(WorldObject(object_id, attributes={"stage": tick}),),
        relations=(("agent", "sequence", object_id),),
    )


def build_records(
    cue: torch.Tensor,
    *,
    initial_latent: torch.Tensor | None = None,
    episode_count: int = 2,
) -> tuple[EpisodicMemoryRecord, ...]:
    """Build two repeated trajectories with real transition evidence."""

    if episode_count < 2:
        raise ValueError("suffix formation requires at least two episodes")
    dimension = cue.numel()
    if initial_latent is None:
        start = torch.zeros(dimension)
        start[0] = 1.0
    else:
        start = initial_latent.detach().cpu().clone()
    if start.numel() != dimension or float(torch.linalg.vector_norm(start)) == 0.0:
        raise ValueError("initial_latent must be a non-zero vector matching cue")
    basis = tuple(torch.roll(start, shifts=index + 1) for index in range(len(ACTION_SEQUENCE)))
    records: list[EpisodicMemoryRecord] = []
    index = 0
    for episode_index in range(episode_count):
        object_id = f"suffix-object-{episode_index}"
        states = (_state(0, start, object_id),) + tuple(
            _state(tick, basis[tick - 1], object_id) for tick in range(1, len(ACTION_SEQUENCE) + 1)
        )
        for tick, action_kind in enumerate(ACTION_SEQUENCE, start=1):
            intent_id = f"suffix-intent-{index}"
            action = WorldAction(
                action_id=intent_id,
                kind=action_kind,
                tick=tick - 1,
                target_id=object_id,
                provenance="experienced",
            )
            outcome = Outcome(
                intent_id=intent_id,
                reward=(0.10, 0.50, 1.0)[tick - 1],
                success=True,
                tick=tick,
            )
            transition = WorldTransition(
                before=states[tick - 1],
                action=action,
                after=states[tick],
                outcome=outcome,
            )
            records.append(
                EpisodicMemoryRecord(
                    memory_id=f"suffix-memory-{index}",
                    episode_id=f"suffix-episode-{episode_index}",
                    tick=tick,
                    cue=cue.detach().clone(),
                    action_intent=ActionIntent(intent_id, action_kind, tick=tick - 1),
                    outcome=outcome,
                    world_transition=transition,
                    prediction_error=PREDICTION_ERRORS[tick - 1],
                    event_ids=(f"suffix-event-{index}",),
                    assembly_ids=(f"suffix-assembly-{index}",),
                    object_ids=(object_id,),
                    relation_ids=(f"agent:sequence:{object_id}",),
                )
            )
            index += 1
    return tuple(records)


def _candidate(rollout_id: str, index: int, kind: str, tick: int) -> PlanningCandidate:
    return PlanningCandidate(
        candidate_id=f"{rollout_id}-step-{index}",
        action=WorldAction(
            action_id=f"{rollout_id}-action-{index}",
            kind=kind,
            tick=tick + index,
            target_id="suffix-holdout",
            parameters={"action_symbol": 10 + ACTION_SEQUENCE.index(kind)},
            provenance="imagined",
        ),
        predicted_reward=0.05 if rollout_id == "suffix-good" else 0.10,
        success_probability=0.50 if rollout_id == "suffix-good" else 0.55,
        expected_progress=(index + 1) / len(ACTION_SEQUENCE),
    )


def _rollouts(tick: int) -> tuple[ImaginedRollout, ImaginedRollout]:
    good = ImaginedRollout(
        rollout_id="suffix-good",
        goal_id="complete-suffix",
        confidence=0.90,
        steps=tuple(
            _candidate("suffix-good", index, kind, tick)
            for index, kind in enumerate(ACTION_SEQUENCE)
        ),
    )
    reversed_rollout = ImaginedRollout(
        rollout_id="suffix-reversed",
        goal_id="complete-suffix",
        confidence=0.90,
        steps=tuple(
            _candidate("suffix-reversed", index, kind, tick)
            for index, kind in enumerate(reversed(ACTION_SEQUENCE))
        ),
    )
    return good, reversed_rollout


class _SuffixEnvironment:
    def __init__(self, states: tuple[WorldState, ...]) -> None:
        self.states = states
        self.index = 0
        self.actions: list[int] = []

    def reset(self) -> tuple[int, tuple[int, ...]]:
        self.index = 0
        self.actions.clear()
        return 97, (10, 11, 12)

    def step(self, action_symbol: int) -> EnvironmentOutcome:
        if self.index >= len(self.states):
            raise RuntimeError("suffix environment received too many actions")
        self.actions.append(action_symbol)
        state = self.states[self.index]
        self.index += 1
        return EnvironmentOutcome(
            sensation=97 + self.index,
            reward=0.05,
            success=True,
            terminal=self.index == len(self.states),
            world_state=state,
        )


def _runtime_gate() -> dict[str, object]:
    runtime = TSKV8Adapter(_runtime_config(), episode_id="concept-suffix-runtime")
    dimension = runtime.perception.feature_dim
    start = torch.ones(dimension)
    runtime.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="concept-suffix-evaluation",
        ),
        learn=False,
        world_state=_state(1, start, "suffix-holdout"),
    )
    percept = runtime.cognitive_snapshot().percept
    if percept is None:
        raise RuntimeError("suffix runtime did not emit a perception")
    runtime_records = build_records(
        start,
        initial_latent=start,
        episode_count=2,
    )
    runtime.concept_formation.consolidate(runtime_records, tick=runtime.tick)
    runtime._cognitive_state = replace(
        runtime._cognitive_state,
        concepts=runtime.concept_formation.concepts,
    )
    runtime._refresh_concept_memory()
    runtime.attach_goal_planner(GoalPlanner(PlanningConfig(concept_sequence_weight=2.0)))
    runtime.set_goals((Goal("complete-suffix", "complete the learned suffix", 1.0),))
    rollouts = _rollouts(runtime.tick)
    decision = runtime.plan_rollouts(rollouts)
    selected = decision.selected.rollout_id
    checkpoint = runtime.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_selected = restored.plan_rollouts(rollouts).selected.rollout_id
    states = tuple(
        _state(
            runtime.tick + index,
            runtime.concept_formation.concepts[0].sequence_traces[0].after_prototypes[index - 1],
            "suffix-holdout",
        )
        for index in range(1, len(ACTION_SEQUENCE) + 1)
    )
    environment = _SuffixEnvironment(states)
    runtime.execute_imagined_rollout_step(
        environment,
        available_actions=(10, 11, 12),
        action_kinds=ACTION_SEQUENCE,
        learn=False,
    )
    remaining = runtime._planned_rollout
    suffix_affinity = 0.0 if remaining is None else remaining.concept_sequence_affinity
    post_execution = TSKV8Adapter.from_native_checkpoint(runtime.native_checkpoint())
    restored_remaining = post_execution._planned_rollout
    restored_suffix_affinity = (
        0.0 if restored_remaining is None else restored_remaining.concept_sequence_affinity
    )
    return {
        "selected_full_rollout": selected,
        "selected_after_checkpoint": restored_selected,
        "remaining_action_kinds": (
            () if remaining is None else tuple(step.action.kind for step in remaining.steps)
        ),
        "suffix_affinity_after_execution": suffix_affinity,
        "suffix_affinity_after_checkpoint": restored_suffix_affinity,
        "environment_actions": environment.actions,
    }


def evaluate() -> dict[str, object]:
    cue = torch.tensor([1.0, 0.0, 0.0, 0.0])
    records = build_records(cue)
    organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
    concepts = organ.consolidate(records, tick=3)
    if len(concepts) != 1 or not concepts[0].sequence_traces:
        raise RuntimeError("suffix evaluation did not form a sequence trace")
    concept = concepts[0]
    trace = concept.sequence_traces[0]
    initial = records[0].world_transition.before if records[0].world_transition else None
    after_first = records[0].world_transition.after if records[0].world_transition else None
    if initial is None or after_first is None:
        raise RuntimeError("suffix evaluation records lack transition states")
    full_affinity = organ.suffix_sequence_affinity(concept, ACTION_SEQUENCE, current_state=initial)
    suffix_affinity = organ.suffix_sequence_affinity(
        concept, ACTION_SEQUENCE[1:], current_state=after_first
    )
    reversed_affinity = organ.suffix_sequence_affinity(
        concept, tuple(reversed(ACTION_SEQUENCE)), current_state=initial
    )
    reordered_affinity = organ.suffix_sequence_affinity(
        concept,
        (ACTION_SEQUENCE[2], ACTION_SEQUENCE[0], ACTION_SEQUENCE[1]),
        current_state=initial,
    )
    wrong_state = _state(
        initial.tick,
        torch.roll(after_first.latent, shifts=1),
        "suffix-object-wrong-state",
    )
    wrong_state_affinity = organ.suffix_sequence_affinity(
        concept, ACTION_SEQUENCE[1:], current_state=wrong_state
    )
    restored = ConceptFormationOrgan.from_checkpoint(organ.checkpoint())
    restored_trace = restored.concepts[0].sequence_traces[0]
    checkpoint_recovery = bool(
        restored_trace.action_kinds == trace.action_kinds
        and restored_trace.visits == trace.visits
        and torch.equal(restored_trace.before_prototype, trace.before_prototype)
        and all(
            torch.equal(left, right)
            for left, right in zip(
                restored_trace.after_prototypes, trace.after_prototypes, strict=True
            )
        )
    )
    runtime = _runtime_gate()
    gate_passed = bool(
        full_affinity > 0.0
        and suffix_affinity > reversed_affinity
        and suffix_affinity > wrong_state_affinity
        and reordered_affinity == 0.0
        and max(trace.step_credit) > min(trace.step_credit)
        and checkpoint_recovery
        and runtime["selected_full_rollout"] == "suffix-good"
        and runtime["selected_after_checkpoint"] == "suffix-good"
        and runtime["remaining_action_kinds"] == ACTION_SEQUENCE[1:]
        and float(runtime["suffix_affinity_after_execution"]) > 0.0
        and runtime["suffix_affinity_after_checkpoint"]
        == runtime["suffix_affinity_after_execution"]
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "concept_count": len(concepts),
            "trace_visits": trace.visits,
            "trace_action_kinds": trace.action_kinds,
            "trace_step_credit": trace.step_credit,
            "trace_prediction_errors": trace.prediction_errors,
            "full_affinity": full_affinity,
            "suffix_affinity": suffix_affinity,
            "reversed_affinity": reversed_affinity,
            "reordered_affinity": reordered_affinity,
            "wrong_state_affinity": wrong_state_affinity,
            "checkpoint_recovery": checkpoint_recovery,
            "runtime": runtime,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "learn after-state-conditioned sequence traces from real transitions, assign future-weighted credit using outcomes and prediction errors, retrieve a remaining suffix after partial execution, and preserve the trace and suffix affinity through native checkpoint recovery",
        },
        "boundary": "This is a closed-world sequence-memory gate; it does not claim open-domain planning or general intelligence.",
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "state-conditioned suffix retrieval and sequence-level credit assignment",
        "records": "two episodes with real WorldTransition after-states, bounded outcomes, and prediction errors",
        "lesions": ["reversed-sequence", "wrong-after-state", "concept-trace-checkpoint"],
        "checkpoint": "ConceptFormationOrgan and native TSKV8 runtime",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_suffix_manifest_20260826.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_suffix_20260826.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
