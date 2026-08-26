"""Evaluate ordered concept sequences in multi-step Taiji rollouts."""

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
    EpisodicMemoryRecord,
    Goal,
    GoalPlanner,
    GoalState,
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
)

MANIFEST_FORMAT = "taiji-concept-sequence-manifest-v1"
REPORT_FORMAT = "taiji-concept-sequence-v1"
SCHEMA_SCALES = (1, 2, 4, 8)


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


def build_sequence_records(
    cue: torch.Tensor, schema_count: int = 4
) -> tuple[EpisodicMemoryRecord, ...]:
    """Build repeated ordered trajectories with varied object identities."""

    if schema_count <= 0:
        raise ValueError("schema_count must be positive")
    patterns = (
        (cue, "near", ("approach", "confirm")),
        (-cue, "far", ("wait", "inspect")),
    )
    records: list[EpisodicMemoryRecord] = []
    index = 0
    for schema_index in range(schema_count):
        for pattern_cue, relation, action_sequence in patterns:
            object_id = f"sequence-object-{schema_index}-{relation}"
            for repeat in range(2):
                episode_id = f"sequence-schema-{schema_index}-{relation}-{repeat}"
                for tick, action_kind in enumerate(action_sequence, start=1):
                    intent_id = f"sequence-intent-{index}"
                    records.append(
                        EpisodicMemoryRecord(
                            memory_id=f"sequence-memory-{index}",
                            episode_id=episode_id,
                            tick=tick,
                            cue=pattern_cue,
                            action_intent=ActionIntent(intent_id, action_kind, tick=tick - 1),
                            outcome=Outcome(intent_id, reward=1.0, success=True, tick=tick),
                            event_ids=(f"sequence-event-{index}",),
                            assembly_ids=(f"sequence-assembly-{index}",),
                            object_ids=(object_id,),
                            relation_ids=(f"agent:{relation}:{object_id}",),
                        )
                    )
                    index += 1
    return tuple(records)


def _query(relation: str) -> tuple[torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    cue = torch.tensor([1.0, 0.0]) if relation == "near" else torch.tensor([-1.0, 0.0])
    object_id = f"sequence-holdout-{relation}"
    return cue, (object_id,), (f"agent:{relation}:{object_id}",)


def _match_prior(
    organ: ConceptFormationOrgan,
    *,
    cue: torch.Tensor,
    object_ids: tuple[str, ...],
    relation_ids: tuple[str, ...],
    action_kind: str,
) -> float:
    matches = organ.retrieve(
        cue,
        object_ids=object_ids,
        relation_ids=relation_ids,
        limit=organ.capacity,
    )
    return max(
        (
            match.score * match.concept.confidence * match.concept.outcome_mean
            for match in matches
            if action_kind in match.concept.action_kinds
        ),
        default=0.0,
    )


def _candidate(
    rollout_id: str,
    index: int,
    kind: str,
    *,
    tick: int,
    predicted_reward: float,
    success_probability: float,
    concept_affinity: float,
) -> PlanningCandidate:
    return PlanningCandidate(
        candidate_id=f"{rollout_id}-step-{index}",
        action=WorldAction(
            action_id=f"{rollout_id}-action-{index}",
            kind=kind,
            tick=tick,
            target_id="sequence-holdout-near",
            provenance="imagined",
        ),
        predicted_reward=predicted_reward,
        success_probability=success_probability,
        expected_progress=0.50,
        concept_affinity=concept_affinity,
    )


def _rollouts(
    organ: ConceptFormationOrgan,
    *,
    cue: torch.Tensor,
    object_ids: tuple[str, ...],
    relation_ids: tuple[str, ...],
    tick: int,
) -> tuple[ImaginedRollout, ImaginedRollout]:
    good_kinds = ("approach", "confirm")
    reversed_kinds = tuple(reversed(good_kinds))
    good_prior = tuple(
        _match_prior(
            organ,
            cue=cue,
            object_ids=object_ids,
            relation_ids=relation_ids,
            action_kind=kind,
        )
        for kind in good_kinds
    )
    good_cue_matches = organ.retrieve(
        cue,
        object_ids=object_ids,
        relation_ids=relation_ids,
        limit=organ.capacity,
    )
    sequence_prior = max(
        (
            match.score
            * match.concept.confidence
            * match.concept.outcome_mean
            * organ.action_sequence_affinity(match.concept, good_kinds)
            for match in good_cue_matches
        ),
        default=0.0,
    )
    reversed_prior = max(
        (
            match.score
            * match.concept.confidence
            * match.concept.outcome_mean
            * organ.action_sequence_affinity(match.concept, reversed_kinds)
            for match in good_cue_matches
        ),
        default=0.0,
    )
    good = ImaginedRollout(
        rollout_id="sequence-good",
        goal_id="complete-sequence",
        confidence=0.90,
        concept_sequence_affinity=sequence_prior,
        steps=tuple(
            _candidate(
                "sequence-good",
                index,
                kind,
                tick=tick,
                predicted_reward=0.05,
                success_probability=0.50,
                concept_affinity=good_prior[index],
            )
            for index, kind in enumerate(good_kinds)
        ),
    )
    reversed_rollout = ImaginedRollout(
        rollout_id="sequence-reversed",
        goal_id="complete-sequence",
        confidence=0.90,
        concept_sequence_affinity=reversed_prior,
        steps=tuple(
            _candidate(
                "sequence-reversed",
                index,
                kind,
                tick=tick,
                predicted_reward=0.25,
                success_probability=0.65,
                concept_affinity=_match_prior(
                    organ,
                    cue=cue,
                    object_ids=object_ids,
                    relation_ids=relation_ids,
                    action_kind=kind,
                ),
            )
            for index, kind in enumerate(reversed_kinds)
        ),
    )
    return reversed_rollout, good


def _plan(rollouts: tuple[ImaginedRollout, ImaginedRollout], *, sequence_weight: float) -> str:
    decision = GoalPlanner(
        PlanningConfig(concept_weight=0.40, concept_sequence_weight=sequence_weight)
    ).plan_rollouts(
        GoalState(
            tick=0,
            goals=(Goal("complete-sequence", "complete the learned sequence", 1.0),),
        ),
        rollouts,
        tick=0,
    )
    return decision.selected.rollout_id


def _runtime_checkpoint_and_feedback(
    organ_records: tuple[EpisodicMemoryRecord, ...],
) -> dict[str, bool]:
    runtime = TSKV8Adapter(_runtime_config(), episode_id="concept-sequence-runtime")
    runtime.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="concept-sequence-evaluation",
        ),
        learn=False,
        world_state=WorldState(
            tick=runtime.tick + 1,
            latent=torch.zeros(runtime.perception.feature_dim),
            objects=(WorldObject("sequence-holdout-near"),),
            relations=(("agent", "near", "sequence-holdout-near"),),
        ),
    )
    percept = runtime.cognitive_snapshot().percept
    if percept is None:
        raise RuntimeError("sequence runtime did not emit a perception")
    records = tuple(
        replace(record, cue=percept.features.detach().clone()) for record in organ_records
    )
    runtime.concept_formation.consolidate(records, tick=runtime.tick)
    runtime.attach_goal_planner(
        GoalPlanner(PlanningConfig(concept_weight=0.40, concept_sequence_weight=0.80))
    )
    runtime.set_goals((Goal("complete-sequence", "complete the learned sequence", 1.0),))
    cue = percept.features.detach().clone()
    rollouts = _rollouts(
        runtime.concept_formation,
        cue=cue,
        object_ids=("sequence-holdout-near",),
        relation_ids=("agent:near:sequence-holdout-near",),
        tick=runtime.tick,
    )
    decision = runtime.plan_rollouts(rollouts)
    selected_before_failure = decision.selected.rollout_id == "sequence-good"
    checkpoint = runtime.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_decision = restored.plan_rollouts(rollouts)
    checkpoint_recovery = restored_decision.selected.rollout_id == "sequence-good"

    runtime.observe(98, learn=False)
    runtime.act(
        (10, 11),
        procedural_action_kinds=("approach", "confirm"),
        use_plan=True,
    )
    runtime.settle_action(-1.0, success=False, learn=False)
    feedback_replan = runtime.replan_required
    feedback_checkpoint = TSKV8Adapter.from_native_checkpoint(runtime.native_checkpoint())
    return {
        "selected_before_failure": selected_before_failure,
        "checkpoint_recovery": checkpoint_recovery,
        "feedback_replan": feedback_replan,
        "feedback_checkpoint_recovery": feedback_checkpoint.replan_required == feedback_replan,
    }


def evaluate() -> dict[str, object]:
    base_cue = torch.tensor([1.0, 0.0])
    records = build_sequence_records(base_cue, schema_count=4)
    organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
    concepts = organ.consolidate(records, tick=1)
    cue, object_ids, relation_ids = _query("near")
    near_concept = next(item for item in concepts if "approach" in item.action_kinds)
    rollouts = _rollouts(
        organ,
        cue=cue,
        object_ids=object_ids,
        relation_ids=relation_ids,
        tick=0,
    )
    baseline_selected = _plan(rollouts, sequence_weight=0.0)
    sequence_selected = _plan(rollouts, sequence_weight=0.80)
    prefix_affinity = organ.action_sequence_affinity(near_concept, ("approach",))
    reversed_affinity = organ.action_sequence_affinity(near_concept, ("confirm", "approach"))
    checkpoint = ConceptFormationOrgan.from_checkpoint(organ.checkpoint())
    checkpoint_recovery = (
        tuple(item.concept_id for item in checkpoint.concepts)
        == tuple(item.concept_id for item in concepts)
        and checkpoint.concepts[0].action_sequences == concepts[0].action_sequences
    )
    lesion = ConceptFormationOrgan.from_checkpoint(organ.checkpoint())
    lesion.lesion((near_concept.concept_id,))
    lesioned_rollouts = _rollouts(
        lesion,
        cue=cue,
        object_ids=object_ids,
        relation_ids=relation_ids,
        tick=0,
    )
    lesion_selected = _plan(lesioned_rollouts, sequence_weight=0.80)
    runtime = _runtime_checkpoint_and_feedback(build_sequence_records(base_cue, schema_count=2))
    schema_scale = []
    for schema_count in SCHEMA_SCALES:
        scaled_records = build_sequence_records(base_cue, schema_count=schema_count)
        scaled_organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
        scaled_concepts = scaled_organ.consolidate(scaled_records, tick=1)
        scaled_cue, scaled_objects, scaled_relations = _query("near")
        scaled_rollouts = _rollouts(
            scaled_organ,
            cue=scaled_cue,
            object_ids=scaled_objects,
            relation_ids=scaled_relations,
            tick=0,
        )
        schema_scale.append(
            {
                "schema_count": schema_count,
                "train_records": len(scaled_records),
                "concept_count": len(scaled_concepts),
                "selected": _plan(scaled_rollouts, sequence_weight=0.80),
            }
        )
    gate_passed = bool(
        len(concepts) == 2
        and baseline_selected == "sequence-reversed"
        and sequence_selected == "sequence-good"
        and prefix_affinity > reversed_affinity
        and reversed_affinity == 0.0
        and checkpoint_recovery
        and lesion_selected == "sequence-reversed"
        and all(item["selected"] == "sequence-good" for item in schema_scale)
        and all(runtime.values())
    )
    return {
        "format": REPORT_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "train_records": len(records),
        "candidate_concepts": len(concepts),
        "metrics": {
            "baseline_without_sequence_prior": baseline_selected,
            "sequence_prior_selection": sequence_selected,
            "learned_prefix_affinity": prefix_affinity,
            "reversed_sequence_affinity": reversed_affinity,
            "concept_lesion_selection": lesion_selected,
            "organ_checkpoint_continuation": checkpoint_recovery,
            "schema_scale": schema_scale,
            "native_runtime": runtime,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "ordered action sequences learned from episode timelines must select the correct multi-step rollout over a higher-immediate-value reversal, transfer across unseen schemas, fail after concept lesion, and preserve runtime recovery with execution-feedback replanning",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "evaluate ordered concept sequences in multi-step imagined planning",
        "schema_scales": list(SCHEMA_SCALES),
        "controls": [
            "no-sequence-prior rollout baseline",
            "ordered-vs-reversed sequence",
            "variable-horizon prefix affinity",
            "concept registry lesion",
            "schema-scale transfer",
            "organ and native runtime checkpoint recovery",
            "execution-feedback replan",
        ],
        "boundary": "closed-world ordered rollout transfer; not open-domain planning or general intelligence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_sequence_manifest_20260826.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_sequence_20260826.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
