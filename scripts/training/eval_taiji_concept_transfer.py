"""Evaluate concept transfer from learned schemas into downstream planning.

The evaluation deliberately keeps the learned content small and explicit.  A
concept is formed from repeated outcomes, queried with unseen object
identities, and consumed as a configurable planning prior.  The report also
checks capacity interference, evidence lesions, and native runtime recovery.
"""

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
    Outcome,
    PlanningCandidate,
    PlanningConfig,
    TaijiConfig,
    TSKV8Adapter,
    WorldAction,
)

MANIFEST_FORMAT = "taiji-concept-transfer-manifest-v1"
REPORT_FORMAT = "taiji-concept-transfer-v1"
SCHEMA_SCALES = (1, 2, 4, 8)


def _record(
    *,
    index: int,
    episode_id: str,
    cue: torch.Tensor,
    relation: str,
    object_id: str,
    action_kind: str,
) -> EpisodicMemoryRecord:
    intent_id = f"transfer-intent-{index}"
    return EpisodicMemoryRecord(
        memory_id=f"transfer-memory-{index}",
        episode_id=episode_id,
        tick=1,
        cue=cue,
        action_intent=ActionIntent(intent_id, action_kind, tick=0),
        outcome=Outcome(intent_id, reward=1.0, success=True, tick=1),
        event_ids=(f"transfer-event-{index}",),
        assembly_ids=(f"transfer-assembly-{index}",),
        object_ids=(object_id,),
        relation_ids=(f"agent:{relation}:{object_id}",),
    )


def build_training_records(schema_count: int = 4) -> tuple[EpisodicMemoryRecord, ...]:
    """Build two learned schemas with varied object identities.

    The action names are evidence from the records, not a fixed action table:
    they are carried by the experienced intents and later attached to the
    resulting concept.  Each semantic schema has two episodes per object so
    formation cannot succeed from a single encounter.
    """

    if schema_count <= 0:
        raise ValueError("schema_count must be positive")
    patterns = (
        (torch.tensor([1.0, 0.0]), "near", "approach"),
        (torch.tensor([-1.0, 0.0]), "far", "wait"),
    )
    records: list[EpisodicMemoryRecord] = []
    index = 0
    for schema_index in range(schema_count):
        for cue, relation, action_kind in patterns:
            for repeat in range(2):
                records.append(
                    _record(
                        index=index,
                        episode_id=f"train-schema-{schema_index}-{relation}-{repeat}",
                        cue=cue,
                        relation=relation,
                        object_id=f"train-object-{schema_index}-{relation}",
                        action_kind=action_kind,
                    )
                )
                index += 1
    return tuple(records)


def _query_for(relation: str) -> tuple[torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    cue = torch.tensor([1.0, 0.0]) if relation == "near" else torch.tensor([-1.0, 0.0])
    object_id = f"holdout-object-{relation}"
    return cue, (object_id,), (f"agent:{relation}:{object_id}",)


def _concept_prior(
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


def _plan_transfer(
    organ: ConceptFormationOrgan,
    *,
    relation: str,
    use_concept_prior: bool = True,
) -> dict[str, object]:
    cue, object_ids, relation_ids = _query_for(relation)
    target_kind = "approach" if relation == "near" else "wait"
    target_prior = (
        _concept_prior(
            organ,
            cue=cue,
            object_ids=object_ids,
            relation_ids=relation_ids,
            action_kind=target_kind,
        )
        if use_concept_prior
        else 0.0
    )
    target = PlanningCandidate(
        candidate_id=f"holdout-{relation}-target",
        action=WorldAction(
            action_id=f"holdout-action-{relation}",
            kind=target_kind,
            tick=0,
            target_id=object_ids[0],
        ),
        predicted_reward=0.05,
        success_probability=0.50,
        expected_progress=0.50,
        concept_affinity=target_prior,
    )
    distractor_kind = "wait" if relation == "near" else "approach"
    distractor = PlanningCandidate(
        candidate_id=f"holdout-{relation}-distractor",
        action=WorldAction(
            action_id=f"holdout-distractor-{relation}",
            kind=distractor_kind,
            tick=0,
            target_id=object_ids[0],
        ),
        predicted_reward=0.30,
        success_probability=0.65,
        expected_progress=0.50,
    )
    decision = GoalPlanner(PlanningConfig(concept_weight=0.40)).plan(
        GoalState(tick=0, goals=(Goal("holdout-goal", "complete the holdout task", 1.0),)),
        (target, distractor),
        tick=0,
    )
    return {
        "relation": relation,
        "target_kind": target_kind,
        "selected_kind": decision.selected.action.kind,
        "selected_candidate": decision.selected.candidate_id,
        "target_prior": target_prior,
        "transferred": decision.selected.action.kind == target_kind,
    }


def _signal_lesion_results(records: tuple[EpisodicMemoryRecord, ...]) -> dict[str, bool]:
    """Ablate each evidence family before formation and require failure closed."""

    variants = {
        "event_assembly": tuple(
            replace(record, event_ids=(), assembly_ids=()) for record in records
        ),
        "world": tuple(
            replace(record, object_ids=(), relation_ids=())
            if record.memory_id.endswith(("0", "2", "4", "6"))
            else record
            for record in records
        ),
        "outcome": tuple(
            replace(
                record,
                outcome=replace(record.outcome, reward=-1.0, success=False)
                if record.memory_id.endswith(("0", "2", "4", "6"))
                else record.outcome,
            )
            for record in records
        ),
    }
    results: dict[str, bool] = {}
    for name, variant in variants.items():
        organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
        results[name] = len(organ.consolidate(variant, tick=1)) == 0
    return results


def _runtime_checkpoint_recovery(
    records: tuple[EpisodicMemoryRecord, ...],
) -> bool:
    """Verify the concept organ survives the real Taiji native checkpoint."""

    config = TaijiConfig(
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
    runtime = TSKV8Adapter(config, episode_id="concept-transfer-runtime")
    before = runtime.concept_formation.consolidate(records, tick=1)
    checkpoint = runtime.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    after = restored.concept_formation.concepts
    if tuple(item.concept_id for item in before) != tuple(item.concept_id for item in after):
        return False
    cue, object_ids, relation_ids = _query_for("near")
    return bool(
        restored.concept_formation.retrieve(
            cue,
            object_ids=object_ids,
            relation_ids=relation_ids,
            limit=4,
        )
    )


def evaluate_schema_scale(schema_count: int) -> dict[str, object]:
    records = build_training_records(schema_count)
    organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
    concepts = organ.consolidate(records, tick=1)
    near = _plan_transfer(organ, relation="near")
    far = _plan_transfer(organ, relation="far")
    return {
        "schema_count": schema_count,
        "train_records": len(records),
        "concept_count": len(concepts),
        "holdout_tasks": [near, far],
        "holdout_transfer_rate": sum(
            bool(item["transferred"]) for item in (near, far)
        )
        / 2.0,
    }


def evaluate_capacity_interference(records: tuple[EpisodicMemoryRecord, ...]) -> dict[str, object]:
    results = []
    for capacity in (1, 2):
        organ = ConceptFormationOrgan(capacity=capacity, prune_threshold=0.0)
        concepts = organ.consolidate(records, tick=1)
        tasks = (
            _plan_transfer(organ, relation="near"),
            _plan_transfer(organ, relation="far"),
        )
        results.append(
            {
                "capacity": capacity,
                "retained_concepts": len(concepts),
                "holdout_hits": sum(bool(item["transferred"]) for item in tasks),
            }
        )
    small, full = results
    return {
        "curve": results,
        "capacity_respected": all(
            int(item["retained_concepts"]) <= int(item["capacity"]) for item in results
        ),
        "interference_detected": int(full["holdout_hits"]) > int(small["holdout_hits"]),
    }


def evaluate() -> dict[str, object]:
    records = build_training_records(4)
    organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
    concepts = organ.consolidate(records, tick=1)
    baseline_near = _plan_transfer(organ, relation="near", use_concept_prior=False)
    transferred_near = _plan_transfer(organ, relation="near")
    transferred_far = _plan_transfer(organ, relation="far")
    checkpoint = ConceptFormationOrgan.from_checkpoint(organ.checkpoint())
    checkpoint_recovery = (
        tuple(item.concept_id for item in checkpoint.concepts)
        == tuple(item.concept_id for item in concepts)
        and len(checkpoint.concepts) == len(concepts)
        and all(
            left.concept_id == right.concept_id
            and torch.equal(left.prototype, right.prototype)
            and left.support_event_ids == right.support_event_ids
            and left.support_assembly_ids == right.support_assembly_ids
            and left.object_ids == right.object_ids
            and left.relation_ids == right.relation_ids
            and left.action_kinds == right.action_kinds
            and left.update_count == right.update_count
            for left, right in zip(checkpoint.concepts, concepts)
        )
    )

    lesion_organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
    lesion_concepts = lesion_organ.consolidate(records, tick=1)
    target_concept = next(item for item in lesion_concepts if "approach" in item.action_kinds)
    pre_lesion = _plan_transfer(lesion_organ, relation="near")
    removed = lesion_organ.lesion((target_concept.concept_id,))
    post_lesion = _plan_transfer(lesion_organ, relation="near")
    concept_lesion = bool(
        removed == (target_concept.concept_id,)
        and bool(pre_lesion["transferred"])
        and not bool(post_lesion["transferred"])
        and float(post_lesion["target_prior"]) == 0.0
    )

    schema_scale = [evaluate_schema_scale(count) for count in SCHEMA_SCALES]
    capacity = evaluate_capacity_interference(records)
    # Use the minimum two-episode-per-concept corpus for lesion controls.  On
    # the schema-scale corpus, removing one signal from only one encounter
    # would correctly leave enough intact encounters to consolidate the same
    # concept, which is an interference test rather than a signal lesion.
    signal_lesions = _signal_lesion_results(build_training_records(1))
    runtime_recovery = _runtime_checkpoint_recovery(records)
    gate_passed = bool(
        len(concepts) == 2
        and not bool(baseline_near["transferred"])
        and bool(transferred_near["transferred"])
        and bool(transferred_far["transferred"])
        and all(float(item["holdout_transfer_rate"]) == 1.0 for item in schema_scale)
        and bool(capacity["capacity_respected"])
        and bool(capacity["interference_detected"])
        and all(signal_lesions.values())
        and checkpoint_recovery
        and runtime_recovery
        and concept_lesion
    )
    return {
        "format": REPORT_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "train_records": len(records),
        "candidate_concepts": 2,
        "metrics": {
            "cross_schema_identity": len(concepts) == 2,
            "baseline_without_concept_prior": baseline_near,
            "unseen_task_transfer": {
                "near": transferred_near,
                "far": transferred_far,
                "rate": sum(
                    bool(item["transferred"])
                    for item in (transferred_near, transferred_far)
                )
                / 2.0,
            },
            "schema_scale": schema_scale,
            "capacity_interference": capacity,
            "signal_lesions": signal_lesions,
            "concept_lesion": {
                "passed": concept_lesion,
                "removed": list(removed),
                "before": pre_lesion,
                "after": post_lesion,
            },
            "organ_checkpoint_continuation": checkpoint_recovery,
            "native_runtime_checkpoint_recovery": runtime_recovery,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "learned concepts must transfer to unseen object schemas and downstream planning, expose bounded capacity interference, fail closed under event/world/outcome evidence lesions, and survive native runtime checkpoint recovery",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "evaluate concept transfer from cross-schema experience into unseen-task planning",
        "schema_scales": list(SCHEMA_SCALES),
        "controls": [
            "no-concept-prior baseline",
            "unseen-object-and-relation-schema transfer",
            "capacity-interference curve",
            "event-assembly/world/outcome evidence lesions",
            "organ-checkpoint-continuation",
            "native-runtime-checkpoint-recovery",
            "concept-registry-lesion",
        ],
        "boundary": "closed-world downstream transfer evidence; not open-domain semantic competence or general intelligence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_transfer_manifest_20260826.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_transfer_20260826.json",
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
