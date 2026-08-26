"""Evaluate unique concept ownership for online sequence branch birth."""

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
    ConceptMatch,
    EpisodicMemoryRecord,
    Observation,
    Outcome,
    TaijiConfig,
    TSKV8Adapter,
    WorldAction,
    WorldObject,
    WorldState,
    WorldTransition,
)

MANIFEST_FORMAT = "taiji-concept-branch-attribution-manifest-v1"
REPORT_FORMAT = "taiji-concept-branch-attribution-v1"


def _state(tick: int, latent: torch.Tensor, object_id: str) -> WorldState:
    return WorldState(
        tick=tick,
        latent=latent.detach().clone(),
        objects=(WorldObject(object_id, attributes={"stage": tick}),),
        relations=(("agent", "sequence", object_id),),
    )


def _records(
    cue: torch.Tensor,
    *,
    prefix: str,
    rescue_after: torch.Tensor,
    handoff_after: torch.Tensor,
) -> tuple[EpisodicMemoryRecord, ...]:
    actions = ("approach", "rescue", "handoff", "archive")
    start = torch.tensor([1.0, 0.0, 0.0, 0.0])
    common_after = torch.tensor([0.0, 1.0, 0.0, 0.0])
    archive_after = torch.tensor([1.0, 0.0, 0.0, 0.0])
    records: list[EpisodicMemoryRecord] = []
    for episode_index in range(2):
        object_id = f"{prefix}-object-{episode_index}"
        states = (
            _state(0, start, object_id),
            _state(1, common_after, object_id),
            _state(2, rescue_after, object_id),
            _state(3, handoff_after, object_id),
            _state(4, archive_after, object_id),
        )
        for index, action_kind in enumerate(actions):
            intent_id = f"{prefix}-intent-{episode_index}-{index}"
            outcome = Outcome(intent_id, 1.0, success=True, tick=index + 1)
            transition = WorldTransition(
                states[index],
                WorldAction(intent_id, action_kind, index, target_id=object_id),
                states[index + 1],
                outcome,
            )
            records.append(
                EpisodicMemoryRecord(
                    memory_id=f"{prefix}-memory-{episode_index}-{index}",
                    episode_id=f"{prefix}-episode-{episode_index}",
                    tick=index + 1,
                    cue=cue.detach().clone(),
                    action_intent=ActionIntent(intent_id, action_kind, tick=index),
                    outcome=outcome,
                    world_transition=transition,
                    prediction_error=0.05,
                    event_ids=(f"{prefix}-event-{episode_index}-{index}",),
                    assembly_ids=(f"{prefix}-assembly-{episode_index}-{index}",),
                    object_ids=(object_id,),
                    relation_ids=(f"agent:sequence:{object_id}",),
                )
            )
    return tuple(records)


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
        concept_branch_owner_min_score=0.65,
        concept_branch_owner_min_margin=0.05,
    )


def _transition(
    before: WorldState,
    action_kind: str,
    after: WorldState,
    *,
    intent_id: str,
) -> WorldTransition:
    return WorldTransition(
        before,
        WorldAction(intent_id, action_kind, before.tick, target_id="holdout"),
        after,
        Outcome(intent_id, 1.0, success=True, tick=after.tick),
    )


def evaluate() -> dict[str, object]:
    cue_a = torch.tensor([0.95, 0.3122499, 0.0, 0.0])
    cue_b = torch.tensor([0.95, -0.3122499, 0.0, 0.0])
    records_a = _records(
        cue_a,
        prefix="owner-a",
        rescue_after=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        handoff_after=torch.tensor([0.0, 0.0, -1.0, 0.0]),
    )
    records_b = _records(
        cue_b,
        prefix="owner-b",
        rescue_after=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        handoff_after=torch.tensor([0.0, 0.0, 0.0, -1.0]),
    )
    organ = ConceptFormationOrgan(
        capacity=4,
        similarity_threshold=0.85,
        prune_threshold=0.0,
        trace_capacity=8,
    )
    concepts = organ.consolidate((*records_a, *records_b), tick=4)
    if len(concepts) != 2:
        raise RuntimeError(f"expected two competing concepts, got {len(concepts)}")
    concept_a = next(
        concept for concept in concepts if any(item.startswith("owner-a-") for item in concept.object_ids)
    )
    concept_b = next(
        concept for concept in concepts if any(item.startswith("owner-b-") for item in concept.object_ids)
    )
    matches = (ConceptMatch(concept_a, 0.95), ConceptMatch(concept_b, 0.95))
    common = _state(1, torch.tensor([0.0, 1.0, 0.0, 0.0]), "holdout")
    rescue_a = _state(2, torch.tensor([0.0, 0.0, 1.0, 0.0]), "holdout")
    handoff_a = _state(3, torch.tensor([0.0, 0.0, -1.0, 0.0]), "holdout")
    ambiguous = _state(2, torch.tensor([0.0, 0.0, 0.7071068, 0.7071068]), "holdout")
    owner_transition = _transition(common, "rescue", rescue_a, intent_id="owner-rescue")
    handoff_transition = _transition(rescue_a, "handoff", handoff_a, intent_id="owner-handoff")
    ambiguous_transition = _transition(common, "rescue", ambiguous, intent_id="owner-ambiguous")
    owner_rescue = organ.select_sequence_owner(matches, owner_transition, 0.05)
    owner_handoff = organ.select_sequence_owner(matches, handoff_transition, 0.05)
    low_confidence = organ.select_sequence_owner(
        (ConceptMatch(concept_a, 0.64), ConceptMatch(concept_b, 0.64)),
        owner_transition,
        0.05,
    )
    interference = organ.select_sequence_owner(matches, ambiguous_transition, 0.05)
    lesioned = ConceptFormationOrgan.from_checkpoint(organ.checkpoint())
    lesioned.lesion_sequence_traces((concept_a.concept_id,))
    owner_after_lesion = lesioned.select_sequence_owner(
        (
            ConceptMatch(lesioned.concepts[0], 0.95),
            ConceptMatch(lesioned.concepts[1], 0.95),
        ),
        owner_transition,
        0.05,
    )

    runtime = TSKV8Adapter(_runtime_config(), episode_id="branch-attribution-runtime")
    runtime.concept_formation.consolidate((*records_a, *records_b), tick=4)
    runtime.observe_event(
        Observation("text-byte", 97, timestamp=0, source="branch-attribution"),
        learn=False,
        world_state=common,
    )
    runtime._cognitive_state = replace(
        runtime._cognitive_state,
        concepts=runtime.concept_formation.concepts,
        memory=replace(
            runtime._cognitive_state.memory,
            concept_ids=tuple(concept.concept_id for concept in runtime.concept_formation.concepts),
            concept_confidence=1.0,
        ),
    )
    runtime.act((10,), procedural_action_kinds=("rescue",))
    runtime.settle_action(
        1.0,
        world_state=rescue_a,
        success=True,
        terminal=False,
        learn=False,
    )
    mid_checkpoint = runtime.native_checkpoint()
    recovered = TSKV8Adapter.from_native_checkpoint(mid_checkpoint)
    recovered.observe_event(
        Observation("text-byte", 98, timestamp=1, source="branch-attribution"),
        learn=False,
        world_state=rescue_a,
    )
    recovered._cognitive_state = replace(
        recovered._cognitive_state,
        concepts=recovered.concept_formation.concepts,
        memory=replace(
            recovered._cognitive_state.memory,
            concept_ids=tuple(concept.concept_id for concept in recovered.concept_formation.concepts),
            concept_confidence=1.0,
        ),
    )
    recovered.act((11,), procedural_action_kinds=("handoff",))
    recovered.settle_action(
        1.0,
        world_state=handoff_a,
        success=True,
        terminal=True,
        learn=False,
    )
    recovered_concepts = recovered.concept_formation.concepts
    recovered_a = next(
        concept
        for concept in recovered_concepts
        if any(item.startswith("owner-a-") for item in concept.object_ids)
    )
    recovered_b = next(
        concept
        for concept in recovered_concepts
        if any(item.startswith("owner-b-") for item in concept.object_ids)
    )
    born_trace_ids = tuple(trace.trace_id for trace in recovered_a.sequence_traces)
    checkpointed = TSKV8Adapter.from_native_checkpoint(recovered.native_checkpoint())
    checkpointed_a = next(
        concept
        for concept in checkpointed.concept_formation.concepts
        if any(item.startswith("owner-a-") for item in concept.object_ids)
    )
    automatic_trace_ids = tuple(trace.trace_id for trace in checkpointed_a.sequence_traces)
    gate_passed = bool(
        owner_rescue == concept_a.concept_id
        and owner_handoff == concept_a.concept_id
        and low_confidence is None
        and interference is None
        and owner_after_lesion is None
        and len(TSKV8Adapter.from_native_checkpoint(mid_checkpoint)._online_concept_branches) == 1
        and len(recovered_a.sequence_traces) == len(concept_a.sequence_traces) + 1
        and len(recovered_b.sequence_traces) == len(concept_b.sequence_traces)
        and len(born_trace_ids) == len(automatic_trace_ids)
        and born_trace_ids == automatic_trace_ids
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "concept_count": len(concepts),
            "owner_rescue": owner_rescue,
            "owner_handoff": owner_handoff,
            "low_confidence_owner": low_confidence,
            "interference_owner": interference,
            "owner_after_lesion": owner_after_lesion,
            "buffer_owner_count_before_checkpoint": len(runtime._online_concept_branches),
            "owner_trace_count_before_birth": len(concept_a.sequence_traces),
            "owner_trace_count_after_birth": len(recovered_a.sequence_traces),
            "other_trace_count_after_birth": len(recovered_b.sequence_traces),
            "automatic_trace_ids": list(automatic_trace_ids),
            "checkpoint_recovery": born_trace_ids == automatic_trace_ids,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "simultaneously active concepts must select one owner from match confidence, learned after-state and prediction-error fit; low confidence, close competition and owner lesion must fail closed, while buffered birth and checkpoint continuation remain intact",
        },
        "boundary": "This is a closed-world branch-attribution gate; it does not claim open-domain planning or general intelligence.",
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "unique owner attribution for online concept branch birth",
        "owner_signals": ["concept match confidence", "learned before/after state", "prediction-error fit"],
        "fail_closed": ["low confidence", "close cross-concept competition", "owner trace lesion"],
        "continuation": "episode buffer and native checkpoint must preserve the selected owner only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("reports/taiji_concept_branch_attribution_20260826.json"))
    parser.add_argument("--manifest", type=Path, default=Path("plans/manifests/taiji_concept_branch_attribution_v1.json"))
    args = parser.parse_args()
    report = evaluate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
