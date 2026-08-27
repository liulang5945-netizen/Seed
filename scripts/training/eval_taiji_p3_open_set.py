"""Evaluate open-set world-schema evolution across carried episodes.

The benchmark starts from a one-action training schema, then presents a new
target object, new action kinds, and new relation predicates at runtime.  The
adapter must expand the learner in place, preserve old weights, calibrate from
real outcomes, and continue from a checkpoint into a second episode segment.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_a3_workspace import build_corpus  # noqa: E402
from taiji import (  # noqa: E402
    CognitiveState,
    Observation,
    Outcome,
    PerceptionConfig,
    TaijiConfig,
    TaijiWorldState,
    TSKV8Adapter,
    WorkspaceCompositionSample,
    WorkspaceRouter,
    WorkspaceRoutingExample,
    WorldAction,
    WorldDynamicsLearner,
    WorldEvent,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldObject,
    WorldSchema,
    WorldState,
)

MANIFEST_FORMAT = "taiji-p3-open-set-manifest-v1"
REPORT_FORMAT = "taiji-p3-open-set-v1"
CONDITIONS = ("learned", "none")
SEGMENT_ASSEMBLIES = 3
TRANSITION_KINDS = ("assemble", "secure", "archive", "archive")


def _config(seed: int) -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(12, 8),
        synapse_fan_in=4,
        motor_fan_in=6,
        memory_units=16,
        memory_fan_in=4,
        memory_readout_fan_in=6,
        memory_meta_dim=6,
        memory_iterations=2,
        memory_time_dim=4,
        memory_episode_dim=4,
        lateral_fan_in=4,
        seed=int(seed),
        perception=PerceptionConfig(maximum_assembly_duration=1),
    )


def _world(
    sample_id: str,
    tick: int,
    *,
    target_id: str,
    phase: int = 0,
    success: bool = True,
) -> WorldState:
    progressed = bool(success)
    effective_phase = int(phase) if progressed else 0
    relations = [("agent", "relates-to", target_id)]
    if effective_phase >= 1:
        relations.append(("agent", "assembled", target_id))
    if effective_phase >= 2:
        relations.append(("agent", "secured", target_id))
    if effective_phase >= 3:
        relations.append(("agent", "archived", target_id))
    return WorldState(
        tick=int(tick),
        latent=torch.zeros(2),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject(
                target_id,
                attributes={
                    "assembled": float(effective_phase >= 1),
                    "secure_count": float(effective_phase >= 2),
                    "archive_count": float(max(0, effective_phase - 2)),
                },
                tags=("open-set-target",),
            ),
        ),
        relations=tuple(relations),
        events=(WorldEvent(f"{sample_id}:world:{tick}", "observed", int(tick)),),
    )


def _transition_state(
    before: WorldState,
    *,
    sample_id: str,
    target_id: str,
    phase: int,
    success: bool,
) -> WorldState:
    base = _world(
        sample_id,
        before.tick + 1,
        target_id=target_id,
        phase=phase,
        success=success,
    )
    return replace(
        base,
        latent=before.latent,
        events=before.events
        + (WorldEvent(f"{sample_id}:transition:{phase}", "world-transition", before.tick),),
        percept_event_id=before.percept_event_id,
        percept_assembly_id=before.percept_assembly_id,
        percept_boundary_closed=before.percept_boundary_closed,
    )


def _observation_world(state: WorldState, *, tick: int) -> WorldState:
    return replace(state, tick=int(tick))


def _training_corpus() -> WorldInterventionCorpus:
    initial = _world("train", 0, target_id="target")
    expected = _transition_state(
        initial,
        sample_id="train",
        target_id="target",
        phase=1,
        success=True,
    )
    action = WorldAction(
        "train:assemble",
        "assemble",
        initial.tick,
        actor_id="agent",
        target_id="target",
        parameters={"workspace_count": 2.0},
        provenance="open-set-training",
    )
    return WorldInterventionCorpus(
        train=(
            WorldInterventionCase(
                case_id="train:assemble",
                initial=initial,
                action=action,
                expected_state=expected,
                expected_outcome=Outcome(
                    intent_id=action.action_id,
                    reward=1.0,
                    success=True,
                    tick=expected.tick,
                ),
            ),
        )
    )


def _fit_world_learner(seed: int) -> WorldDynamicsLearner:
    corpus = _training_corpus()
    schema = WorldSchema.from_corpus(corpus)
    learner = WorldDynamicsLearner(schema, hidden_dim=24, seed=int(seed) + 9000)
    learner.fit(corpus.train, epochs=120, learning_rate=0.01)
    return learner


def _same_world_state(left: WorldState, right: WorldState) -> bool:
    left_payload = left.to_payload()
    right_payload = right.to_payload()
    left_latent = left_payload.pop("latent")
    right_latent = right_payload.pop("latent")
    return bool(torch.equal(left_latent, right_latent) and left_payload == right_payload)


def _observe_segment(
    model: TSKV8Adapter,
    sample: WorkspaceCompositionSample,
    *,
    condition: str,
    segment: str,
    count: int,
    target_id: str,
    initial_world: WorldState | None = None,
) -> tuple[CognitiveState, int, bool]:
    closed_count = 0
    lineage_complete = True
    for index in range(int(count)):
        current = model.cognitive_snapshot().world
        world = (
            _world(sample_id=f"{sample.tick}:{segment}", tick=model.tick + 1, target_id=target_id)
            if initial_world is None and index == 0
            else _observation_world(current, tick=model.tick + 1)
        )
        model.observe_event(
            Observation(
                modality="text-byte",
                value=97 + (index % 26),
                timestamp=index,
                source=f"p3-open-set-{segment}",
            ),
            learn=False,
            world_state=world,
            workspace_candidates=sample.candidates,
            workspace_mode=condition,
        )
        state = model.cognitive_snapshot()
        percept = state.percept
        event = state.events[-1] if state.events else None
        closed = bool(percept is not None and percept.boundary)
        closed_count += int(closed)
        lineage_complete = lineage_complete and bool(
            closed
            and event is not None
            and state.workspace.percept_boundary_closed
            and state.world.percept_boundary_closed
            and state.workspace.percept_event_id == event.event_id
            and state.world.percept_event_id == event.event_id
            and state.workspace.percept_assembly_id == percept.assembly_id
            and state.world.percept_assembly_id == percept.assembly_id
        )
    return model.cognitive_snapshot(), closed_count, lineage_complete


def _observe_bridge(
    model: TSKV8Adapter,
    sample: WorkspaceCompositionSample,
    *,
    condition: str,
    source: str,
) -> tuple[CognitiveState, bool]:
    current = model.cognitive_snapshot().world
    model.observe_event(
        Observation(
            modality="text-byte",
            value=123,
            timestamp=model.tick,
            source=source,
        ),
        learn=False,
        world_state=_observation_world(current, tick=model.tick + 1),
        workspace_candidates=sample.candidates,
        workspace_mode=condition,
    )
    state = model.cognitive_snapshot()
    return state, bool(state.percept is not None and state.percept.boundary)


def _route_success(state: CognitiveState, sample: WorkspaceCompositionSample) -> bool:
    if state.workspace.selection is None:
        raise RuntimeError("open-set evaluation requires a workspace selection")
    return set(state.workspace.selection.selected_ids) == set(sample.relevant_ids)


def _run_transition(
    model: TSKV8Adapter,
    owner: TaijiWorldState,
    sample: WorkspaceCompositionSample,
    *,
    target_id: str,
    phase: int,
    kind: str,
    route_success: bool,
) -> tuple[TaijiWorldState, bool]:
    before = model.cognitive_snapshot().world
    action = WorldAction(
        f"{sample.tick}:{kind}:{phase}",
        kind,
        before.tick,
        actor_id="agent",
        target_id=target_id,
        parameters={
            "workspace_count": float(
                0
                if model.cognitive_snapshot().workspace.selection is None
                else len(model.cognitive_snapshot().workspace.selection.selected_ids)
            )
        },
        provenance="open-set-runtime",
    )
    model.act((97, 98), sample=False, world_action=action)
    after = _transition_state(
        before,
        sample_id=str(sample.tick),
        target_id=target_id,
        phase=phase,
        success=route_success,
    )
    model.settle_action(
        1.0 if route_success else -1.0,
        learn=False,
        learn_world=True,
        world_state=after,
        success=route_success,
    )
    transition = model.cognitive_snapshot().world_transition
    if transition is None:
        raise RuntimeError(f"open-set evaluation lost {kind} transition")
    owner.apply(transition)
    return owner, bool(transition.outcome.success)


def evaluate_seed(
    seed: int,
    train: tuple[WorkspaceCompositionSample, ...],
    holdout: tuple[WorkspaceCompositionSample, ...],
    *,
    capacity: int = 2,
    epochs: int = 100,
    learning_rate: float = 0.2,
) -> dict[str, object]:
    feature_dim = train[0].candidates[0].features.numel()
    router = WorkspaceRouter(feature_dim, capacity=capacity, seed=int(seed))
    router.fit(
        tuple(
            WorkspaceRoutingExample(sample.candidates, sample.relevant_ids, sample.tick)
            for sample in train
        ),
        epochs=epochs,
        learning_rate=learning_rate,
    )
    base_learner = _fit_world_learner(seed)
    condition_totals = {
        condition: {
            "episodes": 0,
            "route_success": 0,
            "world_success": 0,
            "lineage": 0,
            "closed_assemblies": 0,
            "relation_progression": 0,
            "schema_object": 0,
            "schema_relations": 0,
            "schema_actions": 0,
            "schema_checkpoint": 0,
            "cross_episode": 0,
            "history": 0,
            "roundtrip": 0,
            "calibration": 0,
        }
        for condition in CONDITIONS
    }
    episode_details: list[dict[str, object]] = []
    for index, sample in enumerate(holdout):
        for condition in CONDITIONS:
            target_id = f"target:holdout:{sample.tick}"
            model = TSKV8Adapter(
                _config(seed + index),
                episode_id=f"open-set:{seed}:{condition}:{index}:a",
            )
            model.attach_workspace_router(router)
            model.attach_world_dynamics(copy.deepcopy(base_learner))
            schema_before = model._world_dynamics.schema if model._world_dynamics else None
            state, closed_a, lineage_a = _observe_segment(
                model,
                sample,
                condition=condition,
                segment="episode-a",
                count=SEGMENT_ASSEMBLIES,
                target_id=target_id,
            )
            route_a = _route_success(state, sample)
            owner = TaijiWorldState(state.world)
            owner, first_success = _run_transition(
                model,
                owner,
                sample,
                target_id=target_id,
                phase=1,
                kind=TRANSITION_KINDS[0],
                route_success=route_a,
            )
            state, bridge_a = _observe_bridge(
                model,
                sample,
                condition=condition,
                source="p3-open-set-bridge-a1",
            )
            owner.synchronize_observation(state.world)
            owner, second_success = _run_transition(
                model,
                owner,
                sample,
                target_id=target_id,
                phase=2,
                kind=TRANSITION_KINDS[1],
                route_success=route_a,
            )
            state, bridge_b = _observe_bridge(
                model,
                sample,
                condition=condition,
                source="p3-open-set-bridge-a2",
            )
            owner.synchronize_observation(state.world)
            mid_adapter_checkpoint = model.native_checkpoint()
            mid_owner_checkpoint = owner.checkpoint()
            mid_schema = None if model._world_dynamics is None else model._world_dynamics.schema
            restored = TSKV8Adapter.from_native_checkpoint(mid_adapter_checkpoint)
            restored_owner = TaijiWorldState.from_checkpoint(mid_owner_checkpoint)
            checkpoint_continuation = bool(
                restored._world_dynamics is not None
                and mid_schema is not None
                and restored._world_dynamics.schema.payload() == mid_schema.payload()
                and restored._world_dynamics.online_updates == 2
                and restored_owner.state.tick == owner.state.tick
                and len(restored_owner.history) == 2
            )
            restored.begin_episode(f"open-set:{seed}:{condition}:{index}:b")
            state, closed_b, lineage_b = _observe_segment(
                restored,
                sample,
                condition=condition,
                segment="episode-b",
                count=SEGMENT_ASSEMBLIES,
                target_id=target_id,
            )
            restored_owner.advance_observation(state.world)
            route_b = _route_success(state, sample)
            restored_owner, third_success = _run_transition(
                restored,
                restored_owner,
                sample,
                target_id=target_id,
                phase=3,
                kind=TRANSITION_KINDS[2],
                route_success=route_b,
            )
            state, bridge_c = _observe_bridge(
                restored,
                sample,
                condition=condition,
                source="p3-open-set-bridge-b1",
            )
            restored_owner.synchronize_observation(state.world)
            restored_owner, fourth_success = _run_transition(
                restored,
                restored_owner,
                sample,
                target_id=target_id,
                phase=4,
                kind=TRANSITION_KINDS[3],
                route_success=route_b,
            )
            state, bridge_d = _observe_bridge(
                restored,
                sample,
                condition=condition,
                source="p3-open-set-bridge-b2",
            )
            restored_owner.synchronize_observation(state.world)
            final_checkpoint = restored.native_checkpoint()
            final_restored = TSKV8Adapter.from_native_checkpoint(final_checkpoint)
            final_owner = TaijiWorldState.from_checkpoint(restored_owner.checkpoint())
            final_state = final_restored.cognitive_snapshot()
            learner = final_restored._world_dynamics
            if learner is None or schema_before is None:
                raise RuntimeError("open-set evaluation lost world learner")
            final_schema = learner.schema
            old_objects = set(schema_before.object_ids)
            old_relations = set(schema_before.relation_slots)
            old_actions = set(schema_before.action_kinds)
            new_objects = set(final_schema.object_ids) - old_objects
            new_relations = set(final_schema.relation_slots) - old_relations
            new_actions = set(final_schema.action_kinds) - old_actions
            relation_complete = {
                ("agent", "secured", target_id),
                ("agent", "archived", target_id),
            }.issubset(set(final_state.world.relations))
            lineage_complete = bool(
                lineage_a
                and lineage_b
                and bridge_a
                and bridge_b
                and bridge_c
                and bridge_d
                and lineage_b
                and final_state.world.percept_boundary_closed
                and final_state.world.percept_event_id.startswith(
                    f"open-set:{seed}:{condition}:{index}:b"
                )
                and any(
                    event.event_id.startswith(f"open-set:{seed}:{condition}:{index}:a")
                    for event in final_state.events
                )
            )
            calibration = bool(
                learner.online_updates == 4
                and len(final_state.world_calibration_trace) == 4
                and tuple(
                    item.online_update_count_after for item in final_state.world_calibration_trace
                )
                == (1, 2, 3, 4)
                and all(item.calibration_applied for item in final_state.world_calibration_trace)
            )
            values = condition_totals[condition]
            values["episodes"] += 1
            values["route_success"] += int(route_a and route_b)
            values["world_success"] += int(
                first_success and second_success and third_success and fourth_success
            )
            values["lineage"] += int(lineage_complete)
            values["closed_assemblies"] += closed_a + closed_b
            values["relation_progression"] += int(relation_complete)
            values["schema_object"] += int(target_id in new_objects)
            values["schema_relations"] += int(
                any(relation[1] == "secured" for relation in new_relations)
                and any(relation[1] == "archived" for relation in new_relations)
            )
            values["schema_actions"] += int({"secure", "archive"}.issubset(new_actions))
            values["schema_checkpoint"] += int(checkpoint_continuation)
            values["cross_episode"] += int(
                final_state.episode_id.endswith(":b")
                and any(
                    event.event_id.startswith(f"open-set:{seed}:{condition}:{index}:a")
                    for event in final_state.events
                )
            )
            values["history"] += int(len(final_owner.history) == 4)
            values["roundtrip"] += int(_same_world_state(final_owner.state, restored_owner.state))
            values["calibration"] += int(calibration)
            episode_details.append(
                {
                    "index": index,
                    "condition": condition,
                    "route_a": route_a,
                    "route_b": route_b,
                    "closed_assemblies": closed_a + closed_b,
                    "new_objects": sorted(new_objects),
                    "new_relations": sorted(new_relations),
                    "new_actions": sorted(new_actions),
                    "relation_progression": relation_complete,
                    "schema_evolution_count": learner.schema_evolution_count,
                }
            )
    normalized = {
        condition: {
            key: value / float(value_map["episodes"])
            for key, value in value_map.items()
            if key not in {"episodes", "closed_assemblies"}
        }
        | {
            "episodes": value_map["episodes"],
            "closed_assemblies": value_map["closed_assemblies"],
        }
        for condition, value_map in condition_totals.items()
    }
    return {
        "seed": int(seed),
        "router_fit_updates": router.fit_updates,
        "world_training_updates": base_learner.online_updates,
        "conditions": normalized,
        "episodes": episode_details,
    }


def build_manifest(
    *,
    train_count: int = 64,
    holdout_count: int = 32,
    seeds: tuple[int, ...] = (11, 29, 47),
) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "open-set world schema evolution across two carried episode segments",
        "train_count": int(train_count),
        "holdout_count": int(holdout_count),
        "seeds": list(seeds),
        "assemblies_per_segment": SEGMENT_ASSEMBLIES,
        "world_transitions": list(TRANSITION_KINDS),
        "training_schema": {
            "object_ids": ["agent", "target"],
            "relation_predicates": ["relates-to", "assembled"],
            "action_kinds": ["assemble"],
        },
        "open_set_holdout": {
            "object_pattern": "target:holdout:<sample_tick>",
            "relation_predicates": ["secured", "archived"],
            "action_kinds": ["secure", "archive"],
        },
        "lesion": "workspace mode none",
        "controls": [
            "semantic-key network expansion without old-weight reset",
            "open object registration",
            "open relation registration",
            "open action-kind registration",
            "native checkpoint schema continuation",
            "cross-episode world and event carryover",
            "TaijiWorldState four-transition ownership and roundtrip",
            "runtime outcome-feedback calibration trace",
        ],
        "boundary": "open-set schema/runtime capability; not open-domain semantics or general intelligence",
    }


def evaluate(
    train: tuple[WorkspaceCompositionSample, ...],
    holdout: tuple[WorkspaceCompositionSample, ...],
    *,
    seeds: tuple[int, ...] = (11, 29, 47),
) -> dict[str, object]:
    runs = [evaluate_seed(seed, train, holdout) for seed in seeds]
    learned = [run["conditions"]["learned"] for run in runs]
    none = [run["conditions"]["none"] for run in runs]
    metric_names = (
        "route_success",
        "world_success",
        "lineage",
        "relation_progression",
        "schema_object",
        "schema_relations",
        "schema_actions",
        "schema_checkpoint",
        "cross_episode",
        "history",
        "roundtrip",
        "calibration",
    )
    rates = {
        f"learned_{name}_min": min(float(item[name]) for item in learned) for name in metric_names
    }
    rates["none_route_success_max"] = max(float(item["route_success"]) for item in none)
    rates["learned_closed_assemblies_min"] = min(
        float(item["closed_assemblies"]) / float(item["episodes"] * 2) for item in learned
    )
    passed = bool(
        rates["learned_route_success_min"] >= 1.0
        and rates["learned_world_success_min"] >= 1.0
        and rates["none_route_success_max"] <= 0.0
        and all(rates[f"learned_{name}_min"] >= 1.0 for name in metric_names[2:])
        and rates["learned_closed_assemblies_min"] >= float(SEGMENT_ASSEMBLIES)
    )
    return {
        "format": REPORT_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "seeds": runs,
        "aggregate": {**rates, "passed": passed},
        "gate": {
            "passed": passed,
            "criterion": (
                "learned route/world, open object/relation/action registration, lineage, "
                "cross-episode carryover, checkpoint, four-transition ownership, roundtrip "
                "and outcome-feedback calibration must all be 1.0; workspace lesion route 0.0"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p3_open_set_manifest_20260827.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p3_open_set_20260827.json",
    )
    args = parser.parse_args()
    train, holdout = build_corpus(seed=20260827, train_count=64, holdout_count=32)
    report = evaluate(train, holdout)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
