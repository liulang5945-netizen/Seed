"""Evaluate variable-horizon perception-to-world continuation.

This benchmark extends the narrow P2 -> P3 closure without changing its
baseline report.  Each episode first closes a variable number of perceptual
assemblies, then performs two causally linked world interventions.  The
second intervention is continued from a native adapter checkpoint so that
lineage, world state, and online world calibration must survive persistence.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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

MANIFEST_FORMAT = "taiji-p2-p3-variable-horizon-manifest-v1"
REPORT_FORMAT = "taiji-p2-p3-variable-horizon-v1"
CONDITIONS = ("learned", "none")
HORIZONS = (3, 4, 5)


def _world_states_equal(left: WorldState, right: WorldState) -> bool:
    left_payload = left.to_payload()
    right_payload = right.to_payload()
    left_latent = left_payload.pop("latent")
    right_latent = right_payload.pop("latent")
    return bool(torch.equal(left_latent, right_latent) and left_payload == right_payload)


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


def _world(sample_id: str, tick: int, *, phase: int = 0, success: bool = True) -> WorldState:
    """Build the stable world schema used by train and holdout episodes.

    The workspace holdout still receives new sampled candidate/object IDs.
    The world learner deliberately consumes the stable roles ``agent`` and
    ``target`` so that transfer tests the learned relation/state dynamics,
    rather than a lookup table over episode-specific object names.
    """

    committed = bool(success and phase >= 2)
    assembled = bool(success and phase >= 1)
    relations = [("agent", "relates-to", "target")]
    if assembled:
        relations.append(("agent", "assembled", "target"))
    if committed:
        relations.append(("agent", "committed", "target"))
    return WorldState(
        tick=int(tick),
        latent=torch.zeros(2),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject(
                "target",
                attributes={
                    "assembled": float(assembled),
                    "commit_count": float(int(committed)),
                },
                tags=("closure-target",),
            ),
        ),
        relations=tuple(relations),
        events=(WorldEvent(f"{sample_id}:world:{tick}", "observed", int(tick)),),
    )


def _world_transition_state(
    before: WorldState,
    *,
    sample_id: str,
    phase: int,
    success: bool,
    holdout_relation: str | None = None,
) -> WorldState:
    after = _world(sample_id, before.tick + 1, phase=phase, success=success)
    if phase >= 2 and success and holdout_relation is not None:
        relations = tuple(
            relation for relation in after.relations if relation[1] != "committed"
        ) + (
            ("agent", holdout_relation, "target"),
        )
        after = WorldState(
            tick=after.tick,
            latent=after.latent,
            entities=after.entities,
            relations=relations,
            objects=after.objects,
            events=after.events,
        )
    return WorldState(
        tick=after.tick,
        latent=before.latent,
        entities=after.entities,
        relations=after.relations,
        objects=after.objects,
        events=before.events
        + (WorldEvent(f"{sample_id}:world-transition:{phase}", "world-transition", before.tick),),
        percept_event_id=before.percept_event_id,
        percept_assembly_id=before.percept_assembly_id,
        percept_boundary_closed=before.percept_boundary_closed,
    )


def _world_training_corpus() -> WorldInterventionCorpus:
    initial = _world("train", 0)
    assembled = _world_transition_state(
        initial,
        sample_id="train",
        phase=1,
        success=True,
    )
    committed = _world_transition_state(
        assembled,
        sample_id="train",
        phase=2,
        success=True,
    )
    assemble_action = WorldAction(
        "train:assemble",
        "assemble",
        initial.tick,
        actor_id="agent",
        target_id="target",
        parameters={"workspace_count": 2.0},
        provenance="variable-horizon-training",
    )
    commit_action = WorldAction(
        "train:commit",
        "commit",
        assembled.tick,
        actor_id="agent",
        target_id="target",
        parameters={"workspace_count": 2.0},
        provenance="variable-horizon-training",
    )
    return WorldInterventionCorpus(
        train=(
            WorldInterventionCase(
                case_id="train:assemble",
                initial=initial,
                action=assemble_action,
                expected_state=assembled,
                expected_outcome=Outcome(
                    intent_id=assemble_action.action_id,
                    reward=1.0,
                    success=True,
                    tick=assembled.tick,
                ),
            ),
            WorldInterventionCase(
                case_id="train:commit",
                initial=assembled,
                action=commit_action,
                expected_state=committed,
                expected_outcome=Outcome(
                    intent_id=commit_action.action_id,
                    reward=1.0,
                    success=True,
                    tick=committed.tick,
                ),
            ),
        )
    )


def _fit_world_learner(seed: int) -> WorldDynamicsLearner:
    corpus = _world_training_corpus()
    schema = WorldSchema.from_corpus(corpus)
    learner = WorldDynamicsLearner(schema, hidden_dim=24, seed=int(seed) + 7000)
    learner.fit(corpus.train, epochs=120, learning_rate=0.01)
    return learner


def _observe_closed_assemblies(
    model: TSKV8Adapter,
    sample: WorkspaceCompositionSample,
    *,
    condition: str,
    horizon: int,
) -> tuple[CognitiveState, int, bool]:
    closed_count = 0
    lineage_complete = True
    for index in range(int(horizon)):
        model.observe_event(
            Observation(
                modality="text-byte",
                value=97 + (index % 26),
                timestamp=index,
                source="p2-p3-variable-horizon",
            ),
            learn=False,
            world_state=_world(str(sample.tick), model.tick + 1),
            workspace_candidates=sample.candidates,
            workspace_mode=condition,
        )
        state = model.cognitive_snapshot()
        percept = state.percept
        closed = bool(percept is not None and percept.boundary)
        closed_count += int(closed)
        event = state.events[-1] if state.events else None
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


def _run_two_world_transitions(
    model: TSKV8Adapter,
    sample: WorkspaceCompositionSample,
    *,
    route_success: bool,
) -> dict[str, object]:
    before = model.cognitive_snapshot().world
    owned = TaijiWorldState(before)
    selected_ids = (
        ()
        if model.cognitive_snapshot().workspace.selection is None
        else model.cognitive_snapshot().workspace.selection.selected_ids
    )
    first_action = WorldAction(
        f"{sample.tick}:assemble",
        "assemble",
        before.tick,
        actor_id="agent",
        target_id="target",
        parameters={
            "workspace_count": float(len(selected_ids)),
            "selected_ids": selected_ids,
        },
        provenance="variable-horizon-runtime",
    )
    model.act((97, 98), sample=False, world_action=first_action)
    first_after = _world_transition_state(
        before,
        sample_id=str(sample.tick),
        phase=1,
        success=route_success,
    )
    model.settle_action(
        1.0 if route_success else -1.0,
        learn=False,
        learn_world=True,
        world_state=first_after,
        success=route_success,
    )
    first_snapshot = model.cognitive_snapshot()
    first_transition = first_snapshot.world_transition
    if first_transition is None:
        raise RuntimeError("variable-horizon evaluation lost first world transition")
    owned.apply(first_transition)
    first_checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(first_checkpoint)
    restored_owned = TaijiWorldState.from_checkpoint(owned.checkpoint())
    checkpoint_continuation = bool(
        restored.cognitive_snapshot().world.tick == first_after.tick
        and restored._world_dynamics is not None
        and restored._world_dynamics.online_updates == 1
        and _world_states_equal(restored_owned.state, owned.state)
    )

    # The native action contract consumes the settled experience on the next
    # observation.  Keep this bridge explicit instead of bypassing the
    # pending-experience guard before the second world intervention.
    restored.observe_event(
        Observation(
            modality="text-byte",
            value=123,
            timestamp=restored.tick,
            source="p2-p3-variable-horizon-bridge",
        ),
        learn=False,
        world_state=first_after,
    )
    restored_owned.synchronize_observation(restored.cognitive_snapshot().world)
    second_before = restored.cognitive_snapshot().world
    second_action = WorldAction(
        f"{sample.tick}:commit",
        "commit",
        second_before.tick,
        actor_id="agent",
        target_id="target",
        parameters={
            "workspace_count": float(len(selected_ids)),
            "selected_ids": selected_ids,
        },
        provenance="variable-horizon-runtime",
    )
    restored.act((97, 98), sample=False, world_action=second_action)
    second_after = _world_transition_state(
        second_before,
        sample_id=str(sample.tick),
        phase=2,
        success=route_success,
        holdout_relation="secured",
    )
    restored.settle_action(
        1.0 if route_success else -1.0,
        learn=False,
        learn_world=True,
        world_state=second_after,
        success=route_success,
    )
    second_snapshot = restored.cognitive_snapshot()
    second_transition = second_snapshot.world_transition
    if second_transition is None:
        raise RuntimeError("variable-horizon evaluation lost second world transition")
    restored_owned.apply(second_transition)
    final_checkpoint = restored.native_checkpoint()
    final_restored = TSKV8Adapter.from_native_checkpoint(final_checkpoint)
    final_owned = TaijiWorldState.from_checkpoint(restored_owned.checkpoint())
    final_state = final_restored.cognitive_snapshot()
    trace = final_state.world_calibration_trace
    lineage_complete = bool(
        final_state.world.percept_boundary_closed
        and final_state.world.percept_event_id
        and final_state.world.percept_assembly_id
        and first_transition.before.percept_assembly_id
        and second_transition.before.percept_assembly_id
        and first_transition.after.percept_assembly_id
        == first_transition.before.percept_assembly_id
        and second_transition.after.percept_assembly_id
        == second_transition.before.percept_assembly_id
    )
    trace_complete = bool(
        len(trace) == 2
        and tuple(item.online_update_count_after for item in trace) == (1, 2)
        and all(item.calibration_applied for item in trace)
    )
    relation_progression_complete = bool(
        ("agent", "assembled", "target") in first_transition.after.relations
        and ("agent", "secured", "target") in second_transition.after.relations
        and ("agent", "committed", "target") not in second_transition.after.relations
    )
    world_roundtrip = bool(
        _world_states_equal(final_owned.state, restored_owned.state)
        and len(final_owned.history) == 2
        and final_state.world.tick == second_after.tick
    )
    return {
        "route_success": route_success,
        "world_transition_success": bool(
            route_success and first_transition.outcome.success and second_transition.outcome.success
        ),
        "history_length": len(final_owned.history),
        "checkpoint_continuation": checkpoint_continuation,
        "lineage_complete": lineage_complete,
        "world_roundtrip": world_roundtrip,
        "runtime_calibration_trace_complete": trace_complete,
        "relation_progression_complete": relation_progression_complete,
        "online_update_count": (
            0
            if final_restored._world_dynamics is None
            else final_restored._world_dynamics.online_updates
        ),
        "relation_progression": [
            list(first_transition.after.relations),
            list(second_transition.after.relations),
        ],
    }


def evaluate_variable_horizon(
    train: tuple[WorkspaceCompositionSample, ...],
    holdout: tuple[WorkspaceCompositionSample, ...],
    *,
    seeds: tuple[int, ...] = (11, 29, 47),
    horizons: tuple[int, ...] = HORIZONS,
    capacity: int = 2,
    epochs: int = 100,
    learning_rate: float = 0.2,
) -> dict[str, object]:
    if not train or not holdout:
        raise ValueError("variable-horizon closure needs train and holdout samples")
    if not horizons or any(int(item) < 3 for item in horizons):
        raise ValueError("variable-horizon closure needs horizons of at least three assemblies")
    reports: list[dict[str, object]] = []
    for seed in seeds:
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
        condition_reports: dict[str, dict[str, float]] = {
            condition: {
                "episodes": 0.0,
                "route_success": 0.0,
                "world_transition_success": 0.0,
                "lineage_complete": 0.0,
                "checkpoint_continuation": 0.0,
                "world_roundtrip": 0.0,
                "runtime_calibration_trace_complete": 0.0,
                "relation_progression_complete": 0.0,
                "two_transition_history": 0.0,
                "closed_assembly_count": 0.0,
                "closed_assemblies_min": 1_000_000_000.0,
            }
            for condition in CONDITIONS
        }
        horizon_reports: list[dict[str, object]] = []
        for horizon in horizons:
            for condition in CONDITIONS:
                for index, sample in enumerate(holdout):
                    model = TSKV8Adapter(
                        _config(seed + index + int(horizon) * 100),
                        episode_id=f"variable:{seed}:{horizon}:{condition}:{index}",
                    )
                    model.attach_workspace_router(router)
                    model.attach_world_dynamics(copy.deepcopy(base_learner))
                    state, closed_count, percept_lineage = _observe_closed_assemblies(
                        model,
                        sample,
                        condition=condition,
                        horizon=horizon,
                    )
                    if state.workspace.selection is None:
                        raise RuntimeError("variable-horizon evaluation has no workspace selection")
                    route_success = set(state.workspace.selection.selected_ids) == set(
                        sample.relevant_ids
                    )
                    transition_report = _run_two_world_transitions(
                        model,
                        sample,
                        route_success=route_success,
                    )
                    values = condition_reports[condition]
                    values["episodes"] += 1.0
                    values["route_success"] += float(route_success)
                    values["world_transition_success"] += float(
                        transition_report["world_transition_success"]
                    )
                    values["lineage_complete"] += float(
                        percept_lineage and transition_report["lineage_complete"]
                    )
                    values["checkpoint_continuation"] += float(
                        transition_report["checkpoint_continuation"]
                    )
                    values["world_roundtrip"] += float(transition_report["world_roundtrip"])
                    values["runtime_calibration_trace_complete"] += float(
                        transition_report["runtime_calibration_trace_complete"]
                    )
                    values["relation_progression_complete"] += float(
                        transition_report["relation_progression_complete"]
                    )
                    values["two_transition_history"] += float(
                        transition_report["history_length"] == 2
                    )
                    values["closed_assembly_count"] += float(closed_count)
                    values["closed_assemblies_min"] = min(
                        values["closed_assemblies_min"], float(closed_count)
                    )
                horizon_reports.append(
                    {
                        "horizon": int(horizon),
                        "condition": condition,
                        "closed_assemblies_per_episode": int(horizon),
                    }
                )
        denominators = {
            "episodes": float(len(holdout) * len(horizons)),
            "closed_assembly_count": float(len(holdout) * len(horizons)),
        }
        normalized = {}
        for condition, values in condition_reports.items():
            episodes = denominators["episodes"]
            normalized[condition] = {
                **{
                    name: value
                    / (
                        denominators["closed_assembly_count"]
                        if name == "closed_assembly_count"
                        else episodes
                    )
                    for name, value in values.items()
                    if name not in {"episodes", "closed_assembly_count"}
                },
                "episodes": int(values["episodes"]),
                "closed_assembly_count": int(values["closed_assembly_count"]),
                "closed_assemblies_min": int(values["closed_assemblies_min"]),
            }
        reports.append(
            {
                "seed": int(seed),
                "router_fit_updates": router.fit_updates,
                "world_training_updates": base_learner.online_updates,
                "conditions": normalized,
                "horizons": horizon_reports,
            }
        )

    learned = [report["conditions"]["learned"] for report in reports]
    none = [report["conditions"]["none"] for report in reports]
    learned_route_min = min(float(item["route_success"]) for item in learned)
    learned_world_min = min(float(item["world_transition_success"]) for item in learned)
    none_route_max = max(float(item["route_success"]) for item in none)
    lineage_min = min(float(item["lineage_complete"]) for item in learned)
    checkpoint_min = min(float(item["checkpoint_continuation"]) for item in learned)
    world_roundtrip_min = min(float(item["world_roundtrip"]) for item in learned)
    trace_min = min(float(item["runtime_calibration_trace_complete"]) for item in learned)
    relation_min = min(float(item["relation_progression_complete"]) for item in learned)
    history_min = min(float(item["two_transition_history"]) for item in learned)
    closed_min = min(float(item["closed_assemblies_min"]) for item in learned)
    aggregate = {
        "learned_route_success_rate_min": learned_route_min,
        "learned_world_transition_success_rate_min": learned_world_min,
        "none_route_success_rate_max": none_route_max,
        "lineage_complete_rate_min": lineage_min,
        "checkpoint_continuation_rate_min": checkpoint_min,
        "world_roundtrip_rate_min": world_roundtrip_min,
        "runtime_calibration_trace_rate_min": trace_min,
        "relation_progression_rate_min": relation_min,
        "two_transition_history_rate_min": history_min,
        "closed_assemblies_min_per_episode": closed_min,
        "closed_assembly_count_total_min": min(
            int(item["closed_assembly_count"]) for item in learned
        ),
        "horizons": list(horizons),
    }
    passed = bool(
        learned_route_min >= 0.9
        and learned_world_min >= 0.9
        and none_route_max <= 0.1
        and lineage_min >= 1.0
        and checkpoint_min >= 1.0
        and world_roundtrip_min >= 1.0
        and trace_min >= 1.0
        and relation_min >= 1.0
        and history_min >= 1.0
        and closed_min >= float(min(horizons))
    )
    aggregate["passed"] = passed
    return {
        "format": REPORT_FORMAT,
        "capacity": int(capacity),
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "seeds": reports,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "criterion": (
                "learned route/world success >= 0.90, none lesion <= 0.10, all horizons "
                "close at least three assemblies, two world transitions survive checkpoint "
                "continuation, relation changes and runtime world calibration remain complete"
            ),
        },
    }


def build_manifest(
    *,
    train_count: int = 64,
    holdout_count: int = 32,
    seeds: tuple[int, ...] = (11, 29, 47),
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "variable-horizon perception-to-world continuation across two transitions",
        "train_count": int(train_count),
        "holdout_count": int(holdout_count),
        "seeds": list(seeds),
        "assembly_horizons": list(horizons),
        "holdout": "new workspace object identity and candidate composition",
        "world_relation_training": ["relates-to", "assembled", "committed"],
        "world_relation_holdout": ["secured"],
        "world_state_roles": ["agent", "target"],
        "lesion": "workspace mode none",
        "controls": [
            "workspace lesion",
            "native checkpoint continuation after first transition",
            "TaijiWorldState two-transition ownership and roundtrip",
            "percept event and assembly lineage",
            "runtime world learner online calibration trace",
            "relation progression across assemble and commit",
        ],
        "checkpoint": "adapter and owned world state are checkpointed after transition one and after transition two",
        "boundary": "runtime contract and causal continuation; not open-domain semantics or general intelligence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p2_p3_variable_horizon_manifest_20260827.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p2_p3_variable_horizon_20260827.json",
    )
    args = parser.parse_args()
    train, holdout = build_corpus(seed=20260827, train_count=64, holdout_count=32)
    report = evaluate_variable_horizon(train, holdout)
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
