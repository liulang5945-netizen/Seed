"""Evaluate the learned-perception to world/workspace runtime closure.

This benchmark keeps the target relation in evaluation metadata only.  The
adapter receives raw observations, a candidate set, and an external world
snapshot; it must preserve the percept-to-workspace/world lineage while the
learned workspace router selects the unseen composition used by a real world
transition.
"""

from __future__ import annotations

import argparse
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
    WorldEvent,
    WorldObject,
    WorldState,
    WorldTransition,
)

MANIFEST_FORMAT = "taiji-p2-p3-closure-manifest-v1"
REPORT_FORMAT = "taiji-p2-p3-closure-v1"
CONDITIONS = ("learned", "none")


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


def _world(sample_id: str, tick: int) -> WorldState:
    object_id = f"object:{sample_id}"
    return WorldState(
        tick=int(tick),
        latent=torch.zeros(2),
        objects=(
            WorldObject(
                object_id,
                attributes={"assembled": 0.0, "commit_count": 0},
                tags=("closure-target",),
            ),
        ),
        relations=(("agent", "relates-to", object_id),),
        events=(WorldEvent(f"{sample_id}:observed:{tick}", "observed", int(tick)),),
    )


def _observe_two_closed_assemblies(
    model: TSKV8Adapter,
    sample: WorkspaceCompositionSample,
    *,
    condition: str,
) -> tuple[CognitiveState, int]:
    closed_count = 0
    for tick, symbol in enumerate((97, 98), start=1):
        model.observe_event(
            Observation(
                modality="text-byte",
                value=symbol,
                timestamp=tick,
                source="p2-p3-closure",
            ),
            learn=False,
            world_state=_world(str(sample.tick), model.tick + 1),
            workspace_candidates=sample.candidates,
            workspace_mode=condition,
        )
        percept = model.cognitive_snapshot().percept
        closed_count += int(percept is not None and percept.boundary)
    state = model.cognitive_snapshot()
    return state, closed_count


def _apply_workspace_transition(
    state: CognitiveState,
    sample: WorkspaceCompositionSample,
) -> tuple[bool, bool, int]:
    if state.workspace.selection is None:
        raise RuntimeError("closure evaluation requires a workspace selection")
    selected_ids = tuple(state.workspace.selection.selected_ids)
    route_success = set(selected_ids) == set(sample.relevant_ids)
    world = TaijiWorldState(state.world)
    before = world.state
    if len(before.objects) != 1:
        raise ValueError("closure world expects one target object")
    target = before.objects[0]
    after = WorldState(
        tick=before.tick + 1,
        latent=before.latent,
        entities=before.entities,
        relations=before.relations,
        objects=(
            WorldObject(
                target.object_id,
                attributes={
                    **dict(target.attributes),
                    "assembled": float(route_success),
                    "commit_count": int(target.attribute("commit_count", 0)) + 1,
                    "selected_ids": selected_ids,
                },
                tags=target.tags,
            ),
        ),
        events=before.events
        + (WorldEvent(f"{target.object_id}:assembly", "workspace-assembly", before.tick),),
        percept_event_id=before.percept_event_id,
        percept_assembly_id=before.percept_assembly_id,
        percept_boundary_closed=before.percept_boundary_closed,
    )
    action = WorldAction(
        f"{target.object_id}:assemble",
        "assemble",
        before.tick,
        target_id=target.object_id,
        parameters={"selected_ids": selected_ids},
    )
    outcome = Outcome(
        intent_id=action.action_id,
        reward=1.0 if route_success else -1.0,
        success=route_success,
        tick=after.tick,
    )
    world.apply(WorldTransition(before, action, after, outcome))
    restored = TaijiWorldState.from_checkpoint(world.checkpoint())
    world_roundtrip = (
        restored.state.objects == world.state.objects
        and len(restored.history) == 1
        and restored.state.percept_event_id == before.percept_event_id
    )
    return route_success, world_roundtrip, len(world.history)


def evaluate_closure(
    train: tuple[WorkspaceCompositionSample, ...],
    holdout: tuple[WorkspaceCompositionSample, ...],
    *,
    seeds: tuple[int, ...] = (11, 29, 47),
    capacity: int = 2,
    epochs: int = 100,
    learning_rate: float = 0.2,
) -> dict[str, object]:
    if not train or not holdout:
        raise ValueError("P2-P3 closure evaluation needs train and holdout samples")
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
        condition_totals = {
            condition: {"route_success": 0, "world_success": 0, "history": 0}
            for condition in CONDITIONS
        }
        lineage_ok = 0
        closed_assembly_count = 0
        checkpoint_ok = False
        for index, sample in enumerate(holdout):
            for condition in CONDITIONS:
                model = TSKV8Adapter(_config(seed + index), episode_id=f"closure:{seed}:{index}")
                model.attach_workspace_router(router)
                state, closed_count = _observe_two_closed_assemblies(
                    model, sample, condition=condition
                )
                if condition == "learned":
                    closed_assembly_count += closed_count
                    if state.percept is None or not state.events:
                        raise RuntimeError("closure evaluation produced no percept lineage")
                    event = state.events[-1]
                    lineage_ok += int(
                        state.percept.boundary
                        and state.workspace.percept_boundary_closed
                        and state.world.percept_boundary_closed
                        and state.workspace.percept_event_id == event.event_id
                        and state.world.percept_event_id == event.event_id
                        and state.workspace.percept_assembly_id == state.percept.assembly_id
                        and state.world.percept_assembly_id == state.percept.assembly_id
                    )
                    if index == 0:
                        restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
                        restored_state = restored.cognitive_snapshot()
                        checkpoint_ok = bool(
                            restored_state.workspace.percept_event_id
                            == state.workspace.percept_event_id
                            and restored_state.world.percept_assembly_id
                            == state.world.percept_assembly_id
                            and restored_state.world.percept_boundary_closed
                            == state.world.percept_boundary_closed
                        )
                route_success, world_roundtrip, history_length = _apply_workspace_transition(
                    state, sample
                )
                condition_totals[condition]["route_success"] += int(route_success)
                condition_totals[condition]["world_success"] += int(
                    route_success and world_roundtrip
                )
                condition_totals[condition]["history"] += int(history_length == 1)
        count = float(len(holdout))
        metrics = {
            condition: {
                "route_success_rate": values["route_success"] / count,
                "world_transition_success_rate": values["world_success"] / count,
                "one_transition_history_rate": values["history"] / count,
            }
            for condition, values in condition_totals.items()
        }
        reports.append(
            {
                "seed": int(seed),
                "router_fit_updates": router.fit_updates,
                "lineage_rate": lineage_ok / count,
                "closed_assembly_count": closed_assembly_count,
                "checkpoint_continuation": checkpoint_ok,
                "conditions": metrics,
            }
        )

    learned_route_min = min(
        float(report["conditions"]["learned"]["route_success_rate"]) for report in reports
    )
    learned_world_min = min(
        float(report["conditions"]["learned"]["world_transition_success_rate"])
        for report in reports
    )
    none_route_max = max(
        float(report["conditions"]["none"]["route_success_rate"]) for report in reports
    )
    lineage_min = min(float(report["lineage_rate"]) for report in reports)
    checkpoint_passed = all(bool(report["checkpoint_continuation"]) for report in reports)
    total_closed = sum(int(report["closed_assembly_count"]) for report in reports)
    aggregate = {
        "learned_route_success_rate_min": learned_route_min,
        "learned_world_transition_success_rate_min": learned_world_min,
        "none_route_success_rate_max": none_route_max,
        "lineage_rate_min": lineage_min,
        "checkpoint_continuation": checkpoint_passed,
        "closed_assembly_count": total_closed,
    }
    passed = bool(
        learned_route_min >= 0.9
        and learned_world_min >= 0.9
        and none_route_max <= 0.1
        and lineage_min >= 1.0
        and checkpoint_passed
        and total_closed == 2 * len(holdout) * len(seeds)
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
                "learned route/world success >= 0.90, none lesion <= 0.10, lineage and "
                "checkpoint continuation complete, and every holdout closes two assemblies"
            ),
        },
    }


def build_manifest(*, train_count: int = 64, holdout_count: int = 32) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "route unseen two-source workspace composition into a Taiji world transition",
        "world_contract": "TaijiWorldState owns one object, one relation, and one workspace transition",
        "perception_contract": "two raw observations must produce two boundary-closed percept events",
        "train_count": int(train_count),
        "holdout_count": int(holdout_count),
        "holdout": "new sampled object identity and candidate composition",
        "lesion": "workspace mode none",
        "checkpoint": "native adapter checkpoint after two closed assemblies",
        "lineage": [
            "percept_event_id",
            "percept_assembly_id",
            "percept_boundary_closed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p2_p3_closure_manifest_20260827.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p2_p3_closure_20260827.json",
    )
    args = parser.parse_args()
    train, holdout = build_corpus(seed=20260827, train_count=64, holdout_count=32)
    report = evaluate_closure(train, holdout)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
