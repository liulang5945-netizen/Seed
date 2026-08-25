"""Evaluate learned workspace routing through a two-step world outcome loop."""

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
    Outcome,
    TaijiWorldState,
    WorkspaceCandidate,
    WorkspaceRouter,
    WorkspaceRoutingExample,
    WorldAction,
    WorldEvent,
    WorldObject,
    WorldState,
    WorldTransition,
)

MANIFEST_FORMAT = "taiji-a3-world-workspace-manifest-v1"
REPORT_FORMAT = "taiji-a3-world-workspace-v1"
CONDITIONS = ("learned", "strongest_single", "dense", "fixed", "random", "none")


def _initial_state(sample_id: str) -> WorldState:
    return WorldState(
        tick=0,
        objects=(
            WorldObject(
                f"task:{sample_id}",
                attributes={"assembled": 0.0, "committed": 0.0},
                tags=("composition-task",),
            ),
        ),
        events=(WorldEvent(f"{sample_id}:start", "task-start", 0),),
    )


def _task_object(state: WorldState) -> WorldObject:
    if len(state.objects) != 1:
        raise ValueError("A3 world task expects exactly one task object")
    return state.objects[0]


def _world_episode(sample, signal: torch.Tensor) -> dict[str, object]:
    sample_id = sample.relevant_ids[0].split(":", maxsplit=1)[0]
    world = TaijiWorldState(_initial_state(sample_id))
    error = float(torch.mean((signal[:2] - sample.target) ** 2))
    assembled = error <= 1e-8
    before = world.state
    task = _task_object(before)
    assembled_state = WorldState(
        tick=1,
        latent=before.latent,
        objects=(
            WorldObject(
                task.object_id,
                attributes={
                    **dict(task.attributes),
                    "assembled": float(assembled),
                    "assemble_error": error,
                },
                tags=task.tags,
            ),
        ),
        events=before.events
        + (WorldEvent(f"{sample_id}:assemble", "assembly", 1, object_id=task.object_id),),
    )
    assemble_action = WorldAction(
        f"{sample_id}:assemble",
        "assemble",
        0,
        target_id=task.object_id,
        parameters={"signal": signal.detach().clone()},
    )
    assemble_outcome = Outcome(
        intent_id=assemble_action.action_id,
        reward=1.0 if assembled else -1.0,
        success=assembled,
        tick=1,
    )
    world.apply(WorldTransition(before, assemble_action, assembled_state, assemble_outcome))

    before = world.state
    task = _task_object(before)
    committed = bool(task.attribute("assembled", 0.0))
    committed_state = WorldState(
        tick=2,
        latent=before.latent,
        objects=(
            WorldObject(
                task.object_id,
                attributes={**dict(task.attributes), "committed": float(committed)},
                tags=task.tags,
            ),
        ),
        events=before.events
        + (WorldEvent(f"{sample_id}:commit", "commit", 2, object_id=task.object_id),),
    )
    commit_action = WorldAction(f"{sample_id}:commit", "commit", 1, target_id=task.object_id)
    commit_outcome = Outcome(
        intent_id=commit_action.action_id,
        reward=1.0 if committed else -1.0,
        success=committed,
        tick=2,
    )
    world.apply(WorldTransition(before, commit_action, committed_state, commit_outcome))
    return {
        "first_success": bool(assembled),
        "final_success": bool(committed),
        "total_reward": float(assemble_outcome.reward + commit_outcome.reward),
        "history_length": len(world.history),
    }


def _mean_signal(candidates: tuple[WorkspaceCandidate, ...]) -> torch.Tensor:
    return torch.stack([candidate.features for candidate in candidates]).mean(dim=0)


def evaluate_world_workspace(
    train,
    holdout,
    *,
    seeds: tuple[int, ...] = (11, 29, 47),
    capacity: int = 2,
    epochs: int = 100,
    learning_rate: float = 0.2,
) -> dict[str, object]:
    if not train or not holdout:
        raise ValueError("A3 world workspace evaluation needs train and holdout samples")
    reports: list[dict[str, object]] = []
    for seed in seeds:
        feature_dim = train[0].candidates[0].features.numel()
        router = WorkspaceRouter(feature_dim, capacity=capacity, seed=seed)
        router.fit(
            tuple(WorkspaceRoutingExample(sample.candidates, sample.relevant_ids, sample.tick) for sample in train),
            epochs=epochs,
            learning_rate=learning_rate,
        )
        totals = {
            condition: {"first_success": 0, "final_success": 0, "total_reward": 0.0}
            for condition in CONDITIONS
        }
        for index, sample in enumerate(holdout):
            learned = router.route(sample.candidates, tick=sample.tick, mode="learned")
            random_route = router.route(
                sample.candidates,
                tick=sample.tick,
                mode="random",
                random_seed=seed + index,
            )
            none_route = router.route(sample.candidates, tick=sample.tick, mode="none")
            signals = {
                "learned": learned.broadcast,
                "dense": _mean_signal(sample.candidates),
                "fixed": _mean_signal(sample.candidates[:capacity]),
                "random": random_route.broadcast,
                "none": none_route.broadcast,
            }
            condition_results = {
                condition: _world_episode(sample, signal) for condition, signal in signals.items()
            }
            single_results = [
                _world_episode(sample, candidate.features) for candidate in sample.candidates
            ]
            best_single = max(single_results, key=lambda result: (result["final_success"], result["total_reward"]))
            condition_results["strongest_single"] = best_single
            for condition, result in condition_results.items():
                totals[condition]["first_success"] += int(result["first_success"])
                totals[condition]["final_success"] += int(result["final_success"])
                totals[condition]["total_reward"] += float(result["total_reward"])
        count = float(len(holdout))
        metrics = {
            condition: {
                "first_success_accuracy": values["first_success"] / count,
                "final_success_accuracy": values["final_success"] / count,
                "mean_total_reward": values["total_reward"] / count,
            }
            for condition, values in totals.items()
        }
        metrics["learned_gain_vs_strongest_single"] = (
            metrics["learned"]["final_success_accuracy"]
            - metrics["strongest_single"]["final_success_accuracy"]
        )
        metrics["learned_gain_vs_dense"] = (
            metrics["learned"]["final_success_accuracy"]
            - metrics["dense"]["final_success_accuracy"]
        )
        metrics["router_fit_updates"] = router.fit_updates
        reports.append(metrics)
    aggregate = {
        "learned_final_success_accuracy": sum(
            float(report["learned"]["final_success_accuracy"]) for report in reports
        )
        / len(reports),
        "strongest_single_final_success_accuracy": sum(
            float(report["strongest_single"]["final_success_accuracy"]) for report in reports
        )
        / len(reports),
        "dense_final_success_accuracy": sum(
            float(report["dense"]["final_success_accuracy"]) for report in reports
        )
        / len(reports),
        "random_final_success_accuracy": sum(
            float(report["random"]["final_success_accuracy"]) for report in reports
        )
        / len(reports),
        "none_final_success_accuracy": sum(
            float(report["none"]["final_success_accuracy"]) for report in reports
        )
        / len(reports),
        "learned_gain_vs_strongest_single_min": min(
            float(report["learned_gain_vs_strongest_single"]) for report in reports
        ),
        "learned_gain_vs_dense_min": min(float(report["learned_gain_vs_dense"]) for report in reports),
        "history_length": 2,
    }
    aggregate["passed"] = bool(
        aggregate["learned_final_success_accuracy"] >= 0.95
        and aggregate["learned_gain_vs_strongest_single_min"] >= 0.2
        and aggregate["learned_gain_vs_dense_min"] >= 0.2
    )
    return {
        "format": REPORT_FORMAT,
        "capacity": capacity,
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "seeds": reports,
        "aggregate": aggregate,
        "gate": {
            "passed": aggregate["passed"],
            "criterion": "learned workspace final success >= 0.95 and beats strongest single/dense by >= 0.2",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "two-step assemble then commit; commit succeeds only after a correct workspace composition",
        "world_state": "TaijiWorldState owns task object and two contiguous transitions",
        "action_kinds": ["assemble", "commit"],
        "outcome_rule": "assemble succeeds only when workspace content exactly reconstructs the hidden two-source target",
        "controls": list(CONDITIONS),
        "split_constraint": "holdout uses new sampled combinations and independent distractor noise",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_a3_world_workspace_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_a3_world_workspace_baseline_20260825.json",
    )
    args = parser.parse_args()
    train, holdout = build_corpus()
    report = evaluate_world_workspace(train, holdout)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
