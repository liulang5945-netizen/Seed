"""Evaluate data-derived world-model imagined rollouts across horizons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_a2_world import build_corpus  # noqa: E402
from taiji import (  # noqa: E402
    Goal,
    GoalPlanner,
    Observation,
    PlanningCandidate,
    TaijiConfig,
    TSKV8Adapter,
    WorldAction,
    WorldDynamicsLearner,
    WorldSchema,
    WorldState,
)

SEEDS = (11, 23, 37)
HORIZONS = (3, 4, 5)
MANIFEST_FORMAT = "taiji-p7-world-model-rollout-manifest-v1"
REPORT_FORMAT = "taiji-p7-world-model-rollout-v1"


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
        seed=seed,
    )


def _initial_state(state: WorldState) -> WorldState:
    return WorldState(
        tick=1,
        latent=state.latent.detach().clone(),
        entities=state.entities,
        relations=state.relations,
        objects=state.objects,
        events=state.events,
        affordances=state.affordances,
        uncertainty=state.uncertainty,
    )


def evaluate_seed(seed: int) -> dict[str, object]:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    learner = WorldDynamicsLearner(schema, hidden_dim=32, seed=seed)
    learner.fit(corpus.train, epochs=250, learning_rate=0.01)

    move_case = next(case for case in corpus.train if case.action.kind == "move")
    actor_id = move_case.action.actor_id
    action_kind = move_case.action.kind
    target_ids = schema.target_ids
    adapter = TSKV8Adapter(_config(seed), episode_id=f"world-rollout-{seed}")
    adapter.attach_world_dynamics(learner)
    adapter.attach_goal_planner(GoalPlanner())
    adapter.set_goals((Goal("reach-world", "reach the world target", priority=1.0),))
    adapter.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="world-rollout-evaluation",
        ),
        learn=False,
        world_state=_initial_state(move_case.initial),
    )

    horizon_runs: list[dict[str, object]] = []
    for horizon in HORIZONS:
        start_tick = adapter.cognitive_snapshot().world.tick
        templates = tuple(
            PlanningCandidate(
                candidate_id=f"h{horizon}-step-{index}",
                action=WorldAction(
                    action_id=f"h{horizon}-action-{index}",
                    kind=action_kind,
                    tick=start_tick + index,
                    actor_id=actor_id,
                    target_id=target_ids[index % len(target_ids)],
                    parameters={"step": 1.0},
                    provenance="candidate",
                ),
                predicted_reward=0.0,
                success_probability=0.0,
                expected_progress=(index + 1) / horizon,
            )
            for index in range(horizon)
        )
        rollout = adapter.imagine_world_rollout(
            f"world-rollout-h{horizon}",
            "reach-world",
            templates,
            confidence=0.9,
        )
        decision = adapter.plan_rollouts((rollout,))
        checkpoint = adapter.native_checkpoint()
        restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
        restored_rollout = restored._planned_rollout
        checkpoint_complete = bool(
            restored_rollout is not None
            and restored_rollout.rollout_id == rollout.rollout_id
            and len(restored_rollout.steps) == horizon
            and all(
                step.prediction_provenance == "world-dynamics" for step in restored_rollout.steps
            )
        )
        lesion = TSKV8Adapter(_config(seed), episode_id=f"world-rollout-lesion-{seed}")
        try:
            lesion.imagine_world_rollout(
                f"lesion-h{horizon}",
                "reach-world",
                templates,
            )
        except RuntimeError as error:
            lesion_complete = "world dynamics is not attached" in str(error)
        else:
            lesion_complete = False
        horizon_runs.append(
            {
                "horizon": horizon,
                "rollout_length": len(rollout.steps),
                "selected_rollout": decision.selected.rollout_id,
                "provenance_complete": all(
                    step.prediction_provenance == "world-dynamics"
                    and step.action.provenance == "world-dynamics"
                    for step in rollout.steps
                ),
                "tick_chain_complete": tuple(step.action.tick for step in rollout.steps)
                == tuple(start_tick + index for index in range(horizon)),
                "checkpoint_complete": checkpoint_complete,
                "world_model_lesion_complete": lesion_complete,
            }
        )
    horizon_gate = all(
        run["rollout_length"] == run["horizon"]
        and run["selected_rollout"] == f"world-rollout-h{run['horizon']}"
        and run["provenance_complete"]
        and run["tick_chain_complete"]
        and run["checkpoint_complete"]
        and run["world_model_lesion_complete"]
        for run in horizon_runs
    )
    return {
        "seed": seed,
        "train_examples": len(corpus.train),
        "schema": schema.payload(),
        "horizon_gate": horizon_gate,
        "horizon_runs": horizon_runs,
    }


def evaluate(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    runs = [evaluate_seed(seed) for seed in seeds]
    rates = {
        "horizon_gate": sum(bool(run["horizon_gate"]) for run in runs) / len(runs),
    }
    return {
        "format": REPORT_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "metrics": {"cross_seed_rates": rates, "runs": runs},
        "gate": {
            "passed": all(rate == 1.0 for rate in rates.values()),
            "criterion": "all seeds must generate and select data-derived world-model rollouts at horizons 3/4/5, preserve per-step prediction provenance and tick chains through native checkpoint continuation, and fail closed when the world-model organ is lesioned",
        },
    }


def build_manifest(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "data-derived world-dynamics imagined rollout into GoalPlanner",
        "seeds": list(seeds),
        "horizons": list(HORIZONS),
        "controls": ["native-checkpoint-continuation", "world-model-lesion"],
        "boundary": "numeric world prediction and structured planning provenance; not general semantics or long-horizon intelligence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p7_world_model_rollout_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p7_world_model_rollout_report_20260825.json",
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
