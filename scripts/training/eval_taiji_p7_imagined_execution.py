"""Evaluate imagined world rollouts consumed by real environment execution."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_a2_world import build_corpus  # noqa: E402
from taiji import (  # noqa: E402
    EnvironmentOutcome,
    Goal,
    GoalPlanner,
    Observation,
    PlanningCandidate,
    PlanningConfig,
    TaijiConfig,
    TSKV8Adapter,
    WorldAction,
    WorldDynamicsLearner,
    WorldObject,
    WorldSchema,
    WorldState,
)

SEEDS = (11, 23, 37)
HORIZONS = (3, 4, 5)
MANIFEST_FORMAT = "taiji-p7-imagined-execution-manifest-v1"
REPORT_FORMAT = "taiji-p7-imagined-execution-v1"


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


def _advance_state(state: WorldState, target_id: str) -> WorldState:
    objects = []
    for item in state.objects:
        attributes = dict(item.attributes)
        if item.object_id == target_id:
            attributes["position"] = float(attributes.get("position", 0.0)) + 1.0
        objects.append(WorldObject(item.object_id, attributes=attributes, tags=item.tags))
    return WorldState(
        tick=state.tick + 1,
        latent=state.latent,
        entities=state.entities,
        relations=state.relations,
        objects=tuple(objects),
        events=state.events,
        affordances=state.affordances,
        uncertainty=state.uncertainty,
    )


class _RolloutEnvironment:
    def __init__(self, states: tuple[WorldState, ...]) -> None:
        self.states = states
        self.index = 0
        self.actions: list[int] = []

    def reset(self) -> tuple[int, tuple[int, ...]]:
        self.index = 0
        self.actions.clear()
        return 97, (10, 11)

    def step(self, action_symbol: int) -> EnvironmentOutcome:
        if self.index >= len(self.states):
            raise RuntimeError("rollout environment received too many actions")
        self.actions.append(action_symbol)
        state = self.states[self.index]
        self.index += 1
        return EnvironmentOutcome(
            sensation=97 + self.index,
            reward=1.0,
            success=True,
            terminal=self.index == len(self.states),
            world_state=state,
        )


def evaluate_seed(seed: int) -> dict[str, object]:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    base_learner = WorldDynamicsLearner(schema, hidden_dim=32, seed=seed)
    base_learner.fit(corpus.train, epochs=250, learning_rate=0.01)
    move_case = next(case for case in corpus.train if case.action.kind == "move")
    action_kind = move_case.action.kind
    actor_id = move_case.action.actor_id
    target_ids = schema.target_ids
    horizon_runs: list[dict[str, object]] = []

    for horizon in HORIZONS:
        adapter = TSKV8Adapter(_config(seed), episode_id=f"imagined-execution-{seed}-{horizon}")
        adapter.attach_world_dynamics(deepcopy(base_learner))
        adapter.attach_goal_planner(
            GoalPlanner(PlanningConfig(replan_error_threshold=1.0))
        )
        adapter.set_goals((Goal("reach-world", "reach the world target", priority=1.0),))
        initial = _initial_state(move_case.initial)
        adapter.observe_event(
            Observation(
                modality="text-byte",
                value=97,
                timestamp=0,
                source="imagined-execution-evaluation",
            ),
            learn=False,
            world_state=initial,
        )
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
                    parameters={"step": 1.0, "action_symbol": 10},
                ),
                predicted_reward=0.0,
                success_probability=0.0,
                expected_progress=(index + 1) / horizon,
            )
            for index in range(horizon)
        )
        rollout = adapter.imagine_world_rollout(
            f"imagined-execution-h{horizon}",
            "reach-world",
            templates,
        )
        adapter.plan_rollouts((rollout,))
        actual_states = []
        current = initial
        for template in templates:
            current = _advance_state(current, template.action.target_id)
            actual_states.append(current)
        environment = _RolloutEnvironment(tuple(actual_states))
        outcomes = []
        for _ in range(horizon):
            outcomes.append(
                adapter.execute_imagined_rollout_step(
                    environment,
                    available_actions=(10, 11),
                    action_kinds=(action_kind, "idle"),
                    learn=False,
                    learn_world=True,
                )
            )
        snapshot = adapter.cognitive_snapshot()
        checkpoint = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
        checkpoint_state = checkpoint.cognitive_snapshot()
        horizon_runs.append(
            {
                "horizon": horizon,
                "executed_steps": len(outcomes),
                "all_success": all(outcome.success is True for outcome in outcomes),
                "terminal_final": outcomes[-1].terminal,
                "action_trace": environment.actions,
                "prediction_trace_complete": bool(
                    len(snapshot.world_calibration_trace) == horizon
                    and all(
                        item.prediction.state_error is not None
                        and item.prediction.reward_error is not None
                        for item in snapshot.world_calibration_trace
                    )
                ),
                "replan_not_required": not adapter.replan_required,
                "remaining_rollout_consumed": adapter._planned_rollout is None,
                "checkpoint_trace_complete": len(checkpoint_state.world_calibration_trace)
                == horizon,
                "checkpoint_learner_updates": (
                    0
                    if checkpoint._world_dynamics is None
                    else checkpoint._world_dynamics.online_updates
                ),
            }
        )
    execution_gate = all(
        run["executed_steps"] == run["horizon"]
        and run["all_success"]
        and run["terminal_final"]
        and run["action_trace"] == [10] * run["horizon"]
        and run["prediction_trace_complete"]
        and run["replan_not_required"]
        and run["remaining_rollout_consumed"]
        and run["checkpoint_trace_complete"]
        and run["checkpoint_learner_updates"] == run["horizon"]
        for run in horizon_runs
    )
    return {"seed": seed, "execution_gate": execution_gate, "horizon_runs": horizon_runs}


def evaluate(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    runs = [evaluate_seed(seed) for seed in seeds]
    rates = {
        "execution_gate": sum(bool(run["execution_gate"]) for run in runs) / len(runs),
    }
    return {
        "format": REPORT_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "metrics": {"cross_seed_rates": rates, "runs": runs},
        "gate": {
            "passed": all(rate == 1.0 for rate in rates.values()),
            "criterion": "all seeds must execute imagined world rollouts at horizons 3/4/5 through the real environment, score each actual transition, consume the remaining plan, preserve prediction traces and learner updates through checkpoint, and avoid unnecessary replanning",
        },
    }


def build_manifest(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "consume data-derived imagined world rollouts in a real environment loop",
        "seeds": list(seeds),
        "horizons": list(HORIZONS),
        "controls": ["prediction-error-trace", "remaining-rollout-consumption", "native-checkpoint-continuation"],
        "boundary": "numeric world prediction to environment execution; not general planning or intelligence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p7_imagined_execution_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p7_imagined_execution_report_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
