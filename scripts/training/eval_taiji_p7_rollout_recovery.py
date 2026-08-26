"""Evaluate imagined-rollout interruption and recovery after world error."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_a2_world import build_corpus  # noqa: E402
from scripts.training.eval_taiji_p7_imagined_execution import (  # noqa: E402
    _advance_state,
    _config,
    _initial_state,
    _RolloutEnvironment,
)
from taiji import (  # noqa: E402
    Goal,
    GoalPlanner,
    Observation,
    PlanningCandidate,
    TSKV8Adapter,
    WorldAction,
    WorldDynamicsLearner,
    WorldObject,
    WorldSchema,
    WorldState,
)

SEEDS = (11, 23, 37)
MANIFEST_FORMAT = "taiji-p7-rollout-recovery-manifest-v1"
REPORT_FORMAT = "taiji-p7-rollout-recovery-v1"


def _corrupt_state(state: WorldState, target_id: str) -> WorldState:
    objects = []
    for item in state.objects:
        attributes = dict(item.attributes)
        if item.object_id == target_id:
            attributes["position"] = 9.0
        objects.append(WorldObject(item.object_id, attributes=attributes, tags=item.tags))
    return WorldState(
        tick=state.tick,
        latent=state.latent,
        entities=state.entities,
        relations=state.relations,
        objects=tuple(objects),
        events=state.events,
        affordances=state.affordances,
        uncertainty=state.uncertainty,
    )


def _templates(
    *,
    start_tick: int,
    target_ids: tuple[str, ...],
    prefix: str,
    length: int,
) -> tuple[PlanningCandidate, ...]:
    return tuple(
        PlanningCandidate(
            candidate_id=f"{prefix}-step-{index}",
            action=WorldAction(
                action_id=f"{prefix}-action-{index}",
                kind="move",
                tick=start_tick + index,
                actor_id="agent",
                target_id=target_ids[index % len(target_ids)],
                parameters={"step": 1.0, "action_symbol": 10},
            ),
            predicted_reward=0.0,
            success_probability=0.0,
            expected_progress=(index + 1) / length,
        )
        for index in range(length)
    )


def evaluate_seed(seed: int) -> dict[str, object]:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    learner = WorldDynamicsLearner(schema, hidden_dim=32, seed=seed)
    learner.fit(corpus.train, epochs=250, learning_rate=0.01)
    move_case = next(case for case in corpus.train if case.action.kind == "move")
    initial = _initial_state(move_case.initial)
    adapter = TSKV8Adapter(_config(seed), episode_id=f"rollout-recovery-{seed}")
    adapter.attach_world_dynamics(learner)
    adapter.attach_goal_planner(GoalPlanner())
    adapter.set_goals((Goal("reach-world", "reach the world target", priority=1.0),))
    adapter.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="rollout-recovery-evaluation",
        ),
        learn=False,
        world_state=initial,
    )
    start_tick = adapter.cognitive_snapshot().world.tick
    first_templates = _templates(
        start_tick=start_tick,
        target_ids=schema.target_ids,
        prefix="initial",
        length=3,
    )
    adapter.plan_rollouts(
        (
            adapter.imagine_world_rollout(
                "initial-rollout",
                "reach-world",
                first_templates,
            ),
        )
    )
    corrupted = _corrupt_state(_advance_state(initial, "red"), "red")
    recovery_one = _advance_state(corrupted, "blue")
    recovery_two = _advance_state(recovery_one, "red")
    environment = _RolloutEnvironment((corrupted, recovery_one, recovery_two))
    first = adapter.execute_imagined_rollout_step(
        environment,
        available_actions=(10, 11),
        action_kinds=("move", "idle"),
        learn=False,
        learn_world=True,
    )
    interrupted_state = adapter.cognitive_snapshot()
    interruption_checkpoint = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    interruption_checkpoint_state = interruption_checkpoint.cognitive_snapshot()
    interruption_gate = bool(
        first.success is True
        and adapter.replan_required
        and adapter._planned_rollout is None
        and len(interrupted_state.world_calibration_trace) == 1
        and interrupted_state.planning_recovery is not None
        and interrupted_state.planning_recovery.mode == "world-error-recovery"
        and interrupted_state.planning_recovery.trigger == "world-prediction-error"
        and interrupted_state.planning_recovery.remaining_rollout_steps == 2
        and interruption_checkpoint_state.planning_recovery == interrupted_state.planning_recovery
        and interrupted_state.world_calibration_trace[0].prediction.state_error is not None
        and interrupted_state.world_calibration_trace[0].prediction.state_error > 0.25
    )
    adapter = interruption_checkpoint
    recovery_start_tick = interrupted_state.world.tick
    adapter.attach_goal_planner(GoalPlanner())
    recovery_templates = _templates(
        start_tick=recovery_start_tick,
        target_ids=("blue", "red"),
        prefix="recovery",
        length=2,
    )
    adapter.plan_rollouts(
        (
            adapter.imagine_world_rollout(
                "recovery-rollout",
                "reach-world",
                recovery_templates,
            ),
        )
    )
    recovery_outcomes = [
        adapter.execute_imagined_rollout_step(
            environment,
            available_actions=(10, 11),
            action_kinds=("move", "idle"),
            learn=False,
            learn_world=True,
        )
        for _ in recovery_templates
    ]
    final_state = adapter.cognitive_snapshot()
    checkpoint = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    checkpoint_state = checkpoint.cognitive_snapshot()
    recovery_gate = bool(
        interruption_gate
        and all(outcome.success is True for outcome in recovery_outcomes)
        and recovery_outcomes[-1].terminal
        and not adapter.replan_required
        and adapter._planned_rollout is None
        and len(final_state.world_calibration_trace) == 3
        and final_state.planning_recovery is None
        and len(checkpoint_state.world_calibration_trace) == 3
        and checkpoint._world_dynamics is not None
        and checkpoint._world_dynamics.online_updates == 3
    )
    return {
        "seed": seed,
        "interruption_gate": interruption_gate,
        "recovery_gate": recovery_gate,
        "action_trace": environment.actions,
        "trace_length": len(final_state.world_calibration_trace),
        "checkpoint_trace_length": len(checkpoint_state.world_calibration_trace),
        "checkpoint_recovery_preserved": (
            interruption_checkpoint_state.planning_recovery is not None
        ),
        "recovery_updates": (
            0 if checkpoint._world_dynamics is None else checkpoint._world_dynamics.online_updates
        ),
    }


def evaluate(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    runs = [evaluate_seed(seed) for seed in seeds]
    rates = {
        "interruption_gate": sum(bool(run["interruption_gate"]) for run in runs) / len(runs),
        "recovery_gate": sum(bool(run["recovery_gate"]) for run in runs) / len(runs),
    }
    return {
        "format": REPORT_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "metrics": {"cross_seed_rates": rates, "runs": runs},
        "gate": {
            "passed": all(rate == 1.0 for rate in rates.values()),
            "criterion": "all seeds must stop an imagined rollout after a positive-reward but high world-state error, preserve the error trace, replan with a recovery rollout, finish the recovery environment episode, and restore the complete trace and learner updates through checkpoint",
        },
    }


def build_manifest(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "recover an imagined rollout after a high-error real world transition",
        "seeds": list(seeds),
        "controls": [
            "positive-reward-world-error",
            "rollout-interruption",
            "recovery-rollout",
            "native-checkpoint-continuation",
        ],
        "boundary": "numeric world prediction error and structured replan recovery; not general failure recovery or intelligence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p7_rollout_recovery_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p7_rollout_recovery_report_20260825.json",
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
