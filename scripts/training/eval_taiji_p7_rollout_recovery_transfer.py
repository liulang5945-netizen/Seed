"""Evaluate runtime recovery across horizons and failure positions."""

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
from scripts.training.eval_taiji_p7_rollout_recovery import (  # noqa: E402
    _advance_state,
    _config,
    _corrupt_state,
    _initial_state,
    _RolloutEnvironment,
    _templates,
)
from taiji import (  # noqa: E402
    Goal,
    GoalPlanner,
    Observation,
    PlanningConfig,
    TSKV8Adapter,
    WorldDynamicsLearner,
    WorldSchema,
)

SEEDS = (11, 23, 37)
HORIZONS = (3, 4, 5)
MANIFEST_FORMAT = "taiji-p7-rollout-recovery-transfer-manifest-v1"
REPORT_FORMAT = "taiji-p7-rollout-recovery-transfer-v1"


def _run_case(
    *,
    seed: int,
    horizon: int,
    failure_position: int,
    base_learner: WorldDynamicsLearner,
    initial,
    target_ids: tuple[str, ...],
) -> dict[str, object]:
    adapter = TSKV8Adapter(
        _config(seed),
        episode_id=f"rollout-recovery-transfer-{seed}-{horizon}-{failure_position}",
    )
    adapter.attach_world_dynamics(deepcopy(base_learner))
    adapter.attach_goal_planner(
        GoalPlanner(PlanningConfig(replan_error_threshold=4.0, recovery_error_threshold=4.0))
    )
    adapter.set_goals((Goal("reach-world", "reach the world target", priority=1.0),))
    adapter.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="rollout-recovery-transfer-evaluation",
        ),
        learn=False,
        world_state=initial,
    )
    start_tick = adapter.cognitive_snapshot().world.tick
    templates = _templates(
        start_tick=start_tick,
        target_ids=target_ids,
        prefix=f"h{horizon}-f{failure_position}-initial",
        length=horizon,
    )
    adapter.plan_rollouts(
        (adapter.imagine_world_rollout(
            f"h{horizon}-f{failure_position}-initial-rollout",
            "reach-world",
            templates,
        ),)
    )

    actual_states = []
    current = initial
    for index, template in enumerate(templates):
        current = _advance_state(current, template.action.target_id)
        if index == failure_position:
            current = _corrupt_state(current, "red")
        actual_states.append(current)
    environment = _RolloutEnvironment(tuple(actual_states))

    pre_failure_outcomes = []
    for _ in range(failure_position + 1):
        pre_failure_outcomes.append(
            adapter.execute_imagined_rollout_step(
                environment,
                available_actions=(10, 11),
                action_kinds=("move", "idle"),
                learn=False,
                learn_world=True,
            )
        )
    interrupted_state = adapter.cognitive_snapshot()
    checkpoint = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    checkpoint_interrupted_state = checkpoint.cognitive_snapshot()
    remaining = horizon - failure_position - 1
    interruption_gate = bool(
        all(outcome.success is True and outcome.terminal is False for outcome in pre_failure_outcomes)
        and adapter.replan_required
        and adapter._planned_rollout is None
        and interrupted_state.planning_recovery is not None
        and interrupted_state.planning_recovery.remaining_rollout_steps == remaining
        and interrupted_state.planning_recovery.prediction_error > 0.25
        and checkpoint_interrupted_state.planning_recovery == interrupted_state.planning_recovery
    )

    adapter = checkpoint
    recovery_templates = _templates(
        start_tick=interrupted_state.world.tick,
        target_ids=("blue", "red"),
        prefix=f"h{horizon}-f{failure_position}-recovery",
        length=remaining,
    )
    adapter.plan_rollouts(
        (adapter.imagine_world_rollout(
            f"h{horizon}-f{failure_position}-recovery-rollout",
            "reach-world",
            recovery_templates,
        ),)
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
    final_checkpoint = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    final_checkpoint_state = final_checkpoint.cognitive_snapshot()
    recovery_gate = bool(
        interruption_gate
        and all(outcome.success is True for outcome in recovery_outcomes)
        and recovery_outcomes[-1].terminal
        and not adapter.replan_required
        and adapter._planned_rollout is None
        and final_state.planning_recovery is None
        and len(final_state.world_calibration_trace) == horizon
        and len(final_checkpoint_state.world_calibration_trace) == horizon
        and final_checkpoint._world_dynamics is not None
        and final_checkpoint._world_dynamics.online_updates == horizon
    )
    return {
        "seed": seed,
        "horizon": horizon,
        "failure_position": failure_position,
        "remaining_recovery_steps": remaining,
        "interruption_gate": interruption_gate,
        "recovery_gate": recovery_gate,
        "trace_length": len(final_state.world_calibration_trace),
        "checkpoint_trace_length": len(final_checkpoint_state.world_calibration_trace),
        "action_trace": environment.actions,
    }


def evaluate_seed(seed: int) -> list[dict[str, object]]:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    base_learner = WorldDynamicsLearner(schema, hidden_dim=32, seed=seed)
    base_learner.fit(corpus.train, epochs=250, learning_rate=0.01)
    move_case = next(case for case in corpus.train if case.action.kind == "move")
    initial = _initial_state(move_case.initial)
    cases = []
    for horizon in HORIZONS:
        for failure_position in range(horizon - 1):
            cases.append(
                _run_case(
                    seed=seed,
                    horizon=horizon,
                    failure_position=failure_position,
                    base_learner=base_learner,
                    initial=initial,
                    target_ids=schema.target_ids,
                )
            )
    return cases


def evaluate(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    runs = [run for seed in seeds for run in evaluate_seed(seed)]
    rates = {
        "interruption_gate": sum(bool(run["interruption_gate"]) for run in runs) / len(runs),
        "recovery_gate": sum(bool(run["recovery_gate"]) for run in runs) / len(runs),
    }
    return {
        "format": REPORT_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "metrics": {
            "cross_seed_rates": rates,
            "case_count": len(runs),
            "horizons": list(HORIZONS),
            "runs": runs,
        },
        "gate": {
            "passed": all(rate == 1.0 for rate in rates.values()),
            "criterion": "all seeds and all non-terminal failure positions across 3/4/5-step imagined rollouts must preserve explicit recovery state through checkpoint and finish without planner replacement",
        },
    }


def build_manifest(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "transfer runtime rollout recovery across horizon and failure position",
        "seeds": list(seeds),
        "horizons": list(HORIZONS),
        "failure_positions": "all non-terminal positions per horizon",
        "controls": ["variable-horizon", "failure-position-transfer", "recovery-checkpoint-continuation"],
        "boundary": "runtime recovery state transfer; not general failure recovery or intelligence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p7_rollout_recovery_transfer_manifest_20260826.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p7_rollout_recovery_transfer_report_20260826.json",
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
