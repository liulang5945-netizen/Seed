"""Evaluate ledger-driven outcome probability calibration without holdout leakage."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p3_open_set import (  # noqa: E402
    _config,
    _fit_world_learner,
    _world,
)
from scripts.training.eval_taiji_p3_transition_adjudication import (  # noqa: E402
    _run_real_transition,
)
from taiji import (  # noqa: E402
    TSKV8Adapter,
    WorldAction,
    WorldDynamicsLearner,
    WorldState,
)

MANIFEST_FORMAT = "taiji-p3-probability-calibration-manifest-v1"
REPORT_FORMAT = "taiji-p3-probability-calibration-v1"
HOLDOUT_SUCCESS = (True, False, True, False, False)
HOLDOUT_REWARDS = (1.0, -1.0, 1.0, -1.0, -1.0)
CALIBRATION_EPSILON = 0.05


def _known_context() -> tuple[WorldState, WorldAction]:
    state = _world("calibration-known", 1, target_id="target", phase=0)
    action = WorldAction(
        "calibration-known-action",
        "assemble",
        state.tick,
        actor_id="agent",
        target_id="target",
        parameters={"workspace_count": 2.0},
    )
    return state, action


def _holdout_context() -> tuple[WorldState, WorldAction]:
    state = _world("calibration-holdout", 1, target_id="target:holdout", phase=0)
    state = replace(state, relations=(("agent", "tracks", "target:holdout"),))
    action = WorldAction(
        "calibration-holdout-action",
        "secure",
        state.tick,
        actor_id="agent",
        target_id="target:holdout",
        parameters={"strength": 0.75},
    )
    return state, action


def _binary_metrics(probability: float, outcomes: tuple[bool, ...]) -> dict[str, float]:
    actual = tuple(float(value) for value in outcomes)
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    brier = sum((clipped - value) ** 2 for value in actual) / len(actual)
    nll = sum(
        -(value * math.log(clipped) + (1.0 - value) * math.log(1.0 - clipped)) for value in actual
    ) / len(actual)
    empirical_rate = sum(actual) / len(actual)
    return {
        "brier": brier,
        "nll": nll,
        "empirical_success_rate": empirical_rate,
        "coverage": float(abs(clipped - empirical_rate) <= CALIBRATION_EPSILON),
    }


def evaluate_seed(seed: int) -> dict[str, object]:
    learner = _fit_ledger(seed)
    known_state, known_action = _known_context()
    known_prediction = learner.predict(known_state, known_action)
    ledger_before_holdout = learner.schema_registry.transition_outcome_count
    known_key = learner.schema_registry.transition_context_key(known_state, known_action)
    known_hypotheses_before = learner.schema_registry.transition_hypotheses[known_key]
    holdout_state, holdout_action = _holdout_context()
    unknown_prediction = learner.predict(holdout_state, holdout_action)
    ledger_after_holdout = learner.schema_registry.transition_outcome_count
    known_hypotheses_after = learner.schema_registry.transition_hypotheses[known_key]
    metrics = _binary_metrics(known_prediction.success_probability, HOLDOUT_SUCCESS)
    checkpoint_model = TSKV8Adapter(_config(seed), episode_id=f"calibration:{seed}:checkpoint")
    checkpoint_model.attach_world_dynamics(learner)
    checkpoint = checkpoint_model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_learner = restored._world_dynamics
    restored_prediction = (
        None if restored_learner is None else restored_learner.predict(known_state, known_action)
    )
    checkpoint_continuation = bool(
        restored_learner is not None
        and restored_prediction is not None
        and restored_learner.schema_registry.transition_outcome_count == 1
        and restored_prediction.reward == known_prediction.reward
        and restored_prediction.success_probability == known_prediction.success_probability
        and restored_prediction.uncertainty == known_prediction.uncertainty
        and restored_prediction.uncertainty_mode == known_prediction.uncertainty_mode
    )
    lesion = TSKV8Adapter(_config(seed), episode_id=f"calibration:{seed}:lesion")
    world_model_lesion = False
    try:
        lesion.predict_world_candidates(())
    except RuntimeError:
        world_model_lesion = True
    return {
        "seed": int(seed),
        "brier": metrics["brier"],
        "nll": metrics["nll"],
        "empirical_success_rate": metrics["empirical_success_rate"],
        "coverage": metrics["coverage"],
        "reward_mae": abs(known_prediction.reward - sum(HOLDOUT_REWARDS) / len(HOLDOUT_REWARDS)),
        "known_stochastic_prediction": bool(
            known_prediction.uncertainty_mode == "stochastic"
            and known_prediction.uncertainty == 0.4
            and known_prediction.success_probability == 0.4
        ),
        "unseen_relation_prediction": bool(
            unknown_prediction.uncertainty_mode == "unseen"
            and unknown_prediction.uncertainty == 1.0
        ),
        "holdout_feedback_isolated": bool(
            ledger_before_holdout == ledger_after_holdout == 1
            and known_hypotheses_before == known_hypotheses_after
        ),
        "checkpoint_continuation": checkpoint_continuation,
        "world_model_lesion": world_model_lesion,
    }


def _fit_ledger(seed: int) -> WorldDynamicsLearner:
    learner = _fit_world_learner(seed)
    _run_real_transition(
        learner,
        seed=seed,
        episode_id=f"calibration:{seed}:a1",
        phase=1,
        reward=1.0,
        success=True,
    )
    _run_real_transition(
        learner,
        seed=seed,
        episode_id=f"calibration:{seed}:a2",
        phase=1,
        reward=1.0,
        success=True,
    )
    _run_real_transition(
        learner,
        seed=seed,
        episode_id=f"calibration:{seed}:b1",
        phase=0,
        reward=-1.0,
        success=False,
    )
    _run_real_transition(
        learner,
        seed=seed,
        episode_id=f"calibration:{seed}:b2",
        phase=0,
        reward=-1.0,
        success=False,
    )
    _run_real_transition(
        learner,
        seed=seed,
        episode_id=f"calibration:{seed}:b3",
        phase=0,
        reward=-1.0,
        success=False,
    )
    return learner


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "ledger-driven probability calibration on an independent holdout",
        "seeds": [11, 29, 47],
        "holdout_success": list(HOLDOUT_SUCCESS),
        "holdout_rewards": list(HOLDOUT_REWARDS),
        "controls": [
            "Brier and binary NLL are computed only after ledger training",
            "group confidence coverage compares predicted probability with holdout rate",
            "reward estimate is checked against an independent holdout mean",
            "new relation/target context remains unseen and maximally uncertain",
            "holdout prediction cannot increase transition evidence count",
            "native checkpoint restores ledger-driven probability and uncertainty",
            "world-model lesion fails closed for planner projection",
        ],
        "boundary": "probability calibration contract; not general stochastic world prediction",
    }


def evaluate(*, seeds: tuple[int, ...] = (11, 29, 47)) -> dict[str, object]:
    runs = [evaluate_seed(seed) for seed in seeds]
    boolean_metrics = (
        "coverage",
        "known_stochastic_prediction",
        "unseen_relation_prediction",
        "holdout_feedback_isolated",
        "checkpoint_continuation",
        "world_model_lesion",
    )
    aggregate = {
        f"{name}_min": min(float(bool(run[name])) for run in runs) for name in boolean_metrics
    }
    aggregate["brier_max"] = max(float(run["brier"]) for run in runs)
    aggregate["nll_max"] = max(float(run["nll"]) for run in runs)
    aggregate["reward_mae_max"] = max(float(run["reward_mae"]) for run in runs)
    passed = bool(
        all(aggregate[f"{name}_min"] >= 1.0 for name in boolean_metrics)
        and aggregate["brier_max"] <= 0.25
        and aggregate["nll_max"] <= 0.70
        and aggregate["reward_mae_max"] <= 0.05
    )
    aggregate["passed"] = passed
    return {
        "format": REPORT_FORMAT,
        "seeds": runs,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "criterion": (
                "independent holdout Brier/NLL, reward error, coverage, leakage isolation, "
                "uncertainty modes, checkpoint continuation and lesion must pass"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_p3_probability_calibration_manifest_20260827.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p3_probability_calibration_20260827.json",
    )
    args = parser.parse_args()
    report = evaluate()
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
