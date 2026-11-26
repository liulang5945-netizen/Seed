"""Evaluate real-adapter transition evidence adjudication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p3_open_set import (  # noqa: E402
    _config,
    _fit_world_learner,
    _transition_state,
    _world,
)
from taiji import (  # noqa: E402
    Observation,
    TSKV8Adapter,
    WorldAction,
    WorldDynamicsLearner,
)

MANIFEST_FORMAT = "taiji-p3-transition-adjudication-manifest-v1"
REPORT_FORMAT = "taiji-p3-transition-adjudication-v1"


def _run_real_transition(
    learner: WorldDynamicsLearner,
    *,
    seed: int,
    episode_id: str,
    phase: int,
    reward: float,
    success: bool,
) -> TSKV8Adapter:
    model = TSKV8Adapter(
        _config(seed),
        episode_id=episode_id,
    )
    model.attach_world_dynamics(learner)
    model.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="p3-transition-adjudication",
        ),
        learn=False,
        world_state=_world(
            "adjudication",
            1,
            target_id="target",
            phase=0,
        ),
    )
    before = model.cognitive_snapshot().world
    action = WorldAction(
        f"{episode_id}:assemble",
        "assemble",
        before.tick,
        actor_id="agent",
        target_id="target",
        parameters={"workspace_count": 2.0},
        provenance="p3-transition-adjudication",
    )
    model.act((97, 98), sample=False, world_action=action)
    after = _transition_state(
        before,
        sample_id="adjudication",
        target_id="target",
        phase=phase,
        success=success,
    )
    model.settle_action(
        reward,
        learn=False,
        learn_world=True,
        world_state=after,
        success=success,
    )
    return model


def evaluate_seed(seed: int) -> dict[str, object]:
    deterministic_learner = _fit_world_learner(seed)
    deterministic_first = _run_real_transition(
        deterministic_learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:deterministic-a",
        phase=1,
        reward=1.0,
        success=True,
    )
    deterministic_first_trace = deterministic_first.cognitive_snapshot().world_calibration_trace[-1]
    deterministic_repeat = _run_real_transition(
        deterministic_learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:deterministic-b",
        phase=1,
        reward=1.0,
        success=True,
    )
    deterministic_repeat_trace = deterministic_repeat.cognitive_snapshot().world_calibration_trace[
        -1
    ]
    deterministic_repeat_state = deterministic_repeat.cognitive_snapshot()
    if deterministic_repeat_state.world_transition is None:
        raise RuntimeError("deterministic repeat lost the real adapter transition")
    deterministic_key = deterministic_learner.schema_registry.transition_evidence_key(
        deterministic_repeat_state.world_transition
    )
    deterministic_repeat_confidence = deterministic_learner.schema_registry.transition_confidence[
        deterministic_key
    ]
    deterministic_rejected = _run_real_transition(
        deterministic_learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:deterministic-c",
        phase=0,
        reward=-1.0,
        success=False,
    )
    deterministic_state = deterministic_rejected.cognitive_snapshot()
    deterministic_trace = deterministic_state.world_calibration_trace[-1]
    if deterministic_state.world_transition is None:
        raise RuntimeError("deterministic adjudication lost the real adapter transition")
    deterministic_key = deterministic_learner.schema_registry.transition_evidence_key(
        deterministic_state.world_transition
    )
    deterministic_hypotheses = deterministic_learner.schema_registry.transition_hypotheses[
        deterministic_key
    ]

    stochastic_learner = _fit_world_learner(seed + 1000)
    _run_real_transition(
        stochastic_learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:stochastic-a1",
        phase=1,
        reward=1.0,
        success=True,
    )
    _run_real_transition(
        stochastic_learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:stochastic-b1",
        phase=0,
        reward=-1.0,
        success=False,
    )
    _run_real_transition(
        stochastic_learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:stochastic-b2",
        phase=0,
        reward=-1.0,
        success=False,
    )
    stochastic_a2 = _run_real_transition(
        stochastic_learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:stochastic-a2",
        phase=1,
        reward=1.0,
        success=True,
    )
    stochastic_b3 = _run_real_transition(
        stochastic_learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:stochastic-b3",
        phase=0,
        reward=-1.0,
        success=False,
    )
    stochastic_state = stochastic_b3.cognitive_snapshot()
    if stochastic_state.world_transition is None:
        raise RuntimeError("stochastic adjudication lost the real adapter transition")
    evidence_key = stochastic_learner.schema_registry.transition_evidence_key(
        stochastic_state.world_transition
    )
    checkpoint = stochastic_b3.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_learner = restored._world_dynamics
    checkpoint_registry = bool(
        restored_learner is not None
        and restored_learner.schema_registry.transition_outcome_count == 1
        and restored_learner.schema_registry.transition_outcome_mode(evidence_key) == "stochastic"
        and restored_learner.schema_registry.transition_confidence[evidence_key]
        == stochastic_learner.schema_registry.transition_confidence[evidence_key]
        and restored_learner.transition_acceptances == 3
        and restored_learner.transition_rejections == 2
        and restored_learner.online_updates == 3
    )
    checkpoint_network = bool(
        restored_learner is not None
        and all(
            torch.equal(value, restored_learner.state_dict()[name])
            for name, value in checkpoint["components"]["world_dynamics"]["state_dict"].items()
        )
    )
    restored.observe(123, learn=False)
    checkpoint_continuation = bool(
        restored._world_dynamics is not None
        and restored._world_dynamics.schema_registry.transition_outcome_count == 1
        and restored._world_dynamics.schema_registry.transition_outcome_mode(evidence_key)
        == "stochastic"
        and restored._world_dynamics.online_updates == 3
    )
    stochastic_hypotheses = stochastic_learner.schema_registry.transition_hypotheses[evidence_key]
    known_prediction = deterministic_repeat_trace.prediction
    conflicted_prediction = stochastic_a2.cognitive_snapshot().world_prediction
    stochastic_prediction = stochastic_b3.cognitive_snapshot().world_prediction
    if conflicted_prediction is None or stochastic_prediction is None:
        raise RuntimeError("prediction uncertainty trace was not retained")
    no_update_on_reject = bool(
        stochastic_a2.cognitive_snapshot().world_calibration_trace[-1].calibration_applied is False
        and stochastic_a2.cognitive_snapshot()
        .world_calibration_trace[-1]
        .online_update_count_before
        == stochastic_a2.cognitive_snapshot().world_calibration_trace[-1].online_update_count_after
        and stochastic_learner.online_updates == 3
        and stochastic_learner.transition_acceptances == 3
        and stochastic_learner.transition_rejections == 2
    )
    return {
        "seed": int(seed),
        "first_calibration": bool(deterministic_first_trace.calibration_applied),
        "cross_episode_calibration": bool(deterministic_repeat_trace.calibration_applied),
        "cross_episode_confidence": float(deterministic_repeat_confidence),
        "known_prediction_uncertainty": bool(
            known_prediction.uncertainty_mode == "deterministic"
            and known_prediction.uncertainty == 0.0
        ),
        "conflicted_prediction_uncertainty": bool(
            conflicted_prediction.uncertainty_mode == "conflicted"
            and conflicted_prediction.uncertainty == 1.0
        ),
        "stochastic_prediction_uncertainty": bool(
            stochastic_prediction.uncertainty_mode == "stochastic"
            and stochastic_prediction.uncertainty == 0.5
        ),
        "relation_specific_holdout": bool(
            not deterministic_trace.calibration_applied
            and deterministic_learner.schema_registry.contradiction_count == 1
            and deterministic_learner.schema_registry.transition_outcome_mode(deterministic_key)
            == "conflicted"
            and sorted(item["evidence_count"] for item in deterministic_hypotheses) == [1, 2]
        ),
        "contradiction_rejected": bool(not deterministic_trace.calibration_applied),
        "stochastic_tie_rejected": bool(
            not stochastic_a2.cognitive_snapshot().world_calibration_trace[-1].calibration_applied
        ),
        "stochastic_mode": bool(
            stochastic_learner.schema_registry.transition_outcome_mode(evidence_key) == "stochastic"
        ),
        "stochastic_clear_leader": bool(
            stochastic_b3.cognitive_snapshot().world_calibration_trace[-1].calibration_applied
            and sorted(item["evidence_count"] for item in stochastic_hypotheses) == [2, 3]
        ),
        "stochastic_confidence": float(
            stochastic_learner.schema_registry.transition_confidence[evidence_key]
        ),
        "no_update_on_reject": no_update_on_reject,
        "checkpoint_registry": checkpoint_registry,
        "checkpoint_network": checkpoint_network,
        "checkpoint_continuation": checkpoint_continuation,
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "real TSKV8Adapter transition outcome adjudication across episodes",
        "seeds": [11, 29, 47],
        "controls": [
            "stable semantic before/action evidence key excludes tick and event ids",
            "outcome ledger stores multiple after-state hypotheses with evidence counts",
            "consistent repeated outcome increases the leading hypothesis share",
            "known deterministic context reports zero ledger outcome uncertainty",
            "one-off contradictory after-state remains conflicted and is recorded",
            "conflicted context reports maximal uncertainty to planning consumers",
            "only a clear leader can pass local SGD adjudication",
            "repeatable stochastic outcomes become an explicit stochastic ledger mode",
            "stochastic uncertainty equals one minus the leading outcome probability",
            "ambiguous stochastic tie fails closed before local SGD update",
            "prediction calibration trace reflects accepted versus rejected feedback",
            "registry and network state survive native checkpoint continuation",
            "relation-specific after-state differences remain distinguishable",
        ],
        "boundary": "transition adjudication safety; not open-domain semantics or general intelligence",
    }


def evaluate(*, seeds: tuple[int, ...] = (11, 29, 47)) -> dict[str, object]:
    runs = [evaluate_seed(seed) for seed in seeds]
    metrics = (
        "first_calibration",
        "cross_episode_calibration",
        "known_prediction_uncertainty",
        "conflicted_prediction_uncertainty",
        "stochastic_prediction_uncertainty",
        "relation_specific_holdout",
        "contradiction_rejected",
        "stochastic_tie_rejected",
        "stochastic_mode",
        "stochastic_clear_leader",
        "no_update_on_reject",
        "checkpoint_registry",
        "checkpoint_network",
        "checkpoint_continuation",
    )
    aggregate = {f"{name}_min": min(float(bool(run[name])) for run in runs) for name in metrics}
    aggregate["cross_episode_confidence_min"] = min(
        float(run["cross_episode_confidence"]) for run in runs
    )
    aggregate["stochastic_confidence_min"] = min(
        float(run["stochastic_confidence"]) for run in runs
    )
    passed = bool(
        all(aggregate[f"{name}_min"] >= 1.0 for name in metrics)
        and aggregate["cross_episode_confidence_min"] >= 1.0
        and aggregate["stochastic_confidence_min"] >= 0.6
    )
    aggregate["passed"] = passed
    return {
        "format": REPORT_FORMAT,
        "seeds": runs,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "criterion": (
                "real adapter deterministic conflicts must fail closed, repeatable stochastic "
                "outcomes must form a clear-ledger mode, and registry/network checkpoint "
                "continuation must pass"
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
        / "taiji_p3_transition_adjudication_manifest_20260827.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p3_transition_adjudication_20260827.json",
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
