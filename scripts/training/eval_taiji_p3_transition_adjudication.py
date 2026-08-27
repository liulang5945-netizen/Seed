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
    learner = _fit_world_learner(seed)
    first = _run_real_transition(
        learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:a",
        phase=1,
        reward=1.0,
        success=True,
    )
    first_state = first.cognitive_snapshot()
    first_trace = first_state.world_calibration_trace[-1]
    second = _run_real_transition(
        learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:b",
        phase=1,
        reward=1.0,
        success=True,
    )
    second_state = second.cognitive_snapshot()
    second_trace = second_state.world_calibration_trace[-1]
    if second_state.world_transition is None:
        raise RuntimeError("transition adjudication lost the real adapter transition")
    evidence_key = learner.schema_registry.transition_evidence_key(second_state.world_transition)
    checkpoint = second.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_learner = restored._world_dynamics
    checkpoint_registry = bool(
        restored_learner is not None
        and restored_learner.schema_registry.transition_outcome_count == 1
        and restored_learner.schema_registry.transition_confidence[evidence_key] >= 0.2
        and restored_learner.transition_acceptances == 2
        and restored_learner.transition_rejections == 0
        and restored_learner.online_updates == 2
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
        and restored._world_dynamics.online_updates == 2
    )

    rejected = _run_real_transition(
        learner,
        seed=seed,
        episode_id=f"adjudication:{seed}:c",
        phase=0,
        reward=-1.0,
        success=False,
    )
    rejected_state = rejected.cognitive_snapshot()
    rejected_trace = rejected_state.world_calibration_trace[-1]
    no_update_on_reject = bool(
        learner.online_updates == 2
        and learner.transition_acceptances == 2
        and learner.transition_rejections == 1
    )
    return {
        "seed": int(seed),
        "first_calibration": bool(first_trace.calibration_applied),
        "cross_episode_calibration": bool(second_trace.calibration_applied),
        "cross_episode_confidence": float(
            learner.schema_registry.transition_confidence[evidence_key]
        ),
        "relation_specific_holdout": bool(
            not rejected_trace.calibration_applied
            and learner.schema_registry.contradiction_count == 1
        ),
        "contradiction_rejected": bool(not rejected_trace.calibration_applied),
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
            "consistent repeated outcome increases transition confidence",
            "contradictory after-state is recorded as a conflict",
            "contradictory outcome fails closed before local SGD update",
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
        "relation_specific_holdout",
        "contradiction_rejected",
        "no_update_on_reject",
        "checkpoint_registry",
        "checkpoint_network",
        "checkpoint_continuation",
    )
    aggregate = {f"{name}_min": min(float(bool(run[name])) for run in runs) for name in metrics}
    aggregate["cross_episode_confidence_min"] = min(
        float(run["cross_episode_confidence"]) for run in runs
    )
    passed = bool(
        all(aggregate[f"{name}_min"] >= 1.0 for name in metrics)
        and aggregate["cross_episode_confidence_min"] >= 0.2
    )
    aggregate["passed"] = passed
    return {
        "format": REPORT_FORMAT,
        "seeds": runs,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "criterion": (
                "real adapter repeated outcomes must calibrate, contradictory relation outcomes "
                "must fail closed, and registry/network checkpoint continuation must pass"
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
