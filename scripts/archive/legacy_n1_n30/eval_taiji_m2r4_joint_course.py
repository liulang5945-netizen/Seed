"""Evaluate the joint native semantic course and protected-owner retention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r3_r3_multistep import (  # noqa: E402
    ENTITIES,
    RESOURCES,
    STATES,
    _content,
    _event,
    _goal,
    _world,
)
from scripts.training.eval_taiji_m2r3_r3_multistep import (  # noqa: E402
    _metrics as _transition_metrics,
)
from scripts.training.eval_taiji_m2r3_r3_multistep import (  # noqa: E402
    build_corpus as build_transition_corpus,
)
from taiji import (  # noqa: E402
    PerceptEvent,
    StructuredSemanticCorpus,
    StructuredSemanticExample,
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    TSKV8Adapter,
    WorldState,
    content_digest,
    semantic_fact_key,
)

REPORT_FORMAT = "taiji-m2r4-joint-course-v1"
REPORT_VERSION = 1
CONTROL_MODES = ("frozen-parent", "static-only", "transition-only", "joint-native")
SNAPSHOT_VARIANTS = (
    (1.00, 0.00, 1.00, 1),
    (0.95, 0.02, 0.95, 2),
    (0.90, 0.05, 0.90, 3),
    (0.85, 0.08, 0.85, 4),
)


def _snapshot_event(
    *,
    seed: int,
    sample_id: str,
    world: WorldState,
    fact_keys: tuple[str, ...],
    confidence: float,
    prediction_error: float,
    boundary_score: float,
    duration: int,
) -> PerceptEvent:
    active = {semantic_fact_key(*relation) for relation in world.relations}
    vector = [1.0 if key in active else 0.0 for key in fact_keys]
    return PerceptEvent(
        event_id=f"seed{seed}:{sample_id}:snapshot",
        observation_tick=int(world.tick),
        modality="m2r4-joint-snapshot",
        features=torch.tensor(vector, dtype=torch.float32),
        assembly_id=f"seed{seed}:{sample_id}:snapshot-assembly",
        boundary_score=float(boundary_score),
        prediction_error=float(prediction_error),
        boundary=True,
        confidence=float(confidence),
        duration=int(duration),
    )


def build_snapshot_corpus(
    seed: int,
    transition_corpus: Any,
) -> tuple[StructuredSemanticCorpus, dict[str, PerceptEvent]]:
    """Build a separate static state-encoding course from transition facts.

    Every state/resource combination appears in train, dev, and test through a
    distinct metadata variant.  The exact input digests and family IDs remain
    disjoint, while the test still checks whether the static owner can retain
    the same semantic state under a changed observation envelope.
    """

    fact_keys = tuple(transition_corpus.fact_keys)
    train: list[StructuredSemanticExample] = []
    dev: list[StructuredSemanticExample] = []
    test: list[StructuredSemanticExample] = []
    snapshots: dict[str, PerceptEvent] = {}
    tick = 100
    for _entity_index, entity in enumerate(ENTITIES):
        for _state_index, state in enumerate(STATES):
            for _resource_index, resource in enumerate(RESOURCES):
                for confidence, prediction_error, boundary_score, duration in SNAPSHOT_VARIANTS:
                    sample_id = f"{entity}-{state}-{resource}-v{duration}"
                    world = _world(entity, state, resource, tick + duration)
                    goal = _goal(state)
                    event = _snapshot_event(
                        seed=seed,
                        sample_id=sample_id,
                        world=world,
                        fact_keys=fact_keys,
                        confidence=confidence,
                        prediction_error=prediction_error,
                        boundary_score=boundary_score,
                        duration=duration,
                    )
                    content = _content(
                        state=state,
                        resource=resource,
                        tick=world.tick,
                        goal=goal,
                        sample_id=sample_id,
                    )
                    example = StructuredSemanticExample(
                        example_id=f"seed{seed}:joint:{sample_id}",
                        family_id=f"seed{seed}:joint-family:{sample_id}",
                        percept=event,
                        world=world,
                        goal=goal,
                        content=content,
                    )
                    snapshots[sample_id] = event
                    if duration < 3:
                        train.append(example)
                    elif duration == 3:
                        dev.append(example)
                    else:
                        test.append(example)
                tick += 5
    return StructuredSemanticCorpus.from_splits(train=train, dev=dev, test=test), snapshots


def _static_metrics(
    learner: StructuredSemanticLearner,
    examples: tuple[StructuredSemanticExample, ...],
) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    goal_hits = content_hits = status_hits = 0
    for example in examples:
        result = learner.predict(example.percept)
        expected = set(example.fact_keys)
        actual = (
            set()
            if result.world is None
            else {semantic_fact_key(*relation) for relation in result.world.relations}
        )
        true_positive += len(expected & actual)
        false_positive += len(actual - expected)
        false_negative += len(expected - actual)
        goal_hits += int(result.goal is not None and result.goal.goal_id == example.goal.goal_id)
        content_hits += int(
            result.content_plan is not None
            and result.content_plan.content_id == example.content.content_id
        )
        expected_status = (
            "clarify" if example.content.intent_kind == "request_information" else "resolved"
        )
        status_hits += int(result.status == expected_status)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "examples": len(examples),
        "fact_f1": 2.0 * precision * recall / max(1e-9, precision + recall),
        "goal_accuracy": goal_hits / len(examples),
        "content_accuracy": content_hits / len(examples),
        "status_accuracy": status_hits / len(examples),
    }


def _run_adapter_sequence(
    adapter: TSKV8Adapter,
    seed: int,
    initial_snapshot: PerceptEvent,
) -> tuple[tuple[dict[str, Any], ...], WorldState | None]:
    semantic = adapter.infer_structured_semantics(initial_snapshot)
    current = semantic.world
    results: list[dict[str, Any]] = []
    if current is None:
        return tuple(results), None
    for tick, kind in enumerate(("release", "move", "block"), 1):
        result = adapter.infer_structured_transition(current, _event(seed, kind, tick))
        results.append(result.to_payload())
        if result.world is None:
            return tuple(results), None
        current = result.world
    return tuple(results), current


def _fact_keys(world: WorldState | None) -> set[str]:
    return (
        set() if world is None else {semantic_fact_key(*relation) for relation in world.relations}
    )


def _run_control(seed: int, mode: str) -> dict[str, Any]:
    transition_corpus = build_transition_corpus(seed)
    static_corpus, snapshots = build_snapshot_corpus(seed, transition_corpus)
    static_learner = StructuredSemanticLearner(static_corpus)
    transition_learner = StructuredSemanticTransitionLearner(transition_corpus)
    static_before = static_learner.owner_digests()
    transition_before = transition_learner.owner_digests()
    static_losses: dict[str, float] = {}
    transition_losses: dict[str, float] = {}
    if mode in {"static-only", "joint-native"}:
        static_losses = static_learner.fit(static_corpus.train, epochs=220, learning_rate=1.6)
    if mode in {"transition-only", "joint-native"}:
        transition_losses = transition_learner.fit(
            transition_corpus.train,
            epochs=240,
            learning_rate=0.5,
        )
    static_after = static_learner.owner_digests()
    transition_after = transition_learner.owner_digests()
    static_train = _static_metrics(static_learner, static_corpus.train)
    static_dev = _static_metrics(static_learner, static_corpus.dev)
    static_test = _static_metrics(static_learner, static_corpus.test)
    transition_train = _transition_metrics(transition_learner, transition_corpus.train)
    transition_dev = _transition_metrics(transition_learner, transition_corpus.dev)
    transition_test = _transition_metrics(transition_learner, transition_corpus.test)

    adapter = TSKV8Adapter()
    plain_checkpoint = adapter.native_checkpoint()
    cognitive_before = content_digest(adapter.cognitive_snapshot().to_payload())
    adapter.attach_structured_semantic_learner(static_learner)
    adapter.attach_structured_semantic_transition_learner(transition_learner)
    joint_checkpoint = adapter.native_checkpoint()
    initial_snapshot = snapshots["alpha-pending-cache-v4"]
    direct_results, direct_final = _run_adapter_sequence(adapter, seed, initial_snapshot)
    cognitive_after = content_digest(adapter.cognitive_snapshot().to_payload())
    restored = TSKV8Adapter.from_native_checkpoint(joint_checkpoint)
    stable_checkpoint = content_digest(joint_checkpoint) == content_digest(
        restored.native_checkpoint()
    )
    restored_results, restored_final = _run_adapter_sequence(
        restored,
        seed,
        initial_snapshot,
    )
    expected_final = {
        semantic_fact_key("agent", "tracks", "alpha"),
        semantic_fact_key("agent", "state", "blocked"),
        semantic_fact_key("agent", "holds", "queue"),
    }
    static_changed = any(static_before[name] != static_after[name] for name in static_before)
    transition_changed = any(
        transition_before[name] != transition_after[name] for name in transition_before
    )
    expected_static_changed = mode in {"static-only", "joint-native"}
    expected_transition_changed = mode in {"transition-only", "joint-native"}
    composition = len(direct_results) == 3 and _fact_keys(direct_final) == expected_final
    expected_composition = mode == "joint-native"
    checks = {
        "static_record_disjoint": static_corpus.manifest()["record_disjoint"],
        "transition_record_disjoint": transition_corpus.manifest()["record_disjoint"],
        "static_owner_write_scope": static_changed == expected_static_changed,
        "transition_owner_write_scope": transition_changed == expected_transition_changed,
        "static_test_learning": (
            static_test["fact_f1"] >= 0.80
            if mode in {"static-only", "joint-native"}
            else static_test["fact_f1"] < 0.80
        ),
        "transition_test_learning": (
            transition_test["fact_f1"] >= 0.80
            if mode in {"transition-only", "joint-native"}
            else transition_test["fact_f1"] < 0.80
        ),
        "joint_composition_scope": composition == expected_composition,
        "joint_checkpoint_roundtrip": (
            len(direct_results) == len(restored_results)
            and content_digest(direct_results) == content_digest(restored_results)
            and content_digest(None if direct_final is None else direct_final.to_payload())
            == content_digest(None if restored_final is None else restored_final.to_payload())
        ),
        "joint_checkpoint_stable": stable_checkpoint,
        "cognitive_state_read_only": cognitive_before == cognitive_after,
        "native_components_present": (
            "structured_semantic" in joint_checkpoint["components"]
            and "structured_semantic_transition" in joint_checkpoint["components"]
        ),
        "plain_parent_has_no_new_components": (
            "structured_semantic" not in plain_checkpoint["components"]
            and "structured_semantic_transition" not in plain_checkpoint["components"]
        ),
        "no_execution_side_effect": (
            adapter.last_task_interpretation is None
            and adapter.last_task_decomposition is None
            and adapter.last_semantic_provider_evidence is None
        ),
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "mode": mode,
        "status": "passed" if passed else "failed",
        "corpus": {
            "static": static_corpus.manifest(),
            "transition": transition_corpus.manifest(),
        },
        "training": {
            "static": {"epochs": 220, "learning_rate": 1.6, "losses": static_losses},
            "transition": {
                "epochs": 240,
                "learning_rate": 0.5,
                "losses": transition_losses,
            },
        },
        "metrics": {
            "static": {"train": static_train, "dev": static_dev, "test": static_test},
            "transition": {
                "train": transition_train,
                "dev": transition_dev,
                "test": transition_test,
            },
            "composition": {
                "steps": len(direct_results),
                "final_world_digest": (
                    None if direct_final is None else content_digest(direct_final.to_payload())
                ),
            },
        },
        "preflight": {
            "static_parameter_count": static_learner.parameter_count,
            "transition_parameter_count": transition_learner.parameter_count,
            "static_owner_changed": static_changed,
            "transition_owner_changed": transition_changed,
        },
        "gate": {"passed": passed, "checks": checks},
    }


def _run_seed(seed: int) -> dict[str, Any]:
    controls = {mode: _run_control(seed, mode) for mode in CONTROL_MODES}
    joint = controls["joint-native"]
    static_only = controls["static-only"]
    transition_only = controls["transition-only"]
    retention_checks = {
        "static_owner_retention": (
            joint["metrics"]["static"]["test"]["fact_f1"]
            >= static_only["metrics"]["static"]["test"]["fact_f1"]
        ),
        "transition_owner_retention": (
            joint["metrics"]["transition"]["test"]["fact_f1"]
            >= transition_only["metrics"]["transition"]["test"]["fact_f1"]
        ),
        "joint_new_composition": joint["gate"]["checks"]["joint_composition_scope"],
        "all_controls_have_expected_scope": all(
            control["gate"]["passed"] for control in controls.values()
        ),
    }
    passed = all(bool(value) for value in retention_checks.values())
    return {
        "seed": seed,
        "status": "passed" if passed else "failed",
        "controls": controls,
        "retention": retention_checks,
        "gate": {"passed": passed, "checks": retention_checks},
    }


def evaluate(seeds: tuple[int, ...] = (11, 29, 47)) -> dict[str, Any]:
    runs = tuple(_run_seed(int(seed)) for seed in seeds)
    passed = all(run["status"] == "passed" for run in runs)
    joint_metrics = [run["controls"]["joint-native"]["metrics"] for run in runs]
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "status": "passed" if passed else "failed",
        "can_promote": False,
        "seeds": list(seeds),
        "runs": list(runs),
        "aggregate": {
            "all_seeds_passed": passed,
            "mean_joint_static_test_fact_f1": sum(
                item["static"]["test"]["fact_f1"] for item in joint_metrics
            )
            / len(joint_metrics),
            "mean_joint_transition_test_fact_f1": sum(
                item["transition"]["test"]["fact_f1"] for item in joint_metrics
            )
            / len(joint_metrics),
            "mean_joint_sequence_steps": sum(item["composition"]["steps"] for item in joint_metrics)
            / len(joint_metrics),
        },
        "gate": {
            "passed": passed,
            "criterion": "every seed preserves both owners and composes them in runtime",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m2r4_joint_course_20260907.json",
    )
    args = parser.parse_args()
    report = evaluate(tuple(args.seeds))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
