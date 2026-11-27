"""M3.R2: verify native Workbench task-state transitions on CPU.

R1 learned a static mapping from one verified Workbench observation to
structured semantics.  R2 keeps the same evidence boundary and adds only a
temporal question: after the project state changes, can a native transition
owner remove stale facts and retain the new Goal/ContentPlan across ticks?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m3r1_native_observation import (  # noqa: E402
    SPLITS,
    _semantic_kind,
    build_course,
)
from seed.persistence import atomic_save  # noqa: E402
from taiji import (  # noqa: E402
    StructuredSemanticTransitionCorpus,
    StructuredSemanticTransitionExample,
    StructuredSemanticTransitionLearner,
    TSKV8Adapter,
    WorkbenchObservation,
    WorldState,
    content_digest,
    semantic_fact_key,
)

REPORT_FORMAT = "taiji-m3r2-native-task-state-sequence-v1"
REPORT_VERSION = 1
SEQUENCE_PATHS = ("missing", "app.py", "index.ts", "main.rs", "shared.h", "missing")


def _world(observation: WorkbenchObservation, *, tick: int) -> WorldState:
    relations = (
        (
            "workbench",
            "read",
            "success" if observation.read_success else "failure",
        ),
        (
            "workbench",
            "target",
            "file" if observation.file_is_file else "missing",
        ),
        ("workbench", "language", observation.language_id),
        ("workbench", "language_state", observation.selection_state),
        (
            "workbench",
            "toolchain",
            "available" if observation.toolchain_available else "missing",
        ),
        (
            "workbench",
            "diagnostics",
            "connected" if observation.diagnostics_connected else "disconnected",
        ),
    )
    event = observation.to_percept_event(tick=tick)
    return WorldState(
        tick=tick,
        latent=torch.empty(0),
        entities=("workbench",),
        relations=relations,
        uncertainty=0.0,
        percept_event_id=event.event_id,
        percept_assembly_id=event.assembly_id,
    )


def _sequence_for_split(
    observations: tuple[WorkbenchObservation, ...],
) -> tuple[WorkbenchObservation, ...]:
    by_path = {item.path: item for item in observations}
    missing = by_path[next(path for path in by_path if path.startswith("missing_"))]
    return (
        missing,
        by_path["app.py"],
        by_path["index.ts"],
        by_path["main.rs"],
        by_path["shared.h"],
        missing,
    )


def _build_split_transitions(
    *,
    split: str,
    observations: tuple[WorkbenchObservation, ...],
    static_corpus: Any,
) -> tuple[StructuredSemanticTransitionExample, ...]:
    static_examples = (
        static_corpus.train
        if split == "train"
        else static_corpus.dev if split == "dev" else static_corpus.test
    )
    catalog = {
        item.example_id.rsplit(":", 1)[-1]: (item.goal, item.content) for item in static_examples
    }
    sequence = _sequence_for_split(observations)
    result = []
    for index in range(1, len(sequence)):
        before_observation = sequence[index - 1]
        current_observation = sequence[index]
        event = current_observation.to_percept_event(tick=index)
        goal, content = catalog[current_observation.path]
        # Recreate the content target at the transition tick.  The target IDs
        # remain catalogued by R1, but tick is part of the transition contract.
        content = type(content).from_payload({**content.to_payload(), "tick": index})
        result.append(
            StructuredSemanticTransitionExample(
                example_id=f"m3r2:{split}:step-{index}:{current_observation.path}",
                family_id=f"m3r2:{split}:sequence",
                before=_world(before_observation, tick=index - 1),
                event=event,
                after=_world(current_observation, tick=index),
                goal=goal,
                content=content,
            )
        )
    return tuple(result)


def build_transition_course() -> tuple[
    StructuredSemanticTransitionCorpus,
    dict[str, tuple[WorkbenchObservation, ...]],
]:
    static_corpus, observations = build_course()
    transitions = {
        split: _build_split_transitions(
            split=split,
            observations=observations[split],
            static_corpus=static_corpus,
        )
        for split in SPLITS
    }
    corpus = StructuredSemanticTransitionCorpus.from_splits(
        train=transitions["train"],
        dev=transitions["dev"],
        test=transitions["test"],
    )
    return corpus, observations


def _metrics(
    learner: StructuredSemanticTransitionLearner,
    examples: tuple[StructuredSemanticTransitionExample, ...],
) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    goal_hits = content_hits = status_hits = 0
    for example in examples:
        result = learner.predict(example.before, example.event)
        expected = set(example.after_fact_keys)
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
    count = max(1, len(examples))
    return {
        "examples": len(examples),
        "fact_f1": 2.0 * precision * recall / max(1e-9, precision + recall),
        "goal_accuracy": goal_hits / count,
        "content_accuracy": content_hits / count,
        "status_accuracy": status_hits / count,
    }


def _sequence_metrics(
    learner: StructuredSemanticTransitionLearner,
    observations: tuple[WorkbenchObservation, ...],
) -> dict[str, Any]:
    current = _world(observations[0], tick=0)
    rows: list[dict[str, Any]] = []
    fact_hits = goal_hits = content_hits = status_hits = 0
    for tick, observation in enumerate(observations[1:], start=1):
        event = observation.to_percept_event(tick=tick)
        result = learner.predict(current, event)
        expected = set(
            semantic_fact_key(*relation) for relation in _world(observation, tick=tick).relations
        )
        actual = (
            set()
            if result.world is None
            else {semantic_fact_key(*relation) for relation in result.world.relations}
        )
        fact_hits += int(expected == actual)
        kind = _semantic_kind(observation)
        expected_goal = f"goal:{kind}"
        expected_content = f"content:{kind}"
        goal_hits += int(result.goal is not None and result.goal.goal_id == expected_goal)
        content_hits += int(
            result.content_plan is not None and result.content_plan.content_id == expected_content
        )
        status_hits += int(
            result.status == ("clarify" if kind != "inspect-language" else "resolved")
        )
        rows.append(
            {
                "tick": tick,
                "path": observation.path,
                "expected_facts": sorted(expected),
                "actual_facts": sorted(actual),
                "goal": None if result.goal is None else result.goal.goal_id,
                "content": None if result.content_plan is None else result.content_plan.content_id,
                "status": result.status,
            }
        )
        if result.world is not None:
            current = result.world
    count = max(1, len(rows))
    return {
        "steps": len(rows),
        "fact_exact_accuracy": fact_hits / count,
        "goal_accuracy": goal_hits / count,
        "content_accuracy": content_hits / count,
        "status_accuracy": status_hits / count,
        "rows": rows,
        "final_world_digest": content_digest(current.to_payload()),
    }


def _static_only_sequence(observations: tuple[WorkbenchObservation, ...]) -> dict[str, Any]:
    """A frozen-state control: it never applies an observation transition."""

    initial = _world(observations[0], tick=0)
    expected = [
        _world(item, tick=index).relations for index, item in enumerate(observations[1:], start=1)
    ]
    actual = [initial.relations for _ in expected]
    exact = sum(int(left == right) for left, right in zip(expected, actual, strict=True)) / max(
        1, len(expected)
    )
    return {"steps": len(expected), "fact_exact_accuracy": exact}


def _sequence_world_digests(
    learner: StructuredSemanticTransitionLearner,
    observations: Sequence[WorkbenchObservation],
) -> tuple[str, ...]:
    current = _world(observations[0], tick=0)
    digests = []
    for tick, observation in enumerate(observations[1:], start=1):
        result = learner.predict(current, observation.to_percept_event(tick=tick))
        current = result.world or current
        digests.append(content_digest(current.to_payload()))
    return tuple(digests)


def _runtime_gate(
    learner: StructuredSemanticTransitionLearner,
    observations: tuple[WorkbenchObservation, ...],
) -> dict[str, Any]:
    adapter = TSKV8Adapter()
    state_before = content_digest(adapter.cognitive_snapshot().to_payload())
    adapter.attach_structured_semantic_transition_learner(learner)
    current = _world(observations[0], tick=0)
    direct = []
    for tick, observation in enumerate(observations[1:], start=1):
        result = adapter.infer_structured_transition(
            current,
            observation.to_percept_event(tick=tick),
        )
        direct.append(result.to_payload())
        current = result.world or current
    state_after = content_digest(adapter.cognitive_snapshot().to_payload())
    checkpoint = adapter.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    current = _world(observations[0], tick=0)
    restored_results = []
    for tick, observation in enumerate(observations[1:], start=1):
        result = restored.infer_structured_transition(
            current,
            observation.to_percept_event(tick=tick),
        )
        restored_results.append(result.to_payload())
        current = result.world or current
    direct_digest = content_digest(direct)
    restored_digest = content_digest(restored_results)
    return {
        "owner_attached": adapter.structured_semantic_transition_learner is learner,
        "checkpoint_component_present": "structured_semantic_transition"
        in checkpoint["components"],
        "restored_owner_present": restored.structured_semantic_transition_learner is not None,
        "output_stable_after_runtime_restore": direct_digest == restored_digest,
        "cognitive_state_read_only": state_before
        == state_after
        == content_digest(restored.cognitive_snapshot().to_payload()),
        "runtime_checkpoint_stable": content_digest(checkpoint)
        == content_digest(restored.native_checkpoint()),
        "output_digest": direct_digest,
        "restored_output_digest": restored_digest,
    }


def run_gate(output_path: Path | None = None) -> dict[str, Any]:
    static_corpus, _ = build_course()
    corpus, observations = build_transition_course()
    test_sequence = _sequence_for_split(observations["test"])
    preflight_path = (output_path or PROJECT_ROOT / "reports" / "m3r2.json").with_suffix(
        ".preflight.pt"
    )
    final_path = (output_path or PROJECT_ROOT / "reports" / "m3r2.json").with_suffix(".final.pt")
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        untrained = StructuredSemanticTransitionLearner(corpus)
        preflight_before = _sequence_world_digests(untrained, test_sequence)
        atomic_save(untrained.checkpoint(), preflight_path)
        preflight_restored = StructuredSemanticTransitionLearner.from_checkpoint(
            torch.load(preflight_path, map_location="cpu", weights_only=False), corpus
        )
        preflight_after = _sequence_world_digests(preflight_restored, test_sequence)

        native = StructuredSemanticTransitionLearner(corpus)
        owner_before = native.owner_digests()
        # The transition input contains a generic fact×event interaction
        # basis.  The earlier R3 course was smaller; its 0.5 learning rate
        # makes this larger observation course diverge on CPU.  Keep the
        # native delta rule and use the measured stable rate instead of
        # allowing non-finite values into a checkpoint.
        losses = native.fit(corpus.train, epochs=280, learning_rate=0.2)
        owner_after = native.owner_digests()
        atomic_save(native.checkpoint(), final_path)
        restored = StructuredSemanticTransitionLearner.from_checkpoint(
            torch.load(final_path, map_location="cpu", weights_only=False), corpus
        )
        test_metrics = _metrics(restored, corpus.test)
        sequence_metrics = _sequence_metrics(restored, test_sequence)
        lesion = StructuredSemanticTransitionLearner.from_checkpoint(native.checkpoint(), corpus)
        lesion_before, lesion_after = lesion.zero_transition_head()
        lesion_metrics = _sequence_metrics(lesion, test_sequence)
        runtime = _runtime_gate(restored, test_sequence)

        full_sequence = _sequence_for_split(observations["test"])
        deleted_sequence = tuple(item for index, item in enumerate(full_sequence) if index != 4)
        shuffled_sequence = (
            full_sequence[0],
            full_sequence[2],
            full_sequence[1],
            full_sequence[3],
            full_sequence[4],
            full_sequence[5],
        )
        full_digests = _sequence_world_digests(restored, full_sequence)
        deleted_digests = _sequence_world_digests(restored, deleted_sequence)
        shuffled_digests = _sequence_world_digests(restored, shuffled_sequence)
        event_order_gate = {
            "deleted_event_changes_intermediate_state": full_digests[3] != deleted_digests[3],
            "shuffled_event_changes_ordered_state": full_digests[1] != shuffled_digests[1],
            "final_missing_clears_language": True,
        }
        # The last check above is evaluated on the expected final state; use the
        # learner output explicitly so stale language facts cannot hide behind a
        # static expected-value check.
        final_relations = set(sequence_metrics["rows"][-1]["actual_facts"])
        event_order_gate["final_missing_clears_language"] = not any(
            fact in final_relations
            for fact in (
                "workbench::language::python",
                "workbench::language::typescript",
                "workbench::language::rust",
            )
        )
        checkpoint_gate = {
            "preflight_saved": preflight_path.exists(),
            "preflight_output_stable": preflight_before == preflight_after,
            "post_training_saved": final_path.exists(),
            "post_training_output_stable": _sequence_world_digests(native, test_sequence)
            == _sequence_world_digests(restored, test_sequence),
            "training_steps": native.training_steps,
        }
        gates = {
            "project_disjoint": corpus.manifest()["record_disjoint"]
            and not (
                {item.family_id for item in corpus.train} & {item.family_id for item in corpus.dev}
            )
            and not (
                {item.family_id for item in corpus.train} & {item.family_id for item in corpus.test}
            )
            and not (
                {item.family_id for item in corpus.dev} & {item.family_id for item in corpus.test}
            )
            and not (
                {item.input_digest for item in corpus.train}
                & {item.input_digest for item in corpus.test}
            ),
            "native_test_fact_f1": float(test_metrics["fact_f1"]) >= 0.80,
            "native_sequence_fact_accuracy": float(sequence_metrics["fact_exact_accuracy"]) >= 0.80,
            "native_sequence_goal_accuracy": float(sequence_metrics["goal_accuracy"]) >= 0.80,
            "native_sequence_content_accuracy": float(sequence_metrics["content_accuracy"]) >= 0.80,
            "native_sequence_status_accuracy": float(sequence_metrics["status_accuracy"]) >= 0.80,
            "native_beats_static_only": float(sequence_metrics["fact_exact_accuracy"])
            > float(_static_only_sequence(test_sequence)["fact_exact_accuracy"]),
            "event_order_and_deletion": all(event_order_gate.values()),
            "checkpoint_save_restore": all(checkpoint_gate.values())
            and checkpoint_gate["training_steps"] > 0,
            "runtime_owner_roundtrip": all(
                runtime[key]
                for key in (
                    "owner_attached",
                    "checkpoint_component_present",
                    "restored_owner_present",
                    "output_stable_after_runtime_restore",
                    "cognitive_state_read_only",
                    "runtime_checkpoint_stable",
                )
            ),
            "transition_lesion_changes_sequence": lesion_before != lesion_after
            and float(lesion_metrics["fact_exact_accuracy"])
            < float(sequence_metrics["fact_exact_accuracy"]),
            "provider_not_attached": corpus.manifest()["provider_attached"] is False,
        }
        report: dict[str, Any] = {
            "format": REPORT_FORMAT,
            "version": REPORT_VERSION,
            "gate": "M3.R2",
            "status": "passed" if all(gates.values()) else "failed",
            "can_promote": False,
            "promotion_reason": (
                "native Workbench task-state transition remains a bounded CPU Gate; "
                "read-only ActionIntent and executable side effects stay frozen."
            ),
            "course": {
                "manifest": corpus.manifest(),
                "static_manifest": static_corpus.manifest(),
                "parameter_count": native.parameter_count,
                "owner_digests_changed": {
                    name: owner_before[name] != owner_after[name] for name in owner_before
                },
                "losses": losses,
                "sequence_paths": list(SEQUENCE_PATHS),
            },
            "metrics": {
                "native_transition_test": test_metrics,
                "native_sequence_test": sequence_metrics,
                "native_lesion_sequence_test": lesion_metrics,
                "static_only_sequence_test": _static_only_sequence(test_sequence),
                "provider_assisted_upper_bound": {
                    "fact_exact_accuracy": 1.0,
                    "goal_accuracy": 1.0,
                    "content_accuracy": 1.0,
                    "status_accuracy": 1.0,
                },
            },
            "checkpoint": checkpoint_gate,
            "runtime": runtime,
            "event_controls": event_order_gate,
            "boundaries": {
                "provider_final_binding": False,
                "raw_text_language_learning": False,
                "workspace_write": False,
                "terminal": False,
                "mcp": False,
                "cuda": False,
            },
            "gates": gates,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return report
    finally:
        preflight_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m3r2_task_state_sequence_20260907.json",
    )
    args = parser.parse_args()
    report = run_gate(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
