"""Evaluate persistent multi-step semantic world transitions across seeds."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ContentPlan,
    Goal,
    PerceptEvent,
    StructuredSemanticTransitionCorpus,
    StructuredSemanticTransitionExample,
    StructuredSemanticTransitionLearner,
    StructuredSemanticTransitionResult,
    WorldState,
    content_digest,
    semantic_fact_key,
)

REPORT_FORMAT = "taiji-m2r3-r3-multistep-semantic-transition-v1"
REPORT_VERSION = 1
ENTITIES = ("alpha", "beta")
STATES = ("pending", "ready", "blocked")
RESOURCES = ("cache", "queue")
EVENTS = ("release", "move", "block", "noop")


def _event_layout(seed: int) -> dict[str, int]:
    values = list(EVENTS)
    random.Random(int(seed)).shuffle(values)
    return {value: index for index, value in enumerate(values)}


def _event(
    seed: int,
    kind: str,
    tick: int,
    *,
    confidence: float = 1.0,
) -> PerceptEvent:
    layout = _event_layout(seed)
    vector = [0.0] * len(EVENTS)
    vector[layout[kind]] = 1.0
    return PerceptEvent(
        event_id=f"seed{seed}:event:{kind}:{tick}",
        observation_tick=int(tick),
        modality="structured-semantic-transition",
        features=torch.tensor(vector, dtype=torch.float32),
        assembly_id=f"seed{seed}:assembly:{kind}:{tick}",
        boundary_score=1.0,
        prediction_error=0.0,
        boundary=True,
        confidence=float(confidence),
    )


def _world(entity: str, state: str, resource: str, tick: int) -> WorldState:
    return WorldState(
        tick=int(tick),
        latent=torch.empty(0),
        entities=("agent", entity, resource),
        relations=(
            ("agent", "tracks", entity),
            ("agent", "state", state),
            ("agent", "holds", resource),
        ),
        uncertainty=0.0,
    )


def _apply_event(state: str, resource: str, kind: str) -> tuple[str, str]:
    if kind == "release":
        return "ready", resource
    if kind == "block":
        return "blocked", resource
    if kind == "move":
        return state, "queue" if resource == "cache" else "cache"
    return state, resource


def _goal(state: str) -> Goal:
    descriptions = {
        "pending": "wait for the pending state",
        "ready": "report the ready state",
        "blocked": "clarify the blocked state",
    }
    return Goal(goal_id=f"goal:{state}", description=descriptions[state], priority=1.0)


def _content(*, state: str, resource: str, tick: int, goal: Goal, sample_id: str) -> ContentPlan:
    return ContentPlan(
        content_id=f"content:{state}:{resource}",
        intent_id=f"intent:{state}",
        intent_kind=(
            "request_information" if state == "blocked" else "report_status"
        ),
        semantic_slots={"state": state, "resource": resource},
        required_terms=(state, resource),
        source_goal_id=goal.goal_id,
        expected_outcome="clarify" if state == "blocked" else "continue",
        confidence=1.0,
        provenance=f"m2r3-r3:{sample_id}",
        tick=int(tick),
    )


def _example(
    *,
    seed: int,
    sample_id: str,
    entity: str,
    state: str,
    resource: str,
    kind: str,
    tick: int,
) -> StructuredSemanticTransitionExample:
    next_state, next_resource = _apply_event(state, resource, kind)
    before = _world(entity, state, resource, tick - 1)
    after = _world(entity, next_state, next_resource, tick)
    goal = _goal(next_state)
    return StructuredSemanticTransitionExample(
        example_id=f"seed{seed}:{sample_id}",
        family_id=f"seed{seed}:family:{sample_id}",
        before=before,
        event=_event(seed, kind, tick),
        after=after,
        goal=goal,
        content=_content(
            state=next_state,
            resource=next_resource,
            tick=tick,
            goal=goal,
            sample_id=sample_id,
        ),
    )


def build_corpus(seed: int) -> StructuredSemanticTransitionCorpus:
    train: list[StructuredSemanticTransitionExample] = []
    dev: list[StructuredSemanticTransitionExample] = []
    test: list[StructuredSemanticTransitionExample] = []
    tick = 1
    for entity_index, entity in enumerate(ENTITIES):
        for state_index, state in enumerate(STATES):
            for resource_index, resource in enumerate(RESOURCES):
                for event_index, kind in enumerate(EVENTS):
                    sample_id = f"{entity}-{state}-{resource}-{kind}"
                    example = _example(
                        seed=seed,
                        sample_id=sample_id,
                        entity=entity,
                        state=state,
                        resource=resource,
                        kind=kind,
                        tick=tick,
                    )
                    tick += 1
                    residue = (entity_index + state_index + resource_index + event_index) % 3
                    if residue != 0:
                        train.append(example)
                    elif (entity_index + resource_index) % 2 == 0:
                        dev.append(example)
                    else:
                        test.append(example)
    return StructuredSemanticTransitionCorpus.from_splits(train=train, dev=dev, test=test)


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
        expected_status = "clarify" if example.content.intent_kind == "request_information" else "resolved"
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


def _run_sequence(
    learner: StructuredSemanticTransitionLearner,
    seed: int,
    initial: WorldState,
    kinds: tuple[str, ...],
    *,
    start_tick: int,
) -> tuple[tuple[StructuredSemanticTransitionResult, ...], WorldState]:
    results = []
    current = initial
    for offset, kind in enumerate(kinds):
        result = learner.predict(current, _event(seed, kind, start_tick + offset))
        results.append(result)
        if result.world is None:
            break
        current = result.world
    return tuple(results), current


def _run_seed(seed: int) -> dict[str, Any]:
    corpus = build_corpus(seed)
    learner = StructuredSemanticTransitionLearner(corpus)
    owners_before = learner.owner_digests()
    losses = learner.fit(corpus.train, epochs=240, learning_rate=0.5)
    owners_after = learner.owner_digests()
    train_metrics = _metrics(learner, corpus.train)
    dev_metrics = _metrics(learner, corpus.dev)
    test_metrics = _metrics(learner, corpus.test)

    initial = _world("alpha", "pending", "cache", 0)
    sequence, final_world = _run_sequence(
        learner,
        seed,
        initial,
        ("release", "move", "block"),
        start_tick=1,
    )
    deletion_sequence, deletion_world = _run_sequence(
        learner,
        seed,
        initial,
        ("release", "block"),
        start_tick=1,
    )
    reverse_sequence, reverse_world = _run_sequence(
        learner,
        seed,
        initial,
        ("block", "release"),
        start_tick=1,
    )
    checkpoint_first = learner.predict(initial, _event(seed, "release", 1))
    restored = StructuredSemanticTransitionLearner.from_checkpoint(learner.checkpoint())
    restored_second = (
        None
        if checkpoint_first.world is None
        else restored.predict(checkpoint_first.world, _event(seed, "move", 2))
    )
    direct_second = (
        None
        if checkpoint_first.world is None
        else learner.predict(checkpoint_first.world, _event(seed, "move", 2))
    )
    unknown = learner.predict(initial, _event(seed, "release", 1, confidence=0.2))
    conflict_world = replace(
        initial,
        relations=initial.relations + (("agent", "state", "ready"),),
    )
    conflict = learner.predict(conflict_world, _event(seed, "noop", 1))
    lesion_before, lesion_after = restored.zero_transition_head()
    lesion_result = restored.predict(initial, _event(seed, "release", 1))
    final_fact_keys = {
        semantic_fact_key(*relation) for relation in final_world.relations
    }
    expected_final = {
        semantic_fact_key("agent", "tracks", "alpha"),
        semantic_fact_key("agent", "state", "blocked"),
        semantic_fact_key("agent", "holds", "queue"),
    }
    deletion_facts = {semantic_fact_key(*relation) for relation in deletion_world.relations}
    reverse_facts = {semantic_fact_key(*relation) for relation in reverse_world.relations}
    expected_deletion = {
        semantic_fact_key("agent", "tracks", "alpha"),
        semantic_fact_key("agent", "state", "blocked"),
        semantic_fact_key("agent", "holds", "cache"),
    }
    expected_reverse = {
        semantic_fact_key("agent", "tracks", "alpha"),
        semantic_fact_key("agent", "state", "ready"),
        semantic_fact_key("agent", "holds", "cache"),
    }
    checks = {
        "record_disjoint": corpus.manifest()["record_disjoint"],
        "dev_transition_learning": dev_metrics["fact_f1"] >= 0.80,
        "test_transition_learning": test_metrics["fact_f1"] >= 0.80,
        "dev_goal_learning": dev_metrics["goal_accuracy"] >= 0.80,
        "test_content_learning": test_metrics["content_accuracy"] >= 0.80,
        "sequence_persistence": len(sequence) == 3 and final_fact_keys == expected_final,
        "event_deletion_changes_state": deletion_facts == expected_deletion,
        "event_order_changes_state": reverse_facts == expected_reverse and reverse_facts != final_fact_keys,
        "checkpoint_mid_sequence": (
            restored_second is not None
            and direct_second is not None
            and content_digest(restored_second.to_payload())
            == content_digest(direct_second.to_payload())
        ),
        "unknown_event_fail_closed": unknown.status == "unknown" and unknown.goal is None,
        "world_conflict_fail_closed": conflict.status == "conflict" and conflict.goal is None,
        "transition_lesion_effective": (
            lesion_before != lesion_after
            and lesion_result.world is not None
            and {
                semantic_fact_key(*relation) for relation in lesion_result.world.relations
            }
            != {
                semantic_fact_key("agent", "tracks", "alpha"),
                semantic_fact_key("agent", "state", "ready"),
                semantic_fact_key("agent", "holds", "cache"),
            }
        ),
        "all_transition_owners_changed": all(owners_before[name] != owners_after[name] for name in owners_before),
        "provider_not_attached": corpus.manifest()["provider_attached"] is False,
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "seed": seed,
        "status": "passed" if passed else "failed",
        "corpus": corpus.manifest(),
        "training": {"epochs": 240, "learning_rate": 0.5, "losses": losses},
        "metrics": {
            "train": train_metrics,
            "dev": dev_metrics,
            "test": test_metrics,
            "unknown_status": unknown.status,
            "conflict_status": conflict.status,
            "sequence_statuses": [item.status for item in sequence],
            "final_world_digest": content_digest(final_world.to_payload()),
        },
        "preflight": {
            "checkpoint_matches_mid_sequence": checks["checkpoint_mid_sequence"],
            "parameter_count": learner.parameter_count,
            "owner_changes": {
                name: owners_before[name] != owners_after[name] for name in owners_before
            },
        },
        "gate": {"passed": passed, "checks": checks},
    }


def evaluate(seeds: tuple[int, ...] = (11, 29, 47)) -> dict[str, Any]:
    runs = tuple(_run_seed(int(seed)) for seed in seeds)
    passed = all(run["status"] == "passed" for run in runs)
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "status": "passed" if passed else "failed",
        "can_promote": False,
        "seeds": list(seeds),
        "runs": list(runs),
        "aggregate": {
            "all_seeds_passed": passed,
            "mean_test_fact_f1": sum(run["metrics"]["test"]["fact_f1"] for run in runs) / len(runs),
            "mean_test_goal_accuracy": sum(
                run["metrics"]["test"]["goal_accuracy"] for run in runs
            )
            / len(runs),
            "mean_test_content_accuracy": sum(
                run["metrics"]["test"]["content_accuracy"] for run in runs
            )
            / len(runs),
        },
        "gate": {"passed": passed, "criterion": "every independent seed passes every transition check"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m2r3_r3_multistep_20260907.json",
    )
    args = parser.parse_args()
    report = evaluate(tuple(args.seeds))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
