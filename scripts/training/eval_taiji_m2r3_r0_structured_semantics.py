"""Run the M2.R3.R0 native structured semantic CPU canary.

The canary deliberately stops at a structured ContentPlan.  It measures a
learned ``PerceptEvent -> facts/WorldState -> Goal -> ContentPlan`` chain and
does not load a language provider or execute any action.
"""

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

from taiji.contracts import Goal, PerceptEvent, WorldState  # noqa: E402
from taiji.generation import ContentPlan  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.semantic_training import (  # noqa: E402
    StructuredSemanticCorpus,
    StructuredSemanticExample,
    StructuredSemanticLearner,
    semantic_fact_key,
)

REPORT_FORMAT = "taiji-m2r3-r0-structured-semantics-v1"
REPORT_VERSION = 1
FEATURE_DIM = 7
OBJECT_FEATURES = {"alpha": 0, "beta": 1}
STATE_FEATURES = {"ready": 2, "blocked": 3}
RESOURCE_FEATURES = {"cache": 4, "queue": 5}
CONTEXT_FEATURE_INDEX = 6


def _percept(sample_id: str, vector: tuple[float, ...], tick: int, *, confidence: float = 1.0) -> PerceptEvent:
    return PerceptEvent(
        event_id=f"{sample_id}:percept",
        observation_tick=int(tick),
        modality="structured-semantic",
        features=torch.tensor(vector, dtype=torch.float32),
        assembly_id=f"{sample_id}:assembly",
        boundary_score=1.0,
        prediction_error=0.0,
        boundary=True,
        confidence=float(confidence),
    )


def _content(
    *,
    sample_id: str,
    tick: int,
    goal: Goal,
    state: str,
    resource: str,
) -> ContentPlan:
    report = state == "ready"
    mode = "report" if report else "clarify"
    return ContentPlan(
        content_id=f"content:{mode}:{resource}",
        intent_id=f"intent:{mode}",
        intent_kind="report_status" if report else "request_information",
        semantic_slots={"state": state, "resource": resource},
        required_terms=(state, resource),
        source_goal_id=goal.goal_id,
        expected_outcome="continue" if report else "request-more-information",
        confidence=1.0,
        provenance=f"m2r3-canary:{sample_id}",
        tick=int(tick),
    )


def _example(
    sample_id: str,
    *,
    object_id: str,
    state: str,
    resource: str,
    tick: int,
    context: int = 0,
) -> StructuredSemanticExample:
    if object_id not in OBJECT_FEATURES or state not in STATE_FEATURES or resource not in RESOURCE_FEATURES:
        raise ValueError("unknown M2.R3 canary factor")
    vector = [0.0] * FEATURE_DIM
    vector[OBJECT_FEATURES[object_id]] = 1.0
    vector[STATE_FEATURES[state]] = 1.0
    vector[RESOURCE_FEATURES[resource]] = 1.0
    vector[CONTEXT_FEATURE_INDEX] = float(context)
    goal = Goal(
        goal_id="goal:report" if state == "ready" else "goal:clarify",
        description="report the current state" if state == "ready" else "clarify the blocked state",
        priority=1.0,
    )
    world = WorldState(
        tick=int(tick),
        latent=torch.empty(0),
        entities=("agent", object_id, resource),
        relations=(
            ("agent", "tracks", object_id),
            ("agent", "state", state),
            ("agent", "holds", resource),
        ),
        uncertainty=0.0,
    )
    return StructuredSemanticExample(
        example_id=sample_id,
        family_id=f"family:{sample_id}",
        percept=_percept(sample_id, tuple(vector), tick),
        world=world,
        goal=goal,
        content=_content(
            sample_id=sample_id,
            tick=tick,
            goal=goal,
            state=state,
            resource=resource,
        ),
    )


def build_corpus() -> StructuredSemanticCorpus:
    """Build a composition split with no exact percept input repeated."""

    train = tuple(
        _example(
            f"train-{object_id}-{state}-{resource}",
            object_id=object_id,
            state=state,
            resource=resource,
            tick=index + 1,
        )
        for index, object_id in enumerate(("alpha", "beta"))
        for state in ("ready", "blocked")
        for resource in ("cache", "queue")
    )
    dev = (
        _example(
            "dev-beta-ready-cache",
            object_id="beta",
            state="ready",
            resource="cache",
            tick=20,
            context=1,
        ),
        _example(
            "dev-alpha-blocked-cache",
            object_id="alpha",
            state="blocked",
            resource="cache",
            tick=21,
            context=1,
        ),
        _example(
            "dev-beta-ready-queue",
            object_id="beta",
            state="ready",
            resource="queue",
            tick=22,
            context=1,
        ),
        _example(
            "dev-alpha-blocked-queue",
            object_id="alpha",
            state="blocked",
            resource="queue",
            tick=23,
            context=1,
        ),
    )
    test = (
        _example(
            "test-alpha-ready-cache",
            object_id="alpha",
            state="ready",
            resource="cache",
            tick=30,
            context=1,
        ),
        _example(
            "test-beta-blocked-cache",
            object_id="beta",
            state="blocked",
            resource="cache",
            tick=31,
            context=1,
        ),
        _example(
            "test-alpha-ready-queue",
            object_id="alpha",
            state="ready",
            resource="queue",
            tick=32,
            context=1,
        ),
        _example(
            "test-beta-blocked-queue",
            object_id="beta",
            state="blocked",
            resource="queue",
            tick=33,
            context=1,
        ),
    )
    return StructuredSemanticCorpus.from_splits(train=train, dev=dev, test=test)


def _fact_metrics(
    learner: StructuredSemanticLearner,
    examples: tuple[StructuredSemanticExample, ...],
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    goal_correct = content_correct = status_correct = 0
    points: list[dict[str, Any]] = []
    for example in examples:
        result = learner.predict(example.percept)
        expected_facts = set(example.fact_keys)
        actual_facts = (
            set()
            if result.world is None
            else {semantic_fact_key(*relation) for relation in result.world.relations}
        )
        true_positive += len(expected_facts & actual_facts)
        false_positive += len(actual_facts - expected_facts)
        false_negative += len(expected_facts - actual_facts)
        goal_hit = result.goal is not None and result.goal.goal_id == example.goal.goal_id
        content_hit = (
            result.content_plan is not None
            and result.content_plan.content_id == example.content.content_id
        )
        expected_status = "clarify" if example.content.intent_kind == "request_information" else "resolved"
        status_hit = result.status == expected_status
        goal_correct += int(goal_hit)
        content_correct += int(content_hit)
        status_correct += int(status_hit)
        points.append(
            {
                "example_id": example.example_id,
                "status": result.status,
                "goal": None if result.goal is None else result.goal.goal_id,
                "content": None
                if result.content_plan is None
                else result.content_plan.content_id,
                "goal_expected": example.goal.goal_id,
                "content_expected": example.content.content_id,
                "fact_recall": 1.0
                if not expected_facts
                else len(expected_facts & actual_facts) / len(expected_facts),
            }
        )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "examples": len(examples),
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": f1,
        "goal_accuracy": goal_correct / len(examples),
        "content_accuracy": content_correct / len(examples),
        "status_accuracy": status_correct / len(examples),
        "points": points,
    }


def evaluate(*, epochs: int = 240, learning_rate: float = 2.0) -> dict[str, Any]:
    corpus = build_corpus()
    learner = StructuredSemanticLearner(corpus)
    before_owners = learner.owner_digests()
    losses = learner.fit(corpus.train, epochs=epochs, learning_rate=learning_rate)
    after_owners = learner.owner_digests()
    train_metrics = _fact_metrics(learner, corpus.train)
    dev_metrics = _fact_metrics(learner, corpus.dev)
    test_metrics = _fact_metrics(learner, corpus.test)

    unknown = _percept("probe-unknown", (0.0,) * FEATURE_DIM, 20, confidence=0.2)
    unknown_result = learner.predict(unknown)
    conflict_vector = (1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0)
    conflict = _percept("probe-conflict", conflict_vector, 21)
    conflict_result = learner.predict(conflict)
    clarification = learner.predict(corpus.test[1].percept)

    checkpoint = learner.checkpoint()
    restored = StructuredSemanticLearner.from_checkpoint(checkpoint, corpus)
    checkpoint_matches = all(
        content_digest(learner.predict(example.percept).to_payload())
        == content_digest(restored.predict(example.percept).to_payload())
        for example in corpus.test
    )
    lesion_before, lesion_after = restored.zero_fact_head()
    lesion_metrics = _fact_metrics(restored, corpus.test)
    owner_changes = {
        name: before_owners[name] != after_owners[name] for name in before_owners
    }
    gate_checks = {
        "record_disjoint": corpus.manifest()["record_disjoint"],
        "train_fact_learning": train_metrics["fact_f1"] >= 0.95,
        "dev_fact_learning": dev_metrics["fact_f1"] >= 0.80,
        "test_fact_learning": test_metrics["fact_f1"] >= 0.80,
        "dev_goal_learning": dev_metrics["goal_accuracy"] >= 0.80,
        "test_goal_learning": test_metrics["goal_accuracy"] >= 0.80,
        "dev_content_learning": dev_metrics["content_accuracy"] >= 0.80,
        "test_content_learning": test_metrics["content_accuracy"] >= 0.80,
        "unknown_fail_closed": unknown_result.status == "unknown" and unknown_result.goal is None,
        "conflict_fail_closed": conflict_result.status == "conflict" and conflict_result.goal is None,
        "clarification_plan": (
            clarification.status == "clarify"
            and clarification.content_plan is not None
            and clarification.content_plan.intent_kind == "request_information"
        ),
        "checkpoint_save_restore": checkpoint_matches,
        "all_declared_owners_changed": all(owner_changes.values()),
        "fact_lesion_effective": (
            lesion_before != lesion_after
            and lesion_metrics["fact_f1"] < test_metrics["fact_f1"]
            and lesion_metrics["goal_accuracy"] < test_metrics["goal_accuracy"]
        ),
        "provider_not_attached": corpus.manifest()["provider_attached"] is False,
    }
    passed = all(bool(value) for value in gate_checks.values())
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "status": "passed" if passed else "failed",
        "can_promote": False,
        "corpus": corpus.manifest(),
        "preflight": {
            "checkpoint_digest": content_digest(checkpoint),
            "checkpoint_matches_after_restore": checkpoint_matches,
            "parameter_count": learner.parameter_count,
            "training_steps": learner.training_steps,
            "owner_digests_before": before_owners,
            "owner_digests_after": after_owners,
        },
        "training": {"epochs": epochs, "learning_rate": learning_rate, "losses": losses},
        "metrics": {
            "train": train_metrics,
            "dev": dev_metrics,
            "test": test_metrics,
            "lesion_test": lesion_metrics,
            "unknown_status": unknown_result.status,
            "conflict_status": conflict_result.status,
            "clarification_status": clarification.status,
            "clarification_content": (
                None
                if clarification.content_plan is None
                else clarification.content_plan.content_id
            ),
        },
        "gate": {"passed": passed, "checks": gate_checks},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=2.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m2r3_r0_structured_semantics_20260907.json",
    )
    args = parser.parse_args()
    report = evaluate(epochs=args.epochs, learning_rate=args.learning_rate)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
