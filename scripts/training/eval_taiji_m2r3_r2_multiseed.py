"""Evaluate multi-entity, multi-relation structured semantics across seeds."""

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
    StructuredSemanticCorpus,
    StructuredSemanticExample,
    StructuredSemanticLearner,
    WorldState,
    content_digest,
    semantic_fact_key,
)

REPORT_FORMAT = "taiji-m2r3-r2-multiseed-structured-semantics-v1"
REPORT_VERSION = 1
ENTITIES = ("alpha", "beta", "gamma")
STATES = ("ready", "blocked", "pending")
RESOURCES = ("cache", "queue", "vault")
CONSTRAINTS = ("safe", "fast")
FACTOR_KEYS = tuple(
    [f"entity:{value}" for value in ENTITIES]
    + [f"state:{value}" for value in STATES]
    + [f"resource:{value}" for value in RESOURCES]
    + [f"constraint:{value}" for value in CONSTRAINTS]
)


def _layout(seed: int) -> dict[str, int]:
    values = list(FACTOR_KEYS)
    random.Random(int(seed)).shuffle(values)
    return {value: index for index, value in enumerate(values)}


def _percept(
    sample_id: str,
    vector: tuple[float, ...],
    tick: int,
    *,
    confidence: float = 1.0,
) -> PerceptEvent:
    return PerceptEvent(
        event_id=f"{sample_id}:percept",
        observation_tick=int(tick),
        modality="structured-semantic-r2",
        features=torch.tensor(vector, dtype=torch.float32),
        assembly_id=f"{sample_id}:assembly",
        boundary_score=1.0,
        prediction_error=0.0,
        boundary=True,
        confidence=float(confidence),
    )


def _goal(state: str) -> Goal:
    descriptions = {
        "ready": "report the ready state",
        "blocked": "clarify the blocked state",
        "pending": "plan the pending state",
    }
    return Goal(
        goal_id=f"goal:{state}",
        description=descriptions[state],
        priority=1.0,
    )


def _content(
    *,
    sample_id: str,
    tick: int,
    goal: Goal,
    state: str,
    resource: str,
    constraint: str,
) -> ContentPlan:
    intent_kind = {
        "ready": "report_status",
        "blocked": "request_information",
        "pending": "plan_action",
    }[state]
    return ContentPlan(
        content_id=f"content:{state}:{resource}:{constraint}",
        intent_id=f"intent:{state}",
        intent_kind=intent_kind,
        semantic_slots={
            "state": state,
            "resource": resource,
            "constraint": constraint,
        },
        required_terms=(state, resource, constraint),
        source_goal_id=goal.goal_id,
        expected_outcome="clarify" if state == "blocked" else "continue",
        confidence=1.0,
        provenance=f"m2r3-r2:{sample_id}",
        tick=int(tick),
    )


def _example(
    *,
    seed: int,
    layout: dict[str, int],
    sample_id: str,
    entity: str,
    state: str,
    resource: str,
    constraint: str,
    tick: int,
) -> StructuredSemanticExample:
    vector = [0.0] * len(FACTOR_KEYS)
    for key in (
        f"entity:{entity}",
        f"state:{state}",
        f"resource:{resource}",
        f"constraint:{constraint}",
    ):
        vector[layout[key]] = 1.0
    goal = _goal(state)
    world = WorldState(
        tick=int(tick),
        latent=torch.empty(0),
        entities=("agent", entity, resource, constraint),
        relations=(
            ("agent", "tracks", entity),
            ("agent", "state", state),
            ("agent", "holds", resource),
            ("agent", "constraint", constraint),
        ),
        uncertainty=0.0,
    )
    return StructuredSemanticExample(
        example_id=f"seed{seed}:{sample_id}",
        family_id=f"seed{seed}:family:{sample_id}",
        percept=_percept(f"seed{seed}:{sample_id}", tuple(vector), tick),
        world=world,
        goal=goal,
        content=_content(
            sample_id=sample_id,
            tick=tick,
            goal=goal,
            state=state,
            resource=resource,
            constraint=constraint,
        ),
    )


def build_corpus(seed: int) -> StructuredSemanticCorpus:
    """Hold out relation combinations while retaining all atomic vocabulary."""

    layout = _layout(seed)
    train: list[StructuredSemanticExample] = []
    dev: list[StructuredSemanticExample] = []
    test: list[StructuredSemanticExample] = []
    tick = 1
    for entity_index, entity in enumerate(ENTITIES):
        for state_index, state in enumerate(STATES):
            for resource_index, resource in enumerate(RESOURCES):
                for constraint_index, constraint in enumerate(CONSTRAINTS):
                    sample_id = f"{entity}-{state}-{resource}-{constraint}"
                    example = _example(
                        seed=seed,
                        layout=layout,
                        sample_id=sample_id,
                        entity=entity,
                        state=state,
                        resource=resource,
                        constraint=constraint,
                        tick=tick,
                    )
                    tick += 1
                    residue = (entity_index + state_index + resource_index + constraint_index) % 3
                    if residue != 0:
                        train.append(example)
                    elif (entity_index + resource_index) % 2 == 0:
                        dev.append(example)
                    else:
                        test.append(example)
    return StructuredSemanticCorpus.from_splits(train=train, dev=dev, test=test)


def _metrics(
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


def _constraint_conflict(
    example: StructuredSemanticExample,
    layout: dict[str, int],
) -> PerceptEvent:
    features = example.percept.features.detach().clone()
    other = next(
        value for value in CONSTRAINTS if value not in example.content.semantic_slots["constraint"]
    )
    features[layout[f"constraint:{other}"]] = 1.0
    return replace(
        example.percept,
        event_id=f"{example.example_id}:constraint-conflict",
        assembly_id=f"{example.example_id}:constraint-conflict-assembly",
        features=features,
    )


def _run_seed(seed: int) -> dict[str, Any]:
    corpus = build_corpus(seed)
    learner = StructuredSemanticLearner(corpus)
    owners_before = learner.owner_digests()
    losses = learner.fit(corpus.train, epochs=220, learning_rate=1.6)
    owners_after = learner.owner_digests()
    train_metrics = _metrics(learner, corpus.train)
    dev_metrics = _metrics(learner, corpus.dev)
    test_metrics = _metrics(learner, corpus.test)
    layout = _layout(seed)
    unknown = _percept(
        f"seed{seed}:unknown",
        (0.0,) * len(FACTOR_KEYS),
        1000,
        confidence=0.2,
    )
    blocked = next(
        example for example in corpus.test if example.content.semantic_slots["state"] == "blocked"
    )
    pending = next(
        example for example in corpus.test if example.content.semantic_slots["state"] == "pending"
    )
    known = corpus.test[0]
    unknown_result = learner.predict(unknown)
    conflict_result = learner.predict(_constraint_conflict(known, layout))
    clarification_result = learner.predict(blocked.percept)
    plan_result = learner.predict(pending.percept)
    checkpoint = learner.checkpoint()
    restored = StructuredSemanticLearner.from_checkpoint(checkpoint)
    checkpoint_matches = all(
        content_digest(learner.predict(example.percept).to_payload())
        == content_digest(restored.predict(example.percept).to_payload())
        for example in corpus.test
    )
    lesion_before, lesion_after = restored.zero_fact_head()
    lesion_metrics = _metrics(restored, corpus.test)
    owner_changes = {name: owners_before[name] != owners_after[name] for name in owners_before}
    checks = {
        "record_disjoint": corpus.manifest()["record_disjoint"],
        "multi_entity_relations": all(
            metric["fact_f1"] >= 0.90 for metric in (dev_metrics, test_metrics)
        ),
        "dev_goal_learning": dev_metrics["goal_accuracy"] >= 0.80,
        "test_goal_learning": test_metrics["goal_accuracy"] >= 0.80,
        "dev_content_learning": dev_metrics["content_accuracy"] >= 0.80,
        "test_content_learning": test_metrics["content_accuracy"] >= 0.80,
        "unknown_fail_closed": unknown_result.status == "unknown" and unknown_result.goal is None,
        "constraint_conflict_fail_closed": (
            conflict_result.status == "conflict" and conflict_result.goal is None
        ),
        "clarification_output": (
            clarification_result.status == "clarify"
            and clarification_result.content_plan is not None
            and clarification_result.content_plan.intent_kind == "request_information"
        ),
        "pending_plan_output": (
            plan_result.status == "resolved"
            and plan_result.content_plan is not None
            and plan_result.content_plan.intent_kind == "plan_action"
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
    return {
        "seed": seed,
        "status": "passed" if all(bool(value) for value in checks.values()) else "failed",
        "corpus": corpus.manifest(),
        "training": {"epochs": 220, "learning_rate": 1.6, "losses": losses},
        "metrics": {
            "train": train_metrics,
            "dev": dev_metrics,
            "test": test_metrics,
            "lesion_test": lesion_metrics,
            "unknown_status": unknown_result.status,
            "constraint_conflict_status": conflict_result.status,
            "clarification_status": clarification_result.status,
            "pending_status": plan_result.status,
        },
        "preflight": {
            "checkpoint_digest": content_digest(checkpoint),
            "checkpoint_matches_after_restore": checkpoint_matches,
            "parameter_count": learner.parameter_count,
            "owner_changes": owner_changes,
        },
        "gate": {"passed": all(bool(value) for value in checks.values()), "checks": checks},
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
            "mean_test_goal_accuracy": sum(run["metrics"]["test"]["goal_accuracy"] for run in runs)
            / len(runs),
            "mean_test_content_accuracy": sum(
                run["metrics"]["test"]["content_accuracy"] for run in runs
            )
            / len(runs),
        },
        "gate": {
            "passed": passed,
            "criterion": "every independent seed passes every declared check",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m2r3_r2_multiseed_20260907.json",
    )
    args = parser.parse_args()
    report = evaluate(tuple(args.seeds))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
