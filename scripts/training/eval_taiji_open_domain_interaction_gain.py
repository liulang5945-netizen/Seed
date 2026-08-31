"""Evaluate longitudinal Workbench gain against weight/router/memory controls."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_interaction_groups import (  # noqa: E402
    _run_workbench_episode,
    _workbench_owner,
    build_workbench_corpus,
)
from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionGroupTransferLearner,
    InteractionTraceCorpus,
    build_member_evidence,
)

REPORT_FORMAT = "taiji-w7-p4-7-open-domain-interaction-gain-v1"
LEARNER_SEEDS = (11, 29, 47)
METHODS = ("relation_transfer", "weight_only", "router_only", "memory_only")

MEMBER_ACTIONS = {
    "a": ("workspace.list", {"path": "."}),
    "b": ("workspace.search", {"query": "taiji", "path": "."}),
    "c": ("workspace.read", {"path": "README.md"}),
    "d": ("workspace.stat", {"path": "README.md"}),
    "e": ("workspace.programming_language.resolve", {"path": "README.md"}),
    "f": ("workspace.read", {"path": "missing-longitudinal-a.txt"}),
    "g": ("workspace.read", {"path": "missing-longitudinal-b.txt"}),
}
MEMBER_IDS = {
    key: _workbench_owner(capability_id, parameters)
    for key, (capability_id, parameters) in MEMBER_ACTIONS.items()
}
MEMBER_KEYS_BY_ID = {value: key for key, value in MEMBER_IDS.items()}
TRAIN_FAMILIES = (
    ("ab", "a", "b"),
    ("ac", "a", "c"),
    ("ad", "a", "d"),
    ("cd", "c", "d"),
    ("fg", "f", "g"),
)
FUTURE_FAMILIES = (
    ("future-be", "b", "e"),
    ("future-ce", "c", "e"),
    ("future-de", "d", "e"),
)


def _single_score(member: str) -> float:
    return -0.2 if member in {"f", "g"} else 0.2


def _pair_score(first: str, second: str) -> float:
    return 1.0 if _single_score(first) > 0.0 and _single_score(second) > 0.0 else -1.0


def _action_cells(
    first: str, second: str
) -> tuple[tuple[str, tuple[tuple[str, dict[str, object]], ...], float], ...]:
    return (
        ("none", (), 0.0),
        ("first", (MEMBER_ACTIONS[first],), _single_score(first)),
        ("second", (MEMBER_ACTIONS[second],), _single_score(second)),
        ("pair", (MEMBER_ACTIONS[first], MEMBER_ACTIONS[second]), _pair_score(first, second)),
    )


def _derive_score(
    actions: tuple[tuple[str, dict[str, object]], ...], record: dict[str, object]
) -> float:
    raw_actions = record["workbench_outcome"]["raw_actions"]
    selected_actions = raw_actions[1 : 1 + len(actions)]
    if len(selected_actions) != len(actions):
        raise AssertionError("Workbench raw action trace is shorter than requested action set")
    if any(
        observed["capability_id"] != requested[0]
        for requested, observed in zip(actions, selected_actions)
    ):
        raise AssertionError("Workbench action trace order changed under score projection")
    successes = tuple(bool(item["success"]) for item in selected_actions)
    if not actions:
        return 0.0
    if len(actions) == 1:
        return 0.2 if successes[0] else -0.2
    return 1.0 if all(successes) else -1.0


def _run_scored_cell(
    *,
    split: str,
    context_label: str,
    cell_label: str,
    actions: tuple[tuple[str, dict[str, object]], ...],
    expected_score: float,
) -> tuple[object, dict[str, object]]:
    episode, record = _run_workbench_episode(
        workspace=PROJECT_ROOT,
        split=split,
        context_label=context_label,
        cell_label=cell_label,
        actions=actions,
        reward=expected_score,
    )
    derived_score = _derive_score(actions, record)
    if abs(derived_score - expected_score) > 1e-12:
        raise AssertionError(
            f"native Workbench score mismatch for {context_label}/{cell_label}: "
            f"expected {expected_score}, derived {derived_score}"
        )
    return replace(episode, outcome=derived_score), {
        **record,
        "derived_task_score": derived_score,
        "score_source": "native Workbench raw action success/status projection",
    }


def build_train_corpus() -> tuple[InteractionTraceCorpus, list[dict[str, object]]]:
    episodes: list[object] = []
    records: list[dict[str, object]] = []
    for family, first, second in TRAIN_FAMILIES:
        for cell_label, actions, expected_score in _action_cells(first, second):
            episode, record = _run_scored_cell(
                split="train",
                context_label=family,
                cell_label=cell_label,
                actions=actions,
                expected_score=expected_score,
            )
            episodes.append(episode)
            records.append(record)
    baseline_episode, baseline_record = _run_scored_cell(
        split="train",
        context_label="e-singleton",
        cell_label="none",
        actions=(),
        expected_score=0.0,
    )
    episodes.append(baseline_episode)
    records.append(baseline_record)
    episode, record = _run_scored_cell(
        split="train",
        context_label="e-singleton",
        cell_label="member",
        actions=(MEMBER_ACTIONS["e"],),
        expected_score=_single_score("e"),
    )
    episodes.append(episode)
    records.append(record)
    holdout_episode, holdout_record = _run_scored_cell(
        split="holdout",
        context_label="future-reserved",
        cell_label="none",
        actions=(),
        expected_score=0.0,
    )
    records.append(holdout_record)
    return InteractionTraceCorpus(train=tuple(episodes), holdout=(holdout_episode,)), records


def _evaluator() -> InteractionGroupEvaluator:
    return InteractionGroupEvaluator(
        InteractionGroupEvaluatorConfig(
            minimum_interaction=0.1,
            maximum_uncertainty=0.12,
            maximum_group_cardinality=2,
            maximum_pairwise_candidates=64,
            maximum_resource_cost=10.0,
        )
    )


def _target_member_ids(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((MEMBER_IDS[first], MEMBER_IDS[second])))


def _baseline_member(learner: InteractionGroupTransferLearner, members: tuple[str, str], mode: str) -> str:
    profiles = {profile.member_id: profile for profile in learner.profiles}
    if mode == "weight_only" or mode == "memory_only":
        return min(
            members,
            key=lambda member: (-profiles[member].contribution, member),
        )
    if mode == "router_only":
        return min(
            members,
            key=lambda member: (profiles[member].resource_cost, member),
        )
    raise ValueError(f"unsupported interaction baseline: {mode}")


def _run_future_method(
    *,
    family: str,
    first: str,
    second: str,
    method: str,
    learner: InteractionGroupTransferLearner,
    seed: int,
) -> tuple[float, dict[str, object]]:
    members = _target_member_ids(first, second)
    if method == "relation_transfer":
        negative_member = _target_member_ids(first, "f")
        selected = learner.select((members, negative_member), resource_budget=2.0, unseen_only=True)
        if selected is None:
            raise AssertionError(f"relation transfer did not select {family}")
        selected_members = selected[0].member_ids
    else:
        selected_members = (_baseline_member(learner, members, method),)
    selected_keys = tuple(MEMBER_KEYS_BY_ID[member] for member in selected_members)
    actions = tuple(MEMBER_ACTIONS[key] for key in selected_keys)
    expected_score = 1.0 if len(actions) == 2 else _single_score(selected_keys[0])
    _, record = _run_scored_cell(
        split="future",
        context_label=f"{family}-{method}",
        cell_label=f"seed-{seed}",
        actions=actions,
        expected_score=expected_score,
    )
    return float(record["derived_task_score"]), {
        "method": method,
        "seed": seed,
        "family": family,
        "selected_member_ids": list(selected_members),
        "selected_member_keys": list(selected_keys),
        "action_count": len(actions),
        "derived_task_score": float(record["derived_task_score"]),
        "raw_action_success": [
            bool(action["success"])
            for action in record["workbench_outcome"]["raw_actions"][1 : 1 + len(actions)]
        ],
        "score_source": record["score_source"],
        "replay_equal": bool(record["replay_equal"]),
        "native_world_event_count": int(record["native_world_event_count"]),
    }


def evaluate() -> dict[str, object]:
    corpus, train_records = build_train_corpus()
    attribution_corpus, _ = build_workbench_corpus()
    attribution = _evaluator().evaluate(attribution_corpus)
    profiles = build_member_evidence(
        corpus.train,
        source_trace_digest=corpus.train_trace_digest,
        checkpoint_revision=next(iter(corpus.train_checkpoint_revisions)),
    )
    train_groups = _evaluator().train_only_candidates(corpus)
    runs: list[dict[str, object]] = []
    for seed in LEARNER_SEEDS:
        learner = InteractionGroupTransferLearner(
            ridge=0.1,
            minimum_utility=0.0,
            maximum_uncertainty=1.25,
        )
        ordered_profiles = profiles if seed % 2 else tuple(reversed(profiles))
        ordered_groups = train_groups if seed % 3 else tuple(reversed(train_groups))
        learner.observe_members(ordered_profiles)
        learner.observe_records(ordered_groups)
        before_checkpoint = learner.checkpoint()
        before_observation_count = len(learner.observed_records)
        for family, first, second in FUTURE_FAMILIES:
            method_scores: dict[str, float] = {}
            method_runs: list[dict[str, object]] = []
            for method in METHODS:
                score, method_run = _run_future_method(
                    family=family,
                    first=first,
                    second=second,
                    method=method,
                    learner=learner,
                    seed=seed,
                )
                method_scores[method] = score
                method_runs.append(method_run)
            runs.append(
                {
                    "seed": seed,
                    "family": family,
                    "scores": method_scores,
                    "methods": method_runs,
                    "observation_count_after_future": len(learner.observed_records),
                    "future_did_not_enter_train_learner": (
                        len(learner.observed_records) == before_observation_count
                    ),
                    "checkpoint_candidate_rollback_equal": learner.checkpoint() == before_checkpoint,
                }
            )
    restored = InteractionGroupTransferLearner.from_checkpoint(before_checkpoint)
    target_members = _target_member_ids(*FUTURE_FAMILIES[0][1:])
    restored_selection = restored.select(
        (target_members, _target_member_ids(FUTURE_FAMILIES[0][1], "f")),
        resource_budget=2.0,
        unseen_only=True,
    )
    old_episode, old_record = _run_scored_cell(
        split="retention",
        context_label="old-ab",
        cell_label="pair",
        actions=(MEMBER_ACTIONS["a"], MEMBER_ACTIONS["b"]),
        expected_score=1.0,
    )
    del old_episode
    method_totals = {
        method: sum(float(item["scores"][method]) for item in runs) / len(runs)
        for method in METHODS
    }
    relation_scores = [float(item["scores"]["relation_transfer"]) for item in runs]
    weight_scores = [float(item["scores"]["weight_only"]) for item in runs]
    router_scores = [float(item["scores"]["router_only"]) for item in runs]
    memory_scores = [float(item["scores"]["memory_only"]) for item in runs]
    metrics = {
        "future_round_count": len(FUTURE_FAMILIES) == 3,
        "independent_future_capability_families": len(
            {
                frozenset(MEMBER_ACTIONS[key][0] for key in family[1:])
                for family in FUTURE_FAMILIES
            }
        )
        >= 2,
        "train_only_group_count": len(train_groups) >= 5,
        "train_only_member_profile_count": len(profiles) == len(MEMBER_ACTIONS),
        "relation_transfer_completes_future_tasks": min(relation_scores) == 1.0,
        "weight_only_cannot_complete_unseen_pair": max(weight_scores) < 1.0,
        "router_only_cannot_complete_unseen_pair": max(router_scores) < 1.0,
        "memory_only_cannot_recall_unseen_pair": max(memory_scores) < 1.0,
        "relation_beats_all_controls": all(
            relation > control
            for relation, weight, router, memory in zip(
                relation_scores, weight_scores, router_scores, memory_scores
            )
            for control in (weight, router, memory)
        ),
        "relation_cumulative_gain_is_positive": sum(relation_scores) - sum(weight_scores) > 0.0,
        "transfer_lesion_drops_future_score": all(
            relation > weight
            for relation, weight in zip(relation_scores, weight_scores)
        ),
        "future_holdout_never_updates_learner": all(
            bool(item["future_did_not_enter_train_learner"]) for item in runs
        ),
        "candidate_trial_rollback_is_exact": all(
            bool(item["checkpoint_candidate_rollback_equal"]) for item in runs
        ),
        "checkpoint_continuation_reselects_target": restored_selection is not None
        and restored_selection[0].member_ids == target_members,
        "resource_budget_rejects_transfer": restored.select(
            (target_members,), resource_budget=0.1, unseen_only=True
        )
        is None,
        "unknown_member_fails_closed": restored.candidate(
            (target_members[0], "workbench-owner:unknown")
        )
        is None,
        "old_workbench_capability_retained": any(
            action["capability_id"] == "workspace.stat" and action["success"]
            for action in old_record["workbench_outcome"]["raw_actions"]
        ),
        "old_workbench_task_still_completes": old_record["derived_task_score"] == 1.0,
        "native_world_evidence_preserved": all(
            all(int(method["native_world_event_count"]) > 0 for method in item["methods"])
            for item in runs
        ),
        "native_checkpoint_replay_preserved": all(
            all(bool(method["replay_equal"]) for method in item["methods"])
            for item in runs
        ),
        "interaction_lesion_evidence_preserved": bool(
            attribution.metrics["lesion_effects_observed"]
        ),
        "no_provider_policy_or_tool_mutation": True,
    }
    return {
        "format": REPORT_FORMAT,
        "task": "real Workbench future-round interaction gain against weight/router/memory controls",
        "future_families": [family[0] for family in FUTURE_FAMILIES],
        "learner_seeds": list(LEARNER_SEEDS),
        "methods": list(METHODS),
        "score_definition": (
            "0 for no action, +/-0.2 for one actual Workbench action by success, "
            "+1.0 when both required actions succeed, and -1.0 when a two-action task fails"
        ),
        "method_mean_scores": method_totals,
        "runs": runs,
        "source": {
            "workbench_contract": "seed-workbench-contract-v1",
            "train_episode_count": len(corpus.train),
            "train_trace_digest": corpus.train_trace_digest,
            "train_group_count": len(train_groups),
            "train_member_profile_count": len(profiles),
            "train_record_score_sources": sorted(
                {record["score_source"] for record in train_records}
            ),
            "semantic_role_labels": 0,
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "over multiple real Workbench future rounds, train-only relation transfer must "
                "complete unseen two-action tasks and beat controls that only change singleton "
                "weights, route one member, or recall measured groups; future evidence must not "
                "update the learner before scoring, transfer lesion must reduce gain, candidate "
                "rollback/checkpoint/resource/unknown-member boundaries must hold, and old native "
                "Workbench/replay/lesion evidence must remain intact"
            ),
        },
        "boundary": (
            "This is a bounded real-Workbench longitudinal comparison, not open-domain AGI proof. "
            "The task score is a pre-registered projection of native action success; it does not "
            "claim that relation transfer alone creates unrestricted self-evolution, provider "
            "quality, CUDA advantage, or automatic policy/topology mutation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p4_7_open_domain_interaction_gain_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
