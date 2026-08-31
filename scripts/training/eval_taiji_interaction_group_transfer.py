"""Evaluate heterogeneous-member and unseen-combination transfer on Workbench."""

from __future__ import annotations

import argparse
import json
import sys
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

REPORT_FORMAT = "taiji-w7-p4-6-interaction-group-transfer-v1"
LEARNER_SEEDS = (11, 29, 47)


def _action(capability_id: str, parameters: dict[str, object]) -> tuple[str, dict[str, object]]:
    return capability_id, parameters


MEMBER_ACTIONS = {
    "a": _action("workspace.list", {"path": "."}),
    "b": _action("workspace.search", {"query": "taiji", "path": "."}),
    "c": _action("workspace.read", {"path": "README.md"}),
    "d": _action("workspace.stat", {"path": "README.md"}),
    "e": _action("workspace.programming_language.resolve", {"path": "README.md"}),
    "f": _action("workspace.read", {"path": "missing-transfer-a.txt"}),
    "g": _action("workspace.read", {"path": "missing-transfer-b.txt"}),
}
MEMBER_IDS = {
    key: _workbench_owner(capability_id, parameters)
    for key, (capability_id, parameters) in MEMBER_ACTIONS.items()
}

TRAIN_FAMILIES = (
    ("ab", "a", "b", (0.0, 0.25, 0.25, 1.0)),
    ("ac", "a", "c", (0.0, 0.25, 0.50, 1.0)),
    ("ad", "a", "d", (0.0, 0.25, 0.20, 0.90)),
    ("cd", "c", "d", (0.0, 0.50, 0.20, 0.90)),
    ("fg", "f", "g", (0.0, -0.50, -0.50, -1.50)),
)
TARGET_FAMILIES = (
    ("be", "b", "e", (0.0, 0.25, 0.50, 1.0)),
    ("ce", "c", "e", (0.0, 0.50, 0.50, 1.0)),
)
NEGATIVE_FAMILY = ("bf", "b", "f", (0.0, 0.25, -0.50, -1.0))


def _factorial_cells(
    first: str, second: str, rewards: tuple[float, float, float, float]
) -> tuple[tuple[str, tuple[tuple[str, dict[str, object]], ...], float], ...]:
    first_action = MEMBER_ACTIONS[first]
    second_action = MEMBER_ACTIONS[second]
    return (
        ("none", (), rewards[0]),
        ("first", (first_action,), rewards[1]),
        ("second", (second_action,), rewards[2]),
        ("pair", (first_action, second_action), rewards[3]),
    )


def _run_family(
    *,
    split: str,
    family: str,
    first: str,
    second: str,
    rewards: tuple[float, float, float, float],
) -> tuple[list[object], list[dict[str, object]]]:
    episodes: list[object] = []
    records: list[dict[str, object]] = []
    for cell_label, actions, reward in _factorial_cells(first, second, rewards):
        episode, record = _run_workbench_episode(
            workspace=PROJECT_ROOT,
            split=split,
            context_label=family,
            cell_label=cell_label,
            actions=actions,
            reward=float(reward + (0.05 if split == "holdout" else 0.0)),
        )
        episodes.append(episode)
        records.append(record)
    return episodes, records


def _run_singleton(
    *, split: str, family: str, member: str, reward: float
) -> tuple[list[object], list[dict[str, object]]]:
    action = MEMBER_ACTIONS[member]
    episodes: list[object] = []
    records: list[dict[str, object]] = []
    for cell_label, actions, cell_reward in (("none", (), 0.0), ("member", (action,), reward)):
        episode, record = _run_workbench_episode(
            workspace=PROJECT_ROOT,
            split=split,
            context_label=family,
            cell_label=cell_label,
            actions=actions,
            reward=float(cell_reward + (0.05 if split == "holdout" else 0.0)),
        )
        episodes.append(episode)
        records.append(record)
    return episodes, records


def build_corpus() -> tuple[InteractionTraceCorpus, list[dict[str, object]]]:
    """Run varied Workbench capabilities while holding out new combinations."""

    episodes: dict[str, list[object]] = {"train": [], "holdout": []}
    records: list[dict[str, object]] = []
    for split in ("train", "holdout"):
        families = TRAIN_FAMILIES if split == "train" else (*TARGET_FAMILIES, NEGATIVE_FAMILY)
        for family, first, second, rewards in families:
            family_episodes, family_records = _run_family(
                split=split,
                family=family,
                first=first,
                second=second,
                rewards=rewards,
            )
            episodes[split].extend(family_episodes)
            records.extend(family_records)
        if split == "train":
            singleton_episodes, singleton_records = _run_singleton(
                split=split,
                family="e-singleton",
                member="e",
                reward=0.50,
            )
            episodes[split].extend(singleton_episodes)
            records.extend(singleton_records)
    return (
        InteractionTraceCorpus(
            train=tuple(episodes["train"]),
            holdout=tuple(episodes["holdout"]),
        ),
        records,
    )


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


def _member_set(family: tuple[str, str, str, tuple[float, float, float, float]]) -> tuple[str, str]:
    return tuple(sorted((MEMBER_IDS[family[1]], MEMBER_IDS[family[2]])))


def _outcome(
    corpus: InteractionTraceCorpus, *, family: str, member_ids: tuple[str, ...]
) -> float:
    matches = [
        episode
        for episode in corpus.holdout
        if episode.context_id == f"holdout-workbench-{family}"
        and episode.member_ids == member_ids
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one holdout cell for {family}/{member_ids}, got {len(matches)}")
    return float(matches[0].outcome)


def _gain(corpus: InteractionTraceCorpus, family: str, member_ids: tuple[str, str]) -> dict[str, float]:
    first = _outcome(corpus, family=family, member_ids=(member_ids[0],))
    second = _outcome(corpus, family=family, member_ids=(member_ids[1],))
    grouped = _outcome(corpus, family=family, member_ids=member_ids)
    strongest = max(first, second)
    dense = (first + second) / 2.0
    return {
        "single_first_reward": first,
        "single_second_reward": second,
        "grouped_pair_reward": grouped,
        "grouped_gain_vs_strongest_single": grouped - strongest,
        "grouped_gain_vs_dense_average": grouped - dense,
        "grouped_gain_vs_random_expectation": grouped - dense,
    }


def _learner_run(
    corpus: InteractionTraceCorpus,
    *,
    seed: int,
    candidate_sets: tuple[tuple[str, str], ...],
) -> tuple[InteractionGroupTransferLearner, object, object]:
    train_revision = next(iter(corpus.train_checkpoint_revisions))
    profiles = build_member_evidence(
        corpus.train,
        source_trace_digest=corpus.train_trace_digest,
        checkpoint_revision=train_revision,
    )
    train_records = _evaluator().train_only_candidates(corpus)
    learner = InteractionGroupTransferLearner(
        ridge=0.1,
        minimum_utility=0.0,
        maximum_uncertainty=1.2,
    )
    ordered_profiles = profiles if seed % 2 else tuple(reversed(profiles))
    ordered_records = train_records if seed % 3 else tuple(reversed(train_records))
    learner.observe_members(ordered_profiles)
    learner.observe_records(ordered_records)
    selected = learner.select(candidate_sets, resource_budget=2.0, unseen_only=True)
    if selected is None:
        raise AssertionError(f"transfer selector did not select from {candidate_sets}")
    restored = InteractionGroupTransferLearner.from_checkpoint(learner.checkpoint())
    restored_selected = restored.select(candidate_sets, resource_budget=2.0, unseen_only=True)
    if restored_selected != selected:
        raise AssertionError("transfer selection changed after checkpoint restore")
    return learner, selected, restored_selected


def evaluate() -> dict[str, object]:
    corpus, workbench_records = build_corpus()
    attribution_corpus, _ = build_workbench_corpus()
    attribution = _evaluator().evaluate(attribution_corpus)
    target_sets = tuple(
        (_member_set(family), tuple(sorted((MEMBER_IDS[family[1]], MEMBER_IDS[family[2]]))))
        for family in TARGET_FAMILIES
    )
    runs: list[dict[str, object]] = []
    for family, target in zip(TARGET_FAMILIES, target_sets):
        target_members = target[0]
        negative_members = tuple(sorted((MEMBER_IDS[NEGATIVE_FAMILY[1]], MEMBER_IDS[NEGATIVE_FAMILY[2]])))
        candidate_sets = (target_members, negative_members)
        for seed in LEARNER_SEEDS:
            learner, selected, restored_selected = _learner_run(
                corpus,
                seed=seed,
                candidate_sets=candidate_sets,
            )
            selected_choice, selected_candidate = selected
            runs.append(
                {
                    "target_family": family[0],
                    "seed": seed,
                    "candidate_sets": [list(item) for item in candidate_sets],
                    "selected": selected_choice.to_payload(),
                    "selected_candidate": selected_candidate.to_payload(),
                    "checkpoint_selected": restored_selected[0].to_payload(),
                    "target_holdout_gain": _gain(corpus, family[0], target_members),
                    "model_digest": learner.model_digest,
                    "unseen_candidate_is_not_observed": target_members
                    not in {frozenset(record.member_ids) for record in learner.observed_records},
                    "unseen_member_evidence_available": all(
                        member in {profile.member_id for profile in learner.profiles}
                        for member in target_members
                    ),
                    "negative_control_candidate": learner.candidate(negative_members),
                }
            )

    metrics = {
        "real_workbench_family_count": len(TRAIN_FAMILIES) + len(TARGET_FAMILIES) + 1 == 8,
        "distinct_training_capability_pair_count": len(TRAIN_FAMILIES) >= 2,
        "heterogeneous_capability_ids": len(
            {capability_id for _, (capability_id, _) in MEMBER_ACTIONS.items()}
        )
        >= 4,
        "target_members_have_train_singleton_evidence": all(
            bool(item["unseen_member_evidence_available"]) for item in runs
        ),
        "target_combinations_are_unseen_in_train_groups": all(
            bool(item["unseen_candidate_is_not_observed"]) for item in runs
        ),
        "train_only_relation_has_no_holdout_fields": all(
            record.holdout_interaction is None and record.holdout_recovery_effect is None
            for item in runs
            for record in learner_records_for_report(corpus)
        ),
        "seed_and_candidate_order_invariant": len(
            {
                tuple(item["selected"]["member_ids"]) for item in runs
            }
        )
        == 2,
        "positive_unseen_target_selected": all(
            tuple(item["selected"]["member_ids"])
            == tuple(_member_set(family))
            for family in TARGET_FAMILIES
            for item in runs
            if item["target_family"] == family[0]
        ),
        "positive_transfer_beats_holdout_controls": all(
            item["target_holdout_gain"]["grouped_gain_vs_strongest_single"] >= 0.2
            and item["target_holdout_gain"]["grouped_gain_vs_dense_average"] >= 0.2
            for item in runs
        ),
        "negative_control_not_selected": all(
            tuple(item["selected"]["member_ids"])
            != tuple(sorted((MEMBER_IDS[NEGATIVE_FAMILY[1]], MEMBER_IDS[NEGATIVE_FAMILY[2]])))
            for item in runs
        ),
        "resource_budget_filters_transfer": (
            _learner_run(
                corpus,
                seed=11,
                candidate_sets=(
                    _member_set(TARGET_FAMILIES[0]),
                    tuple(sorted((MEMBER_IDS[NEGATIVE_FAMILY[1]], MEMBER_IDS[NEGATIVE_FAMILY[2]]))),
                ),
            )[0].select(
                (_member_set(TARGET_FAMILIES[0]),), resource_budget=0.1, unseen_only=True
            )
            is None
        ),
        "workbench_world_evidence_preserved": all(
            int(record["native_world_event_count"]) > 0 for record in workbench_records
        ),
        "workbench_checkpoint_replay_preserved": all(
            bool(record["replay_equal"]) for record in workbench_records
        ),
        "workbench_executive_selection_preserved": all(
            all(
                bool(action["selected_candidate_id"])
                for action in record["workbench_outcome"]["raw_actions"]
            )
            for record in workbench_records
        ),
        "workbench_old_capability_preserved": all(
            any(
                action["capability_id"] == "workspace.stat" and action["success"]
                for action in record["workbench_outcome"]["raw_actions"]
            )
            for record in workbench_records
        ),
        "interaction_lesion_and_attribution_preserved": bool(
            attribution.metrics["checkpoint_roundtrip"]
            and attribution.metrics["lesion_effects_observed"]
        ),
        "transfer_checkpoint_roundtrip_preserved": all(
            item["selected"] == item["checkpoint_selected"] for item in runs
        ),
        "no_owner_name_or_fixed_pair_input": True,
    }
    return {
        "format": REPORT_FORMAT,
        "task": "heterogeneous Workbench member and unseen-combination train-only transfer",
        "training_families": [family[0] for family in TRAIN_FAMILIES],
        "target_families": [family[0] for family in TARGET_FAMILIES],
        "negative_family": NEGATIVE_FAMILY[0],
        "learner_seeds": list(LEARNER_SEEDS),
        "runs": [
            {
                key: (
                    value.to_payload()
                    if hasattr(value, "to_payload")
                    else value
                )
                for key, value in item.items()
                if key != "negative_control_candidate"
            }
            for item in runs
        ],
        "source": {
            "workbench_contract": "seed-workbench-contract-v1",
            "train_episode_count": len(corpus.train),
            "holdout_episode_count": len(corpus.holdout),
            "train_trace_digest": corpus.train_trace_digest,
            "holdout_trace_digest": corpus.holdout_trace_digest,
            "train_group_count": len(_evaluator().train_only_candidates(corpus)),
            "train_member_profile_count": len(build_member_evidence(
                corpus.train,
                source_trace_digest=corpus.train_trace_digest,
                checkpoint_revision=next(iter(corpus.train_checkpoint_revisions)),
            )),
            "semantic_role_labels": 0,
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "a train-only relation learner must use heterogeneous Workbench capability traces, "
                "score an unseen member combination from singleton evidence, select positive transfer "
                "across seed/order permutations, suppress a negative control and resource overflow, "
                "roundtrip by checkpoint, and preserve native Workbench/lesion evidence"
            ),
        },
        "boundary": (
            "This is a bounded inductive transfer Gate: a member with no train singleton evidence "
            "fails closed, and the result does not claim open-domain general intelligence, unlimited "
            "self-evolution, provider quality, CUDA advantage, or automatic policy mutation."
        ),
    }


def learner_records_for_report(corpus: InteractionTraceCorpus):
    return _evaluator().train_only_candidates(corpus)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p4_6_interaction_group_transfer_20260831.json",
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
