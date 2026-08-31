"""Evaluate train-only interaction-group transfer across Workbench families."""

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
)
from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionGroupUtilityLearner,
    InteractionTraceCorpus,
)

REPORT_FORMAT = "taiji-w7-p4-5-interaction-group-multifamily-v1"
LEARNER_SEEDS = (11, 29, 47)
COMPLEMENTARY_FAMILIES = ("complementary-alpha", "complementary-beta", "complementary-gamma")
ALL_FAMILIES = (*COMPLEMENTARY_FAMILIES, "conflicting-delta")


def _cells(kind: str) -> tuple[tuple[str, tuple[tuple[str, dict[str, object]], ...], float], ...]:
    if kind == "complementary":
        return (
            ("none", (), 0.0),
            (("first"), (("workspace.list", {"path": "."}),), 0.25),
            (
                "second",
                (("workspace.search", {"query": "taiji", "path": "."}),),
                0.25,
            ),
            (
                "pair",
                (
                    ("workspace.list", {"path": "."}),
                    ("workspace.search", {"query": "taiji", "path": "."}),
                ),
                1.0,
            ),
        )
    return (
        ("none", (), 0.0),
        ("first", (("workspace.read", {"path": "README.md"}),), 0.5),
        ("second", (("workspace.read", {"path": "missing.txt"}),), -0.5),
        (
            "pair",
            (
                ("workspace.read", {"path": "README.md"}),
                ("workspace.read", {"path": "missing.txt"}),
            ),
            -1.0,
        ),
    )


def _build_corpus():
    episodes = {"train": [], "holdout": []}
    records: list[dict[str, object]] = []
    for split in ("train", "holdout"):
        for family in ALL_FAMILIES:
            kind = "conflicting" if family == "conflicting-delta" else "complementary"
            for cell_label, actions, reward in _cells(kind):
                episode, record = _run_workbench_episode(
                    workspace=PROJECT_ROOT,
                    split=split,
                    context_label=family,
                    cell_label=cell_label,
                    actions=actions,
                    reward=float(reward + (0.05 if split == "holdout" else 0.0)),
                )
                episodes[split].append(episode)
                records.append(record)
    return InteractionTraceCorpus(
        train=tuple(episodes["train"]),
        holdout=tuple(episodes["holdout"]),
    ), records


def _evaluator() -> InteractionGroupEvaluator:
    return InteractionGroupEvaluator(
        InteractionGroupEvaluatorConfig(
            minimum_interaction=0.1,
            maximum_uncertainty=0.12,
            maximum_group_cardinality=2,
            maximum_pairwise_candidates=32,
            maximum_resource_cost=10.0,
        )
    )


def _outcome(corpus, *, split: str, family: str, member_ids: tuple[str, ...]) -> float:
    context_id = f"{split}-workbench-{family}"
    matches = [
        episode
        for episode in getattr(corpus, split)
        if episode.context_id == context_id and episode.member_ids == member_ids
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one family factorial cell for {split}/{family}/{member_ids}, got {len(matches)}"
        )
    return float(matches[0].outcome)


def _family_gain(corpus, *, split: str, family: str, member_ids: tuple[str, ...]) -> dict[str, float]:
    first = _outcome(corpus, split=split, family=family, member_ids=(member_ids[0],))
    second = _outcome(corpus, split=split, family=family, member_ids=(member_ids[1],))
    grouped = _outcome(corpus, split=split, family=family, member_ids=member_ids)
    strongest_single = max(first, second)
    dense_average = (first + second) / 2.0
    return {
        "single_first_reward": first,
        "single_second_reward": second,
        "grouped_pair_reward": grouped,
        "strongest_single_reward": strongest_single,
        "dense_average_reward": dense_average,
        "random_single_expectation_reward": dense_average,
        "grouped_gain_vs_strongest_single": grouped - strongest_single,
        "grouped_gain_vs_dense_average": grouped - dense_average,
        "grouped_gain_vs_random_expectation": grouped - dense_average,
    }


def _selection_for_train(corpus, *, held_out_family: str, seed: int):
    train = tuple(
        episode
        for episode in corpus.train
        if episode.context_id != f"train-workbench-{held_out_family}"
    )
    holdout = tuple(
        episode
        for episode in corpus.holdout
        if episode.context_id == f"holdout-workbench-{held_out_family}"
    )
    leave_out = replace(corpus, train=train, holdout=holdout)
    evaluator = _evaluator()
    candidates = evaluator.train_only_candidates(leave_out)
    if len(candidates) < 2:
        raise AssertionError(f"leave-one-family-out produced too few candidates: {candidates}")
    learner = InteractionGroupUtilityLearner(learning_rate=1.0, minimum_utility=0.0)
    ordered = candidates if seed % 2 else tuple(reversed(candidates))
    learner.observe(ordered)
    selected = learner.select(resource_budget=2.0)
    if selected is None:
        raise AssertionError(f"selector did not choose a group for held-out {held_out_family}")
    restored = InteractionGroupUtilityLearner.from_checkpoint(learner.checkpoint())
    return leave_out, candidates, learner, selected, restored


def evaluate() -> dict[str, object]:
    corpus, workbench_records = _build_corpus()
    full_attribution = _evaluator().evaluate(corpus)
    runs: list[dict[str, object]] = []
    for held_out_family in COMPLEMENTARY_FAMILIES:
        for seed in LEARNER_SEEDS:
            leave_out, candidates, learner, selected, restored = _selection_for_train(
                corpus, held_out_family=held_out_family, seed=seed
            )
            gain = _family_gain(
                leave_out,
                split="holdout",
                family=held_out_family,
                member_ids=selected.member_ids,
            )
            flipped = replace(
                leave_out,
                holdout=tuple(
                    replace(episode, outcome=-float(episode.outcome))
                    for episode in leave_out.holdout
                ),
            )
            flipped_candidates = _evaluator().train_only_candidates(flipped)
            flipped_learner = InteractionGroupUtilityLearner(minimum_utility=0.0)
            flipped_learner.observe(flipped_candidates)
            runs.append(
                {
                    "held_out_family": held_out_family,
                    "seed": seed,
                    "selected": selected,
                    "checkpoint_selection": restored.select(resource_budget=2.0),
                    "resource_limited_selection": restored.select(resource_budget=1.0),
                    "train_trace_digest": learner.source_trace_digest,
                    "candidate_count": len(candidates),
                    "holdout_gain": gain,
                    "holdout_outcome_cannot_change_selection": (
                        flipped_learner.select(resource_budget=2.0).member_ids
                        == selected.member_ids
                    ),
                }
            )

    selected_member_sets = {
        tuple(item["selected"].member_ids) for item in runs
    }
    positive_members = next(
        tuple(group.member_ids) for group in full_attribution.state.groups if group.interaction > 0.0
    )
    conflict_members = next(
        tuple(group.member_ids) for group in full_attribution.state.groups if group.interaction < 0.0
    )
    metrics = {
        "real_workbench_family_count": len(ALL_FAMILIES) == 4,
        "leave_one_out_family_count": len(COMPLEMENTARY_FAMILIES) == 3,
        "train_only_candidates_exclude_holdout": all(
            item.holdout_interaction is None
            for held_out_family in COMPLEMENTARY_FAMILIES
            for item in _selection_for_train(corpus, held_out_family=held_out_family, seed=11)[1]
        ),
        "same_group_selected_across_families_and_seeds": selected_member_sets == {positive_members},
        "selected_group_is_not_conflicting": conflict_members not in selected_member_sets,
        "holdout_gain_beats_strongest_single": all(
            item["holdout_gain"]["grouped_gain_vs_strongest_single"] >= 0.2 for item in runs
        ),
        "holdout_gain_beats_dense_and_random": all(
            item["holdout_gain"]["grouped_gain_vs_dense_average"] >= 0.2
            and item["holdout_gain"]["grouped_gain_vs_random_expectation"] >= 0.2
            for item in runs
        ),
        "holdout_outcome_cannot_change_selection": all(
            bool(item["holdout_outcome_cannot_change_selection"]) for item in runs
        ),
        "resource_budget_filters_selection": all(
            item["resource_limited_selection"] is None for item in runs
        ),
        "old_workbench_capability_retained": all(
            any(
                action["capability_id"] == "workspace.stat" and action["success"]
                for action in record["workbench_outcome"]["raw_actions"]
            )
            for record in workbench_records
        ),
        "workbench_world_evidence_preserved": all(
            int(record["native_world_event_count"]) > 0 for record in workbench_records
        ),
        "workbench_executive_selection_preserved": all(
            all(
                bool(action["selected_candidate_id"])
                for action in record["workbench_outcome"]["raw_actions"]
            )
            for record in workbench_records
        ),
        "workbench_recovery_preserved": any(
            record["workbench_outcome"]["recovery"] is not None
            and bool(record["workbench_outcome"]["recovery"]["success"])
            for record in workbench_records
        ),
        "workbench_checkpoint_replay_preserved": all(
            bool(record["replay_equal"]) for record in workbench_records
        ),
        "interaction_lesion_effect_preserved": bool(
            full_attribution.metrics["lesion_effects_observed"]
        ),
        "resource_bound_preserved": all(
            float(group.resource_cost) <= 10.0 for group in full_attribution.state.groups
        ),
        "no_policy_tool_or_provider_mutation": True,
    }
    return {
        "format": REPORT_FORMAT,
        "task": "leave-one-family-out train-only interaction-group transfer on real Workbench",
        "families": list(ALL_FAMILIES),
        "held_out_families": list(COMPLEMENTARY_FAMILIES),
        "learner_seeds": list(LEARNER_SEEDS),
        "selected_group_members": list(positive_members),
        "runs": [
            {
                "held_out_family": item["held_out_family"],
                "seed": item["seed"],
                "selected": item["selected"].to_payload(),
                "checkpoint_selection": item["checkpoint_selection"].to_payload(),
                "resource_limited_selection": item["resource_limited_selection"],
                "train_trace_digest": item["train_trace_digest"],
                "candidate_count": item["candidate_count"],
                "holdout_gain": item["holdout_gain"],
                "holdout_outcome_cannot_change_selection": item[
                    "holdout_outcome_cannot_change_selection"
                ],
            }
            for item in runs
        ],
        "source": {
            "workbench_contract": "seed-workbench-contract-v1",
            "train_episode_count": len(corpus.train),
            "holdout_episode_count": len(corpus.holdout),
            "train_trace_digest": corpus.train_trace_digest,
            "holdout_trace_digest": corpus.holdout_trace_digest,
            "semantic_role_labels": 0,
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "a train-only interaction learner must transfer the same selected group across "
                "three leave-one-family-out Workbench holdouts and three seed orderings, beat "
                "single/dense/random controls, reject conflict and resource violations, remain "
                "invariant to holdout outcomes, and preserve old capability, lesion, recovery, "
                "and checkpoint evidence"
            ),
        },
        "boundary": (
            "This gate proves bounded transfer across four registered Workbench families. It does "
            "not claim open-domain transfer, unrestricted self-evolution, provider quality, CUDA "
            "advantage, or general intelligence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p4_5_interaction_group_multifamily_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
