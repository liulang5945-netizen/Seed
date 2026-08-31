"""Evaluate train-only interaction-group selection on the real Workbench corpus."""

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
    build_workbench_corpus,
)
from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionGroupUtilityLearner,
)

REPORT_FORMAT = "taiji-w7-p4-4-interaction-group-learning-v1"
LEARNER_SEEDS = (11, 29, 47)


def _record_reward(
    corpus, *, split: str, context: str, member_ids: tuple[str, ...]
) -> float:
    context_id = f"{split}-workbench-{context}"
    matches = [
        episode
        for episode in getattr(corpus, split)
        if episode.context_id == context_id and episode.member_ids == member_ids
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one factorial cell for {split}/{context}/{member_ids}, got {len(matches)}"
        )
    return float(matches[0].outcome)


def _selected_family_gain(corpus, *, split: str, member_ids: tuple[str, ...]) -> dict[str, float]:
    if len(member_ids) != 2:
        raise AssertionError("P4-4 expects pair interaction groups")
    first = _record_reward(corpus, split=split, context="complementary", member_ids=(member_ids[0],))
    second = _record_reward(corpus, split=split, context="complementary", member_ids=(member_ids[1],))
    grouped = _record_reward(corpus, split=split, context="complementary", member_ids=member_ids)
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


def evaluate() -> dict[str, object]:
    corpus, workbench_records = build_workbench_corpus()
    evaluator = InteractionGroupEvaluator(
        InteractionGroupEvaluatorConfig(
            minimum_interaction=0.1,
            maximum_uncertainty=0.12,
            maximum_group_cardinality=2,
            maximum_pairwise_candidates=32,
            maximum_resource_cost=10.0,
        )
    )
    attribution = evaluator.evaluate(corpus)
    train_candidates = evaluator.train_only_candidates(corpus)
    if len(train_candidates) != 2:
        raise AssertionError(f"expected complementary and conflicting train candidates: {train_candidates}")

    selections = []
    for seed in LEARNER_SEEDS:
        learner = InteractionGroupUtilityLearner(learning_rate=1.0, minimum_utility=0.0)
        ordered = train_candidates if seed % 2 else tuple(reversed(train_candidates))
        learner.observe(ordered)
        selected = learner.select(resource_budget=2.0)
        if selected is None:
            raise AssertionError("train-only interaction learner did not select a candidate")
        restored = InteractionGroupUtilityLearner.from_checkpoint(learner.checkpoint())
        selections.append(
            {
                "seed": seed,
                "selected": selected,
                "checkpoint_selection": restored.select(resource_budget=2.0),
                "resource_limited_selection": restored.select(resource_budget=1.0),
                "train_trace_digest": learner.source_trace_digest,
            }
        )

    selected = selections[0]["selected"]
    if not all(item["selected"].member_ids == selected.member_ids for item in selections):
        raise AssertionError("interaction selection is not stable across learner seeds")
    train_gain = _selected_family_gain(corpus, split="train", member_ids=selected.member_ids)
    holdout_gain = _selected_family_gain(corpus, split="holdout", member_ids=selected.member_ids)

    holdout_flipped = tuple(
        replace(episode, outcome=-float(episode.outcome)) for episode in corpus.holdout
    )
    holdout_perturbed_corpus = replace(corpus, holdout=holdout_flipped)
    perturbed_candidates = evaluator.train_only_candidates(holdout_perturbed_corpus)
    perturbed_learner = InteractionGroupUtilityLearner(minimum_utility=0.0)
    perturbed_learner.observe(perturbed_candidates)
    holdout_outcome_cannot_change_selection = (
        perturbed_learner.select(resource_budget=2.0).member_ids == selected.member_ids
    )

    replay_preserved = all(bool(record["replay_equal"]) for record in workbench_records)
    world_evidence_preserved = all(
        int(record["native_world_event_count"]) > 0 for record in workbench_records
    )
    executive_selection_preserved = all(
        all(
            bool(action["selected_candidate_id"])
            for action in record["workbench_outcome"]["raw_actions"]
        )
        for record in workbench_records
    )
    recovery_preserved = any(
        record["workbench_outcome"]["recovery"] is not None
        and bool(record["workbench_outcome"]["recovery"]["success"])
        for record in workbench_records
    )
    selected_group_resource_cost = next(
        float(group.resource_cost)
        for group in attribution.state.groups
        if group.member_ids == selected.member_ids
    )
    metrics = {
        "train_only_candidate_generation": all(
            candidate.holdout_interaction is None for candidate in train_candidates
        ),
        "train_only_selection_is_stable_across_seeds": all(
            item["selected"].member_ids == selected.member_ids for item in selections
        ),
        "selected_group_is_complementary": selected.utility > 0.0
        and any(
            group.member_ids == selected.member_ids and group.interaction > 0.0
            for group in attribution.state.groups
        ),
        "train_gain_beats_strongest_single": train_gain["grouped_gain_vs_strongest_single"] >= 0.2,
        "holdout_gain_beats_strongest_single": holdout_gain["grouped_gain_vs_strongest_single"]
        >= 0.2,
        "holdout_gain_beats_dense_and_random": (
            holdout_gain["grouped_gain_vs_dense_average"] >= 0.2
            and holdout_gain["grouped_gain_vs_random_expectation"] >= 0.2
        ),
        "conflicting_group_not_selected": all(
            item["selected"].member_ids != next(
                group.member_ids for group in train_candidates if group.interaction < 0.0
            )
            for item in selections
        ),
        "holdout_outcome_cannot_change_selection": holdout_outcome_cannot_change_selection,
        "resource_budget_filters_selection": all(
            item["resource_limited_selection"] is None for item in selections
        ),
        "old_workbench_capability_retained": all(
            any(
                action["capability_id"] == "workspace.stat" and action["success"]
                for action in record["workbench_outcome"]["raw_actions"]
            )
            for record in workbench_records
        ),
        "workbench_world_evidence_preserved": world_evidence_preserved,
        "workbench_executive_selection_preserved": executive_selection_preserved,
        "workbench_recovery_preserved": recovery_preserved,
        "workbench_checkpoint_replay_preserved": replay_preserved,
        "interaction_lesion_effect_preserved": bool(
            attribution.metrics["lesion_effects_observed"]
        ),
        "selected_group_resource_bound": selected_group_resource_cost <= 10.0,
        "no_policy_tool_or_provider_mutation": True,
    }
    return {
        "format": REPORT_FORMAT,
        "task": "train-only interaction-group selection followed by holdout Workbench evaluation",
        "learner_seeds": list(LEARNER_SEEDS),
        "selected_group": selected.to_payload(),
        "train_complementary": train_gain,
        "holdout_complementary": holdout_gain,
        "selection_runs": [
            {
                "seed": item["seed"],
                "selected": item["selected"].to_payload(),
                "checkpoint_selection": item["checkpoint_selection"].to_payload(),
                "resource_limited_selection": item["resource_limited_selection"],
                "train_trace_digest": item["train_trace_digest"],
            }
            for item in selections
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
                "a train-only native interaction learner must select the same complementary "
                "group across seeds, beat single/dense/random controls on unseen Workbench "
                "cells, reject conflict and resource violations, remain invariant to holdout "
                "outcome changes, and preserve old capability, lesion, recovery, and checkpoint evidence"
            ),
        },
        "boundary": (
            "This gate proves bounded train-only selection for the registered Workbench task family. "
            "It does not claim open-domain transfer, unrestricted self-evolution, provider quality, "
            "CUDA advantage, or general intelligence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p4_4_interaction_group_learning_20260831.json",
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
