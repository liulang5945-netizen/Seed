"""Evaluate trace-grounded interaction groups on a deterministic S0 corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionTraceCorpus,
    InteractionTraceEpisode,
    InteractionTraceEvent,
)


def _episode(
    *,
    split: str,
    episode_id: str,
    context_id: str,
    owner_ids: tuple[str, ...],
    outcome: float,
    recovery_effect: float,
) -> InteractionTraceEpisode:
    full_episode_id = f"{split}-{episode_id}"
    outcome_id = f"{full_episode_id}:outcome"
    return InteractionTraceEpisode(
        episode_id=full_episode_id,
        checkpoint_revision=7,
        outcome_id=outcome_id,
        events=tuple(
            InteractionTraceEvent(
                event_id=f"{full_episode_id}:event:{index}",
                owner_id=owner_id,
                episode_id=full_episode_id,
                checkpoint_revision=7,
                outcome_id=outcome_id,
                resource_cost=0.4 + 0.1 * index,
            )
            for index, owner_id in enumerate(owner_ids)
        ),
        outcome=outcome,
        recovery_effect=recovery_effect,
        context_id=context_id,
    )


def build_corpus() -> InteractionTraceCorpus:
    """Build two unseen task contexts without semantic role labels."""

    train = (
        _episode(
            split="train",
            episode_id="ab-none",
            context_id="task-family-ab",
            owner_ids=(),
            outcome=0.0,
            recovery_effect=0.0,
        ),
        _episode(
            split="train",
            episode_id="ab-a",
            context_id="task-family-ab",
            owner_ids=("surface-a",),
            outcome=0.2,
            recovery_effect=0.1,
        ),
        _episode(
            split="train",
            episode_id="ab-b",
            context_id="task-family-ab",
            owner_ids=("surface-b",),
            outcome=0.3,
            recovery_effect=0.1,
        ),
        _episode(
            split="train",
            episode_id="ab-both",
            context_id="task-family-ab",
            owner_ids=("surface-a", "surface-b"),
            outcome=1.0,
            recovery_effect=0.7,
        ),
        _episode(
            split="train",
            episode_id="cd-none",
            context_id="task-family-cd",
            owner_ids=(),
            outcome=0.0,
            recovery_effect=0.0,
        ),
        _episode(
            split="train",
            episode_id="cd-c",
            context_id="task-family-cd",
            owner_ids=("surface-c",),
            outcome=0.6,
            recovery_effect=0.2,
        ),
        _episode(
            split="train",
            episode_id="cd-d",
            context_id="task-family-cd",
            owner_ids=("surface-d",),
            outcome=0.2,
            recovery_effect=0.1,
        ),
        _episode(
            split="train",
            episode_id="cd-both",
            context_id="task-family-cd",
            owner_ids=("surface-c", "surface-d"),
            outcome=-0.2,
            recovery_effect=0.05,
        ),
    )
    holdout = (
        _episode(
            split="holdout",
            episode_id="ab-none",
            context_id="unseen-family-ab",
            owner_ids=(),
            outcome=0.05,
            recovery_effect=0.0,
        ),
        _episode(
            split="holdout",
            episode_id="ab-a",
            context_id="unseen-family-ab",
            owner_ids=("surface-a",),
            outcome=0.25,
            recovery_effect=0.1,
        ),
        _episode(
            split="holdout",
            episode_id="ab-b",
            context_id="unseen-family-ab",
            owner_ids=("surface-b",),
            outcome=0.35,
            recovery_effect=0.1,
        ),
        _episode(
            split="holdout",
            episode_id="ab-both",
            context_id="unseen-family-ab",
            owner_ids=("surface-a", "surface-b"),
            outcome=1.1,
            recovery_effect=0.8,
        ),
        _episode(
            split="holdout",
            episode_id="cd-none",
            context_id="unseen-family-cd",
            owner_ids=(),
            outcome=0.05,
            recovery_effect=0.0,
        ),
        _episode(
            split="holdout",
            episode_id="cd-c",
            context_id="unseen-family-cd",
            owner_ids=("surface-c",),
            outcome=0.55,
            recovery_effect=0.2,
        ),
        _episode(
            split="holdout",
            episode_id="cd-d",
            context_id="unseen-family-cd",
            owner_ids=("surface-d",),
            outcome=0.15,
            recovery_effect=0.1,
        ),
        _episode(
            split="holdout",
            episode_id="cd-both",
            context_id="unseen-family-cd",
            owner_ids=("surface-c", "surface-d"),
            outcome=-0.3,
            recovery_effect=0.05,
        ),
    )
    return InteractionTraceCorpus(train=train, holdout=holdout)


def evaluate() -> dict[str, object]:
    corpus = build_corpus()
    result = InteractionGroupEvaluator(
        InteractionGroupEvaluatorConfig(
            minimum_interaction=0.1,
            maximum_uncertainty=0.12,
            maximum_group_cardinality=2,
            maximum_pairwise_candidates=32,
            maximum_resource_cost=10.0,
        )
    ).evaluate(corpus)
    report = result.to_report()
    report["task"] = "deterministic trace-grounded pair interaction attribution"
    report["input"] = {
        "train_episode_count": len(corpus.train),
        "holdout_episode_count": len(corpus.holdout),
        "train_checkpoint_revisions": sorted(corpus.train_checkpoint_revisions),
        "train_trace_digest": corpus.train_trace_digest,
        "holdout_trace_digest": corpus.holdout_trace_digest,
        "owner_ids_are_opaque": True,
        "semantic_role_labels": 0,
    }
    report["boundary"] = {
        "policy_mutation": False,
        "tool_selection": False,
        "provider_selection": False,
        "high_order_search": False,
        "holdout_can_add_evidence": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r2_interaction_groups_20260829.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": report["gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
