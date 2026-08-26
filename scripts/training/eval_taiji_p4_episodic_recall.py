"""Evaluate one-shot cue-conditioned episodic recall for Taiji P4."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ActionIntent,
    EpisodicMemoryRecord,
    EpisodicMemoryStore,
    Outcome,
)

MANIFEST_FORMAT = "taiji-p4-episodic-recall-manifest-v1"
REPORT_FORMAT = "taiji-p4-episodic-recall-v1"


@dataclass(frozen=True)
class RecallQuery:
    cue: torch.Tensor
    expected_action: str
    expected_reward: float
    episode_id: str


def build_corpus(
    *, cue_dim: int = 8
) -> tuple[tuple[EpisodicMemoryRecord, ...], tuple[RecallQuery, ...]]:
    if int(cue_dim) <= 0:
        raise ValueError("P4 cue_dim must be positive")
    train = tuple(
        EpisodicMemoryRecord(
            memory_id=f"train-memory-{index}",
            episode_id=f"train-episode-{index}",
            tick=1,
            cue=torch.eye(cue_dim)[index],
            action_intent=ActionIntent(
                intent_id=f"train-intent-{index}",
                kind=f"action-{index}",
                parameters={"class": index},
                tick=0,
            ),
            outcome=Outcome(
                intent_id=f"train-intent-{index}",
                reward=float(index - 3),
                success=index % 2 == 0,
                tick=1,
            ),
        )
        for index in range(cue_dim)
    )
    queries = tuple(
        RecallQuery(
            cue=torch.eye(cue_dim)[index],
            expected_action=f"action-{index}",
            expected_reward=float(index - 3),
            episode_id=f"holdout-episode-{index}",
        )
        for index in range(cue_dim)
    )
    return train, queries


def _recall_accuracy(
    store: EpisodicMemoryStore, queries: tuple[RecallQuery, ...]
) -> dict[str, float]:
    action_hits = 0
    reward_errors: list[float] = []
    for query in queries:
        hits = store.retrieve(query.cue, limit=1)
        if not hits or hits[0].record.action_intent is None or hits[0].record.outcome is None:
            continue
        action_hits += int(hits[0].record.action_intent.kind == query.expected_action)
        reward_errors.append(abs(hits[0].record.outcome.reward - query.expected_reward))
    return {
        "action_recall": action_hits / len(queries),
        "mean_reward_error": sum(reward_errors) / len(reward_errors) if reward_errors else 0.0,
    }


def evaluate_recall(
    train: tuple[EpisodicMemoryRecord, ...], queries: tuple[RecallQuery, ...]
) -> dict[str, object]:
    full_store = EpisodicMemoryStore(capacity=len(train))
    for record in train:
        full_store.write(record)
    episode_lesion_store = EpisodicMemoryStore(capacity=len(train))
    for record in train:
        episode_lesion_store.write(replace(record, episode_id="episode-id-lesion"))
    checkpoint_store = EpisodicMemoryStore.from_checkpoint(full_store.checkpoint())
    empty_store = EpisodicMemoryStore(capacity=len(train))

    conditions = {
        "full": full_store,
        "episode_id_lesion": episode_lesion_store,
        "retrieval_lesion": empty_store,
        "write_lesion": EpisodicMemoryStore(capacity=len(train)),
        "checkpoint_continuation": checkpoint_store,
    }
    metrics = {name: _recall_accuracy(store, queries) for name, store in conditions.items()}
    return {
        "format": REPORT_FORMAT,
        "cue_dim": int(train[0].cue.numel()),
        "train_records": len(train),
        "holdout_queries": len(queries),
        "conditions": metrics,
        "gate": {
            "passed": bool(
                metrics["full"]["action_recall"] == 1.0
                and metrics["episode_id_lesion"]["action_recall"] == 1.0
                and metrics["checkpoint_continuation"]["action_recall"] == 1.0
                and metrics["retrieval_lesion"]["action_recall"] < 0.5
                and metrics["write_lesion"]["action_recall"] < 0.5
            ),
            "criterion": "full/episode-ID-lesion/checkpoint recall = 1.0; retrieval/write lesions < 0.5",
        },
    }


def build_manifest(*, cue_dim: int = 8) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "one-shot cue-conditioned action and reward recall",
        "cue_dim": cue_dim,
        "train_episode_ids": [f"train-episode-{index}" for index in range(cue_dim)],
        "holdout_episode_ids": [f"holdout-episode-{index}" for index in range(cue_dim)],
        "controls": [
            "full",
            "episode_id_lesion",
            "retrieval_lesion",
            "write_lesion",
            "checkpoint_continuation",
        ],
        "boundary": "this is episodic recall, not cross-episode semantic composition",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_episodic_recall_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_episodic_recall_baseline_20260825.json",
    )
    args = parser.parse_args()
    train, queries = build_corpus()
    report = evaluate_recall(train, queries)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
