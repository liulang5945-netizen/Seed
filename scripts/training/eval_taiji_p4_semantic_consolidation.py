"""Evaluate cross-episode semantic consolidation from episodic outcomes."""

from __future__ import annotations

import argparse
import json
import sys
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
    SemanticMemoryLearner,
)

MANIFEST_FORMAT = "taiji-p4-semantic-consolidation-manifest-v1"
REPORT_FORMAT = "taiji-p4-semantic-consolidation-v1"


def build_corpus() -> tuple[EpisodicMemoryStore, tuple[torch.Tensor, ...], tuple[float, ...]]:
    store = EpisodicMemoryStore(capacity=3, cue_dim=2)
    observed = ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))
    for index, (left, right, reward) in enumerate(observed):
        intent_id = f"semantic-intent-{index}"
        store.write(
            EpisodicMemoryRecord(
                memory_id=f"semantic-memory-{index}",
                episode_id=f"episode-{index}",
                tick=1,
                cue=torch.tensor([left, right]),
                action_intent=ActionIntent(intent_id, "compose", tick=0),
                outcome=Outcome(intent_id, reward=reward, success=True, tick=1),
            )
        )
    queries = (torch.tensor([1.0, 1.0]),)
    targets = (2.0,)
    return store, queries, targets


def _nearest_episode_error(store: EpisodicMemoryStore, cue: torch.Tensor, target: float) -> float:
    hit = store.retrieve(cue, limit=1)
    if not hit or hit[0].record.outcome is None:
        return abs(target)
    return abs(hit[0].record.outcome.reward - target)


def evaluate_consolidation() -> dict[str, object]:
    store, queries, targets = build_corpus()
    semantic = SemanticMemoryLearner(cue_dim=2)
    semantic.consolidate(store, epochs=300, learning_rate=0.1)
    semantic_checkpoint = SemanticMemoryLearner.from_checkpoint(semantic.checkpoint())
    query, target = queries[0], targets[0]
    consolidated_error = abs(semantic.predict(query) - target)
    checkpoint_error = abs(semantic_checkpoint.predict(query) - target)
    nearest_error = _nearest_episode_error(store, query, target)
    replay_lesion = SemanticMemoryLearner(cue_dim=2)
    episode_lesion = EpisodicMemoryStore(capacity=3, cue_dim=2)
    for record in store.records:
        episode_lesion.write(
            EpisodicMemoryRecord.from_payload(
                {**record.to_payload(), "episode_id": "episode-id-lesion"}
            )
        )
    episode_lesion_model = SemanticMemoryLearner(cue_dim=2)
    episode_lesion_model.consolidate(episode_lesion, epochs=300, learning_rate=0.1)
    return {
        "format": REPORT_FORMAT,
        "train_records": store.count,
        "holdout_queries": 1,
        "metrics": {
            "episodic_nearest_error": nearest_error,
            "semantic_consolidated_error": consolidated_error,
            "replay_lesion_error": abs(replay_lesion.predict(query) - target),
            "episode_id_lesion_error": abs(episode_lesion_model.predict(query) - target),
            "checkpoint_continuation_error": checkpoint_error,
            "consolidation_count": semantic.consolidation_count,
        },
        "gate": {
            "passed": bool(
                consolidated_error < 0.05
                and consolidated_error < nearest_error
                and checkpoint_error < 0.05
                and abs(replay_lesion.predict(query) - target) > 0.5
                and abs(episode_lesion_model.predict(query) - target) < 0.05
            ),
            "criterion": "consolidation beats nearest episode on an unseen composition; replay and checkpoint controls are causal",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "infer the unseen [1,1] composition reward from [0,0], [1,0], [0,1] episodes",
        "train_episodes": [[0, 0, 0], [1, 0, 1], [0, 1, 1]],
        "holdout_composition": [1, 1],
        "holdout_reward": 2,
        "controls": ["episodic_nearest", "replay_lesion", "episode_id_lesion", "checkpoint_continuation"],
        "boundary": "one additive semantic relation only; not a general language or concept benchmark",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_semantic_consolidation_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_semantic_consolidation_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate_consolidation()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
