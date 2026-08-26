"""Scale the P4 semantic consolidation gate to noisy multi-factor episodes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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

MANIFEST_FORMAT = "taiji-p4-semantic-scale-manifest-v1"
REPORT_FORMAT = "taiji-p4-semantic-scale-v1"


def build_corpus(
    *,
    factor_count: int = 4,
    repeats_per_pattern: int = 4,
    noise_scale: float = 0.05,
    seed: int = 20260825,
) -> tuple[EpisodicMemoryStore, torch.Tensor, float]:
    if factor_count <= 0 or repeats_per_pattern <= 0:
        raise ValueError("semantic scale factor_count and repeats must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    store = EpisodicMemoryStore(
        capacity=(2**factor_count) * repeats_per_pattern, cue_dim=factor_count
    )
    index = 0
    for pattern in range(2**factor_count - 1):
        bits = torch.tensor(
            [(pattern >> bit) & 1 for bit in range(factor_count)], dtype=torch.float32
        )
        reward = float(bits.sum())
        for _ in range(repeats_per_pattern):
            intent_id = f"scale-intent-{index}"
            store.write(
                EpisodicMemoryRecord(
                    memory_id=f"scale-memory-{index}",
                    episode_id=f"scale-episode-{index}",
                    tick=1,
                    cue=bits + torch.randn(factor_count, generator=generator) * noise_scale,
                    action_intent=ActionIntent(intent_id, "factor-compose", tick=0),
                    outcome=Outcome(intent_id, reward=reward, success=True, tick=1),
                )
            )
            index += 1
    query = torch.ones(factor_count) + torch.randn(factor_count, generator=generator) * noise_scale
    return store, query, float(factor_count)


def evaluate_scale() -> dict[str, object]:
    store, query, target = build_corpus()
    semantic = SemanticMemoryLearner(cue_dim=query.numel())
    semantic.consolidate(store, epochs=500, learning_rate=0.05)
    checkpoint = SemanticMemoryLearner.from_checkpoint(semantic.checkpoint())
    episode_lesion_store = EpisodicMemoryStore(capacity=store.capacity, cue_dim=query.numel())
    for record in store.records:
        episode_lesion_store.write(replace(record, episode_id="episode-id-lesion"))
    episode_lesion = SemanticMemoryLearner(cue_dim=query.numel())
    episode_lesion.consolidate(episode_lesion_store, epochs=500, learning_rate=0.05)
    nearest = store.retrieve(query, limit=1)
    nearest_prediction = (
        0.0
        if not nearest or nearest[0].record.outcome is None
        else nearest[0].record.outcome.reward
    )
    semantic_error = abs(semantic.predict(query) - target)
    replay_error = abs(SemanticMemoryLearner(query.numel()).predict(query) - target)
    return {
        "format": REPORT_FORMAT,
        "factor_count": query.numel(),
        "train_records": store.count,
        "holdout_combinations": 1,
        "noise_scale": 0.05,
        "metrics": {
            "episodic_nearest_error": abs(nearest_prediction - target),
            "semantic_consolidated_error": semantic_error,
            "replay_lesion_error": replay_error,
            "episode_id_lesion_error": abs(episode_lesion.predict(query) - target),
            "checkpoint_continuation_error": abs(checkpoint.predict(query) - target),
            "consolidation_count": semantic.consolidation_count,
        },
        "gate": {
            "passed": bool(
                semantic_error < 0.2
                and semantic_error < abs(nearest_prediction - target)
                and replay_error > 2.0
                and abs(episode_lesion.predict(query) - target) < 0.2
                and abs(checkpoint.predict(query) - target) < 0.2
            ),
            "criterion": "multi-factor noisy consolidation beats episodic nearest neighbor and preserves replay/episode/checkpoint controls",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "infer the held-out all-active composition from noisy multi-factor episodes",
        "factor_count": 4,
        "seen_combinations": 15,
        "held_out_combination": [1, 1, 1, 1],
        "repeats_per_seen_combination": 4,
        "noise_scale": 0.05,
        "controls": [
            "episodic_nearest",
            "replay_lesion",
            "episode_id_lesion",
            "checkpoint_continuation",
        ],
        "boundary": "additive multi-factor relation only; not general semantic or language competence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_semantic_scale_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_semantic_scale_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate_scale()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
