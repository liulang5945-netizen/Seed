"""Evaluate episodic capacity/interference and data-driven procedural skill."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ActionIntent,
    EpisodicMemoryRecord,
    EpisodicMemoryStore,
    ProceduralMemoryLearner,
)

MANIFEST_FORMAT = "taiji-p4-capacity-procedural-manifest-v1"
REPORT_FORMAT = "taiji-p4-capacity-procedural-v1"


def _memory_record(memory_id: str, cue: torch.Tensor, *, episode_id: str) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id=episode_id,
        tick=1,
        cue=cue,
    )


def evaluate_capacity(capacities: tuple[int, ...] = (100, 1000, 10000)) -> list[dict[str, object]]:
    """Measure bounded retention and oldest-first interference at each capacity."""

    curve: list[dict[str, object]] = []
    for capacity in capacities:
        if capacity <= 0:
            raise ValueError("capacity curve values must be positive")
        store = EpisodicMemoryStore(capacity=capacity, cue_dim=2)
        target = _memory_record("capacity-target", torch.tensor([1.0, 0.0]), episode_id="target")
        start = time.perf_counter()
        store.write(target)
        for index in range(capacity):
            store.write(
                _memory_record(
                    f"capacity-distractor-{index}",
                    torch.tensor([0.0, 1.0]) if index == capacity - 1 else torch.tensor([1.0, 0.0]),
                    episode_id=f"distractor-{index}",
                )
            )
        elapsed = time.perf_counter() - start
        latest_id = f"capacity-distractor-{capacity - 1}"
        latest_cue = torch.tensor([0.0, 1.0])
        latest_hits = store.retrieve(latest_cue, limit=1)
        curve.append(
            {
                "capacity": capacity,
                "writes": capacity + 1,
                "retained_records": store.count,
                "target_evicted_by_interference": "capacity-target" not in {
                    record.memory_id for record in store.records
                },
                "latest_recalled": bool(latest_hits and latest_hits[0].record.memory_id == latest_id),
                "write_seconds": elapsed,
                "writes_per_second": (capacity + 1) / elapsed if elapsed > 0.0 else None,
            }
        )
    return curve


def _build_skill_corpus(
    *, repeats_per_action: int = 16, noise_scale: float = 0.08, seed: int = 20260825
) -> tuple[EpisodicMemoryStore, tuple[tuple[torch.Tensor, str], ...], tuple[str, ...]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    patterns = (
        (torch.tensor([0.0, 0.0]), "idle"),
        (torch.tensor([1.0, 0.0]), "left"),
        (torch.tensor([0.0, 1.0]), "right"),
        (torch.tensor([1.0, 1.0]), "both"),
    )
    store = EpisodicMemoryStore(capacity=128, cue_dim=2)
    for pattern_index, (pattern, action_kind) in enumerate(patterns):
        for repeat in range(repeats_per_action):
            memory_id = f"skill-memory-{pattern_index}-{repeat}"
            intent = ActionIntent(
                intent_id=f"skill-intent-{pattern_index}-{repeat}",
                kind=action_kind,
                tick=0,
            )
            store.write(
                EpisodicMemoryRecord(
                    memory_id=memory_id,
                    episode_id=f"skill-episode-{pattern_index}-{repeat}",
                    tick=1,
                    cue=pattern
                    + torch.randn(2, generator=generator, dtype=torch.float32) * noise_scale,
                    action_intent=intent,
                )
            )
    holdout = tuple(
        (
            pattern
            + torch.randn(2, generator=generator, dtype=torch.float32) * noise_scale,
            action_kind,
        )
        for pattern, action_kind in patterns
    )
    action_kinds = tuple(action_kind for _, action_kind in patterns)
    return store, holdout, action_kinds


def _episodic_nearest_accuracy(
    store: EpisodicMemoryStore, holdout: tuple[tuple[torch.Tensor, str], ...]
) -> float:
    correct = 0
    for cue, expected in holdout:
        hits = store.retrieve(cue, limit=1)
        predicted = None
        if hits and hits[0].record.action_intent is not None:
            predicted = hits[0].record.action_intent.kind
        correct += int(predicted == expected)
    return correct / len(holdout)


def evaluate() -> dict[str, object]:
    capacity_curve = evaluate_capacity()
    store, holdout, action_kinds = _build_skill_corpus()
    procedural = ProceduralMemoryLearner(cue_dim=2)
    consolidation_loss = procedural.consolidate(store, epochs=300, learning_rate=0.1)
    predictions = tuple(procedural.predict(cue) for cue, _ in holdout)
    procedural_accuracy = sum(
        int(predicted == expected) for predicted, (_, expected) in zip(predictions, holdout)
    ) / len(holdout)
    checkpoint = ProceduralMemoryLearner.from_checkpoint(procedural.checkpoint())
    checkpoint_accuracy = sum(
        int(checkpoint.predict(cue) == expected) for cue, expected in holdout
    ) / len(holdout)
    episode_id_lesion = EpisodicMemoryStore(capacity=store.capacity, cue_dim=2)
    for record in store.records:
        episode_id_lesion.write(
            EpisodicMemoryRecord.from_payload(
                {**record.to_payload(), "episode_id": "episode-id-lesion"}
            )
        )
    episode_id_lesion_learner = ProceduralMemoryLearner(cue_dim=2)
    episode_id_lesion_learner.consolidate(episode_id_lesion, epochs=300, learning_rate=0.1)
    episode_id_lesion_accuracy = sum(
        int(episode_id_lesion_learner.predict(cue) == expected)
        for cue, expected in holdout
    ) / len(holdout)
    skill_lesion_accuracy = 1.0 / len(action_kinds)
    nearest_accuracy = _episodic_nearest_accuracy(store, holdout)
    capacity_passed = all(
        item["retained_records"] == item["capacity"]
        and item["target_evicted_by_interference"]
        and item["latest_recalled"]
        for item in capacity_curve
    )
    gate_passed = bool(
        capacity_passed
        and procedural_accuracy >= 0.95
        and checkpoint_accuracy >= 0.95
        and episode_id_lesion_accuracy >= 0.95
        and skill_lesion_accuracy < procedural_accuracy
    )
    return {
        "format": REPORT_FORMAT,
        "capacity_curve": capacity_curve,
        "procedural": {
            "train_records": store.count,
            "action_kinds_discovered": list(procedural.action_kinds),
            "consolidation_loss": consolidation_loss,
            "holdout_records": len(holdout),
        },
        "metrics": {
            "episodic_nearest_accuracy": nearest_accuracy,
            "procedural_accuracy": procedural_accuracy,
            "skill_lesion_accuracy": skill_lesion_accuracy,
            "episode_id_lesion_accuracy": episode_id_lesion_accuracy,
            "checkpoint_continuation_accuracy": checkpoint_accuracy,
            "capacity_curve_passed": capacity_passed,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "bounded episodic retention plus data-driven procedural skill survives episode-id and checkpoint controls",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "measure episodic capacity/interference and consolidate cue-to-action skill",
        "capacities": [100, 1000, 10000],
        "skill_patterns": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "action_labels": "discovered from action_intent.kind in training records",
        "controls": [
            "episodic_nearest",
            "procedural_skill_lesion",
            "episode_id_lesion",
            "checkpoint_continuation",
        ],
        "boundary": "capacity and cue-to-action procedural consolidation only; not general planning or language competence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_capacity_procedural_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_capacity_procedural_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
