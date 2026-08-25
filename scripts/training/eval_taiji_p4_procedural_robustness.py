"""Evaluate multi-step procedural transfer, interference, and bounded forgetting."""

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
    ProceduralSequenceLearner,
)

MANIFEST_FORMAT = "taiji-p4-procedural-robustness-manifest-v1"
REPORT_FORMAT = "taiji-p4-procedural-robustness-v1"
TRAINING_PAIRS = ((0.0, 1.0, "advance"), (1.0, 0.0, "retreat"), (1.0, 2.0, "advance"),
                  (2.0, 1.0, "retreat"), (0.0, 0.0, "hold"), (1.0, 1.0, "hold"),
                  (2.0, 2.0, "hold"))
HOLDOUT_PAIRS = ((0.0, 2.0, "advance"), (2.0, 0.0, "retreat"))


def _record(
    episode_id: str, tick: int, cue: float, action_kind: str, index: int
) -> EpisodicMemoryRecord:
    intent = ActionIntent(
        intent_id=f"{episode_id}:intent:{index}",
        kind=action_kind,
        tick=tick,
    )
    return EpisodicMemoryRecord(
        memory_id=f"{episode_id}:memory:{index}",
        episode_id=episode_id,
        tick=tick,
        cue=torch.tensor([cue], dtype=torch.float32),
        action_intent=intent,
    )


def _build_store(*, capacity: int = 256, repeats: int = 12) -> EpisodicMemoryStore:
    store = EpisodicMemoryStore(capacity=capacity, cue_dim=1)
    index = 0
    for repeat in range(repeats):
        for left, right, transition_kind in TRAINING_PAIRS:
            episode_id = f"skill-{repeat}-{index}"
            store.write(_record(episode_id, 0, left, "prepare", index))
            store.write(_record(episode_id, 1, right, transition_kind, index + 1))
            index += 2
    return store


def _add_similar_interference(store: EpisodicMemoryStore, *, count: int = 24) -> None:
    for index in range(count):
        episode_id = f"interference-{index}"
        left = 0.05 + (index % 3) * 0.9
        right = 1.05 + (index % 3) * 0.9
        store.write(_record(episode_id, 0, left, "prepare", 0))
        store.write(_record(episode_id, 1, right, "inspect", 1))


def _accuracy(
    learner: ProceduralSequenceLearner,
    pairs: tuple[tuple[float, float, str], ...],
) -> float:
    correct_steps = 0
    total_steps = 0
    for left, right, transition_kind in pairs:
        predicted = learner.predict_episode(
            (torch.tensor([left]), torch.tensor([right]))
        )
        expected = ("prepare", transition_kind)
        correct_steps += sum(int(actual == target) for actual, target in zip(predicted, expected))
        total_steps += len(expected)
    return correct_steps / total_steps


def evaluate() -> dict[str, object]:
    store = _build_store()
    learner = ProceduralSequenceLearner(cue_dim=1, hidden_dim=16, seed=20260825)
    baseline_loss = learner.consolidate(store, epochs=250, learning_rate=0.03)
    baseline_accuracy = _accuracy(learner, HOLDOUT_PAIRS)
    checkpoint = ProceduralSequenceLearner.from_checkpoint(learner.checkpoint())
    checkpoint_accuracy = _accuracy(checkpoint, HOLDOUT_PAIRS)

    interference_store = EpisodicMemoryStore.from_checkpoint(store.checkpoint())
    _add_similar_interference(interference_store)
    interference_learner = ProceduralSequenceLearner(cue_dim=1, hidden_dim=16, seed=20260825)
    interference_loss = interference_learner.consolidate(
        interference_store, epochs=250, learning_rate=0.03
    )
    interference_accuracy = _accuracy(interference_learner, HOLDOUT_PAIRS)

    budget_store = EpisodicMemoryStore(capacity=store.count, cue_dim=1)
    for record in store.records:
        budget_store.write(record)
    _add_similar_interference(budget_store, count=store.count // 2 + 1)
    budget_learner = ProceduralSequenceLearner(cue_dim=1, hidden_dim=16, seed=20260825)
    budget_learner.consolidate(budget_store, epochs=250, learning_rate=0.03)
    budget_accuracy = _accuracy(budget_learner, HOLDOUT_PAIRS)
    gate_passed = bool(
        baseline_accuracy == 1.0
        and checkpoint_accuracy == 1.0
        and interference_accuracy >= 0.75
        and budget_accuracy < baseline_accuracy
        and interference_store.count == store.count + 48
        and budget_store.count == store.count
    )
    return {
        "format": REPORT_FORMAT,
        "train_records": store.count,
        "holdout_episodes": len(HOLDOUT_PAIRS),
        "action_kinds": list(learner.action_kinds),
        "metrics": {
            "baseline_loss": baseline_loss,
            "baseline_transfer_accuracy": baseline_accuracy,
            "similar_interference_loss": interference_loss,
            "similar_interference_accuracy": interference_accuracy,
            "checkpoint_continuation_accuracy": checkpoint_accuracy,
            "budgeted_store_records": budget_store.count,
            "budgeted_accuracy_after_forgetting": budget_accuracy,
            "interference_store_records": interference_store.count,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "ordered multi-step skill transfers to unseen transitions, survives similar interference, and shows bounded-memory forgetting",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "learn prepare plus transition actions and transfer to held-out state transitions",
        "training_transitions": [list(item) for item in TRAINING_PAIRS],
        "heldout_transitions": [list(item) for item in HOLDOUT_PAIRS],
        "interference": "nearby cues with a data-derived inspect action",
        "resource_budget": "bounded episodic capacity equal to the original training corpus",
        "controls": ["checkpoint_continuation", "similar_interference", "bounded_forgetting"],
        "boundary": "multi-step procedural transfer only; episode identity remains sequence structure, not answer content",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_procedural_robustness_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_procedural_robustness_baseline_20260825.json",
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
