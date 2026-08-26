"""Evaluate procedural skill ownership inside the Taiji runtime adapter."""

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
    ProceduralMemoryLearner,
    TSKV8Adapter,
)

MANIFEST_FORMAT = "taiji-p4-procedural-runtime-manifest-v1"
REPORT_FORMAT = "taiji-p4-procedural-runtime-v1"
ACTION_SYMBOLS = (10, 11)
ACTION_KINDS = ("right", "left")
TRAINING_SYMBOLS = ((97, "left"), (98, "right"))


def _observe_feature(model: TSKV8Adapter, symbol: int, episode_id: str) -> torch.Tensor:
    model.reset_dynamics(episode_id=episode_id)
    model.observe(symbol, learn=False)
    percept = model.cognitive_snapshot().percept
    if percept is None:
        raise RuntimeError("runtime corpus observation did not produce a perception")
    return percept.features.detach().clone()


def _build_runtime_corpus(model: TSKV8Adapter, *, repeats: int = 8) -> EpisodicMemoryStore:
    store = EpisodicMemoryStore(
        capacity=len(TRAINING_SYMBOLS) * repeats + 16,
        cue_dim=model.perception.feature_dim,
    )
    index = 0
    for repeat in range(repeats):
        for symbol, action_kind in TRAINING_SYMBOLS:
            cue = _observe_feature(model, symbol, f"procedural-train-{repeat}-{symbol}")
            intent = ActionIntent(
                intent_id=f"procedural-train-intent-{index}",
                kind=action_kind,
                tick=model.tick,
            )
            store.write(
                EpisodicMemoryRecord(
                    memory_id=f"procedural-train-memory-{index}",
                    episode_id=f"procedural-train-episode-{repeat}-{symbol}",
                    tick=model.tick,
                    cue=cue,
                    action_intent=intent,
                )
            )
            index += 1
    return store


def _run_runtime_trials(
    model: TSKV8Adapter,
    trials: tuple[tuple[int, str], ...],
    *,
    use_procedural: bool,
    trial_prefix: str,
) -> tuple[float, int]:
    correct = 0
    changed = 0
    for index, (symbol, expected_kind) in enumerate(trials):
        model.reset_dynamics(episode_id=f"{trial_prefix}-{index}")
        model.observe(symbol, learn=False)
        decision = model.act(
            ACTION_SYMBOLS,
            sample=False,
            procedural_action_kinds=ACTION_KINDS,
            use_procedural=use_procedural,
        )
        expected_symbol = ACTION_SYMBOLS[ACTION_KINDS.index(expected_kind)]
        correct += int(decision.action_symbol == expected_symbol)
        changed += int(decision.action_symbol == expected_symbol) if use_procedural else 0
        model.settle_action(1.0, learn=False)
        model.observe(symbol, learn=False)
    return correct / len(trials), changed


def evaluate() -> dict[str, object]:
    model = TSKV8Adapter()
    store = _build_runtime_corpus(model)
    learner = ProceduralMemoryLearner(model.perception.feature_dim)
    model.attach_episodic_memory(store)
    model.attach_procedural_memory(learner)
    consolidation_loss = model.consolidate_procedural_memory(epochs=300, learning_rate=0.1)
    trials = tuple(TRAINING_SYMBOLS)
    procedural_accuracy, selected_count = _run_runtime_trials(
        model, trials, use_procedural=True, trial_prefix="procedural-runtime"
    )
    checkpoint = model.native_checkpoint()
    checkpoint_model = TSKV8Adapter.from_native_checkpoint(checkpoint)
    checkpoint_accuracy, _ = _run_runtime_trials(
        checkpoint_model,
        trials,
        use_procedural=True,
        trial_prefix="procedural-checkpoint",
    )
    lesion_model = TSKV8Adapter.from_native_checkpoint(checkpoint)
    lesion_model.attach_procedural_memory(None)
    runtime_lesion_accuracy, _ = _run_runtime_trials(
        lesion_model, trials, use_procedural=False, trial_prefix="procedural-lesion"
    )

    episode_id_lesion_store = EpisodicMemoryStore(
        capacity=store.capacity, cue_dim=model.perception.feature_dim
    )
    for record in store.records:
        episode_id_lesion_store.write(
            EpisodicMemoryRecord.from_payload(
                {**record.to_payload(), "episode_id": "procedural-episode-id-lesion"}
            )
        )
    episode_id_lesion_learner = ProceduralMemoryLearner(model.perception.feature_dim)
    episode_id_lesion_learner.consolidate(episode_id_lesion_store, epochs=300, learning_rate=0.1)
    episode_id_model = TSKV8Adapter.from_native_checkpoint(checkpoint)
    episode_id_model.attach_episodic_memory(episode_id_lesion_store)
    episode_id_model.attach_procedural_memory(episode_id_lesion_learner)
    episode_id_accuracy, _ = _run_runtime_trials(
        episode_id_model,
        trials,
        use_procedural=True,
        trial_prefix="procedural-episode-id-lesion",
    )
    gate_passed = bool(
        procedural_accuracy == 1.0
        and checkpoint_accuracy == 1.0
        and episode_id_accuracy == 1.0
        and procedural_accuracy > runtime_lesion_accuracy
        and selected_count == len(trials)
    )
    return {
        "format": REPORT_FORMAT,
        "train_records": store.count,
        "action_kinds": list(learner.action_kinds),
        "consolidation_loss": consolidation_loss,
        "metrics": {
            "procedural_runtime_accuracy": procedural_accuracy,
            "runtime_lesion_accuracy": runtime_lesion_accuracy,
            "episode_id_lesion_accuracy": episode_id_accuracy,
            "checkpoint_continuation_accuracy": checkpoint_accuracy,
            "procedural_selections": selected_count,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "procedural learner controls adapter action selection and survives runtime, episode-id, and checkpoint lesions",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "route real adapter action selection from a replayed cue-to-action skill",
        "training_inputs": [[97, "left"], [98, "right"]],
        "available_actions": list(ACTION_SYMBOLS),
        "action_kinds": list(ACTION_KINDS),
        "controls": ["runtime_lesion", "episode_id_lesion", "checkpoint_continuation"],
        "boundary": "explicit action-kind routing only; not general planning or motor competence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_procedural_runtime_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_procedural_runtime_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
