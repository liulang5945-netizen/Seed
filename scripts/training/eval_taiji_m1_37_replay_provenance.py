"""Diagnose whether replay provenance corrupts shared episodic readout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_36_cue_curriculum import (  # noqa: E402
    _config,
    _curriculum,
)
from taiji import DelayedMemoryTask, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-37-replay-provenance-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_37_replay_provenance_20260902.json"
SEEDS = (11, 29, 47)
PROVENANCES = ("replayed", "experienced")


def _actions(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.action) for episode in episodes))


def _probe(
    model: Taiji,
    queries: tuple[Any, ...],
    actions: tuple[int, ...],
) -> tuple[float, float]:
    correct = 0
    provenance_values: list[float] = []
    for query in queries:
        model.reset_dynamics(episode_id=f"m1-37-probe-{query.query_id}")
        model.observe(
            model.config.boundary_symbol,
            learn=False,
            learn_motor=False,
            use_memory=True,
        )
        step = model.observe(
            query.cue,
            learn=False,
            learn_motor=False,
            use_memory=True,
        )
        probabilities = step.probabilities
        prediction = max(
            actions,
            key=lambda action: float(probabilities[action].item()),
        )
        correct += int(prediction == query.expected_action)
        provenance_values.append(
            float(step.memory_recall.provenance_probabilities[2].item())
        )
    return correct / len(queries), sum(provenance_values) / len(provenance_values)


def _condition_record(provenance: str, corpus: Any) -> dict[str, Any]:
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        config = _config(seed)
        actions = _actions(*corpus.phase_a_train, *corpus.phase_b_train)
        phase_a = Taiji(config, episode_id=f"m1-37-phase-a-{provenance}-{seed}")
        for episode in corpus.phase_a_train:
            DelayedMemoryTask._write_episode(phase_a, episode)
        old_before, old_retention_before = _probe(
            phase_a,
            corpus.phase_a_holdout,
            actions,
        )[0], _probe(phase_a, corpus.phase_a_retention, actions)[0]
        phase_a_checkpoint = deepcopy(phase_a.checkpoint())
        phase_a_digest = content_digest(phase_a_checkpoint)

        no_replay = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
        for episode in corpus.phase_b_train:
            DelayedMemoryTask._write_episode(no_replay, episode)
        no_replay_old, _ = _probe(no_replay, corpus.phase_a_holdout, actions)
        no_replay_new, _ = _probe(no_replay, corpus.phase_b_holdout, actions)

        replay = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
        for episode in corpus.phase_b_train:
            DelayedMemoryTask._write_episode(replay, episode)
        for episode in corpus.phase_a_train:
            DelayedMemoryTask._write_episode(
                replay,
                episode,
                provenance=provenance,
                memory_learning_scale=1.0,
                memory_learning_targets="all",
            )
        replay_old, replay_old_provenance = _probe(
            replay,
            corpus.phase_a_holdout,
            actions,
        )
        replay_retention, replay_retention_provenance = _probe(
            replay,
            corpus.phase_a_retention,
            actions,
        )
        replay_new, replay_new_provenance = _probe(
            replay,
            corpus.phase_b_holdout,
            actions,
        )
        replay_checkpoint = deepcopy(replay.checkpoint())
        replay_digest = content_digest(replay_checkpoint)
        restored = Taiji.from_checkpoint(deepcopy(replay_checkpoint))
        restored_digest = content_digest(restored.checkpoint())
        records.append(
            {
                "seed": seed,
                "provenance": provenance,
                "old_before": old_before,
                "old_retention_before": old_retention_before,
                "no_replay_old_after": no_replay_old,
                "no_replay_new_after": no_replay_new,
                "replay_old_after": replay_old,
                "replay_retention_after": replay_retention,
                "replay_new_after": replay_new,
                "replay_backward_transfer": replay_old - old_before,
                "replay_causal_gain": replay_old - no_replay_old,
                "replay_new_delta_vs_no_replay": replay_new - no_replay_new,
                "replay_provenance_readout": {
                    "phase_a_holdout": replay_old_provenance,
                    "phase_a_retention": replay_retention_provenance,
                    "phase_b_holdout": replay_new_provenance,
                },
                "phase_a_checkpoint_digest": phase_a_digest,
                "replay_checkpoint_digest": replay_digest,
                "continued_from_phase_a": phase_a_digest != replay_digest,
                "checkpoint_roundtrip_exact": restored_digest == replay_digest,
                "holdout_updates": 0,
            }
        )
    return {
        "provenance": provenance,
        "records": records,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def _condition_passed(condition: dict[str, Any]) -> bool:
    return all(
        record["continued_from_phase_a"]
        and record["checkpoint_roundtrip_exact"]
        and record["holdout_updates"] == 0
        and record["replay_backward_transfer"] >= 0.0
        and record["replay_retention_after"] >= record["old_retention_before"]
        and record["replay_new_after"] + 0.05 >= record["no_replay_new_after"]
        and record["replay_causal_gain"] > 0.0
        for record in condition["records"]
    )


def run_diagnosis() -> dict[str, Any]:
    corpus = _curriculum(phase_a_start=0, phase_b_start=192)
    conditions = [_condition_record(provenance, corpus) for provenance in PROVENANCES]
    for condition in conditions:
        condition["condition_gate_passed"] = _condition_passed(condition)
    experienced = conditions[1]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if experienced["condition_gate_passed"] else "failed",
        "variable_changed": "replay provenance only",
        "cue_curriculum": "maximally separated byte cues",
        "action_curriculum": "phase-A/phase-B overlap on 48/49",
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "replay_scale": 1.0,
        "corpus_digest": corpus.digest,
        "conditions": conditions,
        "conclusion": {
            "experienced_provenance_passed": experienced["condition_gate_passed"],
            "provenance_is_sufficient_explanation": (
                experienced["condition_gate_passed"]
                and not conditions[0]["condition_gate_passed"]
            ),
            "next_boundary": (
                "preserving experienced provenance did not pass B5; freeze provenance"
                " as the sole explanation and diagnose cue/action event binding"
                if not experienced["condition_gate_passed"]
                else "experienced provenance passed; promote the rule to a held-out review"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_diagnosis()
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
