"""Diagnose whether association and action readout should be written in stages."""

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

FORMAT = "taiji-native-m1-41-staged-write-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_41_staged_write_20260902.json"
SEEDS = (11, 29, 47)
SCHEDULES = ("all", "staged_association_then_readout")
REPLAY_PROVENANCE = "experienced"
REPLAY_SCALE = 1.0
REPLAY_TARGETS = "all"


def _actions(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.action) for episode in episodes))


def _probe(model: Taiji, queries: tuple[Any, ...], actions: tuple[int, ...]) -> float:
    correct = 0
    for query in queries:
        model.reset_dynamics(episode_id=f"m1-41-probe-{query.query_id}")
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
        prediction = max(
            actions,
            key=lambda action: float(step.probabilities[action].item()),
        )
        correct += int(prediction == query.expected_action)
    return correct / len(queries)


def _write(
    model: Taiji,
    episode: Any,
    *,
    targets: str,
) -> None:
    DelayedMemoryTask._write_episode(
        model,
        episode,
        provenance=REPLAY_PROVENANCE,
        memory_learning_scale=REPLAY_SCALE,
        memory_learning_targets=targets,
    )


def _checkpoint_digest(model: Taiji) -> str:
    return content_digest(deepcopy(model.checkpoint()))


def _condition_record(
    schedule: str,
    corpus: Any,
    event_set_digest: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    actions = _actions(*corpus.phase_a_train, *corpus.phase_b_train)
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        model = Taiji(_config(seed), episode_id=f"m1-41-{schedule}-{seed}")
        for episode in corpus.phase_a_train:
            _write(model, episode, targets="all")
        old_before = _probe(model, corpus.phase_a_holdout, actions)
        old_retention_before = _probe(model, corpus.phase_a_retention, actions)
        phase_a_digest = _checkpoint_digest(model)
        phase_a_write_count = int(model.memory.write_count)

        no_replay = Taiji.from_checkpoint(deepcopy(model.checkpoint()))
        for episode in corpus.phase_b_train:
            _write(no_replay, episode, targets="all")
        no_replay_old = _probe(no_replay, corpus.phase_a_holdout, actions)
        no_replay_new = _probe(no_replay, corpus.phase_b_holdout, actions)

        replay = Taiji.from_checkpoint(deepcopy(model.checkpoint()))
        for episode in corpus.phase_b_train:
            _write(replay, episode, targets="all")
        phase_b_digest = _checkpoint_digest(replay)
        phase_b_write_count = int(replay.memory.write_count)
        association_stage_digest: str | None = None
        readout_stage_digest: str | None = None
        if schedule == "all":
            for episode in corpus.replay_train:
                _write(replay, episode, targets=REPLAY_TARGETS)
        else:
            for episode in corpus.replay_train:
                _write(replay, episode, targets="association")
            association_stage_digest = _checkpoint_digest(replay)
            for episode in corpus.replay_train:
                _write(replay, episode, targets="readout")
            readout_stage_digest = _checkpoint_digest(replay)

        replay_old = _probe(replay, corpus.phase_a_holdout, actions)
        replay_retention = _probe(replay, corpus.phase_a_retention, actions)
        replay_new = _probe(replay, corpus.phase_b_holdout, actions)
        replay_digest = _checkpoint_digest(replay)
        restored = Taiji.from_checkpoint(deepcopy(replay.checkpoint()))
        restored_digest = _checkpoint_digest(restored)
        records.append(
            {
                "seed": seed,
                "schedule": schedule,
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
                "phase_a_write_count": phase_a_write_count,
                "phase_b_write_count": phase_b_write_count,
                "final_write_count": int(replay.memory.write_count),
                "physical_replay_write_count": (
                    int(replay.memory.write_count) - phase_b_write_count
                ),
                "phase_a_checkpoint_digest": phase_a_digest,
                "phase_b_checkpoint_digest": phase_b_digest,
                "association_stage_checkpoint_digest": association_stage_digest,
                "readout_stage_checkpoint_digest": readout_stage_digest,
                "replay_checkpoint_digest": replay_digest,
                "continued_from_phase_a": phase_a_digest != replay_digest,
                "checkpoint_roundtrip_exact": restored_digest == replay_digest,
                "active_parameter_count": replay.parameter_count(),
                "planned_active_parameter_count": replay.config.planned_active_parameter_count,
                "parameter_count_matches_plan": (
                    replay.parameter_count() == replay.config.planned_active_parameter_count
                ),
                "holdout_updates": 0,
            }
        )
    return {
        "schedule": schedule,
        "event_set_digest": event_set_digest,
        "records": records,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def _condition_passed(condition: dict[str, Any]) -> bool:
    return all(
        record["continued_from_phase_a"]
        and record["checkpoint_roundtrip_exact"]
        and record["parameter_count_matches_plan"]
        and record["holdout_updates"] == 0
        and record["replay_backward_transfer"] >= 0.0
        and record["replay_retention_after"] >= record["old_retention_before"]
        and record["replay_new_after"] + 0.05 >= record["no_replay_new_after"]
        and record["replay_causal_gain"] > 0.0
        for record in condition["records"]
    )


def run_diagnosis() -> dict[str, Any]:
    corpus = _curriculum(phase_a_start=0, phase_b_start=192)
    event_set_digest = content_digest(
        {
            "phase_a_train": [
                [item.memory_id, item.cue, item.action, item.outcome]
                for item in corpus.phase_a_train
            ],
            "phase_b_train": [
                [item.memory_id, item.cue, item.action, item.outcome]
                for item in corpus.phase_b_train
            ],
            "replay_train": [
                [item.memory_id, item.cue, item.action, item.outcome]
                for item in corpus.replay_train
            ],
            "provenance": REPLAY_PROVENANCE,
            "scale": REPLAY_SCALE,
            "targets": REPLAY_TARGETS,
        }
    )
    conditions = [
        _condition_record(schedule, corpus, event_set_digest)
        for schedule in SCHEDULES
    ]
    for condition in conditions:
        condition["condition_gate_passed"] = _condition_passed(condition)
    staged = conditions[1]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if staged["condition_gate_passed"] else "failed",
        "variable_changed": "association/readout write staging only",
        "baseline_schedule": "all",
        "candidate_schedule": "staged_association_then_readout",
        "cue_curriculum": "maximally separated byte cues",
        "action_curriculum": "phase-A/phase-B overlap on 48/49",
        "replay_provenance": REPLAY_PROVENANCE,
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "replay_scale": REPLAY_SCALE,
        "replay_targets": REPLAY_TARGETS,
        "corpus_digest": corpus.digest,
        "event_set_digest": event_set_digest,
        "conditions": conditions,
        "conclusion": {
            "staged_condition_passed": staged["condition_gate_passed"],
            "staged_write_is_sufficient_explanation": (
                staged["condition_gate_passed"]
                and not conditions[0]["condition_gate_passed"]
            ),
            "next_boundary": (
                "staged write passed; design an atomic write API review"
                if staged["condition_gate_passed"]
                else "staged write did not pass B5; freeze write-target timing and diagnose event encoding/association geometry"
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
