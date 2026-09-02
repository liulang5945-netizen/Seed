"""Diagnose whether replay order preserves cue/action event binding."""

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

FORMAT = "taiji-native-m1-38-event-binding-curriculum-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_38_event_binding_curriculum_20260902.json"
SEEDS = (11, 29, 47)
SCHEDULES = ("posthoc_experienced", "interleaved_experienced")
REPLAY_PROVENANCE = "experienced"
REPLAY_SCALE = 1.0
REPLAY_TARGETS = "all"


def _actions(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.action) for episode in episodes))


def _probe(
    model: Taiji,
    queries: tuple[Any, ...],
    actions: tuple[int, ...],
) -> float:
    correct = 0
    for query in queries:
        model.reset_dynamics(episode_id=f"m1-38-probe-{query.query_id}")
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
    return correct / len(queries)


def _event_payload(episode: Any) -> list[object]:
    return [episode.memory_id, int(episode.cue), int(episode.action), int(episode.outcome)]


def _schedule_events(corpus: Any, schedule: str) -> tuple[tuple[Any, str, float, str], ...]:
    if schedule not in SCHEDULES:
        raise ValueError(f"unsupported event-binding schedule: {schedule}")
    events: list[tuple[Any, str, float, str]] = []
    if schedule == "posthoc_experienced":
        events.extend(
            (episode, "experienced", 1.0, "all")
            for episode in corpus.phase_b_train
        )
        events.extend(
            (episode, REPLAY_PROVENANCE, REPLAY_SCALE, REPLAY_TARGETS)
            for episode in corpus.replay_train
        )
        return tuple(events)
    if len(corpus.phase_b_train) != len(corpus.replay_train):
        raise ValueError("interleaved schedule requires equal phase-B and replay counts")
    for phase_b_episode, replay_episode in zip(
        corpus.phase_b_train,
        corpus.replay_train,
        strict=True,
    ):
        events.append((phase_b_episode, "experienced", 1.0, "all"))
        events.append((replay_episode, REPLAY_PROVENANCE, REPLAY_SCALE, REPLAY_TARGETS))
    return tuple(events)


def _event_set_digest(corpus: Any) -> str:
    return content_digest(
        {
            "phase_b": [_event_payload(item) for item in corpus.phase_b_train],
            "replay": [_event_payload(item) for item in corpus.replay_train],
            "replay_provenance": REPLAY_PROVENANCE,
            "replay_scale": REPLAY_SCALE,
            "replay_targets": REPLAY_TARGETS,
        }
    )


def _event_order_digest(events: tuple[tuple[Any, str, float, str], ...]) -> str:
    return content_digest(
        [
            [_event_payload(episode), provenance, scale, targets]
            for episode, provenance, scale, targets in events
        ]
    )


def _write(
    model: Taiji,
    episode: Any,
    *,
    provenance: str,
    memory_learning_scale: float,
    memory_learning_targets: str,
) -> None:
    DelayedMemoryTask._write_episode(
        model,
        episode,
        provenance=provenance,
        memory_learning_scale=memory_learning_scale,
        memory_learning_targets=memory_learning_targets,
    )


def _condition_record(
    schedule: str,
    corpus: Any,
    event_set_digest: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    events = _schedule_events(corpus, schedule)
    actions = _actions(*corpus.phase_a_train, *corpus.phase_b_train)
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        phase_a = Taiji(_config(seed), episode_id=f"m1-38-phase-a-{seed}")
        for episode in corpus.phase_a_train:
            _write(
                phase_a,
                episode,
                provenance="experienced",
                memory_learning_scale=1.0,
                memory_learning_targets="all",
            )
        old_before = _probe(phase_a, corpus.phase_a_holdout, actions)
        old_retention_before = _probe(phase_a, corpus.phase_a_retention, actions)
        phase_a_checkpoint = deepcopy(phase_a.checkpoint())
        phase_a_digest = content_digest(phase_a_checkpoint)

        no_replay = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
        for episode in corpus.phase_b_train:
            _write(
                no_replay,
                episode,
                provenance="experienced",
                memory_learning_scale=1.0,
                memory_learning_targets="all",
            )
        no_replay_old = _probe(no_replay, corpus.phase_a_holdout, actions)
        no_replay_new = _probe(no_replay, corpus.phase_b_holdout, actions)

        model = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
        for episode, provenance, scale, targets in events:
            _write(
                model,
                episode,
                provenance=provenance,
                memory_learning_scale=scale,
                memory_learning_targets=targets,
            )
        schedule_old = _probe(model, corpus.phase_a_holdout, actions)
        schedule_retention = _probe(model, corpus.phase_a_retention, actions)
        schedule_new = _probe(model, corpus.phase_b_holdout, actions)
        checkpoint = deepcopy(model.checkpoint())
        schedule_digest = content_digest(checkpoint)
        restored = Taiji.from_checkpoint(deepcopy(checkpoint))
        restored_digest = content_digest(restored.checkpoint())
        records.append(
            {
                "seed": seed,
                "schedule": schedule,
                "event_count": len(events),
                "old_before": old_before,
                "old_retention_before": old_retention_before,
                "no_replay_old_after": no_replay_old,
                "no_replay_new_after": no_replay_new,
                "schedule_old_after": schedule_old,
                "schedule_retention_after": schedule_retention,
                "schedule_new_after": schedule_new,
                "schedule_backward_transfer": schedule_old - old_before,
                "schedule_causal_gain": schedule_old - no_replay_old,
                "schedule_new_delta_vs_no_replay": schedule_new - no_replay_new,
                "phase_a_checkpoint_digest": phase_a_digest,
                "schedule_checkpoint_digest": schedule_digest,
                "continued_from_phase_a": phase_a_digest != schedule_digest,
                "checkpoint_roundtrip_exact": restored_digest == schedule_digest,
                "holdout_updates": 0,
            }
        )
    return {
        "schedule": schedule,
        "event_set_digest": event_set_digest,
        "event_order_digest": _event_order_digest(events),
        "event_count": len(events),
        "records": records,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def _condition_passed(condition: dict[str, Any]) -> bool:
    return all(
        record["continued_from_phase_a"]
        and record["checkpoint_roundtrip_exact"]
        and record["holdout_updates"] == 0
        and record["schedule_backward_transfer"] >= 0.0
        and record["schedule_retention_after"] >= record["old_retention_before"]
        and record["schedule_new_after"] + 0.05 >= record["no_replay_new_after"]
        and record["schedule_causal_gain"] > 0.0
        for record in condition["records"]
    )


def run_diagnosis() -> dict[str, Any]:
    corpus = _curriculum(phase_a_start=0, phase_b_start=192)
    event_set_digest = _event_set_digest(corpus)
    conditions = [
        _condition_record(schedule, corpus, event_set_digest)
        for schedule in SCHEDULES
    ]
    for condition in conditions:
        condition["condition_gate_passed"] = _condition_passed(condition)
    interleaved = conditions[1]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if interleaved["condition_gate_passed"] else "failed",
        "variable_changed": "cue/action event order only",
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
            "interleaved_condition_passed": interleaved["condition_gate_passed"],
            "event_order_is_sufficient_explanation": (
                interleaved["condition_gate_passed"]
                and not conditions[0]["condition_gate_passed"]
            ),
            "next_boundary": (
                "interleaved event order passed; hold for M1-39 explicit event-binding"
                if interleaved["condition_gate_passed"]
                else "event order did not pass B5; freeze data-order explanation and diagnose explicit cue/action binding"
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
