"""Evaluate interleaved phase-A rehearsal as a native memory data course."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus  # noqa: E402
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import (  # noqa: E402
    ContinualMemoryCorpus,
    DelayedMemoryTask,
    Taiji,
    TaijiConfig,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-interleaved-rehearsal-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_interleaved_rehearsal_20260902.json"
SCHEDULES = ("no_replay", "posthoc", "interleave_every_1", "interleave_every_4")


def _persistent_digest(model: Taiji) -> str:
    checkpoint = model.checkpoint()
    return content_digest(
        {
            "fabric": checkpoint["fabric"],
            "motor": checkpoint["motor"],
            "memory": checkpoint["memory"],
        }
    )


def _write(
    model: Taiji,
    episode: Any,
    *,
    provenance: str = "experienced",
    memory_learning_scale: float = 1.0,
    memory_learning_targets: str = "all",
) -> None:
    DelayedMemoryTask._write_episode(
        model,
        episode,
        provenance=provenance,
        memory_learning_scale=memory_learning_scale,
        memory_learning_targets=memory_learning_targets,
    )


def _train_schedule(
    model: Taiji,
    corpus: ContinualMemoryCorpus,
    schedule: str,
) -> int:
    replay_count = 0
    replay_cursor = 0
    for index, episode in enumerate(corpus.phase_b_train):
        _write(model, episode)
        if schedule == "interleave_every_1":
            _write(
                model,
                corpus.replay_train[replay_cursor],
                provenance="replayed",
                memory_learning_scale=model.config.replay_memory_learning_scale,
            )
            replay_cursor = (replay_cursor + 1) % len(corpus.replay_train)
            replay_count += 1
        elif schedule == "interleave_every_4" and (index + 1) % 4 == 0:
            for _ in range(4):
                _write(
                    model,
                    corpus.replay_train[replay_cursor],
                    provenance="replayed",
                    memory_learning_scale=model.config.replay_memory_learning_scale,
                )
                replay_cursor = (replay_cursor + 1) % len(corpus.replay_train)
                replay_count += 1
    if schedule == "posthoc":
        for episode in corpus.replay_train:
            _write(
                model,
                episode,
                provenance="replayed",
                memory_learning_scale=model.config.replay_memory_learning_scale,
            )
            replay_count += 1
    return replay_count


def _scores(
    model: Taiji,
    corpus: ContinualMemoryCorpus,
    actions: tuple[int, ...],
) -> dict[str, float]:
    return {
        "old_holdout": DelayedMemoryTask._recall_accuracy(
            model, corpus.phase_a_holdout, actions, use_memory=True
        ),
        "old_retention": DelayedMemoryTask._recall_accuracy(
            model, corpus.phase_a_retention, actions, use_memory=True
        ),
        "new_holdout": DelayedMemoryTask._recall_accuracy(
            model, corpus.phase_b_holdout, actions, use_memory=True
        ),
    }


def _seed_record(
    seed: int,
    corpus: ContinualMemoryCorpus,
    schedule: str,
) -> dict[str, object]:
    config_values = _memory_config(seed).to_dict()
    config_values.update(
        {
            "memory_action_decoder": "shared",
            "memory_confidence_decay": 0.0,
            "replay_memory_learning_scale": 0.25,
        }
    )
    config = TaijiConfig.from_dict(config_values)
    actions = tuple(
        dict.fromkeys(
            episode.action for episode in (*corpus.phase_a_train, *corpus.phase_b_train)
        )
    )
    phase_a = Taiji(
        _memory_config(seed),
        episode_id=f"m1-20-phase-a-{seed}",
    )
    for episode in corpus.phase_a_train:
        _write(phase_a, episode)
    parent_scores = _scores(phase_a, corpus, actions)
    phase_a_payload = deepcopy(phase_a.checkpoint())
    phase_a_digest = content_digest(phase_a_payload)

    model = Taiji(config, episode_id=f"m1-20-{schedule}-{seed}")
    model.restore(deepcopy(phase_a_payload))
    replay_count = _train_schedule(model, corpus, schedule)
    child_scores = _scores(model, corpus, actions)
    checkpoint_payload = model.checkpoint()
    checkpoint_digest = content_digest(checkpoint_payload)
    restored = Taiji(config, episode_id=f"m1-20-restored-{schedule}-{seed}")
    restored.restore(deepcopy(checkpoint_payload))
    restored_scores = _scores(restored, corpus, actions)
    persistent_before = _persistent_digest(restored)
    restored_scores = _scores(restored, corpus, actions)
    persistent_after = _persistent_digest(restored)
    return {
        "seed": seed,
        "schedule": schedule,
        "replay_count": replay_count,
        "parent": parent_scores,
        "child": child_scores,
        "restored": restored_scores,
        "replay_causal_gain_vs_parent": child_scores["old_holdout"]
        - parent_scores["old_holdout"],
        "checkpoint": {
            "parent_digest": phase_a_digest,
            "child_digest": checkpoint_digest,
            "restore_digest_matches": content_digest(restored.checkpoint()) == checkpoint_digest,
            "read_only_persistent_state": persistent_before == persistent_after,
        },
        "holdout_updates": 0,
    }


def _promotable(record: dict[str, object]) -> bool:
    parent = record["parent"]
    child = record["child"]
    restored = record["restored"]
    checkpoint = record["checkpoint"]
    no_replay_new_holdout = float(record["no_replay_new_holdout"])
    return bool(
        child["old_holdout"] >= parent["old_holdout"]
        and child["old_retention"] >= parent["old_retention"]
        and child["new_holdout"] >= no_replay_new_holdout
        and restored == child
        and checkpoint["restore_digest_matches"]
        and checkpoint["read_only_persistent_state"]
        and record["holdout_updates"] == 0
    )


def run_interleaved_diagnostics(
    *,
    train_count: int,
    holdout_count: int,
    retention_count: int,
    seeds: tuple[int, ...],
    schedules: tuple[str, ...] = SCHEDULES,
) -> dict[str, object]:
    corpus = build_corpus(
        train_count=train_count,
        holdout_count=holdout_count,
        retention_count=retention_count,
    )
    unknown = set(schedules) - set(SCHEDULES)
    if unknown:
        raise ValueError(f"unsupported rehearsal schedule: {sorted(unknown)}")
    records = {
        schedule: [
            _seed_record(seed, corpus, schedule)
            for seed in seeds
        ]
        for schedule in schedules
    }
    no_replay_by_seed = {
        int(record["seed"]): float(record["child"]["new_holdout"])
        for record in records.get("no_replay", ())
    }
    for schedule_records in records.values():
        for record in schedule_records:
            record["no_replay_new_holdout"] = no_replay_by_seed[int(record["seed"])]
    promotable = {
        schedule: all(_promotable(record) for record in values)
        for schedule, values in records.items()
    }
    return {
        "corpus_digest": corpus.digest,
        "sample_counts": corpus.sample_counts,
        "schedule_replay_counts": {
            schedule: sorted({int(record["replay_count"]) for record in values})
            for schedule, values in records.items()
        },
        "promotable_schedules": [schedule for schedule, passed in promotable.items() if passed],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--retention-count", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--schedules", nargs="+", choices=SCHEDULES, default=list(SCHEDULES))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "decoder": "shared",
        "replay_order_only": True,
        "diagnostics": run_interleaved_diagnostics(
            train_count=args.train_count,
            holdout_count=args.holdout_count,
            retention_count=args.retention_count,
            seeds=tuple(int(seed) for seed in args.seeds),
            schedules=tuple(str(schedule) for schedule in args.schedules),
        ),
    }
    result["can_promote"] = bool(result["diagnostics"]["promotable_schedules"])
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
