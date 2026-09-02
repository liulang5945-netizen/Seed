"""Diagnose B5 under an action-alphabet overlap curriculum."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus  # noqa: E402
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import (  # noqa: E402
    ContinualMemoryCorpus,
    ContinualMemoryTask,
    DelayedMemoryQuery,
    MemoryEpisode,
    TaijiConfig,
)

FORMAT = "taiji-native-m1-35-action-curriculum-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_35_action_curriculum_20260902.json"
SEEDS = (11, 29, 47)
TRAIN_COUNT = 64
HOLDOUT_COUNT = 32
RETENTION_COUNT = 32
REPLAY_SCALE = 1.0


def _config(seed: int) -> TaijiConfig:
    values = _memory_config(seed).to_dict()
    values.update(
        {
            "memory_action_decoder": "shared",
            "memory_confidence_decay": 0.0,
            "replay_memory_learning_scale": REPLAY_SCALE,
            "identity_organ_enabled": False,
        }
    )
    return TaijiConfig.from_dict(values)


def _queries(
    prefix: str,
    episodes: tuple[MemoryEpisode, ...],
    *,
    count: int,
    offset: int,
) -> tuple[DelayedMemoryQuery, ...]:
    return tuple(
        DelayedMemoryQuery(
            query_id=f"{prefix}-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index in range(int(count))
        for episode in (episodes[(index + offset) % len(episodes)],)
    )


def _overlap_corpus() -> ContinualMemoryCorpus:
    disjoint = build_corpus(
        train_count=TRAIN_COUNT,
        holdout_count=HOLDOUT_COUNT,
        retention_count=RETENTION_COUNT,
    )
    phase_b_train = tuple(
        MemoryEpisode(
            memory_id=episode.memory_id,
            cue=episode.cue,
            # Only this field changes: phase-B reuses phase-A's action alphabet.
            action=48 + index % 2,
            outcome=episode.outcome,
        )
        for index, episode in enumerate(disjoint.phase_b_train)
    )
    return ContinualMemoryCorpus(
        phase_a_train=disjoint.phase_a_train,
        phase_a_holdout=disjoint.phase_a_holdout,
        phase_a_retention=disjoint.phase_a_retention,
        phase_b_train=phase_b_train,
        phase_b_holdout=_queries(
            "m1-b5-overlap-phase-b-holdout",
            phase_b_train,
            count=HOLDOUT_COUNT,
            offset=TRAIN_COUNT // 2,
        ),
        phase_b_retention=_queries(
            "m1-b5-overlap-phase-b-retention",
            phase_b_train,
            count=RETENTION_COUNT,
            offset=TRAIN_COUNT // 2 + HOLDOUT_COUNT,
        ),
        replay_train=disjoint.replay_train,
    )


def _seed_metrics(measurement: Any) -> list[dict[str, Any]]:
    raw = next(
        item for item in measurement.evidence if item.startswith("seed_metrics=")
    )
    value = json.loads(raw.split("=", 1)[1])
    if not isinstance(value, list):
        raise ValueError("B5 evidence seed_metrics must be a list")
    return [dict(item) for item in value]


def _condition_record(name: str, corpus: ContinualMemoryCorpus) -> dict[str, Any]:
    started = time.perf_counter()
    measurement = ContinualMemoryTask(
        _config(SEEDS[0]),
        seeds=SEEDS,
        replay_learning_targets="all",
    ).evaluate(corpus)
    elapsed = time.perf_counter() - started
    phase_a_actions = {episode.action for episode in corpus.phase_a_train}
    phase_b_actions = {episode.action for episode in corpus.phase_b_train}
    return {
        "condition": name,
        "corpus_digest": corpus.digest,
        "phase_a_actions": sorted(phase_a_actions),
        "phase_b_actions": sorted(phase_b_actions),
        "action_alphabet_overlap": sorted(phase_a_actions.intersection(phase_b_actions)),
        "action_alphabet_overlap_count": len(phase_a_actions.intersection(phase_b_actions)),
        "measurement": measurement.to_payload(),
        "seed_metrics": _seed_metrics(measurement),
        "cpu_seconds": round(elapsed, 3),
    }


def _condition_passed(record: dict[str, Any]) -> bool:
    measurement = record["measurement"]
    return bool(
        measurement["status"] == "passed"
        and measurement["holdout_updates"] == 0
        and all(item["continued_from_phase_a"] for item in record["seed_metrics"])
        and all(
            item["replay_old_after"] >= item["old_before"]
            and item["replay_retention_after"] >= item["old_retention_before"]
            and item["replay_new_after"] + 0.05 >= item["no_replay_new_after"]
            for item in record["seed_metrics"]
        )
    )


def run_diagnosis() -> dict[str, Any]:
    disjoint = build_corpus(
        train_count=TRAIN_COUNT,
        holdout_count=HOLDOUT_COUNT,
        retention_count=RETENTION_COUNT,
    )
    overlap = _overlap_corpus()
    records = [
        _condition_record("disjoint_action_control", disjoint),
        _condition_record("overlapping_action_curriculum", overlap),
    ]
    for record in records:
        record["condition_gate_passed"] = _condition_passed(record)
    overlap_record = records[1]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if overlap_record["condition_gate_passed"] else "failed",
        "variable_changed": "phase-B action alphabet only",
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "replay_scale": REPLAY_SCALE,
        "records": records,
        "conclusion": {
            "overlap_condition_passed": overlap_record["condition_gate_passed"],
            "action_alphabet_is_sufficient_explanation": (
                overlap_record["condition_gate_passed"]
                and not records[0]["condition_gate_passed"]
            ),
            "next_boundary": (
                "action overlap did not pass B5; freeze this curriculum and diagnose"
                " cue representation/data distribution"
                if not overlap_record["condition_gate_passed"]
                else "action overlap passed; continue with a held-out action curriculum"
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
