"""Diagnose B5 under maximally separated byte-cue distributions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import (  # noqa: E402
    ContinualMemoryCorpus,
    ContinualMemoryTask,
    DelayedMemoryQuery,
    MemoryEpisode,
    Taiji,
    TaijiConfig,
)

FORMAT = "taiji-native-m1-36-cue-curriculum-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_36_cue_curriculum_20260902.json"
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


def _curriculum(*, phase_a_start: int, phase_b_start: int) -> ContinualMemoryCorpus:
    phase_a_train = tuple(
        MemoryEpisode(
            memory_id=f"m1-b5-cue-a-{index}",
            cue=phase_a_start + index,
            action=48 + index % 2,
            outcome=43 if index % 2 == 0 else 45,
        )
        for index in range(TRAIN_COUNT)
    )
    phase_b_train = tuple(
        MemoryEpisode(
            memory_id=f"m1-b5-cue-b-{index}",
            cue=phase_b_start + index,
            action=48 + index % 2,
            outcome=43 if index % 2 == 0 else 45,
        )
        for index in range(TRAIN_COUNT)
    )
    return ContinualMemoryCorpus(
        phase_a_train=phase_a_train,
        phase_a_holdout=_queries(
            "m1-b5-cue-a-holdout",
            phase_a_train,
            count=HOLDOUT_COUNT,
            offset=0,
        ),
        phase_a_retention=_queries(
            "m1-b5-cue-a-retention",
            phase_a_train,
            count=RETENTION_COUNT,
            offset=HOLDOUT_COUNT,
        ),
        phase_b_train=phase_b_train,
        phase_b_holdout=_queries(
            "m1-b5-cue-b-holdout",
            phase_b_train,
            count=HOLDOUT_COUNT,
            offset=TRAIN_COUNT // 2,
        ),
        phase_b_retention=_queries(
            "m1-b5-cue-b-retention",
            phase_b_train,
            count=RETENTION_COUNT,
            offset=TRAIN_COUNT // 2 + HOLDOUT_COUNT,
        ),
        replay_train=phase_a_train,
    )


def _cue_context(model: Taiji, cue: int) -> torch.Tensor:
    model.reset_dynamics(episode_id=f"m1-36-context-{cue}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
    model.observe(int(cue), learn=False, learn_motor=False, use_memory=False)
    return model.fabric.cortical_context(model.snapshot().regions).detach().clone()


def _context_record(corpus: ContinualMemoryCorpus, seed: int) -> dict[str, Any]:
    model = Taiji(_config(seed), episode_id=f"m1-36-context-model-{seed}")
    phase_a = torch.stack([_cue_context(model, item.cue) for item in corpus.phase_a_train])
    phase_b = torch.stack([_cue_context(model, item.cue) for item in corpus.phase_b_train])
    similarities = F.normalize(phase_a, dim=1) @ F.normalize(phase_b, dim=1).T
    return {
        "seed": seed,
        "cross_phase_context_cosine_max": float(similarities.max().item()),
        "cross_phase_context_cosine_mean": float(similarities.mean().item()),
        "cross_phase_context_near_collision_count": int((similarities >= 0.90).sum().item()),
        "cross_phase_context_pairs": int(similarities.numel()),
    }


def _condition_record(name: str, corpus: ContinualMemoryCorpus) -> dict[str, Any]:
    started = time.perf_counter()
    measurement = ContinualMemoryTask(
        _config(SEEDS[0]),
        seeds=SEEDS,
        replay_learning_targets="all",
    ).evaluate(corpus)
    contexts = [_context_record(corpus, seed) for seed in SEEDS]
    phase_a_cues = {episode.cue for episode in corpus.phase_a_train}
    phase_b_cues = {episode.cue for episode in corpus.phase_b_train}
    return {
        "condition": name,
        "corpus_digest": corpus.digest,
        "phase_a_cue_min_max": [min(phase_a_cues), max(phase_a_cues)],
        "phase_b_cue_min_max": [min(phase_b_cues), max(phase_b_cues)],
        "cue_value_overlap_count": len(phase_a_cues.intersection(phase_b_cues)),
        "phase_a_actions": sorted({episode.action for episode in corpus.phase_a_train}),
        "phase_b_actions": sorted({episode.action for episode in corpus.phase_b_train}),
        "measurement": measurement.to_payload(),
        "seed_metrics": _seed_metrics(measurement),
        "context_records": contexts,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def _seed_metrics(measurement: Any) -> list[dict[str, Any]]:
    raw = next(
        item for item in measurement.evidence if item.startswith("seed_metrics=")
    )
    value = json.loads(raw.split("=", 1)[1])
    if not isinstance(value, list):
        raise ValueError("B5 evidence seed_metrics must be a list")
    return [dict(item) for item in value]


def _condition_passed(record: dict[str, Any]) -> bool:
    measurement = record["measurement"]
    return bool(
        measurement["status"] == "passed"
        and measurement["holdout_updates"] == 0
        and record["cue_value_overlap_count"] == 0
        and all(item["continued_from_phase_a"] for item in record["seed_metrics"])
        and all(
            item["replay_old_after"] >= item["old_before"]
            and item["replay_retention_after"] >= item["old_retention_before"]
            and item["replay_new_after"] + 0.05 >= item["no_replay_new_after"]
            for item in record["seed_metrics"]
        )
    )


def run_diagnosis() -> dict[str, Any]:
    baseline = _curriculum(phase_a_start=65, phase_b_start=145)
    separated = _curriculum(phase_a_start=0, phase_b_start=192)
    records = [
        _condition_record("overlapping_action_baseline", baseline),
        _condition_record("maximally_separated_byte_cues", separated),
    ]
    for record in records:
        record["condition_gate_passed"] = _condition_passed(record)
    separated_record = records[1]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if separated_record["condition_gate_passed"] else "failed",
        "variable_changed": "phase-A/phase-B cue numeric distribution only",
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "replay_scale": REPLAY_SCALE,
        "records": records,
        "conclusion": {
            "separated_condition_passed": separated_record["condition_gate_passed"],
            "cue_distribution_is_sufficient_explanation": (
                separated_record["condition_gate_passed"]
                and not records[0]["condition_gate_passed"]
            ),
            "next_boundary": (
                "maximum byte-cue separation did not pass B5; freeze input-range"
                " explanations and diagnose event representation/training signal"
                if not separated_record["condition_gate_passed"]
                else "cue separation passed; continue with unseen cue generalization"
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
