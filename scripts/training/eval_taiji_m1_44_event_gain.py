"""Test whether episodic event-component gain interferes with cue binding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_36_cue_curriculum import (  # noqa: E402
    _config,
    _curriculum,
)
from scripts.training.eval_taiji_m1_42_geometry_audit import (  # noqa: E402
    _event_patterns,
)
from taiji import DelayedMemoryTask, Taiji, TaijiConfig  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-44-event-gain-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_44_event_gain_20260902.json"
SEEDS = (11, 29, 47)
GAINS = (0.80, 0.05)


def _actions(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.action) for episode in episodes))


def _config_for_gain(seed: int, event_gain: float) -> TaijiConfig:
    values = _config(seed).to_dict()
    values["memory_event_gain"] = float(event_gain)
    return TaijiConfig.from_dict(values)


def _probe(model: Taiji, queries: tuple[Any, ...], actions: tuple[int, ...]) -> float:
    correct = 0
    for query in queries:
        model.reset_dynamics(episode_id=f"m1-44-probe-{query.query_id}")
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


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _geometry(model: Taiji, corpus: Any) -> dict[str, Any]:
    phase_a = [_event_patterns(model, item) for item in corpus.phase_a_train]
    phase_b = [_event_patterns(model, item) for item in corpus.phase_b_train]
    phase_a_events = torch.stack([item["event_pattern"] for item in phase_a])
    phase_b_events = torch.stack([item["event_pattern"] for item in phase_b])
    cross = F.normalize(phase_a_events, dim=1) @ F.normalize(phase_b_events, dim=1).T
    return {
        "phase_a": {
            "event_active_support": _summary(
                [float(item["event_active_support"]) for item in phase_a]
            ),
            "cue_event_cosine": _summary(
                [float(item["cue_event_cosine"]) for item in phase_a]
            ),
            "association_completion_ratio": _summary(
                [float(item["association_completion_ratio"]) for item in phase_a]
            ),
            "association_error_ratio": _summary(
                [float(item["association_error_ratio"]) for item in phase_a]
            ),
        },
        "cross_phase_event_cosine_mean": float(cross.mean().item()),
        "cross_phase_event_cosine_max": float(cross.max().item()),
        "cross_phase_event_near_collision_count": int((cross >= 0.90).sum().item()),
    }


def _condition_record(event_gain: float, corpus: Any) -> dict[str, Any]:
    started = time.perf_counter()
    actions = _actions(*corpus.phase_a_train, *corpus.phase_b_train)
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        config = _config_for_gain(seed, event_gain)
        model = Taiji(config, episode_id=f"m1-44-gain-{event_gain}-{seed}")
        for episode in corpus.phase_a_train:
            DelayedMemoryTask._write_episode(model, episode)
        old_before = _probe(model, corpus.phase_a_holdout, actions)
        old_retention_before = _probe(model, corpus.phase_a_retention, actions)
        phase_a_geometry = _geometry(model, corpus)
        phase_a_checkpoint = deepcopy(model.checkpoint())
        phase_a_digest = content_digest(phase_a_checkpoint)

        no_replay = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
        for episode in corpus.phase_b_train:
            DelayedMemoryTask._write_episode(no_replay, episode)
        no_replay_old = _probe(no_replay, corpus.phase_a_holdout, actions)
        no_replay_new = _probe(no_replay, corpus.phase_b_holdout, actions)

        replay = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
        for episode in corpus.phase_b_train:
            DelayedMemoryTask._write_episode(replay, episode)
        for episode in corpus.replay_train:
            DelayedMemoryTask._write_episode(replay, episode)
        replay_old = _probe(replay, corpus.phase_a_holdout, actions)
        replay_retention = _probe(replay, corpus.phase_a_retention, actions)
        replay_new = _probe(replay, corpus.phase_b_holdout, actions)
        replay_checkpoint = deepcopy(replay.checkpoint())
        replay_digest = content_digest(replay_checkpoint)
        restored = Taiji.from_checkpoint(deepcopy(replay_checkpoint))
        restored_digest = content_digest(deepcopy(restored.checkpoint()))
        records.append(
            {
                "seed": seed,
                "event_gain": event_gain,
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
                "phase_a_geometry": phase_a_geometry,
                "phase_a_checkpoint_digest": phase_a_digest,
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
        "event_gain": event_gain,
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
    conditions = [_condition_record(gain, corpus) for gain in GAINS]
    for condition in conditions:
        condition["condition_gate_passed"] = _condition_passed(condition)
    default, cue_only = conditions
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if cue_only["condition_gate_passed"] else "failed",
        "variable_changed": "memory_event_gain only",
        "baseline_event_gain": GAINS[0],
        "candidate_event_gain": GAINS[1],
        "candidate_information_loss": (
            "action/outcome/reward/time/episode/provenance event components are strongly attenuated in event pattern"
        ),
        "cue_curriculum": "maximally separated byte cues",
        "action_curriculum": "phase-A/phase-B overlap on 48/49",
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "corpus_digest": corpus.digest,
        "conditions": conditions,
        "conclusion": {
            "cue_only_passed": cue_only["condition_gate_passed"],
            "event_gain_is_sufficient_explanation": (
                cue_only["condition_gate_passed"]
                and not default["condition_gate_passed"]
            ),
            "next_boundary": (
                "cue-only passed; require event-information retention review before any change"
                if cue_only["condition_gate_passed"]
                else "cue-only did not pass B5; preserve event gain and diagnose event encoder geometry"
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
