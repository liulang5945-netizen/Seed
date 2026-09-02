"""Evaluate governed association rewiring as a bounded native candidate."""

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
from scripts.training.eval_taiji_m1_42_geometry_audit import (  # noqa: E402
    _event_patterns,
)
from taiji import DelayedMemoryTask, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-43-association-rewire-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_43_association_rewire_20260902.json"
SEEDS = (11, 29, 47)
CONDITIONS = ("no_rewire", "governed_rewire")


def _actions(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.action) for episode in episodes))


def _probe(model: Taiji, queries: tuple[Any, ...], actions: tuple[int, ...]) -> float:
    correct = 0
    for query in queries:
        model.reset_dynamics(episode_id=f"m1-43-probe-{query.query_id}")
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


def _topology_digest(model: Taiji) -> str:
    return content_digest(
        {
            "association_pre_index": model.memory.association.pre_index.detach().cpu(),
            "row_fan_in": model.memory.association.row_fan_in,
        }
    )


def _capture_ratio(model: Taiji, cue_pattern: Any) -> float:
    energy = cue_pattern.square().sum().clamp_min(1e-8)
    captured = cue_pattern[model.memory.association.pre_index].square().sum(dim=1) / energy
    return float(captured.mean().item())


def _write(
    model: Taiji,
    episode: Any,
    *,
    rewire: bool,
) -> dict[str, float | int]:
    DelayedMemoryTask._write_episode(model, episode)
    if not rewire:
        return {"moved_edges": 0, "capture_before": 0.0, "capture_after": 0.0}
    geometry = _event_patterns(model, episode)
    cue_pattern = geometry["cue_pattern"]
    event_pattern = geometry["event_pattern"]
    association = model.memory.association
    capture_before = _capture_ratio(model, cue_pattern)
    error = event_pattern - association.forward(cue_pattern)
    moved_edges = association.structural_update(
        error,
        cue_pattern,
        turnover_ratio=model.config.structural_turnover_ratio,
        capture_target=model.config.structural_capture_target,
        error_threshold=model.config.structural_error_threshold,
    )
    capture_after = _capture_ratio(model, cue_pattern)
    return {
        "moved_edges": int(moved_edges),
        "capture_before": capture_before,
        "capture_after": capture_after,
    }


def _train(
    model: Taiji,
    episodes: tuple[Any, ...],
    *,
    rewire: bool,
) -> dict[str, Any]:
    events = [_write(model, episode, rewire=rewire) for episode in episodes]
    captures_before = [float(event["capture_before"]) for event in events if event["capture_before"]]
    captures_after = [float(event["capture_after"]) for event in events if event["capture_after"]]
    return {
        "event_count": len(events),
        "moved_edges": sum(int(event["moved_edges"]) for event in events),
        "capture_before_mean": (
            sum(captures_before) / len(captures_before) if captures_before else None
        ),
        "capture_after_mean": (
            sum(captures_after) / len(captures_after) if captures_after else None
        ),
    }


def _condition_record(condition: str, corpus: Any) -> dict[str, Any]:
    started = time.perf_counter()
    rewire = condition == "governed_rewire"
    actions = _actions(*corpus.phase_a_train, *corpus.phase_b_train)
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        model = Taiji(_config(seed), episode_id=f"m1-43-{condition}-{seed}")
        phase_a_train = _train(model, corpus.phase_a_train, rewire=rewire)
        old_before = _probe(model, corpus.phase_a_holdout, actions)
        old_retention_before = _probe(model, corpus.phase_a_retention, actions)
        phase_a_digest = content_digest(deepcopy(model.checkpoint()))
        phase_a_topology_digest = _topology_digest(model)

        no_replay = Taiji.from_checkpoint(deepcopy(model.checkpoint()))
        phase_b_no_replay = _train(no_replay, corpus.phase_b_train, rewire=rewire)
        no_replay_old = _probe(no_replay, corpus.phase_a_holdout, actions)
        no_replay_new = _probe(no_replay, corpus.phase_b_holdout, actions)

        replay = Taiji.from_checkpoint(deepcopy(model.checkpoint()))
        phase_b_replay = _train(replay, corpus.phase_b_train, rewire=rewire)
        replay_train = _train(replay, corpus.replay_train, rewire=rewire)
        replay_old = _probe(replay, corpus.phase_a_holdout, actions)
        replay_retention = _probe(replay, corpus.phase_a_retention, actions)
        replay_new = _probe(replay, corpus.phase_b_holdout, actions)
        replay_checkpoint = deepcopy(replay.checkpoint())
        replay_digest = content_digest(replay_checkpoint)
        replay_topology_digest = _topology_digest(replay)
        restored = Taiji.from_checkpoint(deepcopy(replay_checkpoint))
        restored_digest = content_digest(deepcopy(restored.checkpoint()))
        records.append(
            {
                "seed": seed,
                "condition": condition,
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
                "phase_a_train": phase_a_train,
                "phase_b_no_replay": phase_b_no_replay,
                "phase_b_replay": phase_b_replay,
                "replay_train": replay_train,
                "phase_a_checkpoint_digest": phase_a_digest,
                "phase_a_topology_digest": phase_a_topology_digest,
                "replay_checkpoint_digest": replay_digest,
                "replay_topology_digest": replay_topology_digest,
                "topology_changed": phase_a_topology_digest != replay_topology_digest,
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
        "condition": condition,
        "records": records,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def _condition_passed(condition: dict[str, Any]) -> bool:
    return all(
        record["continued_from_phase_a"]
        and record["checkpoint_roundtrip_exact"]
        and record["parameter_count_matches_plan"]
        and record["holdout_updates"] == 0
        and record["topology_changed"]
        and record["replay_backward_transfer"] >= 0.0
        and record["replay_retention_after"] >= record["old_retention_before"]
        and record["replay_new_after"] + 0.05 >= record["no_replay_new_after"]
        and record["replay_causal_gain"] > 0.0
        for record in condition["records"]
    )


def run_diagnosis() -> dict[str, Any]:
    corpus = _curriculum(phase_a_start=0, phase_b_start=192)
    conditions = [_condition_record(condition, corpus) for condition in CONDITIONS]
    for condition in conditions:
        condition["condition_gate_passed"] = _condition_passed(condition)
    control, candidate = conditions
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if candidate["condition_gate_passed"] else "failed",
        "variable_changed": "governed memory.association topology rewiring only",
        "control_condition": "no_rewire",
        "candidate_condition": "governed_rewire",
        "structural_policy": {
            "turnover_ratio": _config(SEEDS[0]).structural_turnover_ratio,
            "capture_target": _config(SEEDS[0]).structural_capture_target,
            "error_threshold": _config(SEEDS[0]).structural_error_threshold,
        },
        "cue_curriculum": "maximally separated byte cues",
        "action_curriculum": "phase-A/phase-B overlap on 48/49",
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "corpus_digest": corpus.digest,
        "conditions": conditions,
        "conclusion": {
            "candidate_passed": candidate["condition_gate_passed"],
            "candidate_is_sufficient_explanation": (
                candidate["condition_gate_passed"]
                and not control["condition_gate_passed"]
            ),
            "next_boundary": (
                "association topology candidate passed; hold for M1-44 structural review"
                if candidate["condition_gate_passed"]
                else "association rewiring did not pass B5; do not retain topology mutation and return to event encoding geometry"
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
