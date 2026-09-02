"""Attribute per-memory interference caused by phase-B replay writes."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
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
from scripts.training.eval_taiji_m1_45_component_geometry import (  # noqa: E402
    _patterns,
)
from taiji import DelayedMemoryTask, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-52-replay-interference-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_52_replay_interference_20260902.json"
SEEDS = (11, 29, 47)
CONDITIONS = ("no_change", "no_replay", "replay", "repeated_replay")
REPLAY_SCALE = 1.0


def _fresh_process_digest(checkpoint: dict[str, Any]) -> str:
    code = """
import io
import sys
import torch
from taiji import Taiji
from taiji.internalization import content_digest

payload = torch.load(io.BytesIO(sys.stdin.buffer.read()), map_location="cpu", weights_only=False)
model = Taiji.from_checkpoint(payload)
print(content_digest(model.checkpoint()))
"""
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    process = subprocess.run(
        (sys.executable, "-c", code),
        cwd=PROJECT_ROOT,
        input=buffer.getvalue(),
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode(errors="replace")[-2000:])
    return process.stdout.decode().strip().splitlines()[-1]


def _actions(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.action) for episode in episodes))


def _outcomes(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.outcome) for episode in episodes))


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("replay interference summary cannot be empty")
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _recall(model: Taiji, episode: Any, actions: tuple[int, ...], outcomes: tuple[int, ...]) -> dict[str, float | int]:
    model.reset_dynamics(episode_id=f"m1-52-recall-{episode.memory_id}")
    model.observe(
        model.config.boundary_symbol,
        learn=False,
        learn_motor=False,
        use_memory=True,
    )
    step = model.observe(
        episode.cue,
        learn=False,
        learn_motor=False,
        use_memory=True,
    )
    recall = step.memory_recall
    if recall is None:
        raise RuntimeError("memory recall was unexpectedly absent")
    action_probabilities = recall.action_probabilities
    outcome_probabilities = recall.outcome_probabilities
    action_alternatives = [
        float(action_probabilities[action].item())
        for action in actions
        if action != episode.action
    ]
    outcome_alternatives = [
        float(outcome_probabilities[outcome].item())
        for outcome in outcomes
        if outcome != episode.outcome
    ]
    return {
        "action_probability": float(action_probabilities[episode.action].item()),
        "action_margin": float(
            action_probabilities[episode.action].item() - max(action_alternatives)
        ),
        "action_correct": int(
            max(actions, key=lambda action: float(action_probabilities[action].item()))
            == episode.action
        ),
        "outcome_probability": float(outcome_probabilities[episode.outcome].item()),
        "outcome_margin": float(
            outcome_probabilities[episode.outcome].item() - max(outcome_alternatives)
        ),
        "outcome_correct": int(
            max(outcomes, key=lambda outcome: float(outcome_probabilities[outcome].item()))
            == episode.outcome
        ),
    }


def _measure(
    model: Taiji,
    episode: Any,
    actions: tuple[int, ...],
    outcomes: tuple[int, ...],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    pattern = _patterns(model, episode)
    recall = _recall(model, episode, actions, outcomes)
    completion = model.memory.association.forward(pattern["cue"])
    residual = pattern["event"] - completion
    row = {
        "memory_id": episode.memory_id,
        "action": int(episode.action),
        "outcome": int(episode.outcome),
        "association_completion_ratio": float(pattern["association_completion_ratio"]),
        "association_error_ratio": float(pattern["association_error_ratio"]),
        "association_error_norm": float(residual.norm().item()),
        **recall,
    }
    for name in (
        "association_completion_ratio",
        "association_error_ratio",
        "association_error_norm",
        "action_probability",
        "action_margin",
        "outcome_probability",
        "outcome_margin",
    ):
        row[f"delta_{name}_vs_phase_a"] = float(row[name] - baseline[name])
    row["delta_action_outcome_margin_gap"] = float(
        (row["action_margin"] - baseline["action_margin"])
        - (row["outcome_margin"] - baseline["outcome_margin"])
    )
    return row


def _memory_weights(model: Taiji) -> dict[str, torch.Tensor]:
    memory = model.memory
    return {
        "association": memory.association.edge_weight.detach().clone(),
        "action_readout": memory.action_readout.edge_weight.detach().clone(),
        "outcome_readout": memory.outcome_readout.edge_weight.detach().clone(),
        "cortical_readout": memory.cortical_readout.edge_weight.detach().clone(),
    }


def _weight_deltas(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in before:
        delta = (after[name] - before[name]).norm()
        scale = before[name].norm().clamp_min(1e-8)
        result[name] = float((delta / scale).item())
    return result


def _topology_digest(model: Taiji) -> str:
    memory = model.memory
    return content_digest(
        {
            "association": memory.association.pre_index.detach().cpu(),
            "readout_receptors": memory.readout_receptors.to_payload(),
            "action_readout": memory.action_readout.pre_index.detach().cpu(),
            "outcome_readout": memory.outcome_readout.pre_index.detach().cpu(),
            "cortical_readout": memory.cortical_readout.pre_index.detach().cpu(),
        }
    )


def _static_similarity_features(corpus: Any, model: Taiji) -> dict[str, dict[str, float | int]]:
    phase_a_patterns = [_patterns(model, episode) for episode in corpus.phase_a_train]
    phase_b_patterns = [_patterns(model, episode) for episode in corpus.phase_b_train]
    phase_a_cues = torch.stack([item["cue"] for item in phase_a_patterns])
    phase_b_cues = torch.stack([item["cue"] for item in phase_b_patterns])
    phase_a_events = torch.stack([item["event"] for item in phase_a_patterns])
    phase_b_events = torch.stack([item["event"] for item in phase_b_patterns])
    cue_a_a = F.normalize(phase_a_cues, dim=1) @ F.normalize(phase_a_cues, dim=1).T
    event_a_a = F.normalize(phase_a_events, dim=1) @ F.normalize(phase_a_events, dim=1).T
    cue_a_b = F.normalize(phase_a_cues, dim=1) @ F.normalize(phase_b_cues, dim=1).T
    event_a_b = F.normalize(phase_a_events, dim=1) @ F.normalize(phase_b_events, dim=1).T
    result: dict[str, dict[str, float | int]] = {}
    for index, episode in enumerate(corpus.phase_a_train):
        other_mask = torch.ones(len(corpus.phase_a_train), dtype=torch.bool)
        other_mask[index] = False
        result[episode.memory_id] = {
            "max_other_replay_cue_cosine": float(cue_a_a[index, other_mask].max().item()),
            "max_other_replay_event_cosine": float(event_a_a[index, other_mask].max().item()),
            "max_phase_b_cue_cosine": float(cue_a_b[index].max().item()),
            "max_phase_b_event_cosine": float(event_a_b[index].max().item()),
            "phase_b_near_cue_collision_count": int((cue_a_b[index] >= 0.90).sum().item()),
            "phase_b_near_event_collision_count": int((event_a_b[index] >= 0.90).sum().item()),
            "same_action_replay_count": sum(
                int(candidate.action == episode.action)
                for candidate in corpus.replay_train
                if candidate.memory_id != episode.memory_id
            ),
            "same_outcome_replay_count": sum(
                int(candidate.outcome == episode.outcome)
                for candidate in corpus.replay_train
                if candidate.memory_id != episode.memory_id
            ),
        }
    return result


def _row_summary(
    rows: list[dict[str, Any]],
    *,
    include_deltas: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(rows),
        "association_completion_ratio": _summary(
            [float(row["association_completion_ratio"]) for row in rows]
        ),
        "association_error_ratio": _summary(
            [float(row["association_error_ratio"]) for row in rows]
        ),
        "action_margin": _summary([float(row["action_margin"]) for row in rows]),
        "outcome_margin": _summary([float(row["outcome_margin"]) for row in rows]),
        "action_accuracy": float(
            sum(int(row["action_correct"]) for row in rows) / len(rows)
        ),
        "outcome_accuracy": float(
            sum(int(row["outcome_correct"]) for row in rows) / len(rows)
        ),
    }
    if include_deltas:
        summary.update(
            {
                "delta_association_error_ratio_vs_phase_a": _summary(
                    [
                        float(row["delta_association_error_ratio_vs_phase_a"])
                        for row in rows
                    ]
                ),
                "delta_action_margin_vs_phase_a": _summary(
                    [float(row["delta_action_margin_vs_phase_a"]) for row in rows]
                ),
                "delta_outcome_margin_vs_phase_a": _summary(
                    [float(row["delta_outcome_margin_vs_phase_a"]) for row in rows]
                ),
                "delta_action_outcome_margin_gap": _summary(
                    [float(row["delta_action_outcome_margin_gap"]) for row in rows]
                ),
                "negative_action_margin_delta_count": sum(
                    int(float(row["delta_action_margin_vs_phase_a"]) < 0.0)
                    for row in rows
                ),
                "negative_outcome_margin_delta_count": sum(
                    int(float(row["delta_outcome_margin_vs_phase_a"]) < 0.0)
                    for row in rows
                ),
            }
        )
    return summary


def _condition_model(
    phase_a_checkpoint: dict[str, Any],
    phase_b_checkpoint: dict[str, Any],
    condition: str,
    replay_train: tuple[Any, ...],
) -> Taiji:
    if condition == "no_change":
        return Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
    model = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    if condition in {"replay", "repeated_replay"}:
        repeats = 1 if condition == "replay" else 2
        for _ in range(repeats):
            for episode in replay_train:
                DelayedMemoryTask._write_episode(
                    model,
                    episode,
                    provenance="replayed",
                    memory_learning_scale=REPLAY_SCALE,
                    memory_learning_targets="all",
                )
    return model


def _attribution_summary(
    corpus: Any,
    static_features: dict[str, dict[str, float | int]],
    conditions: dict[str, Any],
) -> dict[str, Any]:
    static_rows = list(static_features.values())
    replay = conditions["replay"]
    repeated = conditions["repeated_replay"]
    return {
        "same_cue_overwrite": {
            "phase_b_exact_cue_overlap_count": len(
                {episode.cue for episode in corpus.phase_a_train}
                & {episode.cue for episode in corpus.phase_b_train}
            ),
            "max_other_replay_cue_cosine": max(
                float(row["max_other_replay_cue_cosine"]) for row in static_rows
            ),
            "max_phase_b_cue_cosine": max(
                float(row["max_phase_b_cue_cosine"]) for row in static_rows
            ),
            "phase_b_near_cue_collision_total": sum(
                int(row["phase_b_near_cue_collision_count"]) for row in static_rows
            ),
            "evidence": "exact overlap is absent; geometric near-collision is reported but not promoted to overwrite",
        },
        "shared_readout_rewrite": {
            "replay_weight_delta_from_phase_b": dict(
                replay["weight_delta_from_phase_b"]
            ),
            "repeated_replay_weight_delta_from_phase_b": dict(
                repeated["weight_delta_from_phase_b"]
            ),
            "replay_action_margin_delta_mean": float(
                replay["summary"]["delta_action_margin_vs_phase_a"]["mean"]
            ),
            "replay_outcome_margin_delta_mean": float(
                replay["summary"]["delta_outcome_margin_vs_phase_a"]["mean"]
            ),
            "repeated_action_margin_delta_mean": float(
                repeated["summary"]["delta_action_margin_vs_phase_a"]["mean"]
            ),
            "repeated_outcome_margin_delta_mean": float(
                repeated["summary"]["delta_outcome_margin_vs_phase_a"]["mean"]
            ),
            "evidence": "replay-only readout weight deltas are non-zero while no-replay deltas from the phase-B base are zero",
        },
        "association_residual_unclosed": {
            "no_replay_error_ratio_mean": float(
                conditions["no_replay"]["summary"]["association_error_ratio"]["mean"]
            ),
            "replay_error_ratio_mean": float(
                replay["summary"]["association_error_ratio"]["mean"]
            ),
            "repeated_replay_error_ratio_mean": float(
                repeated["summary"]["association_error_ratio"]["mean"]
            ),
            "replay_delta_error_ratio_mean": float(
                replay["summary"]["delta_association_error_ratio_vs_phase_a"]["mean"]
            ),
            "evidence": "replay increases residual error relative to no-replay in this diagnostic; no association rule is changed",
        },
        "course_not_identifiable": {
            "replay_action_outcome_margin_gap": dict(
                replay["summary"]["delta_action_outcome_margin_gap"]
            ),
            "repeated_replay_action_outcome_margin_gap": dict(
                repeated["summary"]["delta_action_outcome_margin_gap"]
            ),
            "evidence": "action and outcome margin changes are paired within floating-point tolerance under the current two-class course and shared decoder",
        },
    }


def _seed_record(corpus: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    actions = _actions(*corpus.phase_a_train, *corpus.phase_b_train)
    outcomes = _outcomes(*corpus.phase_a_train, *corpus.phase_b_train)
    phase_a = Taiji(_config(seed), episode_id=f"m1-52-phase-a-{seed}")
    for episode in corpus.phase_a_train:
        DelayedMemoryTask._write_episode(phase_a, episode)
    static_features = _static_similarity_features(corpus, phase_a)
    phase_a_rows = {}
    for episode in corpus.phase_a_train:
        pattern = _patterns(phase_a, episode)
        recall = _recall(phase_a, episode, actions, outcomes)
        completion = phase_a.memory.association.forward(pattern["cue"])
        residual = pattern["event"] - completion
        phase_a_rows[episode.memory_id] = {
            "memory_id": episode.memory_id,
            "action": int(episode.action),
            "outcome": int(episode.outcome),
            "association_completion_ratio": float(pattern["association_completion_ratio"]),
            "association_error_ratio": float(pattern["association_error_ratio"]),
            "association_error_norm": float(residual.norm().item()),
            **recall,
        }
    phase_a_checkpoint = deepcopy(phase_a.checkpoint())
    phase_a_digest = content_digest(phase_a_checkpoint)
    phase_a_weights = _memory_weights(phase_a)
    phase_a_topology = _topology_digest(phase_a)

    phase_b = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
    for episode in corpus.phase_b_train:
        DelayedMemoryTask._write_episode(phase_b, episode)
    phase_b_checkpoint = deepcopy(phase_b.checkpoint())
    phase_b_digest = content_digest(phase_b_checkpoint)
    phase_b_weights = _memory_weights(phase_b)
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        model = _condition_model(
            phase_a_checkpoint,
            phase_b_checkpoint,
            condition,
            corpus.replay_train,
        )
        memory_before_probe = content_digest(model.memory.to_payload())
        checkpoint = deepcopy(model.checkpoint())
        checkpoint_digest = content_digest(checkpoint)
        restored = Taiji.from_checkpoint(deepcopy(checkpoint))
        same_process_digest = content_digest(restored.checkpoint())
        fresh_process_digest = _fresh_process_digest(checkpoint)
        before_probe_weights = _memory_weights(model)
        rows = []
        for episode in corpus.phase_a_train:
            row = _measure(
                model,
                episode,
                actions,
                outcomes,
                phase_a_rows[episode.memory_id],
            )
            row.update(static_features[episode.memory_id])
            rows.append(row)
        memory_after_probe = content_digest(model.memory.to_payload())
        after_probe_weights = _memory_weights(model)
        conditions[condition] = {
            "rows": rows,
            "summary": _row_summary(rows),
            "memory_write_count": int(model.memory.write_count),
            "memory_writes_since_phase_a": int(
                model.memory.write_count - phase_a.memory.write_count
            ),
            "memory_payload_unchanged_during_probe": memory_before_probe == memory_after_probe,
            "checkpoint_digest": checkpoint_digest,
            "checkpoint_roundtrip_exact": same_process_digest == checkpoint_digest,
            "fresh_process_digest_matches": fresh_process_digest == checkpoint_digest,
            "same_process_digest": same_process_digest,
            "fresh_process_digest": fresh_process_digest,
            "topology_digest": _topology_digest(model),
            "topology_matches_phase_a": _topology_digest(model) == phase_a_topology,
            "active_parameter_count": model.parameter_count(),
            "planned_active_parameter_count": model.config.planned_active_parameter_count,
            "parameter_count_matches_plan": (
                model.parameter_count() == model.config.planned_active_parameter_count
            ),
            "holdout_updates": 0,
            "weight_delta_from_phase_a": _weight_deltas(phase_a_weights, after_probe_weights),
            "weight_delta_from_phase_b": _weight_deltas(phase_b_weights, after_probe_weights),
            "probe_weight_delta": _weight_deltas(before_probe_weights, after_probe_weights),
        }

    return {
        "seed": seed,
        "phase_a_checkpoint_digest": phase_a_digest,
        "phase_b_checkpoint_digest": phase_b_digest,
        "phase_a_topology_digest": phase_a_topology,
        "phase_a_summary": _row_summary(
            list(phase_a_rows.values()), include_deltas=False
        ),
        "conditions": conditions,
        "attribution": _attribution_summary(corpus, static_features, conditions),
        "phase_b_exact_cue_overlap_count": len(
            {episode.cue for episode in corpus.phase_a_train}
            & {episode.cue for episode in corpus.phase_b_train}
        ),
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def run_audit() -> dict[str, Any]:
    corpus = _curriculum(phase_a_start=0, phase_b_start=192)
    records = [_seed_record(corpus, seed) for seed in SEEDS]
    replay_action_deltas = [
        float(record["conditions"]["replay"]["weight_delta_from_phase_b"]["action_readout"])
        for record in records
    ]
    replay_outcome_deltas = [
        float(record["conditions"]["replay"]["weight_delta_from_phase_b"]["outcome_readout"])
        for record in records
    ]
    replay_margin_gaps = [
        abs(
            float(
                record["conditions"]["replay"]["summary"][
                    "delta_action_outcome_margin_gap"
                ]["max"]
            )
        )
        for record in records
    ]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "promote": False,
        "architecture_unchanged": True,
        "variable_changed": "none; per-event replay interference observation only",
        "conditions": list(CONDITIONS),
        "replay_provenance": "replayed",
        "replay_learning_targets": "all",
        "replay_learning_scale": REPLAY_SCALE,
        "cue_curriculum": "maximally separated byte cues",
        "action_curriculum": "phase-A/phase-B overlap on 48/49",
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "corpus_digest": corpus.digest,
        "records": records,
        "cross_seed_attribution": {
            "replay_action_readout_delta_from_phase_b": _summary(replay_action_deltas),
            "replay_outcome_readout_delta_from_phase_b": _summary(replay_outcome_deltas),
            "stable_nonzero_replay_readout_rewrite": all(
                value > 0.0 for value in (*replay_action_deltas, *replay_outcome_deltas)
            ),
            "max_replay_action_outcome_margin_gap": max(replay_margin_gaps),
            "action_outcome_pairing_within_float_tolerance": max(replay_margin_gaps)
            <= 2e-6,
            "exact_cue_overlap_is_zero": all(
                record["phase_b_exact_cue_overlap_count"] == 0 for record in records
            ),
            "interpretation": "shared readout rewrite is stable and correlated with paired output drift, but association and cortical readout also change; cue geometry and course pairing remain confounds, so this does not authorize a readout-only architecture change",
        },
        "attribution_boundary": {
            "same_cue_overwrite": (
                "phase-A and phase-B cue values have zero exact overlap; report per-event "
                "cue cosine and near-collision evidence instead of inferring overwrite"
            ),
            "shared_readout_rewrite": (
                "compare action/outcome/readout-receptor weight deltas from phase-B base "
                "with per-event margin changes"
            ),
            "association_residual_unclosed": (
                "compare completion/error distributions and their change across no-replay, "
                "replay, and repeated-replay"
            ),
            "course_not_identifiable": (
                "report action-vs-outcome margin gap and paired evidence; do not treat "
                "identical labels as an architectural credit result"
            ),
        },
        "gates": {
            "diagnostic_only": True,
            "holdout_updates": 0,
            "requires_fresh_process_checkpoint": True,
            "does_not_change_default_checkpoint": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_audit()
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "format": result["format"],
                "status": result["status"],
                "report_path": result["report_path"],
                "seeds": [
                    {
                        "seed": record["seed"],
                        "phase_b_exact_cue_overlap_count": record[
                            "phase_b_exact_cue_overlap_count"
                        ],
                        "conditions": {
                            name: record["conditions"][name]["summary"]
                            for name in CONDITIONS
                        },
                    }
                    for record in result["records"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
