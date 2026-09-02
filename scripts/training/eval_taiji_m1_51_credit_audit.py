"""Audit action/outcome credit transfer through native episodic association."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_36_cue_curriculum import (  # noqa: E402
    _config,
    _curriculum,
)
from scripts.training.eval_taiji_m1_45_component_geometry import (  # noqa: E402
    COMPONENTS,
    _patterns,
)
from taiji import DelayedMemoryTask, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-51-credit-audit-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_51_credit_audit_20260902.json"
SEEDS = (11, 29, 47)


def _cosine(left: Any, right: Any) -> float:
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _recall_credit(
    model: Taiji,
    episode: Any,
    actions: tuple[int, ...],
    outcomes: tuple[int, ...],
) -> dict[str, float | int]:
    model.reset_dynamics(episode_id=f"m1-51-recall-{episode.memory_id}")
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
    action_probabilities = step.memory_recall.action_probabilities
    outcome_probabilities = step.memory_recall.outcome_probabilities
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
            action_probabilities[episode.action].item()
            - max(action_alternatives)
        ),
        "action_correct": int(
            max(actions, key=lambda action: float(action_probabilities[action].item()))
            == episode.action
        ),
        "outcome_probability": float(outcome_probabilities[episode.outcome].item()),
        "outcome_margin": float(
            outcome_probabilities[episode.outcome].item()
            - max(outcome_alternatives)
        ),
        "outcome_correct": int(
            max(outcomes, key=lambda outcome: float(outcome_probabilities[outcome].item()))
            == episode.outcome
        ),
    }


def _seed_record(corpus: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    model = Taiji(_config(seed), episode_id=f"m1-51-phase-a-{seed}")
    for episode in corpus.phase_a_train:
        DelayedMemoryTask._write_episode(model, episode)
    actions = tuple(sorted({episode.action for episode in corpus.phase_a_train}))
    outcomes = tuple(sorted({episode.outcome for episode in corpus.phase_a_train}))
    rows: list[dict[str, Any]] = []
    for episode in corpus.phase_a_train:
        pattern = _patterns(model, episode)
        recall = _recall_credit(model, episode, actions, outcomes)
        completion = model.memory.association.forward(pattern["cue"])
        residual = pattern["event"] - completion
        row = {
            "memory_id": episode.memory_id,
            "action": int(episode.action),
            "outcome": int(episode.outcome),
            "association_completion_ratio": float(
                pattern["association_completion_ratio"]
            ),
            "association_error_ratio": float(pattern["association_error_ratio"]),
            "association_error_action_cosine": _cosine(
                residual, pattern["components"]["action"]
            ),
            "association_error_outcome_cosine": _cosine(
                residual, pattern["components"]["outcome"]
            ),
            "association_error_episode_cosine": _cosine(
                residual, pattern["components"]["episode"]
            ),
            **recall,
        }
        rows.append(row)
    groups: dict[str, Any] = {}
    for key, selector in (
        ("action", lambda row: str(row["action"])),
        ("outcome", lambda row: str(row["outcome"])),
    ):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(selector(row), []).append(row)
        groups[key] = {
            group: {
                "sample_count": len(group_rows),
                "association_completion_ratio": _summary(
                    [float(row["association_completion_ratio"]) for row in group_rows]
                ),
                "association_error_ratio": _summary(
                    [float(row["association_error_ratio"]) for row in group_rows]
                ),
                "error_action_cosine": _summary(
                    [float(row["association_error_action_cosine"]) for row in group_rows]
                ),
                "error_outcome_cosine": _summary(
                    [float(row["association_error_outcome_cosine"]) for row in group_rows]
                ),
                "error_episode_cosine": _summary(
                    [float(row["association_error_episode_cosine"]) for row in group_rows]
                ),
                "action_margin": _summary(
                    [float(row["action_margin"]) for row in group_rows]
                ),
                "outcome_margin": _summary(
                    [float(row["outcome_margin"]) for row in group_rows]
                ),
                "action_accuracy": float(
                    sum(int(row["action_correct"]) for row in group_rows)
                    / len(group_rows)
                ),
                "outcome_accuracy": float(
                    sum(int(row["outcome_correct"]) for row in group_rows)
                    / len(group_rows)
                ),
            }
            for group, group_rows in grouped.items()
        }
    phase_a_checkpoint = deepcopy(model.checkpoint())
    checkpoint_digest = content_digest(phase_a_checkpoint)
    restored = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
    restored_digest = content_digest(deepcopy(restored.checkpoint()))
    return {
        "seed": seed,
        "actions": list(actions),
        "outcomes": list(outcomes),
        "groups": groups,
        "rows": rows,
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_roundtrip_exact": restored_digest == checkpoint_digest,
        "active_parameter_count": model.parameter_count(),
        "planned_active_parameter_count": model.config.planned_active_parameter_count,
        "parameter_count_matches_plan": (
            model.parameter_count() == model.config.planned_active_parameter_count
        ),
        "holdout_updates": 0,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def run_audit() -> dict[str, Any]:
    corpus = _curriculum(phase_a_start=0, phase_b_start=192)
    records = [_seed_record(corpus, seed) for seed in SEEDS]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "variable_changed": "none; action/outcome credit observation only",
        "components": list(COMPONENTS),
        "cue_curriculum": "maximally separated byte cues",
        "action_curriculum": "phase-A/phase-B overlap on 48/49",
        "memory_event_gain": _config(SEEDS[0]).memory_event_gain,
        "memory_event_component_gains": list(
            _config(SEEDS[0]).memory_event_component_gains
        ),
        "memory_association_component_gains": list(
            _config(SEEDS[0]).memory_association_component_gains
        ),
        "memory_association_event_target_mix": _config(
            SEEDS[0]
        ).memory_association_event_target_mix,
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "corpus_digest": corpus.digest,
        "records": records,
        "diagnostic_boundary": {
            "does_not_promote": True,
            "interpretation": (
                "Compare residual alignment with action/outcome evidence before "
                "designing a new credit path; this report is not a B5 claim."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_audit()
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
