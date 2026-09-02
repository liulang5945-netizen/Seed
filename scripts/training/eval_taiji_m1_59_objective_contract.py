"""Trace the native episodic objective without using evaluation partitions.

M1-59 deliberately stops before introducing a new learning rule.  It proves
that one auditable objective can name its inputs, positive binding, negative
competition, credit axes, and replay provenance while keeping holdout and
retention partitions outside the objective path.
"""

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

from scripts.training.eval_taiji_m1_45_component_geometry import _patterns  # noqa: E402
from scripts.training.eval_taiji_m1_53_credit_identifiability import (  # noqa: E402
    _checkpoint_record,
    _config,
    _course,
)
from taiji import DelayedMemoryTask, EpisodicObjectiveContract, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-59-objective-contract-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_59_objective_contract_20260902.json"
SEEDS = (11, 29, 47)
ACTION_SYMBOLS = (48, 49)
OUTCOME_SYMBOLS = (43, 45)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("objective trace summary cannot be empty")
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _train_partition_digest(partition: str, episodes: tuple[Any, ...]) -> str:
    return content_digest(
        {
            "format": FORMAT,
            "partition": partition,
            "episodes": [
                {
                    "memory_id": item.memory_id,
                    "cue": int(item.cue),
                    "action": int(item.action),
                    "outcome": int(item.outcome),
                }
                for item in episodes
            ],
        }
    )


def _objective_trace(
    model: Taiji,
    episodes: tuple[Any, ...],
    *,
    partition: str,
    provenance: str,
    contract: EpisodicObjectiveContract,
) -> dict[str, Any]:
    """Materialize objective evidence from one allowed train partition.

    The trace reads model geometry only.  It never calls a write method and
    intentionally accepts only partitions named by the contract as train
    sources; this makes accidental holdout leakage fail loudly.
    """

    if partition not in contract.source_partitions:
        raise ValueError(f"objective trace partition is not an allowed source: {partition}")
    if partition in contract.prohibited_partitions:
        raise ValueError(f"objective trace cannot inspect prohibited partition: {partition}")
    if not episodes:
        raise ValueError("objective trace needs at least one episode")

    patterns = [_patterns(model, episode) for episode in episodes]
    events = torch.stack([item["event"] for item in patterns])
    completions = torch.stack(
        [model.memory.association.forward(item["cue"]).detach() for item in patterns]
    )
    association_similarity = F.normalize(completions, dim=1) @ F.normalize(events, dim=1).T

    rows: list[dict[str, Any]] = []
    for index, (episode, pattern) in enumerate(zip(episodes, patterns, strict=True)):
        similarities = association_similarity[index]
        negative_values = torch.cat((similarities[:index], similarities[index + 1 :]))
        positive = float(similarities[index].item())
        negative = float(negative_values.max().item()) if negative_values.numel() else 0.0
        event = pattern["event"]
        action_credit = F.cosine_similarity(
            event.unsqueeze(0), pattern["components"]["action"].unsqueeze(0)
        )
        outcome_credit = F.cosine_similarity(
            event.unsqueeze(0), pattern["components"]["outcome"].unsqueeze(0)
        )
        rows.append(
            {
                "partition": partition,
                "memory_id": episode.memory_id,
                "cue_identity": int(episode.cue),
                "cue_identity_digest": content_digest({"cue": int(episode.cue)}),
                "action": int(episode.action),
                "outcome": int(episode.outcome),
                "reward": 1.0,
                "time": 2,
                "episode": f"m0-b2-train-{episode.memory_id}",
                "provenance": provenance,
                "positive_binding": {
                    "kind": contract.positive_binding,
                    "positive_cosine": positive,
                    "cross_cue_negative_max_cosine": negative,
                    "margin": positive - negative,
                },
                "negative_competition": {
                    "kind": contract.negative_competition,
                    "competitor_count": int(negative_values.numel()),
                },
                "credit": {
                    "association": float(pattern["association_completion_ratio"]),
                    "action_readout": float(action_credit.item()),
                    "outcome_readout": float(outcome_credit.item()),
                },
            }
        )

    return {
        "partition": partition,
        "provenance": provenance,
        "sample_count": len(rows),
        "digest": _train_partition_digest(partition, episodes),
        "rows": rows,
        "summary": {
            "positive_binding_margin": _summary(
                [row["positive_binding"]["margin"] for row in rows]
            ),
            "positive_binding_cosine": _summary(
                [row["positive_binding"]["positive_cosine"] for row in rows]
            ),
            "cross_cue_negative_max_cosine": _summary(
                [row["positive_binding"]["cross_cue_negative_max_cosine"] for row in rows]
            ),
            "association_completion_ratio": _summary(
                [row["credit"]["association"] for row in rows]
            ),
            "action_readout_credit_cosine": _summary(
                [row["credit"]["action_readout"] for row in rows]
            ),
            "outcome_readout_credit_cosine": _summary(
                [row["credit"]["outcome_readout"] for row in rows]
            ),
        },
    }


def _protected_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    before_summary = before["summary"]
    after_summary = after["summary"]
    keys = (
        "positive_binding_margin",
        "association_completion_ratio",
        "action_readout_credit_cosine",
        "outcome_readout_credit_cosine",
    )
    return {
        key: float(after_summary[key]["mean"] - before_summary[key]["mean"])
        for key in keys
    }


def _seed_record(
    course: Any,
    seed: int,
    contract: EpisodicObjectiveContract,
) -> dict[str, Any]:
    started = time.perf_counter()
    phase_a = Taiji(_config(seed), episode_id=f"m1-59-phase-a-{seed}")
    for episode in course.phase_a_train:
        DelayedMemoryTask._write_episode(phase_a, episode)
    phase_a_trace = _objective_trace(
        phase_a,
        course.phase_a_train,
        partition="phase_a_train",
        provenance="experienced",
        contract=contract,
    )
    phase_a_checkpoint = deepcopy(phase_a.checkpoint())
    phase_a_checkpoint_audit = _checkpoint_record(phase_a)

    phase_b = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
    for episode in course.phase_b_train:
        DelayedMemoryTask._write_episode(phase_b, episode)
    phase_b_trace = _objective_trace(
        phase_b,
        course.phase_b_train,
        partition="phase_b_train",
        provenance="experienced",
        contract=contract,
    )
    phase_b_checkpoint = deepcopy(phase_b.checkpoint())
    phase_b_checkpoint_audit = _checkpoint_record(phase_b)

    replay = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
    for episode in course.replay_train:
        DelayedMemoryTask._write_episode(
            replay,
            episode,
            provenance=contract.replay_provenance,
            memory_learning_targets="all",
        )
    replay_trace = _objective_trace(
        replay,
        course.replay_train,
        partition="replay_train",
        provenance=contract.replay_provenance,
        contract=contract,
    )
    protected_after_replay = _objective_trace(
        replay,
        course.phase_b_train,
        partition=contract.protected_partition,
        provenance="experienced",
        contract=contract,
    )
    replay_checkpoint_audit = _checkpoint_record(replay)

    return {
        "seed": seed,
        "phase_a": {
            "trace": phase_a_trace,
            "checkpoint": phase_a_checkpoint_audit,
        },
        "phase_b": {
            "trace": phase_b_trace,
            "checkpoint": phase_b_checkpoint_audit,
        },
        "replay": {
            "trace": replay_trace,
            "checkpoint": replay_checkpoint_audit,
        },
        "protected_partition": contract.protected_partition,
        "protected_before_replay": phase_b_trace,
        "protected_after_replay": protected_after_replay,
        "protected_delta_after_replay": _protected_delta(
            phase_b_trace, protected_after_replay
        ),
        "write_count": int(replay.memory.write_count),
        "active_parameter_count": replay.parameter_count(),
        "planned_active_parameter_count": replay.config.planned_active_parameter_count,
        "parameter_count_matches_plan": (
            replay.parameter_count() == replay.config.planned_active_parameter_count
        ),
        "holdout_updates": 0,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def run(report_path: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    started = time.perf_counter()
    contract = EpisodicObjectiveContract()
    course = _course("factorial", factorial=True)
    records = [_seed_record(course, seed, contract) for seed in seeds]
    result = {
        "format": FORMAT,
        "status": "trace_only_complete",
        "objective_contract": contract.to_dict(),
        "objective_contract_digest": contract.digest,
        "course": "factorial",
        "source_partitions_used": list(contract.source_partitions),
        "prohibited_partitions_used": [],
        "holdout_updates": 0,
        "trace_only": True,
        "default_training_changed": False,
        "records": records,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args()
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    result = run(args.report, seeds)
    print(
        json.dumps(
            {
                "format": result["format"],
                "status": result["status"],
                "report_path": str(args.report),
                "objective_contract_digest": result["objective_contract_digest"],
                "prohibited_partitions_used": result["prohibited_partitions_used"],
                "parameter_count_matches_plan": all(
                    record["parameter_count_matches_plan"] for record in result["records"]
                ),
                "checkpoint_audits_pass": all(
                    record[stage]["checkpoint"]["same_process_digest_matches"]
                    and record[stage]["checkpoint"]["fresh_process_digest_matches"]
                    for record in result["records"]
                    for stage in ("phase_a", "phase_b", "replay")
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
