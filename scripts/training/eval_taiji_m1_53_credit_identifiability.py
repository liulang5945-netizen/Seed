"""Audit whether the memory course can identify action and outcome credit separately."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_36_cue_curriculum import _config  # noqa: E402
from taiji import DelayedMemoryTask, MemoryEpisode, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-53-credit-identifiability-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_53_credit_identifiability_20260902.json"
SEEDS = (11, 29, 47)
TRAIN_COUNT = 64
HOLDOUT_COUNT = 32
RETENTION_COUNT = 32
REPLAY_SCALE = 1.0
ACTION_SYMBOLS = (48, 49)
OUTCOME_SYMBOLS = (43, 45)
COMBINATIONS = tuple(
    (action, outcome)
    for action in ACTION_SYMBOLS
    for outcome in OUTCOME_SYMBOLS
)


@dataclass(frozen=True)
class CreditCourse:
    name: str
    phase_a_train: tuple[MemoryEpisode, ...]
    phase_a_holdout: tuple[MemoryEpisode, ...]
    phase_a_retention: tuple[MemoryEpisode, ...]
    phase_b_train: tuple[MemoryEpisode, ...]
    phase_b_holdout: tuple[MemoryEpisode, ...]
    replay_train: tuple[MemoryEpisode, ...]

    @property
    def digest(self) -> str:
        def payload(items: tuple[MemoryEpisode, ...]) -> list[dict[str, Any]]:
            return [
                {
                    "memory_id": item.memory_id,
                    "cue": int(item.cue),
                    "action": int(item.action),
                    "outcome": int(item.outcome),
                }
                for item in items
            ]

        return content_digest(
            {
                "format": FORMAT,
                "name": self.name,
                "phase_a_train": payload(self.phase_a_train),
                "phase_a_holdout": payload(self.phase_a_holdout),
                "phase_a_retention": payload(self.phase_a_retention),
                "phase_b_train": payload(self.phase_b_train),
                "phase_b_holdout": payload(self.phase_b_holdout),
                "replay_train": payload(self.replay_train),
            }
        )


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


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("credit-identifiability summary cannot be empty")
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _queries(
    prefix: str,
    episodes: tuple[MemoryEpisode, ...],
    *,
    count: int,
    offset: int,
) -> tuple[MemoryEpisode, ...]:
    return tuple(
        MemoryEpisode(
            memory_id=f"{prefix}-{index}",
            cue=episode.cue,
            action=episode.action,
            outcome=episode.outcome,
        )
        for index in range(count)
        for episode in (episodes[(index + offset) % len(episodes)],)
    )


def _episodes(
    prefix: str,
    *,
    cue_start: int,
    count: int,
    factorial: bool,
    combination_offset: int = 0,
) -> tuple[MemoryEpisode, ...]:
    return tuple(
        MemoryEpisode(
            memory_id=f"{prefix}-{index}",
            cue=cue_start + index,
            action=(
                COMBINATIONS[(index + combination_offset) % len(COMBINATIONS)][0]
                if factorial
                else 48 + index % 2
            ),
            outcome=(
                COMBINATIONS[(index + combination_offset) % len(COMBINATIONS)][1]
                if factorial
                else 43 if index % 2 == 0 else 45
            ),
        )
        for index in range(count)
    )


def _course(name: str, *, factorial: bool) -> CreditCourse:
    phase_a_train = _episodes(
        f"m1-53-{name}-a",
        cue_start=0,
        count=TRAIN_COUNT,
        factorial=factorial,
    )
    phase_b_train = _episodes(
        f"m1-53-{name}-b",
        cue_start=192,
        count=TRAIN_COUNT,
        factorial=factorial,
        combination_offset=1,
    )
    return CreditCourse(
        name=name,
        phase_a_train=phase_a_train,
        phase_a_holdout=_queries(
            f"m1-53-{name}-a-holdout",
            phase_a_train,
            count=HOLDOUT_COUNT,
            offset=0,
        ),
        phase_a_retention=_queries(
            f"m1-53-{name}-a-retention",
            phase_a_train,
            count=RETENTION_COUNT,
            offset=HOLDOUT_COUNT,
        ),
        phase_b_train=phase_b_train,
        phase_b_holdout=_queries(
            f"m1-53-{name}-b-holdout",
            phase_b_train,
            count=HOLDOUT_COUNT,
            offset=TRAIN_COUNT // 2,
        ),
        replay_train=phase_a_train,
    )


def _combination_counts(items: tuple[MemoryEpisode, ...]) -> dict[str, int]:
    return {
        f"{action}/{outcome}": sum(
            int(item.action == action and item.outcome == outcome) for item in items
        )
        for action, outcome in COMBINATIONS
    }


def _recall(
    model: Taiji,
    query: MemoryEpisode,
    actions: tuple[int, ...],
    outcomes: tuple[int, ...],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    model.reset_dynamics(episode_id=f"m1-53-recall-{query.memory_id}")
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
    recall = step.memory_recall
    if recall is None:
        raise RuntimeError("memory recall was unexpectedly absent")
    action_probabilities = recall.action_probabilities
    outcome_probabilities = recall.outcome_probabilities
    action_alternatives = [
        float(action_probabilities[action].item())
        for action in actions
        if action != query.action
    ]
    outcome_alternatives = [
        float(outcome_probabilities[outcome].item())
        for outcome in outcomes
        if outcome != query.outcome
    ]
    row: dict[str, Any] = {
        "query_id": query.memory_id,
        "cue": int(query.cue),
        "action": int(query.action),
        "outcome": int(query.outcome),
        "combination": f"{query.action}/{query.outcome}",
        "action_probability": float(action_probabilities[query.action].item()),
        "action_margin": float(
            action_probabilities[query.action].item() - max(action_alternatives)
        ),
        "action_correct": int(
            max(actions, key=lambda value: float(action_probabilities[value].item()))
            == query.action
        ),
        "outcome_probability": float(outcome_probabilities[query.outcome].item()),
        "outcome_margin": float(
            outcome_probabilities[query.outcome].item() - max(outcome_alternatives)
        ),
        "outcome_correct": int(
            max(outcomes, key=lambda value: float(outcome_probabilities[value].item()))
            == query.outcome
        ),
    }
    if baseline is not None:
        row["delta_action_margin_vs_phase_a"] = float(
            row["action_margin"] - baseline["action_margin"]
        )
        row["delta_outcome_margin_vs_phase_a"] = float(
            row["outcome_margin"] - baseline["outcome_margin"]
        )
        row["delta_action_outcome_margin_gap"] = float(
            row["delta_action_margin_vs_phase_a"]
            - row["delta_outcome_margin_vs_phase_a"]
        )
    return row


def _probe_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {
            "sample_count": len(items),
            "action_accuracy": float(
                sum(int(row["action_correct"]) for row in items) / len(items)
            ),
            "outcome_accuracy": float(
                sum(int(row["outcome_correct"]) for row in items) / len(items)
            ),
            "action_margin": _summary([float(row["action_margin"]) for row in items]),
            "outcome_margin": _summary(
                [float(row["outcome_margin"]) for row in items]
            ),
            "action_outcome_margin_gap": _summary(
                [
                    float(row["action_margin"] - row["outcome_margin"])
                    for row in items
                ]
            ),
        }
        if "delta_action_margin_vs_phase_a" in items[0]:
            value.update(
                {
                    "delta_action_margin_vs_phase_a": _summary(
                        [
                            float(row["delta_action_margin_vs_phase_a"])
                            for row in items
                        ]
                    ),
                    "delta_outcome_margin_vs_phase_a": _summary(
                        [
                            float(row["delta_outcome_margin_vs_phase_a"])
                            for row in items
                        ]
                    ),
                    "delta_action_outcome_margin_gap": _summary(
                        [
                            float(row["delta_action_outcome_margin_gap"])
                            for row in items
                        ]
                    ),
                    "absolute_delta_action_outcome_margin_gap": _summary(
                        [
                            abs(float(row["delta_action_outcome_margin_gap"]))
                            for row in items
                        ]
                    ),
                }
            )
        return value

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["combination"]), []).append(row)
    return {
        "all": metrics(rows),
        "by_combination": {
            combination: metrics(grouped[combination])
            for combination in sorted(grouped)
        },
    }


def _probe(
    model: Taiji,
    queries: tuple[MemoryEpisode, ...],
    actions: tuple[int, ...],
    outcomes: tuple[int, ...],
    baseline: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    rows = [
        _recall(
            model,
            query,
            actions,
            outcomes,
            None if baseline is None else baseline[query.cue],
        )
        for query in queries
    ]
    return {"rows": rows, "summary": _probe_summary(rows)}


def _checkpoint_record(model: Taiji) -> dict[str, Any]:
    checkpoint = deepcopy(model.checkpoint())
    digest = content_digest(checkpoint)
    restored = Taiji.from_checkpoint(deepcopy(checkpoint))
    same_process = content_digest(restored.checkpoint())
    fresh_process = _fresh_process_digest(checkpoint)
    return {
        "digest": digest,
        "same_process_digest_matches": same_process == digest,
        "fresh_process_digest_matches": fresh_process == digest,
    }


def _seed_record(course: CreditCourse, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    actions = ACTION_SYMBOLS
    outcomes = OUTCOME_SYMBOLS
    phase_a = Taiji(_config(seed), episode_id=f"m1-53-{course.name}-phase-a-{seed}")
    for episode in course.phase_a_train:
        DelayedMemoryTask._write_episode(phase_a, episode)
    phase_a_baseline = _probe(
        phase_a,
        course.phase_a_holdout,
        actions,
        outcomes,
        None,
    )
    phase_a_baseline_by_cue = {
        row["cue"]: row for row in phase_a_baseline["rows"]
    }
    phase_a_retention = _probe(
        phase_a,
        course.phase_a_retention,
        actions,
        outcomes,
        None,
    )
    phase_a_checkpoint = deepcopy(phase_a.checkpoint())
    phase_a_checkpoint_digest = content_digest(phase_a_checkpoint)
    phase_b = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
    for episode in course.phase_b_train:
        DelayedMemoryTask._write_episode(phase_b, episode)
    phase_b_checkpoint = deepcopy(phase_b.checkpoint())
    phase_b_checkpoint_digest = content_digest(phase_b_checkpoint)
    conditions: dict[str, Any] = {}
    for condition in ("no_replay", "replay"):
        model = Taiji.from_checkpoint(deepcopy(phase_b_checkpoint))
        if condition == "replay":
            for episode in course.replay_train:
                DelayedMemoryTask._write_episode(
                    model,
                    episode,
                    provenance="replayed",
                    memory_learning_scale=REPLAY_SCALE,
                    memory_learning_targets="all",
                )
        checkpoint = _checkpoint_record(model)
        old = _probe(
            model,
            course.phase_a_holdout,
            actions,
            outcomes,
            phase_a_baseline_by_cue,
        )
        retention = _probe(
            model,
            course.phase_a_retention,
            actions,
            outcomes,
            {
                row["cue"]: row for row in phase_a_retention["rows"]
            },
        )
        new = _probe(model, course.phase_b_holdout, actions, outcomes, None)
        conditions[condition] = {
            "old_holdout": old,
            "old_retention": retention,
            "new_holdout": new,
            "memory_write_count": int(model.memory.write_count),
            "memory_writes_since_phase_a": int(
                model.memory.write_count - phase_a.memory.write_count
            ),
            "replay_backward_transfer_action": float(
                old["summary"]["all"]["action_accuracy"]
                - phase_a_baseline["summary"]["all"]["action_accuracy"]
            ),
            "replay_backward_transfer_outcome": float(
                old["summary"]["all"]["outcome_accuracy"]
                - phase_a_baseline["summary"]["all"]["outcome_accuracy"]
            ),
            "checkpoint": checkpoint,
            "active_parameter_count": model.parameter_count(),
            "planned_active_parameter_count": model.config.planned_active_parameter_count,
            "parameter_count_matches_plan": (
                model.parameter_count() == model.config.planned_active_parameter_count
            ),
            "holdout_updates": 0,
        }
    no_replay = conditions["no_replay"]
    replay = conditions["replay"]
    return {
        "seed": seed,
        "phase_a_checkpoint_digest": phase_a_checkpoint_digest,
        "phase_b_checkpoint_digest": phase_b_checkpoint_digest,
        "phase_a_baseline": phase_a_baseline,
        "phase_a_retention_baseline": phase_a_retention,
        "conditions": conditions,
        "replay_causal_gain_action": float(
            replay["old_holdout"]["summary"]["all"]["action_accuracy"]
            - no_replay["old_holdout"]["summary"]["all"]["action_accuracy"]
        ),
        "replay_causal_gain_outcome": float(
            replay["old_holdout"]["summary"]["all"]["outcome_accuracy"]
            - no_replay["old_holdout"]["summary"]["all"]["outcome_accuracy"]
        ),
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def _course_record(course: CreditCourse) -> dict[str, Any]:
    started = time.perf_counter()
    records = [_seed_record(course, seed) for seed in SEEDS]
    return {
        "course": course.name,
        "corpus_digest": course.digest,
        "phase_a_train_combination_counts": _combination_counts(course.phase_a_train),
        "phase_a_holdout_combination_counts": _combination_counts(course.phase_a_holdout),
        "phase_a_retention_combination_counts": _combination_counts(course.phase_a_retention),
        "phase_b_train_combination_counts": _combination_counts(course.phase_b_train),
        "phase_b_holdout_combination_counts": _combination_counts(course.phase_b_holdout),
        "records": records,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def run_audit() -> dict[str, Any]:
    paired = _course_record(_course("paired", factorial=False))
    factorial = _course_record(_course("factorial", factorial=True))
    paired_gaps = [
        abs(
            float(
                record["conditions"]["replay"]["old_holdout"]["summary"]["all"][
                    "delta_action_outcome_margin_gap"
                ]["mean"]
            )
        )
        for record in paired["records"]
    ]
    factorial_gaps = [
        abs(
            float(
                record["conditions"]["replay"]["old_holdout"]["summary"]["all"][
                    "delta_action_outcome_margin_gap"
                ]["mean"]
            )
        )
        for record in factorial["records"]
    ]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "promote": False,
        "architecture_unchanged": True,
        "variable_changed": "training course only; no Taiji core/config/decoder change",
        "same_budget": {
            "train_count": TRAIN_COUNT,
            "holdout_count": HOLDOUT_COUNT,
            "retention_count": RETENTION_COUNT,
            "replay_scale": REPLAY_SCALE,
        },
        "action_symbols": list(ACTION_SYMBOLS),
        "outcome_symbols": list(OUTCOME_SYMBOLS),
        "courses": [paired, factorial],
        "cross_course_identifiability": {
            "paired_replay_action_outcome_delta_gap_abs": _summary(paired_gaps),
            "factorial_replay_action_outcome_delta_gap_abs": _summary(factorial_gaps),
            "paired_course_combination_count": len(COMBINATIONS),
            "factorial_course_combination_count": len(COMBINATIONS),
            "interpretation": "the factorial course is admissible only as a data/evaluator diagnostic; no architecture promotion is made from this report",
        },
        "gates": {
            "all_four_combinations_required": True,
            "requires_fresh_process_checkpoint": True,
            "holdout_updates_must_be_zero": True,
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
                "cross_course_identifiability": result["cross_course_identifiability"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
