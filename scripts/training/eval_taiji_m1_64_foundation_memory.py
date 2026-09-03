"""Re-measure B2 delayed memory at manifest scale on the promoted identity substrate.

M1-63 promoted ``identity_organ_enabled=True`` on the strength of a 16-key course
the evaluator built for itself.  A full-accuracy score on sixteen keys is not
evidence that memory holds, so M1-64 rebuilds B2 at the floors the manifest
actually registers -- 1000 train, 200 holdout, 200 retention -- and re-asks the
question there.

Three properties make this course honest rather than winnable by shortcut:

* every recall key is a ``context`` sequence plus a tail cue, so 1000 keys are
  genuinely mutually distinct instead of colliding inside the 256-symbol
  alphabet that capped the old cue space;
* the action and outcome of a key are drawn from a digest of the *whole* key, so
  neither the tail cue alone nor the context alone carries any usable signal --
  only the conjunction does;
* real filler symbols separate write from read, so the delay the manifest
  declares in ``cue_event_delay_interference_episode`` is actually experienced.

The substrate is not touched.  This script only measures it, against two
separate ablations (memory off, identity organ off), with per-row action margins
rather than accuracy means, and it reports B1/B3/B4/B5 alongside so a flipped
default cannot quietly regress a neighbouring ability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_foundation_baseline import (  # noqa: E402
    _memory_config,
    _model_config,
    build_continual_learning_smoke_corpus,
    build_goal_action_smoke_corpus,
    build_sequence_corpus,
    build_world_transition_smoke_corpus,
)
from scripts.training.eval_taiji_m1_53_credit_identifiability import (  # noqa: E402
    _checkpoint_record,
)
from scripts.training.eval_taiji_m1_63_identity_organ_promotion import (  # noqa: E402
    _organ_telemetry,
)
from taiji import Taiji  # noqa: E402
from taiji.foundation_evaluation import FoundationManifest  # noqa: E402
from taiji.foundation_tasks import (  # noqa: E402
    ContinualLearningTask,
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    DelayedMemoryTask,
    GoalActionTask,
    MemoryEpisode,
    SequencePredictionTask,
    WorldTransitionTask,
    _persistent_digest,
)

FORMAT = "taiji-native-m1-64-foundation-memory-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_64_foundation_memory_20260903.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_foundation_baseline_v1.json"

CUE_BASE = 65
CUE_SPAN = 40
CONTEXT_BASE = 150
INTERFERENCE_BASE = 200
INTERFERENCE_LENGTH = 4
ACTION_SYMBOLS = (48, 49)
OUTCOME_SYMBOLS = (43, 45)
ABLATIONS: tuple[tuple[str, bool, bool | None], ...] = (
    ("taiji", True, None),
    ("memory_lesion", False, None),
    ("identity_lesion", True, False),
)

# Pinned from reports/taiji_foundation_baseline_20260901.json (profile=foundation,
# model_tier=micro).  M1-64 flips no default itself, but it runs on the substrate
# whose identity default M1-63 flipped, so these are the numbers a regression
# would show up against.
PRIOR_BASELINE: dict[str, tuple[float, str]] = {
    "b1_sequence_prediction": (6.497044792701461, "lower_is_better"),
    "b3_world_transition": (0.01758887919602614, "lower_is_better"),
    "b4_goal_action": (1.0, "higher_is_better"),
    "b5_continual_learning": (-0.24420859958696206, "higher_is_better"),
}
REGRESSION_TOLERANCE = 1e-6


def _key_choice(key: tuple[int, ...], salt: str, options: tuple[int, ...]) -> int:
    """Pick an option from a digest of the whole key.

    Deriving the label from the full key is what keeps the course honest: with
    ``CUE_SPAN`` tails each appearing across many contexts, and each context
    spanning many tails, both marginals sit at chance and only the conjunction
    predicts the label.
    """
    digest = hashlib.sha256(f"{salt}\0{key}".encode()).digest()
    return options[digest[0] % len(options)]


def build_foundation_delayed_memory_corpus(
    *,
    train_units: int = 1000,
    holdout_units: int = 200,
    retention_units: int = 200,
    interference_length: int = INTERFERENCE_LENGTH,
) -> DelayedMemoryCorpus:
    if train_units < 1 or holdout_units < 1 or retention_units < 1:
        raise ValueError("B2 foundation partitions must be positive")
    if holdout_units + retention_units > train_units:
        raise ValueError("B2 read partitions cannot exceed the written key space")
    if interference_length < 1:
        raise ValueError("B2 foundation corpus must declare a real delay")

    keys: list[tuple[tuple[int, ...], int]] = []
    for index in range(int(train_units)):
        cue = CUE_BASE + index % CUE_SPAN
        context = (CONTEXT_BASE + index // CUE_SPAN,)
        keys.append((context, cue))
    recall_keys = {(*context, cue) for context, cue in keys}
    if len(recall_keys) != int(train_units):
        raise ValueError("B2 foundation keys must be mutually distinct")

    train = tuple(
        MemoryEpisode(
            memory_id=f"m1-64-train-{index}",
            cue=cue,
            action=_key_choice((*context, cue), "action", ACTION_SYMBOLS),
            outcome=_key_choice((*context, cue), "outcome", OUTCOME_SYMBOLS),
            context=context,
        )
        for index, (context, cue) in enumerate(keys)
    )
    holdout = tuple(
        DelayedMemoryQuery(
            query_id=f"m1-64-holdout-{index}",
            cue=episode.cue,
            expected_action=episode.action,
            context=episode.context,
        )
        for index, episode in enumerate(train[: int(holdout_units)])
    )
    retention = tuple(
        DelayedMemoryQuery(
            query_id=f"m1-64-retention-{index}",
            cue=episode.cue,
            expected_action=episode.action,
            context=episode.context,
        )
        for index, episode in enumerate(
            train[int(holdout_units) : int(holdout_units) + int(retention_units)]
        )
    )
    return DelayedMemoryCorpus(
        train=train,
        holdout=holdout,
        retention=retention,
        interference_symbols=tuple(
            INTERFERENCE_BASE + offset for offset in range(int(interference_length))
        ),
    )


def _majority_share(groups: dict[Any, list[int]], total: int) -> float:
    correct = sum(
        max(actions.count(symbol) for symbol in ACTION_SYMBOLS) for actions in groups.values()
    )
    return correct / total


def _marginal_predictability(
    corpus: DelayedMemoryCorpus, *, permutations: int = 64
) -> dict[str, Any]:
    """How far a tail-only or context-only reader can get on this course.

    A post-hoc majority vote is a biased estimator: with only ``train/CUE_SPAN``
    rows per group it lands well above ``chance`` even on perfectly unbiased
    labels, purely from finite-sample noise.  Comparing the observed share
    against ``chance`` would therefore condemn an honest course.  So the
    reference is a permutation null instead -- the same majority-vote statistic
    recomputed on shuffled labels, which carries exactly the same group-size
    bias.  A marginal only counts as a shortcut when it beats that null.
    """
    chance = 1.0 / len(ACTION_SYMBOLS)
    expected = {episode.recall_key: episode.action for episode in corpus.train}
    total = len(expected)
    actions = list(expected.values())
    result: dict[str, Any] = {"chance": chance, "permutations": int(permutations)}
    for name, projection in (
        ("tail_only", lambda key: key[-1]),
        ("context_only", lambda key: key[:-1]),
    ):
        projected = [projection(key) for key in expected]
        groups: dict[Any, list[int]] = {}
        for label, action in zip(projected, actions):
            groups.setdefault(label, []).append(action)
        observed = _majority_share(groups, total)

        rng = random.Random(f"m1-64-null\0{name}")
        null_shares: list[float] = []
        for _ in range(int(permutations)):
            shuffled = actions[:]
            rng.shuffle(shuffled)
            null_groups: dict[Any, list[int]] = {}
            for label, action in zip(projected, shuffled):
                null_groups.setdefault(label, []).append(action)
            null_shares.append(_majority_share(null_groups, total))

        null_mean = statistics.fmean(null_shares)
        null_max = max(null_shares)
        result[name] = observed
        result[f"{name}_null_mean"] = null_mean
        result[f"{name}_null_max"] = null_max
        result[f"{name}_exceeds_null"] = observed > null_max
    return result


def _read_rows(
    model: Taiji,
    queries: tuple[DelayedMemoryQuery, ...],
    actions: tuple[int, ...],
    *,
    use_memory: bool,
    use_identity: bool | None,
    interference_symbols: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Read every query and keep the per-row margin, not just the hit count.

    The observation sequence mirrors ``DelayedMemoryTask._recall_accuracy``
    exactly; ``read_path_matches_task_metric`` below re-derives the task's own
    accuracy from these rows so the two paths cannot silently drift apart.

    When the identity organ routes a cue (M1-66 organ-first verdict), the margin
    is read from the organ's own evidence restricted to the queried action set.
    A plain motor-softmax over the 256-symbol alphabet would squeeze the two
    action probabilities to ~1e-4 and flatten the margin to noise; the verdict
    lives in the organ's action evidence, so the margin is expanded in that
    space the same way the evaluator already reduces prediction to ``actions``.
    """

    rows: list[dict[str, Any]] = []
    # M1-66 organ-first only applies to the full integral arm; each ablation
    # keeps the original motor synthesis (lesion semantics).
    verdict_enabled = bool(use_memory) and (use_identity is None or bool(use_identity))
    for query in queries:
        model.reset_dynamics(episode_id=f"m1-64-query-{query.query_id}")
        for symbol in (
            model.config.boundary_symbol,
            *query.context,
            query.cue,
            *interference_symbols,
        ):
            step = model.observe(
                symbol,
                learn=False,
                learn_motor=False,
                use_memory=use_memory,
                use_identity=use_identity,
                use_delayed_memory_verdict=verdict_enabled,
            )
        identity_recall = step.identity_recall
        probabilities = model.snapshot().motor_probabilities
        evidence = identity_recall.action_evidence
        if (
            verdict_enabled
            and identity_recall is not None
            and identity_recall.used
            and evidence is not None
        ):
            # organ verdict: expand the margin over the action set only
            logits = tuple(float(evidence[int(action)].item()) for action in actions)
            peak = max(logits)
            expanded = tuple(math.exp(value - peak) for value in logits)
            total = sum(expanded)
            scores = {
                action: expanded[index] / total
                for index, action in enumerate(actions)
            }
        else:
            scores = {action: float(probabilities[action].item()) for action in actions}
        prediction = max(actions, key=lambda action: scores[action])
        alternatives = [
            value for action, value in scores.items() if action != query.expected_action
        ]
        rows.append(
            {
                "query_id": query.query_id,
                "expected_action": int(query.expected_action),
                "predicted_action": int(prediction),
                "action_correct": bool(prediction == query.expected_action),
                "expected_probability": scores[query.expected_action],
                "action_margin": scores[query.expected_action] - max(alternatives),
            }
        )
    return rows


def _row_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("m1-64 row summary cannot be empty")
    margins = [float(row["action_margin"]) for row in rows]
    correct = sum(1 for row in rows if row["action_correct"])
    return {
        "count": len(rows),
        "accuracy": correct / len(rows),
        "margin_mean": statistics.fmean(margins),
        "margin_std": statistics.pstdev(margins) if len(margins) > 1 else 0.0,
        "margin_min": min(margins),
        "margin_max": max(margins),
        "positive_margin_share": sum(1 for value in margins if value > 0.0) / len(margins),
    }


def _seed_record(corpus: DelayedMemoryCorpus, seed: int) -> dict[str, Any]:
    actions = tuple(dict.fromkeys(episode.action for episode in corpus.train))
    interference = corpus.interference_symbols
    config = _memory_config(seed)

    model = Taiji(config, episode_id=f"m1-64-seed-{seed}")
    checkpoint_preflight = _checkpoint_record(model)

    started = time.perf_counter()
    for episode in corpus.train:
        DelayedMemoryTask._write_episode(model, episode)
    train_seconds = time.perf_counter() - started
    checkpoint_after_train = _checkpoint_record(model)

    digest_before_holdout = _persistent_digest(model)
    channels: dict[str, dict[str, float | int]] = {}
    for name, use_memory, use_identity in ABLATIONS:
        rows = _read_rows(
            model,
            corpus.holdout,
            actions,
            use_memory=use_memory,
            use_identity=use_identity,
            interference_symbols=interference,
        )
        channels[name] = _row_summary(rows)
    digest_after_holdout = _persistent_digest(model)

    retention_rows = _read_rows(
        model,
        corpus.retention,
        actions,
        use_memory=True,
        use_identity=None,
        interference_symbols=interference,
    )
    channels["retention"] = _row_summary(retention_rows)
    digest_after_retention = _persistent_digest(model)

    frozen = Taiji(config, episode_id=f"m1-64-frozen-{seed}")
    channels["frozen_parent"] = _row_summary(
        _read_rows(
            frozen,
            corpus.holdout,
            actions,
            use_memory=True,
            use_identity=None,
            interference_symbols=interference,
        )
    )

    telemetry = _organ_telemetry(model)
    capacity = telemetry.get("capacity")
    return {
        "seed": int(seed),
        "channels": channels,
        "beats_memory_lesion": (
            float(channels["taiji"]["accuracy"]) > float(channels["memory_lesion"]["accuracy"])
        ),
        "beats_identity_lesion": (
            float(channels["taiji"]["accuracy"]) > float(channels["identity_lesion"]["accuracy"])
        ),
        "beats_frozen_parent": (
            float(channels["taiji"]["accuracy"]) > float(channels["frozen_parent"]["accuracy"])
        ),
        "holdout_updates": int(digest_before_holdout != digest_after_holdout),
        "retention_updates": int(digest_after_holdout != digest_after_retention),
        "partitions_written": ["train"],
        "checkpoint_preflight": checkpoint_preflight,
        "checkpoint_after_train": checkpoint_after_train,
        "identity_organ": telemetry,
        "slot_pressure": {
            "distinct_keys": len(corpus.train),
            "organ_capacity": capacity,
            "keys_per_slot": (
                len(corpus.train) / float(capacity)
                if isinstance(capacity, int) and capacity
                else None
            ),
        },
        "parameter_count": model.parameter_count(),
        "train_seconds": train_seconds,
    }


def _worst(records: list[dict[str, Any]], channel: str, field: str) -> float:
    return min(float(record["channels"][channel][field]) for record in records)


def _summary(records: list[dict[str, Any]], channel: str, field: str) -> dict[str, float]:
    values = [float(record["channels"][channel][field]) for record in records]
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _not_regressed(value: float, prior: float, direction: str) -> bool:
    if direction == "lower_is_better":
        return value <= prior + REGRESSION_TOLERANCE
    return value >= prior - REGRESSION_TOLERANCE


def _neighbour_abilities(
    manifest: FoundationManifest, *, b1_corpus: list[Path] | None
) -> dict[str, Any]:
    """Re-measure B1/B3/B4/B5 so a flipped default cannot regress them unseen."""
    measured: dict[str, Any] = {}
    if b1_corpus:
        corpus = build_sequence_corpus(
            b1_corpus,
            train_bytes=1_048_576,
            holdout_bytes=131_072,
            retention_bytes=131_072,
            seed=manifest.seeds[0],
        )
        measured["b1_sequence_prediction"] = SequencePredictionTask(
            _model_config("micro", manifest.seeds[0]),
            seeds=manifest.seeds,
            epochs=1,
        ).evaluate(corpus)
    measured["b3_world_transition"] = WorldTransitionTask(seeds=manifest.seeds, epochs=50).evaluate(
        build_world_transition_smoke_corpus()
    )
    measured["b4_goal_action"] = GoalActionTask(
        _memory_config(manifest.seeds[0]), seeds=manifest.seeds
    ).evaluate(build_goal_action_smoke_corpus())
    measured["b5_continual_learning"] = ContinualLearningTask(
        _memory_config(manifest.seeds[0]), seeds=manifest.seeds, epochs=1
    ).evaluate(build_continual_learning_smoke_corpus())

    result: dict[str, Any] = {}
    for ability_id, (prior, direction) in PRIOR_BASELINE.items():
        measurement = measured.get(ability_id)
        if measurement is None:
            result[ability_id] = {
                "status": "not_measured",
                "prior_metric_value": prior,
                "metric_direction": direction,
                "not_regressed": None,
            }
            continue
        value = measurement.metric_value
        result[ability_id] = {
            "status": measurement.status,
            "primary_metric": measurement.primary_metric,
            "metric_direction": direction,
            "metric_value": value,
            "prior_metric_value": prior,
            "not_regressed": (
                None if value is None else _not_regressed(float(value), prior, direction)
            ),
        }
    return result


def run(
    *,
    manifest_path: Path,
    seeds: tuple[int, ...] | None = None,
    train_units: int = 1000,
    holdout_units: int = 200,
    retention_units: int = 200,
    b1_corpus: list[Path] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = FoundationManifest.load(manifest_path)
    spec = manifest.task("b2_delayed_memory")
    active_seeds = tuple(seeds) if seeds else manifest.seeds

    corpus = build_foundation_delayed_memory_corpus(
        train_units=train_units,
        holdout_units=holdout_units,
        retention_units=retention_units,
    )
    marginals = _marginal_predictability(corpus)
    measurement = DelayedMemoryTask(_memory_config(active_seeds[0]), seeds=active_seeds).evaluate(
        corpus
    )
    records = [_seed_record(corpus, seed) for seed in active_seeds]

    task_seed_metrics = {
        int(entry["seed"]): float(entry["taiji"])
        for entry in json.loads(
            next(
                line.split("seed_metrics=", 1)[1]
                for line in measurement.evidence
                if line.startswith("seed_metrics=")
            )
        )
    }
    read_path_matches = all(
        abs(float(record["channels"]["taiji"]["accuracy"]) - task_seed_metrics[int(record["seed"])])
        <= 1e-12
        for record in records
        if int(record["seed"]) in task_seed_metrics
    )

    neighbours = _neighbour_abilities(manifest, b1_corpus=b1_corpus)

    gates = {
        "train_units_meet_manifest": len(corpus.train) >= spec.minimum_train_units,
        "holdout_units_meet_manifest": len(corpus.holdout) >= spec.minimum_holdout_units,
        "retention_units_meet_manifest": len(corpus.retention) >= spec.minimum_retention_units,
        "recall_keys_distinct": len({episode.recall_key for episode in corpus.train})
        == len(corpus.train),
        "interference_declared": len(corpus.interference_symbols) > 0,
        "tail_only_reader_at_chance": marginals["tail_only_exceeds_null"] is False,
        "context_only_reader_at_chance": marginals["context_only_exceeds_null"] is False,
        "read_path_matches_task_metric": read_path_matches,
        "beats_memory_lesion_every_seed": all(record["beats_memory_lesion"] for record in records),
        "beats_identity_lesion_every_seed": all(
            record["beats_identity_lesion"] for record in records
        ),
        "beats_frozen_parent_every_seed": all(record["beats_frozen_parent"] for record in records),
        "worst_seed_margin_positive": _worst(records, "taiji", "margin_mean") > 0.0,
        "task_measurement_passed": measurement.status == "passed",
        "holdout_updates_zero": all(int(record["holdout_updates"]) == 0 for record in records),
        "retention_updates_zero": all(int(record["retention_updates"]) == 0 for record in records),
        "checkpoint_same_process_matches": all(
            record[stage]["same_process_digest_matches"]
            for record in records
            for stage in ("checkpoint_preflight", "checkpoint_after_train")
        ),
        "checkpoint_fresh_process_matches": all(
            record[stage]["fresh_process_digest_matches"]
            for record in records
            for stage in ("checkpoint_preflight", "checkpoint_after_train")
        ),
    }
    for ability_id, entry in neighbours.items():
        gates[f"{ability_id}_not_regressed"] = entry["not_regressed"] is True
    gates["memory_ability_established"] = all(gates.values())

    blocking = sorted(name for name, passed in gates.items() if not passed)
    return {
        "format": FORMAT,
        "status": "gate_passed" if not blocking else "blocked",
        "seeds": list(active_seeds),
        "manifest_path": str(manifest_path),
        "manifest_digest": manifest.digest,
        "blocking_gates": blocking,
        "gates": gates,
        "corpus": {
            "sample_counts": corpus.sample_counts,
            "minimum_train_units": spec.minimum_train_units,
            "minimum_holdout_units": spec.minimum_holdout_units,
            "minimum_retention_units": spec.minimum_retention_units,
            "interference_symbols": list(corpus.interference_symbols),
            "distinct_recall_keys": len({episode.recall_key for episode in corpus.train}),
            "action_symbols": list(ACTION_SYMBOLS),
            "marginal_predictability": marginals,
        },
        "measurement": {
            "ability_id": measurement.ability_id,
            "status": measurement.status,
            "primary_metric": measurement.primary_metric,
            "metric_direction": measurement.metric_direction,
            "metric_value": measurement.metric_value,
            "baseline_metrics": dict(measurement.baseline_metrics),
            "holdout_updates": measurement.holdout_updates,
        },
        "margins": {
            channel: _summary(records, channel, "margin_mean")
            for channel in ("taiji", "memory_lesion", "identity_lesion", "retention")
        },
        "accuracy": {
            channel: _summary(records, channel, "accuracy")
            for channel in (
                "taiji",
                "memory_lesion",
                "identity_lesion",
                "retention",
                "frozen_parent",
            )
        },
        "worst_seed": {
            "taiji_accuracy": _worst(records, "taiji", "accuracy"),
            "taiji_margin_mean": _worst(records, "taiji", "margin_mean"),
            "retention_accuracy": _worst(records, "retention", "accuracy"),
        },
        "neighbour_abilities": neighbours,
        "seed_records": records,
        "prohibited_partitions_used": [],
        "holdout_updates": max(int(record["holdout_updates"]) for record in records),
        "retention_updates": max(int(record["retention_updates"]) for record in records),
        "substrate_changed": False,
        "cpu_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--train-units", type=int, default=1000)
    parser.add_argument("--holdout-units", type=int, default=200)
    parser.add_argument("--retention-units", type=int, default=200)
    parser.add_argument("--b1-corpus", nargs="+", type=Path)
    args = parser.parse_args()

    result = run(
        manifest_path=args.manifest,
        seeds=tuple(args.seeds) if args.seeds else None,
        train_units=args.train_units,
        holdout_units=args.holdout_units,
        retention_units=args.retention_units,
        b1_corpus=args.b1_corpus,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "format": result["format"],
                "status": result["status"],
                "report_path": str(args.report),
                "blocking_gates": result["blocking_gates"],
                "worst_seed": result["worst_seed"],
                "measurement": result["measurement"],
                "margins": result["margins"],
                "neighbour_abilities": result["neighbour_abilities"],
                "corpus": {
                    "sample_counts": result["corpus"]["sample_counts"],
                    "marginal_predictability": result["corpus"]["marginal_predictability"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "gate_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
