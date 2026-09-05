"""M0 foundation evaluation entry point for the five Taiji abilities.

The task runners are intentionally not hidden behind this command yet.  Until
they produce real measurements, this entry point emits ``not_evaluated`` and
never manufactures a capability pass from a fixture or a provider response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    JOINT_TRAINING_PHASES,
    FoundationTrainingDataset,
    Outcome,
    Taiji,
    WorldAction,
    WorldInterventionCase,
    WorldObject,
    WorldState,
)
from taiji.foundation_evaluation import (
    FOUNDATION_REQUIRED_ABILITIES,
    FoundationEvaluation,
    FoundationManifest,
    FoundationMeasurement,
)  # noqa: E402
from taiji.foundation_tasks import (  # noqa: E402
    ContinualLearningCorpus,
    ContinualLearningTask,
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    DelayedMemoryTask,
    GoalActionCorpus,
    GoalActionEpisode,
    GoalActionTask,
    MemoryEpisode,
    SequencePredictionCorpus,
    SequencePredictionTask,
    WorldTransitionCorpus,
    WorldTransitionTask,
    _hash_goal_action_accuracy,
    _hash_memory_accuracy,
    _hash_world_error,
    _majority_accuracy,
    _majority_goal_action_accuracy,
    _no_change_error,
    _persistent_digest,
    _random_world_error,
)
from taiji.foundation_training import _code_revision  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_foundation_baseline_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_foundation_baseline_20260901.json"


def _checkpoint_gate_status(manifest: FoundationManifest, path: Path | None) -> str:
    if path is None:
        return "not_run"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != manifest.checkpoint_gate:
        return "failed"
    if payload.get("status") != "passed":
        return "failed"
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        return "failed"
    return "passed"


def _indexed_checkpoint_paths(
    values: list[list[str]] | None,
    *,
    seeds: tuple[int, ...],
) -> dict[int, Path]:
    """Parse the explicit seed→checkpoint mapping used by child audits."""

    if values is None:
        return {}
    indexed: dict[int, Path] = {}
    for value in values:
        if len(value) != 2:
            raise ValueError("each --checkpoint entry needs SEED PATH")
        seed = int(value[0])
        if seed not in seeds:
            raise ValueError(f"unsupported checkpoint seed: {seed}")
        if seed in indexed:
            raise ValueError(f"duplicate checkpoint seed: {seed}")
        path = Path(value[1])
        if not path.is_file():
            raise FileNotFoundError(path)
        indexed[seed] = path
    if set(indexed) != set(seeds):
        raise ValueError(f"expected exactly one checkpoint for seeds {seeds}")
    return indexed


def _indexed_integer_values(
    values: list[list[str]] | None,
    *,
    seeds: tuple[int, ...],
    name: str,
) -> dict[int, int]:
    """Parse a seed-specific integer mapping without hiding partition policy."""

    if values is None:
        return {}
    indexed: dict[int, int] = {}
    for value in values:
        if len(value) != 2:
            raise ValueError(f"each {name} entry needs SEED VALUE")
        seed = int(value[0])
        if seed not in seeds:
            raise ValueError(f"unsupported {name} seed: {seed}")
        if seed in indexed:
            raise ValueError(f"duplicate {name} seed: {seed}")
        indexed[seed] = int(value[1])
    if set(indexed) != set(seeds):
        raise ValueError(f"expected exactly one {name} value for seeds {seeds}")
    return indexed


def _load_joint_child(path: Path, *, expected_seed: int) -> tuple[dict[str, Any], Any, Any]:
    """Load and verify a joint child without mutating it or inventing metrics."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"joint child checkpoint must contain a mapping: {path}")
    expected_digest = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    if str(payload.get("checkpoint_digest", "")) != expected_digest:
        raise ValueError(f"joint child checkpoint digest mismatch: {path}")
    if payload.get("format") != "taiji-native-joint-training-v1":
        raise ValueError(f"unsupported joint child checkpoint format: {path}")
    if int(payload.get("version", -1)) < 4:
        raise ValueError("M2-2j child audit requires a v4 private-context checkpoint")
    phases = payload.get("training_phases")
    if not isinstance(phases, list) or not phases or any(
        not isinstance(phase, str) for phase in phases
    ):
        raise ValueError("M2-2j child audit requires a non-empty phase plan")
    canonical_phases = [phase for phase in JOINT_TRAINING_PHASES if phase in phases]
    if phases != canonical_phases:
        raise ValueError("M2-2j child audit requires a canonical joint phase plan")
    if payload.get("sequence_fabric_learning") is not False:
        raise ValueError("M2-2j child audit requires sequence_fabric_learning=false")
    if payload.get("sequence_predictive_context_mode") != "private-plastic-temporal-v1":
        raise ValueError("M2-2j child audit requires the private predictive context")
    model_payload = payload.get("model")
    parent_payload = payload.get("parent_model")
    if not isinstance(model_payload, Mapping) or not isinstance(parent_payload, Mapping):
        raise ValueError(f"joint child checkpoint is missing model lineage: {path}")
    model = Taiji.from_checkpoint(model_payload)
    parent = Taiji.from_checkpoint(parent_payload)
    if int(model.config.seed) != int(expected_seed):
        raise ValueError(
            f"checkpoint seed mismatch for {path}: "
            f"expected {expected_seed}, got {model.config.seed}"
        )
    return dict(payload), model, parent


def _score_loaded_model(model: Any, data: bytes) -> float:
    before = content_digest(model.checkpoint())
    score = model.score_bytes(data)
    after = content_digest(model.checkpoint())
    if before != after:
        raise RuntimeError("loaded child score mutated the checkpoint")
    return float(score["mean_surprise"]) / math.log(2.0)


def _evaluate_loaded_b1(
    checkpoints: Mapping[int, Path],
    corpus: SequencePredictionCorpus | Mapping[int, SequencePredictionCorpus],
    *,
    expected_dataset_digest: str | Mapping[int, str] | None = None,
    expected_protected_dataset_digest: str | Mapping[int, str] | None = None,
) -> FoundationMeasurement:
    """Evaluate B1 on trained children and their own frozen parent controls."""

    seed_records: list[dict[str, float | int | str | bool]] = []
    for seed, path in sorted(checkpoints.items()):
        payload, model, parent = _load_joint_child(path, expected_seed=seed)
        seed_corpus = corpus[seed] if isinstance(corpus, Mapping) else corpus
        expected_dataset = (
            expected_dataset_digest.get(seed)
            if isinstance(expected_dataset_digest, Mapping)
            else expected_dataset_digest
        )
        expected_protected = (
            expected_protected_dataset_digest.get(seed)
            if isinstance(expected_protected_dataset_digest, Mapping)
            else expected_protected_dataset_digest
        )
        if expected_dataset is not None and str(payload.get("dataset_digest")) != expected_dataset:
            raise ValueError(f"child dataset digest does not match B1 corpus: {path}")
        if expected_protected is not None and str(payload.get("protected_dataset_digest")) != (
            expected_protected
        ):
            raise ValueError(f"child protected dataset digest does not match B1 corpus: {path}")
        child_bpb = _score_loaded_model(model, seed_corpus.holdout)
        retention_bpb = _score_loaded_model(model, seed_corpus.retention)
        frozen_bpb = _score_loaded_model(parent, seed_corpus.holdout)
        seed_records.append(
            {
                "seed": seed,
                "taiji": child_bpb,
                "retention": retention_bpb,
                "frozen_parent": frozen_bpb,
                "holdout_updates": 0,
                "parameter_count": model.parameter_count(),
                "checkpoint_digest": str(payload["checkpoint_digest"]),
                "dataset_digest": str(payload["dataset_digest"]),
                "protected_dataset_digest": str(payload.get("protected_dataset_digest")),
                "trained_child_checkpoint": True,
            }
        )

    native_values = [float(record["taiji"]) for record in seed_records]
    frozen_values = [float(record["frozen_parent"]) for record in seed_records]
    seeds = tuple(int(record["seed"]) for record in seed_records)
    config = model.config
    first_corpus = corpus[next(iter(sorted(checkpoints)))] if isinstance(corpus, Mapping) else corpus
    baseline_metrics = {
        "random": math.log2(float(config.alphabet_size)),
        "frozen_parent": min(frozen_values),
        "simple_rule": min(
            _unigram_bpb(
                corpus[seed].train if isinstance(corpus, Mapping) else corpus.train,
                corpus[seed].holdout if isinstance(corpus, Mapping) else corpus.holdout,
                config,
            )
            for seed in checkpoints
        ),
        "hash_only": min(
            _hash_only_bpb(corpus.holdout, seed=seed, alphabet_size=config.alphabet_size)
            if not isinstance(corpus, Mapping)
            else _hash_only_bpb(corpus[seed].holdout, seed=seed, alphabet_size=config.alphabet_size)
            for seed in seeds
        ),
    }
    worst_native = max(native_values)
    return FoundationMeasurement(
        ability_id="b1_sequence_prediction",
        status=(
            "passed"
            if max(native_values) < min(baseline_metrics.values())
            else "failed"
        ),
        primary_metric="bits_per_byte",
        metric_direction="lower_is_better",
        metric_value=worst_native,
        baseline_metrics=baseline_metrics,
        sample_counts=first_corpus.sample_counts,
        holdout_updates=max(int(record["holdout_updates"]) for record in seed_records),
        evidence=(
            "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
            "trained_child_checkpoint_evaluation=true",
            "native_model_checkpoint_is_read_only_during_score=true",
            "strictly_beats_strongest_control="
            + str(worst_native < min(baseline_metrics.values())),
        ),
    )


def _unigram_bpb(train: bytes, holdout: bytes, config: Any) -> float:
    counts = [1.0] * int(config.alphabet_size)
    symbols = (config.boundary_symbol, *tuple(int(value) for value in train))
    for symbol in symbols:
        counts[symbol] += 1.0
    total = sum(counts)
    targets = tuple(int(value) for value in holdout) + (config.boundary_symbol,)
    return sum(-math.log2(counts[symbol] / total) for symbol in targets) / len(targets)


def _hash_only_bpb(data: bytes, *, seed: int, alphabet_size: int) -> float:
    symbols = tuple(int(value) for value in data) + (alphabet_size - 1,)
    epsilon = 1.0 / float(alphabet_size * alphabet_size)
    losses: list[float] = []
    for index, target in enumerate(symbols[1:], start=1):
        previous = symbols[index - 1]
        digest = hashlib.sha256(f"{int(seed)}\0{index}\0{previous}".encode()).digest()
        prediction = int.from_bytes(digest[:8], "big") % int(alphabet_size)
        probability = 1.0 - (alphabet_size - 1) * epsilon if prediction == target else epsilon
        losses.append(-math.log2(probability))
    return sum(losses) / len(losses)


def _evaluate_loaded_b2(
    checkpoints: Mapping[int, Path],
    *,
    train_units: int = 1_000,
    holdout_units: int = 200,
    retention_units: int = 200,
) -> FoundationMeasurement:
    """Read B2 from trained children on the manifest-scale delayed course."""

    from scripts.training.eval_taiji_m1_64_foundation_memory import (
        _read_rows,
        _row_summary,
        build_foundation_delayed_memory_corpus,
    )

    corpus = build_foundation_delayed_memory_corpus(
        train_units=train_units,
        holdout_units=holdout_units,
        retention_units=retention_units,
    )
    actions = tuple(dict.fromkeys(episode.action for episode in corpus.train))
    seed_records: list[dict[str, float | int | str | bool]] = []
    for seed, path in sorted(checkpoints.items()):
        payload, model, parent = _load_joint_child(path, expected_seed=seed)
        before = _persistent_digest(model)
        native_rows = _read_rows(
            model,
            corpus.holdout,
            actions,
            use_memory=True,
            use_identity=None,
            interference_symbols=corpus.interference_symbols,
        )
        after_holdout = _persistent_digest(model)
        retention_rows = _read_rows(
            model,
            corpus.retention,
            actions,
            use_memory=True,
            use_identity=None,
            interference_symbols=corpus.interference_symbols,
        )
        after_retention = _persistent_digest(model)
        memory_lesion = _row_summary(
            _read_rows(
                model,
                corpus.holdout,
                actions,
                use_memory=False,
                use_identity=None,
                interference_symbols=corpus.interference_symbols,
            )
        )
        identity_lesion = _row_summary(
            _read_rows(
                model,
                corpus.holdout,
                actions,
                use_memory=True,
                use_identity=False,
                interference_symbols=corpus.interference_symbols,
            )
        )
        frozen = _row_summary(
            _read_rows(
                parent,
                corpus.holdout,
                actions,
                use_memory=True,
                use_identity=None,
                interference_symbols=corpus.interference_symbols,
            )
        )
        native = _row_summary(native_rows)
        retention = _row_summary(retention_rows)
        seed_records.append(
            {
                "seed": seed,
                "taiji": float(native["accuracy"]),
                "retention": float(retention["accuracy"]),
                "memory_lesion": float(memory_lesion["accuracy"]),
                "identity_lesion": float(identity_lesion["accuracy"]),
                "frozen_parent": float(frozen["accuracy"]),
                "holdout_updates": int(before != after_holdout),
                "retention_updates": int(after_holdout != after_retention),
                "checkpoint_digest": str(payload["checkpoint_digest"]),
                "trained_child_checkpoint": True,
            }
        )

    native_values = [float(record["taiji"]) for record in seed_records]
    retention_values = [float(record["retention"]) for record in seed_records]
    baseline_metrics = {
        "random": 1.0 / len(actions),
        "frozen_parent": min(float(record["frozen_parent"]) for record in seed_records),
        "simple_rule": _majority_accuracy(corpus.train, corpus.holdout),
        "hash_only": min(
            _hash_memory_accuracy(corpus.holdout, actions, seed=seed)
            for seed in checkpoints
        ),
        "memory_lesion": min(float(record["memory_lesion"]) for record in seed_records),
        "identity_lesion": min(
            float(record["identity_lesion"]) for record in seed_records
        ),
    }
    worst_native = min(native_values)
    beats_controls = worst_native > max(baseline_metrics.values())
    causal_memory_gain = all(
        float(record["taiji"]) > float(record["memory_lesion"]) for record in seed_records
    )
    causal_identity_gain = all(
        float(record["taiji"]) > float(record["identity_lesion"])
        for record in seed_records
    )
    retention_preserved = all(
        retention >= native - 0.05
        for retention, native in zip(retention_values, native_values, strict=True)
    )
    return FoundationMeasurement(
        ability_id="b2_delayed_memory",
        status=(
            "passed"
            if (
                beats_controls
                and causal_memory_gain
                and causal_identity_gain
                and retention_preserved
                and max(int(record["holdout_updates"]) for record in seed_records) == 0
                and max(int(record["retention_updates"]) for record in seed_records) == 0
            )
            else "failed"
        ),
        primary_metric="recall_accuracy",
        metric_direction="higher_is_better",
        metric_value=worst_native,
        baseline_metrics=baseline_metrics,
        sample_counts=corpus.sample_counts,
        holdout_updates=max(int(record["holdout_updates"]) for record in seed_records),
        evidence=(
            "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
            "trained_child_checkpoint_evaluation=true",
            "delayed_interference_symbols=" + json.dumps(corpus.interference_symbols),
            "checkpoint_read_only="
            + str(
                all(
                    int(record["holdout_updates"]) == 0
                    and int(record["retention_updates"]) == 0
                    for record in seed_records
                )
            ),
        ),
    )


def _evaluate_loaded_b3(
    checkpoints: Mapping[int, Path],
    *,
    train_units: int = 1_000,
    holdout_units: int = 500,
    retention_units: int = 500,
) -> FoundationMeasurement:
    """Read B3 from trained child world learners without registering holdout data."""

    from scripts.training.train_taiji_joint import build_world_corpus
    from taiji.foundation_training import (
        _world_action_error,
        _world_learner_from_payload,
        _world_learner_payload,
    )

    corpus = build_world_corpus(count=train_units)
    # ``build_world_corpus`` fixes the two read partitions to count // 2.
    if len(corpus.holdout) != holdout_units or len(corpus.retention) != retention_units:
        raise ValueError("B3 child course requires train_units=2*holdout_units=2*retention_units")
    seed_records: list[dict[str, float | int | str | bool]] = []
    for seed, path in sorted(checkpoints.items()):
        payload, _model, _parent = _load_joint_child(path, expected_seed=seed)
        learner_payload = payload.get("world_learner")
        parent_payload = payload.get("parent_world_learner")
        if not isinstance(learner_payload, Mapping) or not isinstance(parent_payload, Mapping):
            raise ValueError(f"joint child checkpoint is missing world learner lineage: {path}")
        learner = _world_learner_from_payload(learner_payload)
        frozen = _world_learner_from_payload(parent_payload)
        before = content_digest(_world_learner_payload(learner))
        native_error = _world_action_error(learner, corpus.holdout)
        after_holdout = content_digest(_world_learner_payload(learner))
        retention_error = _world_action_error(learner, corpus.retention)
        after_retention = content_digest(_world_learner_payload(learner))
        frozen_error = _world_action_error(frozen, corpus.holdout)
        schema = learner.schema
        seed_records.append(
            {
                "seed": seed,
                "taiji": native_error,
                "retention": retention_error,
                "frozen_parent": frozen_error,
                "random": _random_world_error(corpus.holdout, schema, seed=seed),
                "simple_rule": _no_change_error(corpus.holdout, schema),
                "hash_only": _hash_world_error(corpus.holdout, schema, seed=seed),
                "holdout_updates": int(before != after_holdout),
                "retention_updates": int(after_holdout != after_retention),
                "checkpoint_digest": str(payload["checkpoint_digest"]),
                "trained_child_checkpoint": True,
            }
        )
    native_values = [float(record["taiji"]) for record in seed_records]
    baseline_metrics = {
        "random": min(float(record["random"]) for record in seed_records),
        "frozen_parent": min(float(record["frozen_parent"]) for record in seed_records),
        "simple_rule": min(float(record["simple_rule"]) for record in seed_records),
        "hash_only": min(float(record["hash_only"]) for record in seed_records),
    }
    worst_native = max(native_values)
    return FoundationMeasurement(
        ability_id="b3_world_transition",
        status=(
            "passed"
            if worst_native < min(baseline_metrics.values())
            and max(int(record["holdout_updates"]) for record in seed_records) == 0
            and max(int(record["retention_updates"]) for record in seed_records) == 0
            else "failed"
        ),
        primary_metric="transition_error",
        metric_direction="lower_is_better",
        metric_value=worst_native,
        baseline_metrics=baseline_metrics,
        sample_counts=corpus.sample_counts,
        holdout_updates=max(int(record["holdout_updates"]) for record in seed_records),
        evidence=(
            "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
            "trained_child_checkpoint_evaluation=true",
            "world_schema_register_parameters=false",
        ),
    )


def _evaluate_loaded_b4(
    checkpoints: Mapping[int, Path],
    *,
    train_units: int = 1_000,
    holdout_units: int = 500,
    retention_units: int = 500,
) -> FoundationMeasurement:
    """Read B4 from trained child action organs on the foundation course."""

    from scripts.training.train_taiji_joint import build_goal_corpus

    corpus = build_goal_corpus(count=train_units)
    if len(corpus.holdout) != holdout_units or len(corpus.retention) != retention_units:
        raise ValueError("B4 child course requires train_units=2*holdout_units=2*retention_units")
    seed_records: list[dict[str, float | int | str | bool]] = []
    for seed, path in sorted(checkpoints.items()):
        payload, model, parent = _load_joint_child(path, expected_seed=seed)
        before = _persistent_digest(model)
        native = GoalActionTask._evaluate_partition(model, corpus.holdout)
        after_holdout = _persistent_digest(model)
        retention = GoalActionTask._evaluate_partition(model, corpus.retention)
        after_retention = _persistent_digest(model)
        frozen = GoalActionTask._evaluate_partition(parent, corpus.holdout)
        seed_records.append(
            {
                "seed": seed,
                "taiji": native,
                "retention": retention,
                "frozen_parent": frozen,
                "holdout_updates": int(before != after_holdout),
                "retention_updates": int(after_holdout != after_retention),
                "checkpoint_digest": str(payload["checkpoint_digest"]),
                "trained_child_checkpoint": True,
            }
        )
    native_values = [float(record["taiji"]) for record in seed_records]
    baseline_metrics = {
        "random": 0.5,
        "frozen_parent": min(float(record["frozen_parent"]) for record in seed_records),
        "simple_rule": _majority_goal_action_accuracy(corpus.train, corpus.holdout),
        "hash_only": min(
            _hash_goal_action_accuracy(corpus.holdout, seed=seed) for seed in checkpoints
        ),
    }
    worst_native = min(native_values)
    retention_preserved = all(
        float(record["retention"]) >= float(record["taiji"]) - 0.05
        for record in seed_records
    )
    return FoundationMeasurement(
        ability_id="b4_goal_action",
        status=(
            "passed"
            if worst_native > max(baseline_metrics.values())
            and retention_preserved
            and max(int(record["holdout_updates"]) for record in seed_records) == 0
            and max(int(record["retention_updates"]) for record in seed_records) == 0
            else "failed"
        ),
        primary_metric="success_rate",
        metric_direction="higher_is_better",
        metric_value=worst_native,
        baseline_metrics=baseline_metrics,
        sample_counts=corpus.sample_counts,
        holdout_updates=max(int(record["holdout_updates"]) for record in seed_records),
        evidence=(
            "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
            "trained_child_checkpoint_evaluation=true",
            "action_readout_evaluation_is_read_only=true",
        ),
    )


def _b5_phase_b_stream(seed: int, *, length: int, offset: int) -> bytes:
    """Create a deterministic phase-B byte stream disjoint from phase-A slices."""

    return bytes(32 + ((index * 37 + int(seed) + int(offset)) % 224) for index in range(length))


def _evaluate_loaded_b5(
    checkpoints: Mapping[int, Path],
    protected_datasets: Mapping[int, FoundationTrainingDataset],
) -> FoundationMeasurement:
    """Run the dedicated B5 continuation/replay contrast from each child."""

    seed_records: list[dict[str, float | int | str | bool]] = []
    course_sample_counts: dict[str, int] | None = None
    for seed, path in sorted(checkpoints.items()):
        payload, source_model, _parent = _load_joint_child(path, expected_seed=seed)
        protected = protected_datasets[seed]
        corpus = ContinualLearningCorpus(
            phase_a_train=protected.train[:4_096],
            phase_a_holdout=protected.holdout[:200],
            phase_b_train=_b5_phase_b_stream(seed, length=4_096, offset=0),
            phase_b_holdout=_b5_phase_b_stream(seed, length=200, offset=1),
            retention=protected.retention[:200],
        )
        if course_sample_counts is None:
            course_sample_counts = corpus.sample_counts
        no_replay = Taiji.from_checkpoint(source_model.checkpoint())
        old_before = _score_loaded_model(no_replay, corpus.phase_a_holdout)
        no_replay.learn_bytes(corpus.phase_b_train, epochs=1, learn_fabric=False)
        old_after_no_replay = _score_loaded_model(no_replay, corpus.phase_a_holdout)
        new_after_no_replay = _score_loaded_model(no_replay, corpus.phase_b_holdout)
        no_replay_bwt = old_before - old_after_no_replay

        replay = Taiji.from_checkpoint(source_model.checkpoint())
        replay_old_before = _score_loaded_model(replay, corpus.phase_a_holdout)
        replay.learn_bytes(corpus.phase_b_train, epochs=1, learn_fabric=False)
        replay_after_phase_b = _score_loaded_model(replay, corpus.phase_a_holdout)
        replay.learn_bytes(corpus.phase_a_train, epochs=1, learn_fabric=False)
        old_after_replay = _score_loaded_model(replay, corpus.phase_a_holdout)
        new_after_replay = _score_loaded_model(replay, corpus.phase_b_holdout)
        retention_after_replay = _score_loaded_model(replay, corpus.retention)
        replay_bwt = replay_old_before - old_after_replay
        seed_records.append(
            {
                "seed": seed,
                "old_before": replay_old_before,
                "old_after_no_replay": old_after_no_replay,
                "old_after_replay": old_after_replay,
                "new_after_no_replay": new_after_no_replay,
                "new_after_replay": new_after_replay,
                "retention_after_replay": retention_after_replay,
                "backward_transfer_no_replay": no_replay_bwt,
                "backward_transfer_replay": replay_bwt,
                "replay_gain_vs_no_replay": replay_bwt - no_replay_bwt,
                "phase_b_old_loss_before_replay": replay_old_before - replay_after_phase_b,
                "continued_from_child": True,
                "replay_executed": True,
                "checkpoint_digest": str(payload["checkpoint_digest"]),
                "replay_corpus_digest": content_digest(corpus.phase_a_train),
                "holdout_updates": 0,
                "trained_with_private_context_only": True,
            }
        )

    replay_values = [float(record["backward_transfer_replay"]) for record in seed_records]
    no_replay_values = [
        float(record["backward_transfer_no_replay"]) for record in seed_records
    ]
    baseline_metrics = {
        "random": 0.0,
        "frozen_parent": 0.0,
        "simple_rule": 0.0,
        "hash_only": 0.0,
        "no_replay": min(no_replay_values),
    }
    worst_replay = min(replay_values)
    replay_beats_no_replay = all(
        float(record["backward_transfer_replay"])
        > float(record["backward_transfer_no_replay"])
        for record in seed_records
    )
    new_capability_preserved = all(
        float(record["new_after_replay"])
        <= float(record["new_after_no_replay"]) + 0.5
        for record in seed_records
    )
    return FoundationMeasurement(
        ability_id="b5_continual_learning",
        status=(
            "passed"
            if (
                worst_replay > max(baseline_metrics.values())
                and replay_beats_no_replay
                and new_capability_preserved
            )
            else "failed"
        ),
        primary_metric="backward_transfer",
        metric_direction="higher_is_better",
        metric_value=worst_replay,
        baseline_metrics=baseline_metrics,
        sample_counts=course_sample_counts or {},
        holdout_updates=0,
        evidence=(
            "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
            "trained_child_checkpoint_evaluation=true",
            "phase_a_replay_is_exact_protected_train=true",
            "sequence_learning_fabric_write=false",
            "checkpoint_read_only_during_score=true",
        ),
    )


def _reuse_b1_measurement(
    path: Path,
    *,
    manifest: FoundationManifest,
    checkpoints: Mapping[int, Path],
    datasets: Mapping[int, FoundationTrainingDataset],
) -> FoundationMeasurement:
    """Reuse an immutable B1 child report only after checking its provenance."""

    if not path.is_file():
        raise ValueError(f"reused B1 report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "taiji-foundation-evaluation-v1":
        raise ValueError("reused B1 report format is not supported")
    if payload.get("manifest_digest") != manifest.digest:
        raise ValueError("reused B1 report manifest digest does not match")
    if payload.get("checkpoint_gate_status") != "passed":
        raise ValueError("reused B1 report checkpoint gate is not passed")
    checkpoint_evaluation = payload.get("checkpoint_evaluation")
    if not isinstance(checkpoint_evaluation, Mapping) or not checkpoint_evaluation.get("mode"):
        raise ValueError("reused B1 report is not a checkpoint evaluation")
    measurements = payload.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError("reused B1 report is missing measurements")
    b1_payload = next(
        (
            item
            for item in measurements
            if isinstance(item, Mapping) and item.get("ability_id") == "b1_sequence_prediction"
        ),
        None,
    )
    if not isinstance(b1_payload, Mapping):
        raise ValueError("reused B1 report is missing the B1 measurement")
    measurement = FoundationMeasurement.from_payload(b1_payload)
    if measurement.status != "passed" or measurement.holdout_updates != 0:
        raise ValueError("reused B1 report is not a passed read-only child measurement")
    if not any("trained_child_checkpoint_evaluation=true" in item for item in measurement.evidence):
        raise ValueError("reused B1 report is not a trained-child evaluation")
    stored_checkpoints = {
        int(item["seed"]): Path(item["path"])
        for item in checkpoint_evaluation.get("checkpoints", [])
        if isinstance(item, Mapping) and "seed" in item and "path" in item
    }
    if {
        seed: str(path)
        for seed, path in sorted(stored_checkpoints.items())
    } != {seed: str(path) for seed, path in sorted(checkpoints.items())}:
        raise ValueError("reused B1 report checkpoints do not match the requested child set")
    stored_digests = checkpoint_evaluation.get("b1_dataset_digests", {})
    stored_protected = checkpoint_evaluation.get("b1_protected_dataset_digests", {})
    expected_digests = {str(seed): dataset.digest for seed, dataset in datasets.items()}
    expected_protected = {
        str(seed): dataset.excluded_dataset_digest for seed, dataset in datasets.items()
    }
    if stored_digests != expected_digests or stored_protected != expected_protected:
        raise ValueError("reused B1 report data digests do not match the requested child set")
    return measurement


def build_contract_report(
    manifest: FoundationManifest,
    *,
    checkpoint_gate_status: str,
    b1_measurement: FoundationMeasurement | None = None,
    b2_measurement: FoundationMeasurement | None = None,
    b3_measurement: FoundationMeasurement | None = None,
    b4_measurement: FoundationMeasurement | None = None,
    b5_measurement: FoundationMeasurement | None = None,
) -> FoundationEvaluation:
    measurements = {
        ability_id: FoundationMeasurement(
            ability_id=ability_id,
            status="not_evaluated",
            primary_metric=manifest.task(ability_id).primary_metric,
            metric_direction=manifest.task(ability_id).metric_direction,
            metric_value=None,
            baseline_metrics={},
            sample_counts={},
            holdout_updates=0,
            evidence=("m0-1-contract-only; task runner pending",),
        )
        for ability_id in FOUNDATION_REQUIRED_ABILITIES
    }
    if b1_measurement is not None:
        measurements[b1_measurement.ability_id] = b1_measurement
    if b2_measurement is not None:
        measurements[b2_measurement.ability_id] = b2_measurement
    if b3_measurement is not None:
        measurements[b3_measurement.ability_id] = b3_measurement
    if b4_measurement is not None:
        measurements[b4_measurement.ability_id] = b4_measurement
    if b5_measurement is not None:
        measurements[b5_measurement.ability_id] = b5_measurement
    return FoundationEvaluation.evaluate(
        manifest,
        measurements,
        checkpoint_gate_status=checkpoint_gate_status,
    )


def _text_from_record(record: Any) -> str | None:
    if isinstance(record, str):
        return record.strip() or None
    if not isinstance(record, dict):
        return None
    for key in ("text", "content", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_sequence_corpus(
    paths: list[Path],
    *,
    train_bytes: int,
    holdout_bytes: int,
    retention_bytes: int,
    seed: int,
) -> SequencePredictionCorpus:
    budgets = {
        "train": int(train_bytes),
        "holdout": int(holdout_bytes),
        "retention": int(retention_bytes),
    }
    if any(value <= 0 for value in budgets.values()):
        raise ValueError("B1 byte budgets must be positive")
    buffers = {partition: bytearray() for partition in budgets}
    seen_text_digests: set[str] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _text_from_record(record)
                if text is None:
                    continue
                text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if text_digest in seen_text_digests:
                    continue
                seen_text_digests.add(text_digest)
                bucket = int.from_bytes(
                    hashlib.sha256(f"{int(seed)}\0{text}".encode()).digest()[:4],
                    "big",
                ) % 10_000
                partition = (
                    "train"
                    if bucket < 8_000
                    else "holdout"
                    if bucket < 9_000
                    else "retention"
                )
                remaining = budgets[partition] - len(buffers[partition])
                if remaining > 0:
                    buffers[partition].extend(text.encode("utf-8")[:remaining])
                if all(len(buffers[name]) >= budgets[name] for name in budgets):
                    break
        if all(len(buffers[name]) >= budgets[name] for name in budgets):
            break
    missing = {
        name: f"{len(buffers[name])}/{budgets[name]}"
        for name in budgets
        if len(buffers[name]) < budgets[name]
    }
    if missing:
        raise ValueError("B1 corpus did not meet byte budgets: " + json.dumps(missing))
    return SequencePredictionCorpus(
        train=bytes(buffers["train"]),
        holdout=bytes(buffers["holdout"]),
        retention=bytes(buffers["retention"]),
    )


def build_delayed_memory_smoke_corpus(*, count: int = 8) -> DelayedMemoryCorpus:
    if int(count) < 4:
        raise ValueError("B2 smoke corpus needs at least four memory episodes")
    train = tuple(
        MemoryEpisode(
            memory_id=f"m0-b2-smoke-{index}",
            cue=65 + index,
            action=48 + index % 2,
            outcome=43 if index % 2 == 0 else 45,
        )
        for index in range(int(count))
    )
    holdout = tuple(
        DelayedMemoryQuery(
            query_id=f"m0-b2-holdout-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(train)
    )
    retention = tuple(
        DelayedMemoryQuery(
            query_id=f"m0-b2-retention-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(train)
    )
    return DelayedMemoryCorpus(train=train, holdout=holdout, retention=retention)


def build_world_transition_smoke_corpus(*, count: int = 8) -> WorldTransitionCorpus:
    if int(count) < 4:
        raise ValueError("B3 smoke corpus needs at least four transitions")

    def case(case_id: str, position: float) -> WorldInterventionCase:
        before = WorldState(
            tick=0,
            latent=torch.zeros(1),
            objects=(
                WorldObject("agent", attributes={"energy": 1.0}),
                WorldObject("target", attributes={"position": position}),
            ),
        )
        action = WorldAction(
            action_id=case_id,
            kind="push",
            tick=0,
            actor_id="agent",
            target_id="target",
            parameters={"amount": 1.0},
        )
        after = WorldState(
            tick=1,
            latent=torch.zeros(1),
            objects=(
                WorldObject("agent", attributes={"energy": 1.0}),
                WorldObject("target", attributes={"position": position + 1.0}),
            ),
        )
        return WorldInterventionCase(
            case_id=case_id,
            initial=before,
            action=action,
            expected_state=after,
            expected_outcome=Outcome(
                intent_id=case_id,
                reward=1.0,
                success=True,
                tick=1,
            ),
        )

    return WorldTransitionCorpus(
        train=tuple(case(f"m0-b3-train-{index}", float(index)) for index in range(int(count))),
        holdout=tuple(
            case(f"m0-b3-holdout-{index}", 10.0 + index) for index in range(max(3, count // 2))
        ),
        retention=tuple(
            case(f"m0-b3-retention-{index}", 20.0 + index)
            for index in range(max(3, count // 2))
        ),
    )


def build_goal_action_smoke_corpus(*, count: int = 32) -> GoalActionCorpus:
    if int(count) < 4 or int(count) % 2:
        raise ValueError("B4 smoke corpus needs an even count of at least four episodes")
    episodes = tuple(
        GoalActionEpisode(
            episode_id=f"m0-b4-smoke-{index}",
            cue=65 + index % 2,
            preferred_action=48 + index % 2,
            alternate_action=49 - index % 2,
        )
        for index in range(int(count))
    )
    half = int(count) // 2
    return GoalActionCorpus(
        train=episodes,
        holdout=tuple(
            GoalActionEpisode(
                episode_id=f"m0-b4-holdout-{index}",
                cue=65 + index % 2,
                preferred_action=48 + index % 2,
                alternate_action=49 - index % 2,
            )
            for index in range(half)
        ),
        retention=tuple(
            GoalActionEpisode(
                episode_id=f"m0-b4-retention-{index}",
                cue=65 + index % 2,
                preferred_action=48 + index % 2,
                alternate_action=49 - index % 2,
            )
            for index in range(half)
        ),
    )


def build_continual_learning_smoke_corpus() -> ContinualLearningCorpus:
    return ContinualLearningCorpus(
        phase_a_train=(b"ABCD1234-" * 64),
        phase_a_holdout=(b"ABCD1234+" * 16),
        phase_b_train=(b"wxyz5678:" * 64),
        phase_b_holdout=(b"wxyz5678;" * 16),
        retention=(b"ABCD1234?" * 16),
    )


def _model_config(tier: str, seed: int) -> Any:
    from taiji import TaijiConfig

    if tier == "default":
        values = TaijiConfig().to_dict()
    elif tier == "micro":
        values = TaijiConfig(
            region_sizes=(8,),
            synapse_fan_in=2,
            motor_fan_in=4,
            memory_units=16,
            memory_fan_in=2,
            memory_readout_fan_in=2,
            memory_meta_dim=4,
            memory_time_dim=2,
            memory_episode_dim=2,
            lateral_fan_in=2,
            concept_capacity=8,
        ).to_dict()
    else:
        raise ValueError(f"unsupported B1 model tier: {tier}")
    values["seed"] = int(seed)
    return TaijiConfig.from_dict(values)


def _memory_config(seed: int) -> Any:
    from taiji import TaijiConfig

    values = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_meta_dim=32,
        memory_readout_fan_in=32,
        memory_iterations=3,
    ).to_dict()
    values["seed"] = int(seed)
    return TaijiConfig.from_dict(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--b1-corpus", nargs="+", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("SEED", "PATH"),
        help="Evaluate a trained joint child; provide exactly one path for every manifest seed.",
    )
    parser.add_argument(
        "--child-foundation",
        action="store_true",
        help="Also evaluate child-bound B2/B3/B4 at manifest sample floors.",
    )
    parser.add_argument(
        "--b5-child",
        action="store_true",
        help="Run the dedicated child continuation no-replay/replay contrast.",
    )
    parser.add_argument(
        "--reuse-b1-report",
        type=Path,
        help="Reuse a previously validated B1 child measurement after provenance checks.",
    )
    parser.add_argument(
        "--b1-partition-seed",
        action="append",
        nargs=2,
        metavar=("SEED", "PARTITION_SEED"),
        help="Content-addressed phase-B partition seed, explicitly mapped per child seed.",
    )
    parser.add_argument(
        "--b1-protected-corpus",
        nargs="+",
        type=Path,
        help="The protected phase-A source used to verify a child checkpoint's lineage.",
    )
    parser.add_argument(
        "--b1-protected-partition-seed",
        action="append",
        nargs=2,
        metavar=("SEED", "PARTITION_SEED"),
        help="Content-addressed phase-A partition seed, explicitly mapped per child seed.",
    )
    parser.add_argument("--b2-smoke", action="store_true")
    parser.add_argument("--b2-corpus", nargs="+", type=Path)
    parser.add_argument("--b3-smoke", action="store_true")
    parser.add_argument("--b4-smoke", action="store_true")
    parser.add_argument("--b5-smoke", action="store_true")
    parser.add_argument("--model-tier", choices=("micro", "default"), default="micro")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--profile", choices=("smoke", "foundation"), default="smoke")
    parser.add_argument("--checkpoint-report", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    manifest = FoundationManifest.load(args.manifest)
    checkpoint_status = _checkpoint_gate_status(manifest, args.checkpoint_report)
    checkpoint_paths = _indexed_checkpoint_paths(args.checkpoint, seeds=manifest.seeds)
    if (args.child_foundation or args.b5_child) and not checkpoint_paths:
        parser.error("--child-foundation/--b5-child require --checkpoint")
    b1_datasets: dict[int, FoundationTrainingDataset] = {}
    protected_datasets: dict[int, FoundationTrainingDataset] = {}
    b1_measurement = None
    b2_measurement = None
    b3_measurement = None
    b4_measurement = None
    b5_measurement = None
    if checkpoint_paths:
        if not args.b1_corpus:
            parser.error("--checkpoint requires --b1-corpus")
        if args.profile != "foundation":
            parser.error("--checkpoint evaluation requires --profile foundation")
        phase_b_seeds = _indexed_integer_values(
            args.b1_partition_seed,
            seeds=manifest.seeds,
            name="--b1-partition-seed",
        )
        protected_seeds = _indexed_integer_values(
            args.b1_protected_partition_seed,
            seeds=manifest.seeds,
            name="--b1-protected-partition-seed",
        )
        if not phase_b_seeds:
            parser.error("--checkpoint requires --b1-partition-seed")
        if args.b1_protected_corpus is None or not protected_seeds:
            parser.error(
                "--checkpoint requires --b1-protected-corpus and "
                "--b1-protected-partition-seed"
            )
        for seed in manifest.seeds:
            protected_dataset = FoundationTrainingDataset.from_jsonl(
                args.b1_protected_corpus,
                profile="foundation",
                partition_seed=protected_seeds[seed],
            )
            protected_datasets[seed] = protected_dataset
            b1_datasets[seed] = FoundationTrainingDataset.from_jsonl(
                args.b1_corpus,
                profile="foundation",
                partition_seed=phase_b_seeds[seed],
                exclude_dataset=protected_dataset,
            )
        if args.reuse_b1_report is not None:
            b1_measurement = _reuse_b1_measurement(
                args.reuse_b1_report,
                manifest=manifest,
                checkpoints=checkpoint_paths,
                datasets=b1_datasets,
            )
        else:
            b1_measurement = _evaluate_loaded_b1(
                checkpoint_paths,
                {seed: dataset.as_sequence_corpus() for seed, dataset in b1_datasets.items()},
                expected_dataset_digest={
                    seed: dataset.digest for seed, dataset in b1_datasets.items()
                },
                expected_protected_dataset_digest={
                    seed: str(dataset.excluded_dataset_digest)
                    for seed, dataset in b1_datasets.items()
                },
            )
        if args.child_foundation:
            b2_measurement = _evaluate_loaded_b2(checkpoint_paths)
            b3_measurement = _evaluate_loaded_b3(checkpoint_paths)
            b4_measurement = _evaluate_loaded_b4(checkpoint_paths)
        if args.b5_child:
            b5_measurement = _evaluate_loaded_b5(checkpoint_paths, protected_datasets)
    elif args.b1_corpus:
        if args.profile == "smoke":
            budgets = (4_096, 1_024, 1_024)
        else:
            budgets = (1_048_576, 131_072, 131_072)
        corpus = build_sequence_corpus(
            args.b1_corpus,
            train_bytes=budgets[0],
            holdout_bytes=budgets[1],
            retention_bytes=budgets[2],
            seed=manifest.seeds[0],
        )
        b1_measurement = SequencePredictionTask(
            _model_config(args.model_tier, manifest.seeds[0]),
            seeds=manifest.seeds,
            epochs=args.epochs,
        ).evaluate(corpus)
    if args.b2_smoke:
        b2_measurement = DelayedMemoryTask(
            _memory_config(manifest.seeds[0]),
            seeds=manifest.seeds,
        ).evaluate(build_delayed_memory_smoke_corpus())
    if args.b2_corpus:
        # Foundation-scale B2 on the M1-64 delayed-memory course.  The task's
        # own read path carries the M1-66 organ-first verdict for the full arm
        # and keeps the lesion arms on the original synthesis (M1-66b), so this
        # measurement is the real B2 result, not the smoke placeholder.
        from scripts.training.eval_taiji_m1_64_foundation_memory import (  # noqa: PLC0415
            build_foundation_delayed_memory_corpus,
        )

        b2_measurement = DelayedMemoryTask(
            _memory_config(manifest.seeds[0]),
            seeds=manifest.seeds,
        ).evaluate(build_foundation_delayed_memory_corpus())
    if args.b3_smoke:
        b3_measurement = WorldTransitionTask(
            seeds=manifest.seeds,
            epochs=10 if args.profile == "smoke" else 50,
        ).evaluate(build_world_transition_smoke_corpus())
    if args.b4_smoke:
        b4_measurement = GoalActionTask(
            _memory_config(manifest.seeds[0]),
            seeds=manifest.seeds,
        ).evaluate(build_goal_action_smoke_corpus())
    if args.b5_smoke:
        b5_measurement = ContinualLearningTask(
            _memory_config(manifest.seeds[0]),
            seeds=manifest.seeds,
            epochs=args.epochs,
        ).evaluate(build_continual_learning_smoke_corpus())
    evaluation = build_contract_report(
        manifest,
        checkpoint_gate_status=checkpoint_status,
        b1_measurement=b1_measurement,
        b2_measurement=b2_measurement,
        b3_measurement=b3_measurement,
        b4_measurement=b4_measurement,
        b5_measurement=b5_measurement,
    )
    result = evaluation.to_payload()
    result["manifest_path"] = str(args.manifest)
    result["contract_status"] = "validated"
    measured = [
        ability_id
        for ability_id, measurement in (
            ("b1_sequence_prediction", b1_measurement),
            ("b2_delayed_memory", b2_measurement),
            ("b3_world_transition", b3_measurement),
            ("b4_goal_action", b4_measurement),
            ("b5_continual_learning", b5_measurement),
        )
        if measurement is not None
    ]
    result["capability_measurements"] = "; ".join(measured) if measured else "not_evaluated"
    result["profile"] = args.profile
    result["model_tier"] = (
        "joint-child-v4" if checkpoint_paths else args.model_tier if b1_measurement is not None else None
    )
    result["code_revision"] = _code_revision()
    result["checkpoint_evaluation"] = {
        "mode": bool(checkpoint_paths),
        "checkpoints": [
            {"seed": seed, "path": str(path)}
            for seed, path in sorted(checkpoint_paths.items())
        ],
        "child_foundation": bool(args.child_foundation),
        "b5_child": bool(args.b5_child),
        "b1_reused_report": str(args.reuse_b1_report) if args.reuse_b1_report else None,
        "b1_dataset_digests": {
            str(seed): dataset.digest for seed, dataset in sorted(b1_datasets.items())
        },
        "b1_protected_dataset_digests": {
            str(seed): dataset.excluded_dataset_digest
            for seed, dataset in sorted(b1_datasets.items())
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["report_written"] = args.report.is_file()
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["report_written"] and result["contract_status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
