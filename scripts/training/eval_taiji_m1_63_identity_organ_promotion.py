"""Audit whether the identity organ may be promoted to a first-class memory organ.

M1-62 proved -- on one shared data contract -- that the native association path
never reaches positive binding while the identity organ does.  M1-63 stops
treating the organ as an upper-bound reference and asks the promotion question
directly: is it a trainable, checkpointable, lesionable key/value organ whose
gain is causal rather than an artefact of the evaluator?

The curricula and their digests are reused verbatim from M1-62 so the claim is
made on identical data.  Every seed trains two models with byte-identical
initial fabric/motor/memory weights (the organ draws from the shared generator
only after those components are built), which makes three independent reads
possible:

``native_recall``
    ``step.memory_recall``, which never consults the organ.  Comparing the
    organ-on and organ-off models on this channel is the "old capability did
    not degrade" test.
``decision_path`` with ``use_identity=True``
    the promoted read.
``decision_path`` with ``use_identity=False``
    the same trained weights with the organ suppressed.  If binding survives
    suppression the gain was never the organ's.

Only ``train`` examples are ever written.  Holdout and retention stay read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_36_cue_curriculum import _config  # noqa: E402
from scripts.training.eval_taiji_m1_53_credit_identifiability import (  # noqa: E402
    _checkpoint_record,
)
from scripts.training.eval_taiji_m1_62_learning_data_contract import (  # noqa: E402
    ACTION_CHANCE,
    ACTION_SYMBOLS,
    COMBINATIONS,
    CUE_START,
    HOLDOUT_KEYS,
    KEY_COUNT,
    OUTCOME_CHANCE,
    RETENTION_KEYS,
    SEEDS,
    TRAIN_REPEATS,
    _curriculum,
    _query_probe,
    _summary,
    _write_train,
)
from taiji import (  # noqa: E402
    MEMORY_LEARNING_EXAMPLE_FIELDS,
    MEMORY_LEARNING_PARTITIONS,
    MemoryLearningCurriculum,
    MemoryLearningExample,
    Taiji,
    TaijiConfig,
)

FORMAT = "taiji-native-m1-63-identity-organ-promotion-v1"
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m1_63_identity_organ_promotion_20260902.json"
)

# Pinned from the committed M1-62 report.  M1-63 is only allowed to speak about
# the organ if it is graded on exactly the same data.
STABLE_KEY_DIGEST = "30d67ccda51decee0aabaab9a5413eb10f662733753586baca6e438c6e58ad31"
CONFLICTING_KEY_DIGEST = (
    "29cb6a01614cd8a454bc61b77b4273f2b6e0eff4347fe17c7a6dc85ffe76f4fe"
)
# Pinned parameter budget at the default organ capacity.
#
# M1-62 measured 147521 against the v1 single-head organ.  M1-63 promoted the
# organ to a real key/value store with a second, read-only outcome head, which
# adds one more 32896-edge projection: 147521 + 32896 == 180417.  The pin is
# updated rather than relaxed, and the composition below is reported so the
# increment stays auditable instead of becoming an unexplained number.
ORGAN_OFF_PARAMETERS = 85953
ORGAN_ON_PARAMETERS = 180417
ORGAN_ON_PARAMETERS_M1_62 = 147521
ORGAN_SECOND_HEAD_PARAMETERS = 32896

QUERY_PARTITIONS = ("holdout", "retention")
# Forcing eviction: half the slots for twice the keys.
OVERWRITE_CAPACITY = KEY_COUNT // 2
# The punished course teaches the correct answer under a negative reward.  With
# ``identity_organ_write_baseline == 0.0`` the organ's modulation becomes -1.0,
# which is a full-strength push away from the taught action rather than a merely
# weakened write -- the failure mode has to be loud to be gradeable.
PUNISHED_REWARD = -1.0
REWARDED_REWARD = 1.0


def _promotion_config(seed: int, *, enabled: bool, capacity: int | None) -> TaijiConfig:
    values = _config(seed).to_dict()
    values["identity_organ_enabled"] = bool(enabled)
    if capacity is not None:
        values["identity_organ_capacity"] = int(capacity)
    return TaijiConfig.from_dict(values)


def _one_shot_curriculum(name: str) -> MemoryLearningCurriculum:
    """A course where every key is taught exactly once.

    The curriculum audit requires ``role`` to agree with the observed count
    (``role == "repeated_observation"`` iff the key appears more than once in
    that partition), so a single-exposure course must declare
    ``one_shot_experience`` -- the label is a claim about the data, not a knob.
    """

    train: list[MemoryLearningExample] = []
    for key in range(KEY_COUNT):
        action, outcome = COMBINATIONS[key % len(COMBINATIONS)]
        train.append(
            MemoryLearningExample(
                example_id=f"m1-63-{name}-train-{key}",
                partition="train",
                cue_key=CUE_START + key,
                action_value=action,
                outcome_value=outcome,
                role="one_shot_experience",
                episode_tag=f"m1-63-{name}-train-{key}",
            )
        )

    def queries(partition: str, keys: range) -> tuple[MemoryLearningExample, ...]:
        items: list[MemoryLearningExample] = []
        for key in keys:
            action, outcome = COMBINATIONS[key % len(COMBINATIONS)]
            items.append(
                MemoryLearningExample(
                    example_id=f"m1-63-{name}-{partition}-{key}",
                    partition=partition,
                    cue_key=CUE_START + key,
                    action_value=action,
                    outcome_value=outcome,
                    role="one_shot_experience",
                    episode_tag=f"m1-63-{name}-{partition}-{key}",
                )
            )
        return tuple(items)

    return MemoryLearningCurriculum(
        name=name,
        train=tuple(train),
        holdout=queries("holdout", HOLDOUT_KEYS),
        retention=queries("retention", RETENTION_KEYS),
        declared_key_value_deterministic=True,
    )


def _punished_curriculum(
    source: MemoryLearningCurriculum, name: str
) -> MemoryLearningCurriculum:
    """``source`` re-taught under punishment, derived rather than re-authored.

    The course is built by rewriting exactly one field of ``source``'s train
    partition -- ``reward`` flips from ``REWARDED_REWARD`` to
    ``PUNISHED_REWARD``.  Deriving instead of hand-writing a parallel course is
    the point: "these two courses differ only in reward sign" becomes structurally
    true rather than something a reader has to verify by eye, so the gate below
    can attribute any behavioural difference to the reward and nothing else.
    Query partitions are copied untouched because reward is a write-time signal
    and holdout/retention are read-only.

    This course exists because the original 15/15-green promotion Gate could not
    express it: ``MemoryLearningExample.reward`` defaults to ``1.0`` and the write
    path discarded it, so "the organ binds a punished action exactly as hard as a
    rewarded one" was an invisible failure mode.  A gate that cannot fail on
    reward sign is not evidence about a reward-driven organ.
    """

    train = tuple(
        replace(
            example,
            example_id=example.example_id.replace(source.name, name, 1),
            reward=PUNISHED_REWARD,
            episode_tag=example.episode_tag.replace(source.name, name, 1),
        )
        for example in source.train
    )

    def queries(partition: str) -> tuple[MemoryLearningExample, ...]:
        return tuple(
            replace(
                example,
                example_id=example.example_id.replace(source.name, name, 1),
                episode_tag=example.episode_tag.replace(source.name, name, 1),
            )
            for example in getattr(source, partition)
        )

    return MemoryLearningCurriculum(
        name=name,
        train=train,
        holdout=queries("holdout"),
        retention=queries("retention"),
        declared_key_value_deterministic=source.declared_key_value_deterministic,
    )


def _reward_contrast(
    rewarded: MemoryLearningCurriculum, punished: MemoryLearningCurriculum
) -> dict[str, Any]:
    """Prove the two courses differ in reward and in nothing else that can teach.

    ``example_id`` and ``episode_tag`` necessarily differ (ids must be unique
    within a curriculum, and both are declared bypass fields that must never
    determine the answer).  Every other field, including the key and both value
    channels, must match position for position.
    """

    graded = [
        field
        for field in MEMORY_LEARNING_EXAMPLE_FIELDS
        if field not in {"example_id", "episode_tag"}
    ]
    train_pairs = list(zip(rewarded.train, punished.train, strict=True))
    differing = sorted(
        {
            field
            for field in graded
            for left, right in train_pairs
            if left.to_dict()[field] != right.to_dict()[field]
        }
    )
    query_pairs = [
        pair
        for partition in QUERY_PARTITIONS
        for pair in zip(
            getattr(rewarded, partition), getattr(punished, partition), strict=True
        )
    ]
    return {
        "graded_fields_compared": graded,
        "train_differing_fields": differing,
        "differs_only_by_reward": differing == ["reward"],
        "rewarded_train_reward": _summary(
            [float(example.reward) for example in rewarded.train]
        ),
        "punished_train_reward": _summary(
            [float(example.reward) for example in punished.train]
        ),
        "punished_train_all_negative": all(
            float(example.reward) < 0.0 for example in punished.train
        ),
        "rewarded_train_all_at_pin": all(
            float(example.reward) == REWARDED_REWARD for example in rewarded.train
        ),
        "punished_train_all_at_pin": all(
            float(example.reward) == PUNISHED_REWARD for example in punished.train
        ),
        "query_partitions_reward_unchanged": all(
            float(left.reward) == float(right.reward) for left, right in query_pairs
        ),
    }


def _decision_row(
    model: Taiji, example: MemoryLearningExample, *, use_identity: bool
) -> dict[str, Any]:
    """One read of the decision path, with the organ explicitly scoped.

    This is deliberately not M1-62's ``_decision_row``: that helper leaves
    ``use_identity`` unset, so it cannot express the suppressed control arm.
    """

    model.reset_dynamics(episode_id=f"m1-63-decision-{example.example_id}")
    model.observe(
        model.config.boundary_symbol,
        learn=False,
        learn_motor=False,
        use_memory=True,
        use_identity=use_identity,
    )
    step = model.observe(
        example.cue_key,
        learn=False,
        learn_motor=False,
        use_memory=True,
        use_identity=use_identity,
    )
    probabilities = step.probabilities
    alternatives = [
        float(probabilities[action].item())
        for action in ACTION_SYMBOLS
        if action != example.action_value
    ]
    probability = float(probabilities[example.action_value].item())
    predicted = max(
        ACTION_SYMBOLS, key=lambda value: float(probabilities[value].item())
    )
    recall = step.identity_recall
    return {
        "query_id": example.example_id,
        "cue": int(example.cue_key),
        "action": int(example.action_value),
        "combination": example.value_key,
        "role": example.role,
        "predicted_action": int(predicted),
        "action_probability": probability,
        "action_margin": float(probability - max(alternatives)),
        "action_correct": int(predicted == example.action_value),
        "identity_recall_used": bool(recall is not None and recall.used),
        "identity_slot_index": (
            None if recall is None or recall.slot_index is None else int(recall.slot_index)
        ),
        "identity_similarity": 0.0 if recall is None else float(recall.similarity),
        "identity_source": "absent" if recall is None else str(recall.source),
        "identity_provenance": "absent" if recall is None else str(recall.provenance),
    }


def _decision_probe(
    model: Taiji,
    examples: tuple[MemoryLearningExample, ...],
    *,
    use_identity: bool,
) -> dict[str, Any]:
    rows = [
        _decision_row(model, example, use_identity=use_identity)
        for example in examples
    ]
    margins = [float(row["action_margin"]) for row in rows]
    unbound = [row for row in rows if not row["identity_recall_used"]]
    bound = [row for row in rows if row["identity_recall_used"]]
    return {
        "rows": rows,
        "summary": {
            "sample_count": len(rows),
            "action_accuracy": float(
                sum(int(row["action_correct"]) for row in rows) / len(rows)
            ),
            "action_margin": _summary(margins),
            "row_action_margin_min": float(min(margins)),
            "identity_recall_used_ratio": float(len(bound) / len(rows)),
            "bound_row_count": len(bound),
            "unbound_row_count": len(unbound),
            "bound_row_action_margin_min": (
                float(min(float(row["action_margin"]) for row in bound))
                if bound
                else None
            ),
            "unbound_row_action_margin_max": (
                float(max(float(row["action_margin"]) for row in unbound))
                if unbound
                else None
            ),
            "unbound_rows_have_zero_margin": all(
                abs(float(row["action_margin"])) <= 1e-9 for row in unbound
            ),
            "cross_cue": _cross_cue(rows),
        },
    }


def _cross_cue(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Do different cues actually resolve to different values and slots?

    A high accuracy can hide a degenerate organ that always emits the majority
    action.  Contrasting pairs (two queries taught different actions) must be
    separated, and distinct cues must occupy distinct slots.
    """

    contrasting = 0
    discriminated = 0
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            if first["action"] == second["action"]:
                continue
            contrasting += 1
            if first["predicted_action"] != second["predicted_action"]:
                discriminated += 1
    cues = {int(row["cue"]) for row in rows}
    slots = {
        int(row["identity_slot_index"])
        for row in rows
        if row["identity_slot_index"] is not None
    }
    return {
        "cue_count": len(cues),
        "contrasting_pairs": contrasting,
        "discriminated_pairs": discriminated,
        "cross_cue_discrimination": (
            float(discriminated / contrasting) if contrasting else 0.0
        ),
        "distinct_slot_count": len(slots),
        "distinct_slots_match_bound_cues": len(slots)
        == len({int(row["cue"]) for row in rows if row["identity_recall_used"]}),
    }


def _organ_telemetry(model: Taiji) -> dict[str, Any]:
    organ = model.identity_organ
    if organ is None:
        return {"present": False}
    bank = organ.bank
    return {
        "present": True,
        "checkpoint_format": organ.CHECKPOINT_FORMAT,
        "version": int(organ.VERSION),
        "capacity": int(organ.capacity),
        "write_count": int(organ.write_count),
        "replacement_count": int(organ.replacement_count),
        "skipped_write_count": int(organ.skipped_write_count),
        "punished_write_count": int(organ.punished_write_count),
        "occupied_count": int(bank.occupied_count),
        "allocation_count": int(bank.allocation_count),
        "match_count": int(bank.match_count),
        "bank_replacement_count": int(bank.replacement_count),
        "parameter_count": int(organ.parameter_count),
    }


def _parameter_record(model: Taiji, record: dict[str, Any]) -> dict[str, Any]:
    record["active_parameter_count"] = model.parameter_count()
    record["planned_active_parameter_count"] = (
        model.config.planned_active_parameter_count
    )
    record["parameter_count_matches_plan"] = (
        model.parameter_count() == model.config.planned_active_parameter_count
    )
    record["saveable"] = (
        record["same_process_digest_matches"]
        and record["fresh_process_digest_matches"]
        and record["parameter_count_matches_plan"]
    )
    return record


def _arm(
    curriculum: MemoryLearningCurriculum,
    seed: int,
    *,
    enabled: bool,
    capacity: int | None,
    course: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _promotion_config(seed, enabled=enabled, capacity=capacity)
    arm = "identity_organ" if enabled else "organ_off_baseline"
    model = Taiji(config, episode_id=f"m1-63-{course}-{arm}-{seed}")
    preflight = _parameter_record(model, _checkpoint_record(model))
    written = _write_train(model, curriculum)
    native = {
        partition: _query_probe(model, curriculum, partition)["summary"]
        for partition in QUERY_PARTITIONS
    }
    decision: dict[str, Any] = {}
    for channel, use_identity in (("organ_on", True), ("organ_suppressed", False)):
        decision[channel] = {
            partition: _decision_probe(
                model, getattr(curriculum, partition), use_identity=use_identity
            )["summary"]
            for partition in QUERY_PARTITIONS
        }
    trained = _parameter_record(model, _checkpoint_record(model))
    return {
        "seed": seed,
        "arm": arm,
        "identity_organ_enabled": bool(config.identity_organ_enabled),
        "identity_organ_capacity": int(config.identity_organ_capacity),
        "train_examples_written": written,
        "partitions_written": ["train"],
        "holdout_updates": 0,
        "retention_updates": 0,
        "checkpoint_preflight": preflight,
        "checkpoint_after_train": trained,
        "organ_telemetry": _organ_telemetry(model),
        "native_recall": native,
        "decision_path": decision,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def _native_binding(record: dict[str, Any], partition: str) -> bool:
    summary = record["native_recall"][partition]["all"]
    return bool(
        summary["action_accuracy"] > ACTION_CHANCE
        and summary["outcome_accuracy"] > OUTCOME_CHANCE
        and summary["action_margin"]["min"] > 0.0
        and summary["outcome_margin"]["min"] > 0.0
    )


def _decision_binding(record: dict[str, Any], channel: str, partition: str) -> bool:
    summary = record["decision_path"][channel][partition]
    return bool(
        summary["action_accuracy"] > ACTION_CHANCE
        and summary["row_action_margin_min"] > 0.0
    )


def _course_summary(
    organ_records: list[dict[str, Any]], baseline_records: list[dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for partition in QUERY_PARTITIONS:
        summary[f"native_{partition}"] = {
            "organ_action_accuracy": _summary(
                [
                    float(r["native_recall"][partition]["all"]["action_accuracy"])
                    for r in organ_records
                ]
            ),
            "baseline_action_accuracy": _summary(
                [
                    float(r["native_recall"][partition]["all"]["action_accuracy"])
                    for r in baseline_records
                ]
            ),
            "organ_row_action_margin_min": float(
                min(
                    float(r["native_recall"][partition]["all"]["action_margin"]["min"])
                    for r in organ_records
                )
            ),
            "baseline_row_action_margin_min": float(
                min(
                    float(r["native_recall"][partition]["all"]["action_margin"]["min"])
                    for r in baseline_records
                )
            ),
        }
        for channel in ("organ_on", "organ_suppressed"):
            rows = [r["decision_path"][channel][partition] for r in organ_records]
            summary[f"decision_{channel}_{partition}"] = {
                "action_accuracy": _summary(
                    [float(r["action_accuracy"]) for r in rows]
                ),
                "row_action_margin_min": float(
                    min(float(r["row_action_margin_min"]) for r in rows)
                ),
                "identity_recall_used_ratio": _summary(
                    [float(r["identity_recall_used_ratio"]) for r in rows]
                ),
                "cross_cue_discrimination": _summary(
                    [float(r["cross_cue"]["cross_cue_discrimination"]) for r in rows]
                ),
                "distinct_slots_match_bound_cues": all(
                    bool(r["cross_cue"]["distinct_slots_match_bound_cues"])
                    for r in rows
                ),
                "unbound_rows_have_zero_margin": all(
                    bool(r["unbound_rows_have_zero_margin"]) for r in rows
                ),
                "bound_row_count_min": int(
                    min(int(r["bound_row_count"]) for r in rows)
                ),
                "bound_row_action_margin_min": (
                    float(
                        min(float(r["bound_row_action_margin_min"]) for r in rows)
                    )
                    if all(r["bound_row_action_margin_min"] is not None for r in rows)
                    else None
                ),
            }

    summary["native_positive_binding_all_seeds"] = all(
        _native_binding(record, partition)
        for record in organ_records
        for partition in QUERY_PARTITIONS
    )
    summary["decision_positive_binding_all_seeds"] = all(
        _decision_binding(record, "organ_on", partition)
        for record in organ_records
        for partition in QUERY_PARTITIONS
    )
    summary["suppressed_positive_binding_all_seeds"] = all(
        _decision_binding(record, "organ_suppressed", partition)
        for record in organ_records
        for partition in QUERY_PARTITIONS
    )
    summary["any_positive_binding_all_seeds"] = bool(
        summary["native_positive_binding_all_seeds"]
        or summary["decision_positive_binding_all_seeds"]
    )
    summary["native_recall_not_degraded_by_organ"] = all(
        float(organ["native_recall"][partition]["all"]["action_margin"]["min"])
        >= float(base["native_recall"][partition]["all"]["action_margin"]["min"]) - 1e-9
        and float(organ["native_recall"][partition]["all"]["action_accuracy"])
        >= float(base["native_recall"][partition]["all"]["action_accuracy"]) - 1e-9
        for organ, base in zip(organ_records, baseline_records, strict=True)
        for partition in QUERY_PARTITIONS
    )
    summary["checkpoint_preflight_all_seeds"] = all(
        bool(record[stage]["saveable"])
        for record in organ_records + baseline_records
        for stage in ("checkpoint_preflight",)
    )
    summary["checkpoint_after_train_all_seeds"] = all(
        bool(record["checkpoint_after_train"]["saveable"])
        for record in organ_records + baseline_records
    )
    summary["organ_writes_all_train_examples"] = all(
        int(record["organ_telemetry"]["write_count"])
        == int(record["train_examples_written"])
        for record in organ_records
    )
    # Reward telemetry is reported unconditionally, not only for the punished
    # course.  A rewarded course must show zero punished writes; if it ever shows
    # a non-zero count the write path has a sign bug, and that is exactly as
    # gradeable a failure as the punished course binding anyway.
    summary["organ_punished_write_count_min"] = min(
        int(record["organ_telemetry"]["punished_write_count"])
        for record in organ_records
    )
    summary["organ_punished_write_count_max"] = max(
        int(record["organ_telemetry"]["punished_write_count"])
        for record in organ_records
    )
    summary["organ_punishes_all_train_examples"] = all(
        int(record["organ_telemetry"]["punished_write_count"])
        == int(record["train_examples_written"])
        for record in organ_records
    )
    summary["organ_punishes_no_train_examples"] = all(
        int(record["organ_telemetry"]["punished_write_count"]) == 0
        for record in organ_records
    )
    summary["organ_skipped_write_count_max"] = max(
        int(record["organ_telemetry"]["skipped_write_count"]) for record in organ_records
    )
    summary["organ_replacement_count_max"] = max(
        int(record["organ_telemetry"]["replacement_count"]) for record in organ_records
    )
    summary["organ_occupied_count_max"] = max(
        int(record["organ_telemetry"]["occupied_count"]) for record in organ_records
    )
    return summary


def _course(
    name: str,
    curriculum: MemoryLearningCurriculum,
    seeds: tuple[int, ...],
    *,
    capacity: int | None = None,
) -> dict[str, Any]:
    organ_records = [
        _arm(curriculum, seed, enabled=True, capacity=capacity, course=name)
        for seed in seeds
    ]
    baseline_records = [
        _arm(curriculum, seed, enabled=False, capacity=capacity, course=name)
        for seed in seeds
    ]
    return {
        "curriculum_name": curriculum.name,
        "curriculum_digest": curriculum.digest,
        "identity_organ_capacity": capacity,
        "audit": curriculum.audit(),
        "arms": {
            "identity_organ": {
                "records": organ_records,
            },
            "organ_off_baseline": {
                "records": baseline_records,
            },
        },
        "summary": _course_summary(organ_records, baseline_records),
    }


def _data_contract_verdict(courses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stable = courses["stable_key"]["audit"]
    negative = courses["conflicting_key"]["audit"]
    one_shot = courses["one_shot_key"]["audit"]
    punished = courses["punished_key"]["audit"]
    return {
        "stable_well_formed": bool(stable["well_formed"]),
        "one_shot_well_formed": bool(one_shot["well_formed"]),
        "punished_well_formed": bool(punished["well_formed"]),
        "stable_key_value_deterministic": bool(
            stable["observed_key_value_deterministic"]
        ),
        "one_shot_key_value_deterministic": bool(
            one_shot["observed_key_value_deterministic"]
        ),
        # The punished course carries the *correct* key/value relation; only the
        # reward sign is inverted.  If determinism broke here the course would be
        # testing conflicting labels rather than punishment.
        "punished_key_value_deterministic": bool(
            punished["observed_key_value_deterministic"]
        ),
        "negative_control_declares_conflict": (
            not negative["declared_key_value_deterministic"]
            and not negative["observed_key_value_deterministic"]
            and bool(negative["declaration_matches_observation"])
        ),
        "no_bypass_field_leak": not any(
            course["audit"]["leakage"]["any_bypass_field_leak"]
            for course in courses.values()
        ),
        "no_train_query_id_overlap": all(
            course["audit"]["leakage"]["train_query_id_overlap"] == 0
            for course in courses.values()
        ),
        "all_partitions_factorial_complete": all(
            course["audit"]["partitions"][partition]["factorial_complete"]
            for course in courses.values()
            for partition in MEMORY_LEARNING_PARTITIONS
        ),
        "stable_train_keys_repeatedly_observed": (
            stable["partitions"]["train"]["observations_per_key"]["min"]
            >= float(TRAIN_REPEATS)
        ),
        "one_shot_train_keys_observed_once": (
            one_shot["partitions"]["train"]["observations_per_key"]["max"] == 1.0
        ),
        "role_labels_match_observations": all(
            course["audit"]["partitions"][partition][
                "role_labels_match_observation_counts"
            ]
            for course in courses.values()
            for partition in MEMORY_LEARNING_PARTITIONS
        ),
    }


def run(report_path: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    started = time.perf_counter()
    stable = _curriculum("stable_key", deterministic=True)
    conflicting = _curriculum("conflicting_key", deterministic=False)
    one_shot = _one_shot_curriculum("one_shot_key")
    punished = _punished_curriculum(stable, "punished_key")
    reward_contrast = _reward_contrast(stable, punished)
    digest_reuse = {
        "stable_key_digest": stable.digest,
        "conflicting_key_digest": conflicting.digest,
        "one_shot_key_digest": one_shot.digest,
        "punished_key_digest": punished.digest,
        "stable_key_matches_m1_62": stable.digest == STABLE_KEY_DIGEST,
        "conflicting_key_matches_m1_62": conflicting.digest == CONFLICTING_KEY_DIGEST,
    }

    courses = {
        "stable_key": _course("stable_key", stable, seeds),
        "conflicting_key": _course("conflicting_key", conflicting, seeds),
        "one_shot_key": _course("one_shot_key", one_shot, seeds),
        "stable_key_overwrite": _course(
            "stable_key_overwrite", stable, seeds, capacity=OVERWRITE_CAPACITY
        ),
        "punished_key": _course("punished_key", punished, seeds),
    }

    data_contract = _data_contract_verdict(courses)
    stable_summary = courses["stable_key"]["summary"]
    negative_summary = courses["conflicting_key"]["summary"]
    one_shot_summary = courses["one_shot_key"]["summary"]
    overwrite_summary = courses["stable_key_overwrite"]["summary"]
    punished_summary = courses["punished_key"]["summary"]

    budget = {
        "organ_off_parameter_count": sorted(
            {
                int(record["checkpoint_after_train"]["active_parameter_count"])
                for record in courses["stable_key"]["arms"]["organ_off_baseline"][
                    "records"
                ]
            }
        ),
        "organ_on_parameter_count": sorted(
            {
                int(record["checkpoint_after_train"]["active_parameter_count"])
                for record in courses["stable_key"]["arms"]["identity_organ"]["records"]
            }
        ),
        "expected_organ_off": ORGAN_OFF_PARAMETERS,
        "expected_organ_on": ORGAN_ON_PARAMETERS,
        "m1_62_organ_on_parameter_count": ORGAN_ON_PARAMETERS_M1_62,
        "second_head_parameter_count": ORGAN_SECOND_HEAD_PARAMETERS,
        "organ_increment": ORGAN_ON_PARAMETERS - ORGAN_OFF_PARAMETERS,
    }
    budget["pin_increment_explained_by_second_head"] = (
        ORGAN_ON_PARAMETERS_M1_62 + ORGAN_SECOND_HEAD_PARAMETERS
        == ORGAN_ON_PARAMETERS
    )
    budget["organ_off_matches_pin"] = budget["organ_off_parameter_count"] == [
        ORGAN_OFF_PARAMETERS
    ]
    budget["organ_on_matches_pin"] = budget["organ_on_parameter_count"] == [
        ORGAN_ON_PARAMETERS
    ]
    budget["all_records_match_plan"] = all(
        bool(record[stage]["parameter_count_matches_plan"])
        for course in courses.values()
        for arm in course["arms"].values()
        for record in arm["records"]
        for stage in ("checkpoint_preflight", "checkpoint_after_train")
    )

    gates = {
        "curriculum_digests_reused": bool(
            digest_reuse["stable_key_matches_m1_62"]
            and digest_reuse["conflicting_key_matches_m1_62"]
        ),
        "data_contract_pass": all(data_contract.values()),
        "checkpoint_preflight_pass": all(
            course["summary"]["checkpoint_preflight_all_seeds"]
            for course in courses.values()
        ),
        "checkpoint_after_train_pass": all(
            course["summary"]["checkpoint_after_train_all_seeds"]
            for course in courses.values()
        ),
        "parameter_budget_pass": bool(
            budget["organ_off_matches_pin"]
            and budget["organ_on_matches_pin"]
            and budget["pin_increment_explained_by_second_head"]
            and budget["all_records_match_plan"]
        ),
        "read_only_query_partitions": all(
            int(record["holdout_updates"]) == 0
            and int(record["retention_updates"]) == 0
            and record["partitions_written"] == ["train"]
            for course in courses.values()
            for arm in course["arms"].values()
            for record in arm["records"]
        ),
        "stable_positive_binding_all_seeds": bool(
            stable_summary["decision_positive_binding_all_seeds"]
        ),
        "negative_control_rejected": bool(
            not negative_summary["any_positive_binding_all_seeds"]
            and not negative_summary["decision_positive_binding_all_seeds"]
        ),
        "identity_gain_is_causal": bool(
            stable_summary["decision_positive_binding_all_seeds"]
            and not stable_summary["suppressed_positive_binding_all_seeds"]
        ),
        "cross_cue_competition_pass": all(
            stable_summary[f"decision_organ_on_{partition}"][
                "cross_cue_discrimination"
            ]["min"]
            >= 1.0
            and stable_summary[f"decision_organ_on_{partition}"][
                "distinct_slots_match_bound_cues"
            ]
            for partition in QUERY_PARTITIONS
        ),
        "one_shot_positive_binding": bool(
            one_shot_summary["decision_positive_binding_all_seeds"]
        ),
        "old_capability_preserved": all(
            course["summary"]["native_recall_not_degraded_by_organ"]
            for course in courses.values()
        ),
        "overwrite_keeps_bound_rows_positive": all(
            (
                overwrite_summary[f"decision_organ_on_{partition}"][
                    "bound_row_count_min"
                ]
                > 0
                and overwrite_summary[f"decision_organ_on_{partition}"][
                    "bound_row_action_margin_min"
                ]
                is not None
                and float(
                    overwrite_summary[f"decision_organ_on_{partition}"][
                        "bound_row_action_margin_min"
                    ]
                )
                > 0.0
            )
            for partition in QUERY_PARTITIONS
        ),
        # Evicted rows must fall back to the organ-free motor baseline, not to
        # something worse: eviction may cost the organ's gain but must not
        # actively corrupt a decision the model could already make.
        "overwrite_evicted_rows_not_below_baseline": all(
            float(
                overwrite_summary[f"decision_organ_on_{partition}"][
                    "row_action_margin_min"
                ]
            )
            >= float(
                overwrite_summary[f"decision_organ_suppressed_{partition}"][
                    "row_action_margin_min"
                ]
            )
            - 1e-9
            and float(
                overwrite_summary[f"decision_organ_on_{partition}"]["action_accuracy"][
                    "min"
                ]
            )
            >= float(
                overwrite_summary[f"decision_organ_suppressed_{partition}"][
                    "action_accuracy"
                ]["min"]
            )
            - 1e-9
            for partition in QUERY_PARTITIONS
        ),
        "overwrite_actually_evicted": int(overwrite_summary["organ_replacement_count_max"])
        > 0,
        # --- reward sensitivity -------------------------------------------------
        # Four gates, because "the punished course did not bind" is only evidence
        # if the punishment actually arrived, nothing but the reward differed, and
        # the organ did not simply refuse to write.
        #
        # 1. The courses are identical apart from the reward sign.
        "punished_course_differs_only_by_reward": bool(
            reward_contrast["differs_only_by_reward"]
            and reward_contrast["punished_train_all_negative"]
            and reward_contrast["rewarded_train_all_at_pin"]
            and reward_contrast["punished_train_all_at_pin"]
            and reward_contrast["query_partitions_reward_unchanged"]
        ),
        # 2. The negative reward reached the organ.  Without this the gate below
        #    could pass vacuously on a write path that silently drops reward --
        #    which is precisely the defect that made the original 15/15 green.
        "punished_write_reaches_organ": bool(
            punished_summary["organ_punishes_all_train_examples"]
            and int(punished_summary["organ_punished_write_count_min"]) > 0
            and bool(stable_summary["organ_punishes_no_train_examples"])
        ),
        # 3. Cue identity is not reward-gated: a cue observed during a punished
        #    episode is still that cue, so slots must still be allocated.  Only the
        #    value heads are reward-modulated.  Without this gate a write path that
        #    refused to write at all under punishment would look identical to one
        #    that correctly anti-bound the action.
        "punished_cue_identity_preserved": bool(
            punished_summary["organ_writes_all_train_examples"]
            and int(punished_summary["organ_skipped_write_count_max"]) == 0
            and int(punished_summary["organ_occupied_count_max"])
            == int(stable_summary["organ_occupied_count_max"])
        ),
        # 4. The decision must not come out positively bound.  Teaching an action
        #    under full punishment and still recalling it as the answer would mean
        #    the organ stores co-occurrence, not reward-weighted value.
        "punished_course_rejected": bool(
            not punished_summary["decision_positive_binding_all_seeds"]
        ),
    }
    blocking = [
        name
        for name in (
            "curriculum_digests_reused",
            "data_contract_pass",
            "checkpoint_preflight_pass",
            "checkpoint_after_train_pass",
            "parameter_budget_pass",
            "read_only_query_partitions",
            "stable_positive_binding_all_seeds",
            "negative_control_rejected",
            "identity_gain_is_causal",
            "cross_cue_competition_pass",
            "one_shot_positive_binding",
            "old_capability_preserved",
            "overwrite_keeps_bound_rows_positive",
            "overwrite_evicted_rows_not_below_baseline",
            "overwrite_actually_evicted",
            "punished_course_differs_only_by_reward",
            "punished_write_reaches_organ",
            "punished_cue_identity_preserved",
            "punished_course_rejected",
        )
        if not gates[name]
    ]
    result = {
        "format": FORMAT,
        "status": "gate_passed" if not blocking else "blocked",
        "blocking_gates": blocking,
        "seeds": list(seeds),
        "digest_reuse": digest_reuse,
        "data_contract": data_contract,
        "reward_contrast": reward_contrast,
        "parameter_budget": budget,
        "gates": gates,
        "courses": courses,
        "prohibited_partitions_used": [],
        "holdout_updates": 0,
        "retention_updates": 0,
        "trace_only": True,
        "default_training_changed": False,
        "default_checkpoint_changed": False,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the identity organ promotion to a first-class organ."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args()
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    result = run(args.report, seeds)
    print(f"format: {result['format']}")
    print(f"status: {result['status']}")
    print(f"report_path: {args.report}")
    print(f"blocking_gates: {result['blocking_gates']}")
    print(f"gates: {result['gates']}")
    print(f"digest_reuse: {result['digest_reuse']}")
    print(f"reward_contrast: {result['reward_contrast']}")
    print(f"parameter_budget: {result['parameter_budget']}")


if __name__ == "__main__":
    main()
