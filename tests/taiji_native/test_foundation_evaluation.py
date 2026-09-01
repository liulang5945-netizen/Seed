from __future__ import annotations

from pathlib import Path

import pytest

from taiji.foundation_evaluation import (
    FOUNDATION_BASELINE_KINDS,
    FOUNDATION_REQUIRED_ABILITIES,
    FoundationEvaluation,
    FoundationManifest,
    FoundationMeasurement,
)


def _manifest() -> FoundationManifest:
    return FoundationManifest.from_payload(
        {
            "format": "taiji-foundation-baseline-v1",
            "version": 1,
            "split_policy": "source-disjoint",
            "partitions": ["train", "holdout", "retention", "control"],
            "seeds": [11, 29, 47],
            "baselines": [*FOUNDATION_BASELINE_KINDS, "taiji"],
            "checkpoint_gate": "taiji-m0-checkpoint-preflight",
            "data_contract": {
                "required_fields": [
                    "sample_id",
                    "source",
                    "license",
                    "objective",
                    "partition",
                    "payload",
                    "target_or_outcome",
                    "provenance",
                    "taint",
                    "content_digest",
                ],
                "forbidden_sources": ["provider_answer"],
                "partition_rule": "source-disjoint",
                "holdout_rule": "read-only",
            },
            "baseline_protocol": {
                "required_controls": list(FOUNDATION_BASELINE_KINDS),
                "seeds_are_shared": True,
                "report_mean_std_worst_seed": True,
                "strictly_beats_strongest_control": True,
            },
            "tasks": [
                {
                    "ability_id": ability_id,
                    "name": ability_id,
                    "input_kind": "fixture",
                    "primary_metric": "accuracy",
                    "metric_direction": "higher_is_better",
                    "minimum_train_units": 1,
                    "minimum_holdout_units": 1,
                    "minimum_retention_units": 1,
                }
                for ability_id in FOUNDATION_REQUIRED_ABILITIES
            ],
            "report_schema": {
                "required_fields": [
                    "manifest_digest",
                    "checkpoint_gate_status",
                    "status",
                    "can_promote",
                    "failure_reasons",
                    "measurements",
                ],
                "measurement_required_fields": [
                    "ability_id",
                    "status",
                    "primary_metric",
                    "metric_direction",
                    "metric_value",
                    "baseline_metrics",
                    "sample_counts",
                    "holdout_updates",
                    "evidence",
                ],
            },
        }
    )


def test_repository_foundation_manifest_is_loadable_and_versioned() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "plans"
        / "manifests"
        / "taiji_foundation_baseline_v1.json"
    )

    manifest = FoundationManifest.load(manifest_path)

    assert manifest.format == "taiji-foundation-baseline-v1"
    assert manifest.version == 1
    assert tuple(task.ability_id for task in manifest.tasks) == FOUNDATION_REQUIRED_ABILITIES


def _measurement(
    ability_id: str,
    *,
    status: str = "passed",
    holdout_updates: int = 0,
) -> FoundationMeasurement:
    return FoundationMeasurement(
        ability_id=ability_id,
        status=status,
        primary_metric="accuracy",
        metric_direction="higher_is_better",
        metric_value=0.9,
        baseline_metrics={
            "random": 0.1,
            "frozen_parent": 0.2,
            "simple_rule": 0.3,
            "hash_only": 0.4,
        },
        sample_counts={"train": 100, "holdout": 20, "retention": 20},
        holdout_updates=holdout_updates,
        evidence=[f"fixture:{ability_id}"],
    )


def test_manifest_requires_exactly_the_five_foundation_abilities() -> None:
    payload = _manifest().to_payload()
    payload["tasks"] = payload["tasks"][:-1]

    with pytest.raises(ValueError, match="abilities"):
        FoundationManifest.from_payload(payload)


def test_manifest_rejects_duplicate_ability_ids() -> None:
    payload = _manifest().to_payload()
    payload["tasks"][1]["ability_id"] = payload["tasks"][0]["ability_id"]

    with pytest.raises(ValueError, match="duplicate"):
        FoundationManifest.from_payload(payload)


def test_evaluation_requires_all_tasks_and_fails_closed_on_holdout_mutation() -> None:
    measurements = {
        ability_id: _measurement(ability_id)
        for ability_id in FOUNDATION_REQUIRED_ABILITIES
    }
    measurements[FOUNDATION_REQUIRED_ABILITIES[-1]] = _measurement(
        FOUNDATION_REQUIRED_ABILITIES[-1], holdout_updates=1
    )

    evaluation = FoundationEvaluation.evaluate(
        _manifest(),
        measurements,
        checkpoint_gate_status="passed",
    )

    assert evaluation.status == "failed"
    assert evaluation.can_promote is False
    assert "holdout_side_effect" in evaluation.failure_reasons

    incomplete = dict(measurements)
    incomplete.pop(FOUNDATION_REQUIRED_ABILITIES[-1])
    with pytest.raises(ValueError, match="missing"):
        FoundationEvaluation.evaluate(
            _manifest(),
            incomplete,
            checkpoint_gate_status="passed",
        )


def test_promotion_requires_checkpoint_gate_and_all_five_measurements() -> None:
    measurements = {
        ability_id: _measurement(ability_id)
        for ability_id in FOUNDATION_REQUIRED_ABILITIES
    }

    blocked = FoundationEvaluation.evaluate(
        _manifest(), measurements, checkpoint_gate_status="not_run"
    )
    assert blocked.status == "passed"
    assert blocked.can_promote is False
    assert "checkpoint_gate_not_passed" in blocked.failure_reasons

    promoted = FoundationEvaluation.evaluate(
        _manifest(), measurements, checkpoint_gate_status="passed"
    )
    assert promoted.status == "passed"
    assert promoted.can_promote is True
    assert promoted.to_payload()["manifest_digest"] == _manifest().digest
