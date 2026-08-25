from __future__ import annotations

import math

import pytest

from taiji import (
    A1EvaluationConfig,
    PerceptionConfig,
    PerceptionCorpus,
    PerceptionEvaluator,
    TaijiConfig,
)


def _config() -> TaijiConfig:
    return TaijiConfig(
        alphabet_size=8,
        boundary_symbol=7,
        seed=11,
        perception=PerceptionConfig(
            feature_dim=8,
            local_window=3,
            minimum_assembly_duration=1,
            maximum_assembly_duration=4,
            boundary_threshold=0.85,
            change_gain=0.5,
            surprise_gain=0.5,
            learning_rate=0.05,
            seed_offset=101,
        ),
    )


def _corpus() -> PerceptionCorpus:
    return PerceptionCorpus(
        train=(
            (1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2),
            (3, 4, 3, 4, 5, 6, 5, 6, 3, 4, 3, 4),
        ),
        unseen_composition=(
            (1, 2, 5, 6, 1, 2, 5, 6),
            (3, 4, 1, 2, 3, 4, 1, 2),
        ),
        boundary_perturbed=(
            (1, 2, 7, 1, 2, 3, 7, 3, 4),
            (3, 4, 7, 5, 6, 3, 4),
        ),
        random_chunk=(
            (2, 5, 1, 4, 3, 6, 2, 5),
            (6, 3, 2, 4, 1, 5, 6, 3),
        ),
    )


def test_a1_report_keeps_unseen_and_control_roles_separate() -> None:
    evaluator = PerceptionEvaluator(
        _config(),
        evaluation=A1EvaluationConfig(seeds=(11, 29), maximum_cross_seed_std=1.0),
    )

    report = evaluator.evaluate(_corpus())

    assert report["contract"] == "taiji-a1-perception-v1"
    assert report["gate"] == "A1"
    assert isinstance(report["gate_passed"], bool)
    assert set(report["primary"]) == {
        "seed",
        "predictive_training",
        "unseen_composition",
        "boundary_perturbed",
        "random_chunk_control",
        "assembly",
    }
    unseen = report["primary"]["unseen_composition"]
    assert set(unseen) == {
        "learned_accuracy",
        "byte_only_accuracy",
        "generalization_gain",
    }
    assert report["primary"]["assembly"]["unique_durations"] >= 1
    assert len(report["cross_seed"]["runs"]) == 2
    assert all(
        math.isfinite(float(run["unseen_composition"]["learned_accuracy"]))
        for run in report["cross_seed"]["runs"]
    )


def test_a1_corpus_rejects_empty_or_negative_sequences() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        PerceptionCorpus(
            train=(),
            unseen_composition=((1, 2),),
            boundary_perturbed=((1, 2),),
            random_chunk=((1, 2),),
        )

    with pytest.raises(ValueError, match="negative"):
        PerceptionCorpus(
            train=((-1, 2),),
            unseen_composition=((1, 2),),
            boundary_perturbed=((1, 2),),
            random_chunk=((1, 2),),
        )


def test_a1_gate_keeps_boundary_and_chunk_lesion_requirements() -> None:
    report = PerceptionEvaluator(
        _config(),
        evaluation=A1EvaluationConfig(
            seeds=(11,),
            minimum_boundary_rate_delta=1.0,
            minimum_random_chunk_drop=1.0,
        ),
    ).evaluate(_corpus())

    assert report["gate_passed"] is False
    assert set(report["diagnostics"]) == {"boundary_rate_delta", "random_chunk_drop"}
