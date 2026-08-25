from __future__ import annotations

import math

from taiji import (
    AssemblyRelationCorpus,
    AssemblyRelationEvaluationConfig,
    AssemblyRelationEvaluator,
    AssemblyRelationExample,
    PerceptionConfig,
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


def _example(left: int, right: int, perturbation: str = "clean") -> AssemblyRelationExample:
    atoms = {
        0: (1, 2),
        1: (3, 4),
        2: (5, 6),
        3: (2, 3),
    }
    sequence = (*atoms[left], *atoms[right])
    if perturbation == "boundary":
        sequence = (*sequence[:2], 7, *sequence[2:])
        split_index = 3
    elif perturbation == "random_chunk":
        sequence = (*sequence[:1], 7, *sequence[1:])
        split_index = 2
    else:
        split_index = 2
    return AssemblyRelationExample(left, right, sequence, split_index, perturbation)


def _corpus() -> AssemblyRelationCorpus:
    train_pairs = ((0, 1), (0, 2), (1, 2), (2, 3))
    unseen_pairs = ((1, 0), (2, 1))
    return AssemblyRelationCorpus(
        atom_count=4,
        train=tuple(_example(*pair) for pair in train_pairs),
        unseen_composition=tuple(_example(*pair) for pair in unseen_pairs),
        boundary_perturbed=tuple(_example(*pair, "boundary") for pair in unseen_pairs),
        random_chunk=tuple(_example(*pair, "random_chunk") for pair in unseen_pairs),
    )


def test_relation_evaluator_reports_slot_binding_and_fair_byte_control() -> None:
    report = AssemblyRelationEvaluator(
        _config(), evaluation=AssemblyRelationEvaluationConfig(seeds=(11, 29))
    ).evaluate(_corpus())

    assert report["contract"] == "taiji-a1-assembly-relation-v1"
    assert isinstance(report["gate_passed"], bool)
    assert set(report["primary"]["slot_binding"]) == {
        "left_accuracy_learned",
        "right_accuracy_learned",
        "pair_exact_accuracy_learned",
        "pair_exact_accuracy_byte_bag",
        "pair_exact_generalization_gain",
    }
    assert len(report["cross_seed"]["runs"]) == 2
    assert math.isfinite(float(report["primary"]["consistency"]["boundary_consistency"]))
