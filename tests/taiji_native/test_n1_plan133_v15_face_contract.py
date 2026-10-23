from __future__ import annotations

from collections import Counter

import pytest

from scripts.training import build_taiji_n1_v14_after_n123 as v14
from scripts.training import build_taiji_n1_v15_after_n132 as v15


def _base_exclusions() -> tuple[set[str], set[str]]:
    return set(), {*v14.DIRECTIONS, v14.ABSTENTION}


def test_v15_face_has_frozen_seed_size_classes_lengths_and_target_positions() -> None:
    names, words = _base_exclusions()
    face = v15.build_v15_face(names, words)

    assert face["schema"] == "n1-hard-neighbor-v15-confirmation-v1"
    assert face["seed"] == 115
    assert face["items"] == 40
    assert face["observations_per_item"] == 16
    assert face["name_neighbor_distance"] == 1
    assert face["subset_sizes"] == {"sacrifice": 20, "confirm": 20}
    assert v14.SEED == 114
    assert Counter(item["kind"] for item in face["episodes"]) == Counter(
        {kind: 10 for kind in v15.KINDS}
    )
    assert all(len(item["observations"]) == 16 for item in face["episodes"])

    for kind, expected_starts in v15.TARGET_STARTS.items():
        selected = [item for item in face["episodes"] if item["kind"] == kind]
        assert Counter(item["subset"] for item in selected) == {"sacrifice": 5, "confirm": 5}
        if kind == "unknown":
            assert all(item["target_start_index"] is None for item in selected)
        else:
            assert [item["target_start_index"] for item in selected[:5]] == list(expected_starts)

    checks = v15.audit_v15_face(face, excluded_names=names, excluded_words=words)
    assert checks["accepted"] is True
    assert checks["checks"]["answer_length_split"] is True
    assert checks["checks"]["confusable_name_distance"] is True
    assert checks["checks"]["variable_answers_disjoint_from_prior_values"] is True
    assert checks["checks"]["fixed_answer_vocabulary_preexcluded"] is True


def test_v15_face_generation_is_deterministic() -> None:
    names, words = _base_exclusions()

    first = v15.build_v15_face(names, words)
    second = v15.build_v15_face(names, words)

    assert first == second


def test_v15_source_union_revalidates_consumed_v14_inputs_and_endpoint_set() -> None:
    names, words, sources, counts = v15._prior_sources_and_union(require_unused_seed=False)
    endpoints = v15._endpoint_specs()

    assert len(names) == 7308
    assert len(words) == 5808
    assert counts["source_hash_count"] == len(sources)
    assert counts["prior_stopped_readiness_count"] == 2
    assert sources[str(v15.INITIAL_READINESS.resolve())] == v15.INITIAL_READINESS_SHA256
    assert sources[str(v15.SECOND_READINESS.resolve())] == v15.SECOND_READINESS_SHA256
    assert len(endpoints) == 12
    assert sum(endpoint["source"] == "N1-120" for endpoint in endpoints) == 9
    assert sum(endpoint["source"] == "N1-132" for endpoint in endpoints) == 3
    assert v15.SOURCE_EXCLUSIONS.is_relative_to(v15.ROOT)
    assert v15.FACE.is_relative_to(v15.ROOT)
    if v15.FACE.exists():
        face = v15.read_json(v15.FACE)
        assert face.get("seed") == 115
        assert face.get("schema") == "n1-hard-neighbor-v15-confirmation-v1"
    else:
        assert not v15.SOURCE_EXCLUSIONS.exists()


def test_v15_source_union_still_refuses_a_consumed_seed() -> None:
    with pytest.raises(ValueError, match="硬邻居账本出现"):
        v15._prior_sources_and_union()


def test_v15_face_audit_rejects_prior_material_and_missing_fixed_answer_exclusions() -> None:
    names, words = _base_exclusions()
    face = v15.build_v15_face(names, words)
    first = face["episodes"][0]
    subject = first["entities"][0]
    material = next(
        row.removesuffix(".").split()[-1]
        for row in first["observations"]
        if row.startswith(subject + " uses ")
    )

    repeated_material = v15.audit_v15_face(
        face, excluded_names=names, excluded_words=words | {material}
    )
    missing_fixed_words = v15.audit_v15_face(face, excluded_names=names, excluded_words=set())

    assert repeated_material["checks"]["disjoint_observation_materials"] is False
    assert repeated_material["accepted"] is False
    assert missing_fixed_words["checks"]["fixed_answer_vocabulary_preexcluded"] is False
    assert missing_fixed_words["accepted"] is False
