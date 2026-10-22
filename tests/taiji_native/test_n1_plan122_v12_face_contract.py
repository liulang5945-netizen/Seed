from __future__ import annotations

from collections import Counter

from scripts.training import build_taiji_n1_v12_after_n121 as builder


def test_v12_face_contract_uses_seed112_four_observations_and_d1() -> None:
    face = builder.build_v12_face([], set(), set())

    assert face["schema"] == "n1-hard-neighbor-v12-confirmation-v1"
    assert face["seed"] == 112
    assert face["items"] == 40
    assert face["observations_per_item"] == 4
    assert face["name_neighbor_distance"] == 1
    assert face["subset_sizes"] == {"sacrifice": 20, "confirm": 20}
    for kind in builder.KINDS:
        selected = [item for item in face["episodes"] if item["kind"] == kind]
        assert len(selected) == 10
        assert Counter(item["subset"] for item in selected) == {"sacrifice": 5, "confirm": 5}
        assert all(len(item["observations"]) == 4 for item in selected)

    checks = builder.audit_v12_face(face, excluded_names=set(), excluded_words=set())
    assert checks["accepted"] is True
    assert checks["checks"]["confusable_name_distance"] is True
    assert checks["checks"]["answer_length_split"] is True
    assert checks["checks"]["disjoint_observation_materials"] is True


def test_v12_face_audit_rejects_a_reused_observation_material() -> None:
    face = builder.build_v12_face([], set(), set())
    first = face["episodes"][0]
    subject = first["entities"][0]
    material = next(
        row.removesuffix(".").split()[-1]
        for row in first["observations"]
        if row.startswith(subject + " uses ")
    )

    checks = builder.audit_v12_face(face, excluded_names=set(), excluded_words={material})

    assert checks["checks"]["disjoint_observation_materials"] is False
    assert checks["accepted"] is False


def test_v12_face_audit_rejects_a_non_neighbor_distractor() -> None:
    face = builder.build_v12_face([], set(), set())
    subject = face["episodes"][0]["entities"][0]
    distractor = list(face["episodes"][0]["entities"][1])
    changed = next(
        index
        for index, (left, right) in enumerate(zip(subject, distractor, strict=False))
        if left != right
    )
    extra = next(index for index in range(len(subject)) if index != changed)
    distractor[extra] = next(letter for letter in builder.v9.LETTERS if letter != subject[extra])
    face["episodes"][0]["entities"][1] = "".join(distractor)

    checks = builder.audit_v12_face(face, excluded_names=set(), excluded_words=set())

    assert checks["checks"]["confusable_name_distance"] is False
    assert checks["accepted"] is False
