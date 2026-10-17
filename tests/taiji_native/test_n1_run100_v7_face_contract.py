from __future__ import annotations

from collections import Counter

from scripts.training import (
    build_taiji_n1_hard_neighbor_training_schedule as train_data,
)
from scripts.training import (
    build_taiji_n1_hard_neighbor_v7 as builder,
)


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def test_v7_face_is_fresh_balanced_d1_and_partitioned():
    train_items, _ = train_data.train_items_from_frozen_manifest()
    train_names = {name for item in train_items for name in item["entities"]}
    train_values = set(train_data._values(train_items))
    prior_names = {f"priorname{i:03d}" for i in range(40)}
    prior_words = {f"prior{i:04d}" for i in range(40)}
    excluded_names = train_names | prior_names
    excluded_words = train_values | prior_words | {"unknown"}
    face = builder.build_face(
        train_items=train_items,
        excluded_names=excluded_names,
        excluded_words=excluded_words,
    )
    audit = builder.audit_face(
        face,
        train_items=train_items,
        excluded_names=excluded_names,
        excluded_words=excluded_words,
    )
    assert audit["accepted"] is True
    assert all(audit["checks"].values())
    assert len(face["episodes"]) == 40
    assert face["subset_sizes"] == {"sacrifice": 20, "confirm": 20}
    assert face["seed"] == 107
    assert face["name_neighbor_distance"] == 1
    assert audit["name_distance_values"] == [1]
    assert audit["entity_overlap_count"] == 0
    assert audit["material_word_overlap_count"] == 0

    per_kind = Counter(episode["kind"] for episode in face["episodes"])
    assert per_kind == {kind: 10 for kind in builder.KINDS}
    per_subset_kind = {
        subset: Counter(
            episode["kind"] for episode in face["episodes"] if episode["subset"] == subset
        )
        for subset in ("sacrifice", "confirm")
    }
    assert all(counts == {kind: 5 for kind in builder.KINDS} for counts in per_subset_kind.values())
    assert all(len(episode["observations"]) == 4 for episode in face["episodes"])
    assert all(
        _hamming(episode["entities"][0], name) == 1
        for episode in face["episodes"]
        for name in episode["entities"][1:]
    )


def test_previous_face_exclusions_use_only_names_and_answer_materials():
    v5 = {
        "episodes": [
            {
                "entities": ["oldentitya", "oldentityb"],
                "answer": "oldword",
                "observations": ["oldentitya uses priorval.", "oldentityb uses otherval."],
                "question": "ignored question",
            }
        ]
    }
    v6 = {
        "episodes": [
            {
                "entities": ["faceentitya", "faceentityb"],
                "answer": "faceword",
                "observations": ["faceentitya uses faceval."],
                "question": "ignored other question",
            }
        ]
    }
    names, words = builder.exclusions_from_faces(v5, v6)
    assert names == {"oldentitya", "oldentityb", "faceentitya", "faceentityb"}
    assert words >= {"oldword", "faceword", "priorval", "otherval", "faceval", "unknown"}
