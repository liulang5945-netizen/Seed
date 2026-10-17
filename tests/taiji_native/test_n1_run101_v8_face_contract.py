from __future__ import annotations

import json
from collections import Counter

from scripts.training import (
    build_taiji_n1_hard_neighbor_training_schedule as train_data,
)
from scripts.training import build_taiji_n1_hard_neighbor_v8 as builder


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def test_v8_face_is_fresh_balanced_d1_and_partitioned():
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
    assert face["seed"] == 108
    assert face["observations_per_item"] == 8
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
    assert all(len(episode["observations"]) == 8 for episode in face["episodes"])
    assert all(
        _hamming(episode["entities"][0], name) == 1
        for episode in face["episodes"]
        for name in episode["entities"][1:]
    )


def test_v7_prefix_reader_never_decodes_confirm_episode_tail(tmp_path, monkeypatch):
    source = tmp_path / "v7.json"
    source.write_text(
        json.dumps(
            {
                "schema": "test",
                "episodes": [
                    {
                        "id": "v7-hard-000",
                        "split": "holdout",
                        "subset": "sacrifice",
                        "kind": "lookup",
                        "entities": ["sacrifice1"],
                        "observations": ["sacrifice1 uses value1."],
                        "question": "material of sacrifice1?",
                        "answer": "value1",
                    },
                    {
                        "id": "v7-hard-001",
                        "split": "holdout",
                        "subset": "confirm",
                        "kind": "lookup",
                        "entities": ["CONFIRM_PRIVATE_ENTITY"],
                        "observations": ["CONFIRM_PRIVATE_ENTITY uses secretval."],
                        "question": "CONFIRM_PRIVATE_QUESTION",
                        "answer": "secretval",
                    },
                ],
                "tail": {"ignored": True},
            }
        ),
        encoding="utf-8",
    )
    original_loads = builder.json.loads
    decoded_strings = []

    def guarded_loads(value, *args, **kwargs):
        decoded_strings.append(value)
        assert "CONFIRM_PRIVATE" not in value
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(builder.json, "loads", guarded_loads)
    records = builder._stream_v7_episode_prefixes(source)
    assert records[0][0] == {"id": "v7-hard-000", "split": "holdout", "subset": "sacrifice"}
    assert records[0][1]["entities"] == ["sacrifice1"]
    assert records[1][0] == {"id": "v7-hard-001", "split": "holdout", "subset": "confirm"}
    assert records[1][1] is None
    assert decoded_strings
    assert all("CONFIRM_PRIVATE" not in value for value in decoded_strings)


def test_previous_face_exclusions_use_only_names_and_materials():
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


def test_actual_v7_exclusion_receipt_is_sacrifice_only():
    names, words, provenance = builder.exclusions_from_v7_sacrifice()
    assert len(names) == 75
    assert "unknown" in words
    assert provenance["sacrifice_item_count"] == 20
    assert provenance["confirm_prefix_count_skipped"] == 20
    assert provenance["confirm_payload_fields_decoded"] is False
    assert provenance["confirm_model_queries"] == 0
