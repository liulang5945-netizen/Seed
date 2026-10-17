from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from scripts.training import build_taiji_n1_hard_neighbor_training_schedule as schedule


@pytest.fixture(scope="module")
def frozen_schedule():
    items, curriculum, pairs = schedule.load_training_sources()
    result = schedule.build_schedule(items, curriculum)
    return items, curriculum, pairs, result


def _subject(item: dict) -> str:
    question = item["question"]
    return question.removeprefix("material of ").removesuffix("?")


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def test_non_train_episode_payloads_are_skipped_without_json_decode():
    raw = (
        b'{"episodes":['
        b'{"id":"train-000","split":"train","kind":"lookup"},'
        b'{"id":"dev-000","split":"dev","private_payload":not_json},'
        b'{"id":"holdout-000","split":"holdout","private_payload":also_not_json}'
        b"]}"
    )
    assert schedule._train_objects_from_bytes(raw) == [
        {"id": "train-000", "split": "train", "kind": "lookup"}
    ]


def test_schedule_is_complete_balanced_and_deterministic(frozen_schedule):
    items, curriculum, _, first = frozen_schedule
    second = schedule.build_schedule(items, curriculum)
    assert first == second
    assert len(first["items"]) == 3072
    assert first["summary"]["parents"] == 192
    assert first["summary"]["slots_per_parent"] == 16
    assert first["summary"]["kind_counts"] == {kind: 768 for kind in schedule.KINDS}
    assert first["summary"]["condition_counts"] == {
        "lookup/obs2/d1": 192,
        "lookup/obs2/d2": 192,
        "lookup/obs4/d1": 192,
        "lookup/obs4/d2": 192,
        "revision/obs4/d1": 384,
        "revision/obs4/d2": 384,
        "transition/obs2/dNone": 768,
        "unknown/obs2/d1": 192,
        "unknown/obs2/d2": 192,
        "unknown/obs4/d1": 192,
        "unknown/obs4/d2": 192,
    }

    by_parent = defaultdict(list)
    for entry in first["items"]:
        by_parent[entry["parent_id"]].append(entry)
    assert set(by_parent) == {item["id"] for item in items}
    assert all(len(rows) == 16 for rows in by_parent.values())


def test_hard_views_preserve_target_semantics_and_exact_name_distance(frozen_schedule):
    items, curriculum, _, payload = frozen_schedule
    originals = {item["id"]: item for item in items}
    train_values = set(schedule._values(items))
    all_replacement_names = {name for entry in curriculum["stream"] for name in entry["names"]}
    all_replacement_names.update(name for entry in curriculum["probes"] for name in entry["names"])
    seen_generated = set()
    for entry in payload["items"]:
        item = entry["item"]
        source = originals[entry["parent_id"]]
        target = _subject(item)
        assert item["split"] == "train"
        assert item["original_kind"] == source["kind"] == entry["parent_kind"]
        assert len(item["observations"]) == entry["observation_count"]
        assert item["id"] != source["id"]

        if source["kind"] == "transition":
            stream_entry = curriculum["stream"][entry["schedule_index"]]
            expected = schedule.core.renaming.rename_item(source, stream_entry["names"])
            assert item["observations"] == expected["observations"]
            assert item["question"] == expected["question"]
            assert entry["hamming_distance"] is None
            continue

        distance = entry["hamming_distance"]
        assert distance in schedule.DISTANCES
        distractors = [
            line.split()[0] for line in item["observations"] if line.split()[0] != target
        ]
        assert distractors
        assert all(_hamming(target, name) == distance for name in distractors)
        assert all(name not in all_replacement_names for name in distractors)
        assert not (set(distractors) & seen_generated)
        seen_generated.update(distractors)
        assert all(
            line.removesuffix(".").split()[-1] in train_values
            for line in item["observations"]
            if " uses " in line
        )

        target_rows = [line for line in item["observations"] if line.startswith(target + " ")]
        if source["kind"] == "lookup":
            assert len(target_rows) == 1
            assert target_rows[0].removesuffix(".").split()[-1] == item["answer"]
            assert entry["target_positions"] == [item["observations"].index(target_rows[0])]
            assert entry["second_question"]["answer"] in train_values
        elif source["kind"] == "unknown":
            assert not target_rows
            assert item["answer"] == "unknown"
            second = entry["second_question"]
            assert second["parent_id"] == item["id"]
            assert second["observations"] == item["observations"]
            assert second["answer"] in train_values
        else:
            assert len(target_rows) == 2
            assert target_rows[0].startswith(target + " uses ")
            assert target_rows[1].startswith(target + " now uses ")
            assert entry["target_positions"] == [
                item["observations"].index(target_rows[0]),
                item["observations"].index(target_rows[1]),
            ]
            assert entry["second_question"] is None


def test_factorial_cells_and_target_placement_are_balanced_per_parent(frozen_schedule):
    _, _, _, payload = frozen_schedule
    by_parent = defaultdict(list)
    for entry in payload["items"]:
        by_parent[entry["parent_id"]].append(entry)
    for rows in by_parent.values():
        kind = rows[0]["parent_kind"]
        if kind in ("lookup", "unknown"):
            counts = Counter((row["observation_count"], row["hamming_distance"]) for row in rows)
            assert counts == Counter(
                {(observations, distance): 4 for observations in (2, 4) for distance in (1, 2)}
            )
        elif kind == "revision":
            by_distance = defaultdict(Counter)
            for row in rows:
                by_distance[row["hamming_distance"]][row["target_positions"][0]] += 1
            assert all(sum(counts.values()) == 8 for counts in by_distance.values())
            assert all(
                max(counts.values()) - min(counts.values()) <= 1 for counts in by_distance.values()
            )
            all_starts = Counter(row["target_positions"][0] for row in rows)
            assert set(all_starts) == {0, 1, 2}
            assert max(all_starts.values()) - min(all_starts.values()) <= 1
        else:
            assert all(row["observation_count"] == 2 for row in rows)
            assert all(row["hamming_distance"] is None for row in rows)


def test_train_only_source_manifest_hash_is_pinned():
    items, digest = schedule.train_items_from_frozen_manifest()
    assert digest == schedule.MANIFEST_SHA256
    assert len(items) == 192
    assert {item["split"] for item in items} == {"train"}
