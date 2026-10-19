from __future__ import annotations

from collections import Counter

from scripts.training import build_taiji_n1_v16_transition_direction_position_diagnostic as v16
from scripts.training import eval_taiji_n1_v16_transition_direction_position_diagnostic as evaluator


def test_v16_face_crosses_all_directions_and_target_positions() -> None:
    excluded_names: set[str] = set()
    excluded_words = set(v16.DIRECTIONS)
    face = v16.build_v16_face(excluded_names, excluded_words)
    checks = v16.audit_v16_face(face, excluded_names=excluded_names, excluded_words=excluded_words)

    assert checks["accepted"] is True
    assert face["schema"] == "n1-transition-direction-position-diagnostic-v1"
    assert face["seed"] == 116
    assert len(face["episodes"]) == 40
    assert Counter(
        (item["answer_direction"], item["target_start_index"]) for item in face["episodes"]
    ) == Counter(
        {(direction, start): 2 for direction in v16.DIRECTIONS for start in v16.TARGET_STARTS}
    )
    assert checks["checks"]["left_right_balanced_in_each_cell"] is True
    assert all(len(item["observations"]) == 16 for item in face["episodes"])


def test_v16_face_audit_rejects_wrong_answer_direction() -> None:
    excluded_names: set[str] = set()
    excluded_words = set(v16.DIRECTIONS)
    face = v16.build_v16_face(excluded_names, excluded_words)
    face["episodes"][0]["answer"] = "south"

    checks = v16.audit_v16_face(face, excluded_names=excluded_names, excluded_words=excluded_words)

    assert checks["accepted"] is False
    assert checks["checks"]["two_target_rows_and_direction_consistency"] is False


def test_v16_source_union_is_bound_to_consumed_v15_and_frozen_endpoints() -> None:
    names, words, source_hashes, counts, endpoints = v16._prior_sources()

    assert len(names) == 7938
    assert len(words) == 6288
    assert counts["source_hash_count"] == len(source_hashes)
    assert len(endpoints) == 12
    assert sum(item["source"] == "N1-132" for item in endpoints) == 3
    assert sum(item["source"] == "N1-120" for item in endpoints) == 9
    assert not v16.FACE.exists()
    assert not v16.RUN_REPORT.exists()


def test_v16_direction_position_summary_keeps_full_cell_denominator() -> None:
    face = v16.build_v16_face(set(), set(v16.DIRECTIONS))
    endpoint_rows = []
    for source, arms in (("N1-132", ("position",)), ("N1-120", ("position", "terminal", "cut"))):
        for seed in (11, 12, 13):
            for arm in arms:
                rows = [
                    {
                        "id": item["id"],
                        "free_exact": (index % 2 == 0),
                    }
                    for index, item in enumerate(face["episodes"])
                ]
                endpoint_rows.append({"source": source, "seed": seed, "arm": arm, "rows": rows})

    cell_summary, paired = evaluator._summarize_direction_position(endpoint_rows, face)

    assert len(cell_summary) == 12
    assert all(
        cell["items"] == 2
        for endpoint in cell_summary.values()
        for cell in endpoint["by_direction_and_target_start"].values()
    )
    assert len(paired) == 9
    assert all(result["overall"]["candidate_minus_control"] == 0 for result in paired.values())
