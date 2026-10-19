from __future__ import annotations

from scripts.training import analyze_taiji_n1_v23_output_sources_after_n145 as analyzer


def _base_item(kind: str = "revision") -> dict:
    target = "aaaaaaaaaa"
    neighbor = "baaaaaaaaa"
    return {
        "id": "v23-confirm-test-000",
        "subset": "confirm",
        "kind": kind,
        "entities": [target, neighbor],
        "observations": [
            f"{target} uses oldval.",
            f"{target} now uses newval.",
            f"{neighbor} uses otherv.",
        ],
        "answer": "newval" if kind == "revision" else "otherv",
        "answer_direction": None,
        "turn_template": None,
    }


def test_material_parser_preserves_entity_and_revision_role() -> None:
    facts = analyzer.parse_material_facts(_base_item())

    assert facts == [
        {"entity": "aaaaaaaaaa", "value": "oldval", "role": "uses", "observation_index": 0},
        {"entity": "aaaaaaaaaa", "value": "newval", "role": "now_uses", "observation_index": 1},
        {"entity": "baaaaaaaaa", "value": "otherv", "role": "uses", "observation_index": 2},
    ]


def test_classification_distinguishes_stale_target_and_neighbor_values() -> None:
    item = _base_item("revision")
    stale = analyzer.classify_row({"output_hex": b"oldval".hex(), "free_exact": False}, item)
    neighbor = analyzer.classify_row({"output_hex": b"otherv".hex(), "free_exact": False}, item)

    assert stale["output_class"] == "target_entity_stale_value"
    assert stale["matched_entity"] == "aaaaaaaaaa"
    assert stale["hamming_distance_to_target"] == 0
    assert stale["free_exact"] is False
    assert neighbor["output_class"] == "other_entity_material_value"
    assert neighbor["matched_entity"] == "baaaaaaaaa"
    assert neighbor["hamming_distance_to_target"] == 1


def test_transition_unknown_and_decode_error_classes_are_explicit() -> None:
    transition = _base_item("transition")
    transition.update({"answer": "south", "answer_direction": "south", "turn_template": "right"})
    initial = analyzer.classify_row({"output_hex": b"east".hex(), "free_exact": False}, transition)
    other = analyzer.classify_row({"output_hex": b"west".hex(), "free_exact": False}, transition)
    unknown = _base_item("unknown")
    abstention = analyzer.classify_row({"output_hex": b"unknown".hex(), "free_exact": True}, unknown)
    bad_hex = analyzer.classify_row({"output_hex": "0xz1", "free_exact": False}, unknown)
    bad_utf8 = analyzer.classify_row({"output_hex": "ff", "free_exact": False}, unknown)

    assert initial["output_class"] == "initial_direction"
    assert other["output_class"] == "other_valid_direction"
    assert abstention["output_class"] == "unknown_literal"
    assert abstention["free_exact_text_consistent"] is True
    assert bad_hex["output_class"] == "invalid_hex"
    assert bad_utf8["output_class"] == "invalid_utf8"


def test_full_fixture_keeps_all_frozen_scores_and_endpoint_rows() -> None:
    episodes = []
    for kind in analyzer.KINDS:
        for local in range(20):
            item = _base_item(kind)
            item.update({"id": f"v23-confirm-{kind}-{local:03d}"})
            if kind == "lookup":
                item["answer"] = "newval"
            elif kind == "revision":
                item["answer"] = "newval"
            elif kind == "transition":
                item.update({"answer": "south", "answer_direction": "south", "turn_template": "right"})
            else:
                item["answer"] = "unknown"
            episodes.append(item)
    face = {
        "schema": "n1-v23-confirmation-face-v1",
        "seed": 123,
        "plan_sha256": analyzer.N144_PLAN_SHA256,
        "episodes": [{**item, "split": "holdout"} for item in episodes],
    }
    rows = []
    for endpoint_id, (source, branch, seed) in analyzer.expected_endpoints().items():
        for item in episodes:
            rows.append(
                {
                    "endpoint_id": endpoint_id,
                    "source": source,
                    "branch": branch,
                    "seed": seed,
                    "subset": "confirm",
                    "kind": item["kind"],
                    "id": item["id"],
                    "free_exact": True,
                    "output_hex": item["answer"].encode("utf-8").hex(),
                }
            )

    classified = analyzer.validate_and_classify(rows, face)
    summary = analyzer.summarize(classified)

    assert len(classified) == 960
    assert all(row["free_exact"] is True for row in classified)
    assert summary["free_exact_text_consistency_mismatches"] == 0
    assert summary["by_kind"]["revision"] == {"gold_answer_text": 240}
    assert summary["by_kind"]["unknown"] == {"unknown_literal": 240}


def test_frozen_hash_constants_match_plan_and_consumed_inputs() -> None:
    assert analyzer.sha256(analyzer.PLAN) == analyzer.PLAN_SHA256
    assert analyzer.sha256(analyzer.N144_PLAN) == analyzer.N144_PLAN_SHA256
    assert analyzer.sha256(analyzer.N145_PLAN) == analyzer.N145_PLAN_SHA256
    assert analyzer.sha256(analyzer.FACE) == analyzer.FACE_SHA256
    assert analyzer.sha256(analyzer.N144_RESULT) == analyzer.N144_RESULT_SHA256
    assert analyzer.sha256(analyzer.N145_RESULT) == analyzer.N145_RESULT_SHA256
