from __future__ import annotations

from collections import Counter

from scripts.training import build_taiji_n1_v17_matched_context_length as builder
from scripts.training import eval_taiji_n1_v17_matched_context_length as evaluator


def _face() -> dict:
    face = builder.build_v17_face(set(), set(builder.DIRECTIONS))
    checks = builder.audit_v17_face(
        face,
        excluded_names=set(),
        excluded_words=set(builder.DIRECTIONS),
    )
    assert checks["accepted"] is True
    return face


def test_v17_face_has_eight_nested_direction_turn_templates() -> None:
    face = _face()

    assert face["seed"] == 117
    assert len(face["episodes"]) == 24
    assert len(face["matched_templates"]) == 8
    assert Counter(item["context_length"] for item in face["episodes"]) == Counter(
        {2: 8, 8: 8, 16: 8}
    )
    by_template = {}
    for item in face["episodes"]:
        by_template.setdefault(item["template_id"], {})[item["context_length"]] = item
    for matched in by_template.values():
        assert matched[2]["observations"] == matched[8]["observations"][:2]
        assert matched[8]["observations"] == matched[16]["observations"][:8]
        assert matched[2]["answer"] == matched[8]["answer"] == matched[16]["answer"]
        assert matched[2]["question"] == matched[8]["question"] == matched[16]["question"]


def test_v17_face_audit_rejects_changed_target_direction() -> None:
    face = _face()
    face["episodes"][0]["answer"] = "south"

    checks = builder.audit_v17_face(
        face,
        excluded_names=set(),
        excluded_words=set(builder.DIRECTIONS),
    )

    assert checks["accepted"] is False
    assert checks["checks"]["target_rows_compute_answer_direction"] is False


def test_v17_summary_preserves_length_denominators_and_matched_pairs() -> None:
    face = _face()
    endpoint_rows = []
    for source, arms in (
        ("N1-132", ("position",)),
        ("N1-120", ("position", "terminal", "cut")),
    ):
        for seed in (11, 12, 13):
            for arm in arms:
                rows = []
                for index, item in enumerate(face["episodes"]):
                    exact = item["context_length"] == 2 and index % 2 == 0
                    rows.append(
                        {
                            "id": item["id"],
                            "template_id": item["template_id"],
                            "context_length": item["context_length"],
                            "answer_direction": item["answer_direction"],
                            "free_exact": exact,
                            "generated_direction": item["answer"] if exact else None,
                            "stop_reason": "boundary" if exact else "limit",
                        }
                    )
                endpoint_rows.append(
                    {"source": source, "seed": seed, "arm": arm, "rows": rows}
                )

    by_endpoint, paired = evaluator._summarize_matched_lengths(face, endpoint_rows)

    assert len(by_endpoint) == 12
    assert len(paired) == 12
    for summary in by_endpoint.values():
        assert [summary["by_context_length"][str(length)]["items"] for length in (2, 8, 16)] == [
            8,
            8,
            8,
        ]
        assert len(summary["by_matched_template"]) == 8
    for summary in paired.values():
        assert len(summary["2_to_8"]) == 8
        assert sum(result["short_only_correct"] for result in summary["2_to_8"].values()) == 4
        assert sum(result["both_wrong"] for result in summary["2_to_8"].values()) == 4


def test_v17_trace_retains_exact_generation_bytes_and_template_metadata() -> None:
    item = _face()["episodes"][0]
    traced = evaluator._attach_item_trace(
        {
            "id": item["id"],
            "raw_hex": b"north".hex(),
            "output_hex": b"north\x00".hex(),
            "stop_reason": "boundary",
            "generated_steps": 6,
            "free_exact": True,
        },
        item,
    )

    assert traced["raw_hex"] == b"north".hex()
    assert traced["output_hex"] == b"north\x00".hex()
    assert traced["generated_text_utf8"] == "north"
    assert traced["generated_direction"] == "north"
    assert traced["matched_template_id"] == item["template_id"]
    assert traced["context_length"] == item["context_length"]
