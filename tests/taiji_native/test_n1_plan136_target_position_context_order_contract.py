from __future__ import annotations

from collections import Counter

from scripts.training import build_taiji_n1_v18_target_position_context_order as builder
from scripts.training import eval_taiji_n1_v18_target_position_context_order as evaluator


def _face() -> dict:
    face = builder.build_v18_face(set(), set(builder.DIRECTIONS))
    checks = builder.audit_v18_face(
        face,
        excluded_names=set(),
        excluded_words=set(builder.DIRECTIONS),
    )
    assert checks["accepted"] is True
    return face


def test_v18_face_balances_baseline_and_position_conditions() -> None:
    face = _face()

    assert face["seed"] == 118
    assert len(face["episodes"]) == 40
    assert len(face["matched_templates"]) == 8
    assert Counter(
        (item["context_length"], item["placement"]) for item in face["episodes"]
    ) == Counter(
        {
            (2, "baseline"): 8,
            (8, "early"): 8,
            (8, "late"): 8,
            (16, "early"): 8,
            (16, "late"): 8,
        }
    )


def test_v18_early_and_late_conditions_preserve_filler_content() -> None:
    face = _face()
    by_template = {}
    for item in face["episodes"]:
        by_template.setdefault(item["template_id"], {})[
            (item["context_length"], item["placement"])
        ] = item

    for conditions in by_template.values():
        baseline = conditions[(2, "baseline")]
        early8 = conditions[(8, "early")]
        late8 = conditions[(8, "late")]
        early16 = conditions[(16, "early")]
        late16 = conditions[(16, "late")]
        assert baseline["observations"] == early8["observations"][:2]
        assert early8["observations"][2:] == late8["observations"][:6]
        assert early16["observations"][2:] == late16["observations"][:14]
        assert early8["observations"][2:] == early16["observations"][2:8]
        assert late8["observations"][:6] == late16["observations"][:6]
        assert (early8["target_start_index"], late8["target_start_index"]) == (0, 6)
        assert (early16["target_start_index"], late16["target_start_index"]) == (0, 14)


def test_v18_face_audit_rejects_wrong_target_position_metadata() -> None:
    face = _face()
    late8 = next(
        item
        for item in face["episodes"]
        if item["context_length"] == 8 and item["placement"] == "late"
    )
    late8["target_start_index"] = 5

    checks = builder.audit_v18_face(
        face,
        excluded_names=set(),
        excluded_words=set(builder.DIRECTIONS),
    )

    assert checks["accepted"] is False
    assert checks["checks"]["target_positions_match_condition"] is False


def test_v18_summary_keeps_condition_denominators_and_position_pairs() -> None:
    face = _face()
    endpoint_rows = []
    template_ordinals = {
        template["template_id"]: index for index, template in enumerate(face["matched_templates"])
    }
    for source, arms in (
        ("N1-132", ("position",)),
        ("N1-120", ("position", "terminal", "cut")),
    ):
        for seed in (11, 12, 13):
            for arm in arms:
                rows = []
                for item in face["episodes"]:
                    template_ordinal = template_ordinals[item["template_id"]]
                    if item["placement"] == "baseline":
                        exact = True
                    elif item["context_length"] == 8:
                        exact = (template_ordinal % 2 == 0) == (item["placement"] == "early")
                    else:
                        exact = True
                    rows.append(
                        {
                            "id": item["id"],
                            "template_id": item["template_id"],
                            "context_length": item["context_length"],
                            "placement": item["placement"],
                            "target_start_index": item["target_start_index"],
                            "answer_direction": item["answer_direction"],
                            "free_exact": exact,
                            "generated_direction": item["answer"] if exact else None,
                            "stop_reason": "boundary" if exact else "limit",
                        }
                    )
                endpoint_rows.append({"source": source, "seed": seed, "arm": arm, "rows": rows})

    by_endpoint, comparisons = evaluator._summarize_conditions(face, endpoint_rows)

    assert len(by_endpoint) == 12
    assert len(comparisons) == 12
    for summary in by_endpoint.values():
        assert [
            summary["by_condition"][key]["items"]
            for key in ("2/baseline", "8/early", "8/late", "16/early", "16/late")
        ] == [8, 8, 8, 8, 8]
        assert len(summary["by_matched_template"]) == 8
    for summary in comparisons.values():
        assert summary["8_early_vs_late"]["matched_templates"] == 8
        assert summary["8_early_vs_late"]["first_only_correct"] == 4
        assert summary["8_early_vs_late"]["second_only_correct"] == 4
        assert summary["16_early_vs_late"]["both_correct"] == 8


def test_v18_trace_retains_exact_bytes_and_target_position_metadata() -> None:
    item = next(
        item
        for item in _face()["episodes"]
        if item["context_length"] == 16 and item["placement"] == "late"
    )
    traced = evaluator._attach_item_trace(
        {
            "id": item["id"],
            "raw_hex": b"west".hex(),
            "output_hex": b"west\x00".hex(),
            "stop_reason": "boundary",
            "generated_steps": 5,
            "free_exact": True,
        },
        item,
    )

    assert traced["raw_hex"] == b"west".hex()
    assert traced["output_hex"] == b"west\x00".hex()
    assert traced["generated_text_utf8"] == "west"
    assert traced["generated_direction"] == "west"
    assert traced["matched_template_id"] == item["template_id"]
    assert traced["context_length"] == 16
    assert traced["placement"] == "late"
    assert traced["target_start_index"] == 14
    assert traced["target_fact_indices"] == [14, 15]
