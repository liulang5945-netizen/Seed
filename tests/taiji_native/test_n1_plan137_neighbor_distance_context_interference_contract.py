from __future__ import annotations

from collections import Counter

from scripts.training import build_taiji_n1_v19_neighbor_distance_context_interference as builder
from scripts.training import eval_taiji_n1_v19_neighbor_distance_context_interference as evaluator


def _face() -> dict:
    face = builder.build_v19_face(set(), set(builder.DIRECTIONS))
    checks = builder.audit_v19_face(
        face,
        excluded_names=set(),
        excluded_words=set(builder.DIRECTIONS),
    )
    assert checks["accepted"] is True
    return face


def test_v19_face_balances_neighbor_distance_at_fixed_context_sizes() -> None:
    face = _face()

    assert face["seed"] == 119
    assert len(face["episodes"]) == 40
    assert len(face["matched_templates"]) == 8
    assert Counter((item["context_length"], item["condition"]) for item in face["episodes"]) == Counter(
        {
            (2, "baseline"): 8,
            (8, "d1"): 8,
            (8, "d2"): 8,
            (16, "d1"): 8,
            (16, "d2"): 8,
        }
    )


def test_v19_templates_use_exact_unique_hamming_distance_sets() -> None:
    face = _face()
    all_entities = []
    for template in face["matched_templates"]:
        subject = template["subject"]
        for distance in builder.DISTANCES:
            names = template["distractor_sets"][str(distance)]["entities"]
            assert len(names) == 14
            assert len(set(names)) == 14
            assert all(
                sum(left != right for left, right in zip(subject, name, strict=True)) == distance
                for name in names
            )
            all_entities.extend(names)
        all_entities.append(subject)
        assert all(
            template["distractor_sets"]["1"]["observations"][i].rsplit(" ", 1)[-1]
            == template["distractor_sets"]["2"]["observations"][i].rsplit(" ", 1)[-1]
            for i in range(14)
        )
    assert len(all_entities) == 232
    assert len(set(all_entities)) == 232


def test_v19_face_audit_rejects_distance_metadata_mismatch() -> None:
    face = _face()
    d2_item = next(item for item in face["episodes"] if item["condition"] == "d2")
    d2_item["neighbor_distance"] = 1

    checks = builder.audit_v19_face(
        face,
        excluded_names=set(),
        excluded_words=set(builder.DIRECTIONS),
    )

    assert checks["accepted"] is False
    assert checks["checks"]["condition_distance_metadata"] is False


def test_v19_summary_preserves_condition_denominators_and_distance_pairs() -> None:
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
                    ordinal = template_ordinals[item["template_id"]]
                    if item["condition"] == "baseline":
                        exact = True
                    else:
                        exact = (ordinal % 2 == 0) == (item["neighbor_distance"] == 1)
                    rows.append(
                        {
                            "id": item["id"],
                            "template_id": item["template_id"],
                            "context_length": item["context_length"],
                            "condition": item["condition"],
                            "neighbor_distance": item["neighbor_distance"],
                            "answer_direction": item["answer_direction"],
                            "free_exact": exact,
                            "generated_direction": item["answer"] if exact else None,
                            "stop_reason": "boundary" if exact else "limit",
                        }
                    )
                endpoint_rows.append(
                    {"source": source, "seed": seed, "arm": arm, "rows": rows}
                )

    by_endpoint, comparisons = evaluator._summarize_conditions(face, endpoint_rows)

    assert len(by_endpoint) == 12
    assert len(comparisons) == 12
    for summary in by_endpoint.values():
        assert [summary["by_condition"][key]["items"] for key in (
            "2/baseline", "8/d1", "8/d2", "16/d1", "16/d2"
        )] == [8, 8, 8, 8, 8]
        assert len(summary["by_matched_template"]) == 8
    for summary in comparisons.values():
        assert summary["8_d1_vs_d2"]["matched_templates"] == 8
        assert summary["8_d1_vs_d2"]["first_only_correct"] == 4
        assert summary["8_d1_vs_d2"]["second_only_correct"] == 4
        assert summary["16_d1_vs_d2"]["first_only_correct"] == 4
        assert summary["16_d1_vs_d2"]["second_only_correct"] == 4


def test_v19_trace_retains_distance_and_exact_generation_bytes() -> None:
    item = next(item for item in _face()["episodes"] if item["context_length"] == 16 and item["condition"] == "d2")
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
    assert traced["neighbor_distance"] == 2
    assert traced["context_length"] == 16
    assert traced["condition"] == "d2"
    assert traced["target_start_index"] == 0
