from __future__ import annotations

from collections import Counter

from scripts.training import build_taiji_n1_v20_neighbor_distance_dose_curve as builder
from scripts.training import eval_taiji_n1_v20_neighbor_distance_dose_curve as evaluator


def _face() -> dict:
    face = builder.build_v20_face(set(), set(builder.DIRECTIONS))
    checks = builder.audit_v20_face(
        face,
        excluded_names=set(),
        excluded_words=set(builder.DIRECTIONS),
    )
    assert checks["accepted"] is True
    return face


def test_v20_face_balances_all_three_distances_at_fixed_context_sizes() -> None:
    face = _face()

    assert face["seed"] == 120
    assert len(face["episodes"]) == 56
    assert len(face["matched_templates"]) == 8
    assert Counter(
        (item["context_length"], item["condition"]) for item in face["episodes"]
    ) == Counter(
        {
            (2, "baseline"): 8,
            (8, "d1"): 8,
            (8, "d2"): 8,
            (8, "d4"): 8,
            (16, "d1"): 8,
            (16, "d2"): 8,
            (16, "d4"): 8,
        }
    )


def test_v20_templates_use_exact_unique_hamming_distance_sets() -> None:
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
        direction_sequences = [
            [row.rsplit(" ", 1)[-1] for row in template["distractor_sets"][str(d)]["observations"]]
            for d in builder.DISTANCES
        ]
        assert direction_sequences[0] == direction_sequences[1] == direction_sequences[2]
    assert len(all_entities) == 344
    assert len(set(all_entities)) == 344


def test_v20_face_audit_rejects_distance_metadata_mismatch() -> None:
    face = _face()
    d4_item = next(item for item in face["episodes"] if item["condition"] == "d4")
    d4_item["neighbor_distance"] = 2

    checks = builder.audit_v20_face(
        face,
        excluded_names=set(),
        excluded_words=set(builder.DIRECTIONS),
    )

    assert checks["accepted"] is False
    assert checks["checks"]["condition_distance_metadata"] is False


def test_v20_summary_preserves_condition_denominators_and_dose_pairs() -> None:
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
                        exact = (ordinal % 3) == ((item["neighbor_distance"] - 1) % 3)
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
                endpoint_rows.append({"source": source, "seed": seed, "arm": arm, "rows": rows})

    by_endpoint, comparisons = evaluator._summarize_conditions(face, endpoint_rows)

    assert len(by_endpoint) == 12
    assert len(comparisons) == 12
    for summary in by_endpoint.values():
        assert [
            summary["by_condition"][key]["items"]
            for key in ("2/baseline", "8/d1", "8/d2", "8/d4", "16/d1", "16/d2", "16/d4")
        ] == [8, 8, 8, 8, 8, 8, 8]
        assert len(summary["by_matched_template"]) == 8
    for summary in comparisons.values():
        assert summary["8_d1_vs_d2"]["first_only_correct"] == 3
        assert summary["8_d2_vs_d4"]["first_only_correct"] == 3
        assert summary["8_d1_vs_d4"]["both_correct"] == 3


def test_v20_trace_retains_distance_and_exact_generation_bytes() -> None:
    item = next(
        item
        for item in _face()["episodes"]
        if item["context_length"] == 16 and item["condition"] == "d4"
    )
    traced = evaluator._attach_item_trace(
        {
            "id": item["id"],
            "raw_hex": b"east".hex(),
            "output_hex": b"east\x00".hex(),
            "stop_reason": "boundary",
            "generated_steps": 5,
            "free_exact": True,
        },
        item,
    )

    assert traced["raw_hex"] == b"east".hex()
    assert traced["output_hex"] == b"east\x00".hex()
    assert traced["generated_text_utf8"] == "east"
    assert traced["generated_direction"] == "east"
    assert traced["neighbor_distance"] == 4
    assert traced["context_length"] == 16
    assert traced["condition"] == "d4"
    assert traced["target_start_index"] == 0
