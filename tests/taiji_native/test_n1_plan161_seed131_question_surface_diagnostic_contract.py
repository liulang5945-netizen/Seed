from __future__ import annotations

import hashlib
from collections import Counter

from scripts.training import build_taiji_n1_plan161_seed131_question_surface_face as builder
from scripts.training import run_taiji_n1_plan161_seed131_question_surface_diagnostic as runner


def test_plan_and_question_surface_are_frozen() -> None:
    assert hashlib.sha256(builder.PLAN.read_bytes()).hexdigest() == builder.PLAN_SHA256
    assert builder.SEED == 131
    assert builder.TOTAL_QUERIES == 2_880
    assert builder.PRIOR_LINKED_CPU_SECONDS == 10_700.171875
    assert set(builder.QUESTION_TEMPLATES) == {
        "material_active",
        "material_passive",
        "direction_after_turning",
        "direction_after_turn",
    }
    for kind in builder.KINDS:
        template_ids = [builder.question_template_id(kind, index) for index in range(40)]
        counts = Counter(template_ids)
        assert len(counts) == 2
        assert set(counts.values()) == {20}


def test_question_templates_are_absent_from_bound_training_questions() -> None:
    endpoints, sources, _ = builder.source_context()
    audit = builder.audit_training_question_surface(sources)
    assert len(endpoints) == 6
    assert audit["complete"] is True
    assert audit["training_question_count"] > 0
    assert set(audit["unseen_question_template_matches"].values()) == {0}
    assert all(path in sources for path in audit["training_manifest_hashes"])


def test_new_question_templates_and_outputs_have_separate_identity() -> None:
    assert (
        builder.render_question("material_active", "abcde12345")
        == "what material does abcde12345 use?"
    )
    assert (
        builder.render_question("material_passive", "abcde12345")
        == "which material is used by abcde12345?"
    )
    assert builder.render_question("direction_after_turning", "abcde12345") == (
        "which direction does abcde12345 face after turning?"
    )
    assert builder.render_question("direction_after_turn", "abcde12345") == (
        "what direction is abcde12345 facing after its turn?"
    )
    assert builder.FACE.name == "n1_plan161_v33_question_surface_seed131_20261020.json"
    assert runner.RESULT.name.startswith("taiji_n1_plan161_")
    assert runner.ATTEMPT != builder.N160A_RESULT


def test_summary_accepts_one_endpoint_and_checks_each_template() -> None:
    rows = []
    for length in builder.LENGTHS:
        for kind in builder.KINDS:
            template_ids = (
                ("material_active", "material_passive")
                if kind in builder.MATERIAL_KINDS
                else ("direction_after_turning", "direction_after_turn")
            )
            for template_id in template_ids:
                for index in range(builder.ITEMS_PER_KIND // 2):
                    rows.append(
                        {
                            "seed": 11,
                            "branch": "unknown8_rebalanced",
                            "context_length": length,
                            "kind": kind,
                            "question_template_id": template_id,
                            "free_exact": index % 2 == 0,
                            "teacher_exact": index % 2 == 0,
                            "output_hex": (
                                "756e6b6e6f776e" if kind == "unknown" and index % 2 == 0 else ""
                            ),
                        }
                    )
    summary = runner.fresh_summary(rows)
    assert len(summary) == 24
    assert all(item["items"] == 20 for item in summary)
    assert all(item["free_exact"] == 10 for item in summary)
