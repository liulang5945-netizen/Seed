from __future__ import annotations

import pytest

from scripts.training import audit_taiji_n1_v10_output_error_attribution as audit


def _item(
    kind: str = "lookup",
    answer: str = "current",
    *,
    target: str = "target",
    observations: list[str] | None = None,
) -> dict:
    if observations is None:
        observations = [f"{target} uses {answer}.", "distractor uses observed."]
    return {
        "answer": answer,
        "entities": [target, "distractor"],
        "id": "item-0",
        "kind": kind,
        "observations": observations,
        "question": f"material of {target}?",
        "split": "diagnostic",
        "template_index": 0,
        "target_row_position_1based": 1,
        "material_answer_length_chars": len(answer) if kind in {"lookup", "revision"} else None,
    }


def _minimal_face_and_result() -> tuple[dict, dict]:
    episodes = []
    rows = []
    for index in range(248):
        item_id = f"item-{index:03d}"
        target = f"target{index}"
        answer = f"answer{index}"
        item = _item(target=target, answer=answer)
        item.update({"id": item_id, "template_index": index % 4})
        episodes.append(item)
        for seed in audit.SEEDS:
            for arm in audit.ARMS:
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "id": item_id,
                        "kind": item["kind"],
                        "split": "diagnostic",
                        "template_index": item["template_index"],
                        "target_row_position_1based": item["target_row_position_1based"],
                        "material_answer_length_chars": item["material_answer_length_chars"],
                        "output_hex": answer.encode("ascii").hex(),
                        "raw_hex": answer.encode("ascii").hex(),
                        "free_exact": True,
                    }
                )
    face = {
        "schema": audit.FACE_SCHEMA,
        "split": "diagnostic",
        "items": len(episodes),
        "episodes": episodes,
    }
    result = {
        "status": "complete",
        "complete": True,
        "face_sha256": audit.FACE_SHA256,
        "plan_sha256": audit.PLAN105_SHA256,
        "model_queries": 2232,
        "planned_model_queries": 2232,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "existing_confirmation_face_queries": 0,
        "gate_recomputed": False,
        "can_promote": False,
        "checkpoints_unchanged": True,
        "sources_unchanged": True,
        "code_identity_unchanged": True,
        "v7_v8_confirmation_payload_fields_decoded": False,
        "rows": rows,
    }
    return face, result


def test_classifier_separates_current_stale_observed_unknown_empty_and_other():
    revision = _item(
        "revision",
        "current",
        observations=[
            "target uses stale.",
            "distractor uses observed.",
            "target now uses current.",
        ],
    )
    assert audit.classify_output(revision, "current") == "exact_answer"
    assert audit.classify_output(revision, "stale") == "revision_stale_material"
    assert audit.classify_output(revision, "observed") == "distractor_observed_material"
    assert audit.classify_output(revision, "unknown") == "unknown_abstention"
    assert audit.classify_output(revision, "") == "empty_output"
    assert audit.classify_output(revision, "never-seen") == "other_unobserved_output"
    assert audit.classify_output(revision, None) == "nontext_output"


def test_unknown_exact_output_is_not_counted_as_abstention():
    item = _item(
        "unknown",
        "unknown",
        observations=["distractor uses observed."],
    )
    assert audit.classify_output(item, "unknown") == "exact_answer"
    assert audit.classify_output(item, "observed") == "distractor_observed_material"


def test_full_endpoint_face_coverage_is_required():
    face, result = _minimal_face_and_result()
    classified = audit._validate_and_classify(face, result)
    assert len(classified) == 2232
    assert all(row["output_class"] == "exact_answer" for row in classified)
    assert len({(row["seed"], row["arm"], row["id"]) for row in classified}) == 2232

    result["rows"].pop()
    with pytest.raises(ValueError, match="row count"):
        audit._validate_and_classify(face, result)


def test_decoded_answer_must_agree_with_frozen_exact_flag():
    face, result = _minimal_face_and_result()
    result["rows"][0]["free_exact"] = False
    with pytest.raises(ValueError, match="free_exact"):
        audit._validate_and_classify(face, result)


def test_summary_and_pair_strata_conserve_denominators():
    sample = [
        {"output_class": "exact_answer", "free_exact": True},
        {"output_class": "distractor_observed_material", "free_exact": False},
        {"output_class": "other_unobserved_output", "free_exact": False},
    ]
    summary = audit._summary(sample)
    assert summary["denominator"] == 3
    assert summary["exact"] == 1
    assert sum(summary["output_classes"].values()) == 3

    rows = [
        {
            "seed": 11,
            "id": "item-0",
            "kind": "revision",
            "material_answer_length_chars": 6,
            "arm": "position",
            "free_exact": False,
            "output_class": "revision_stale_material",
        },
        {
            "seed": 11,
            "id": "item-0",
            "kind": "revision",
            "material_answer_length_chars": 6,
            "arm": "cut",
            "free_exact": True,
            "output_class": "exact_answer",
        },
    ]
    paired = audit._paired(rows, "cut")
    assert paired[0]["denominator"] == 1
    assert paired[0]["comparison_only_exact"] == 1
    assert paired[0]["position_error_classes_when_comparison_exact"]["revision_stale_material"] == 1
