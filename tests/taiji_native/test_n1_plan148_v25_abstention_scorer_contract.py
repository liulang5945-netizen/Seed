from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.training import build_taiji_n1_v25_abstention_after_n147 as builder
from scripts.training import eval_taiji_n1_v25_abstention_after_n147 as evaluator


def _excluded() -> tuple[set[str], set[str]]:
    names = {f"excluded{index:04d}" for index in range(40)}
    words = {"unknown", "alreadyused", *builder.n147_builder.n144_builder.DIRECTIONS}
    return names, words


def test_v25_face_uses_fresh_seed_and_preserves_the_matched_pair_contract() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)
    checks = builder.validate_face(face, excluded_names, excluded_words)
    validated = evaluator.validate_diagnostic_items(face["episodes"])

    assert face["seed"] == 125
    assert face["schema"] == "n1-v25-matched-abstention-diagnostic-face-v1"
    assert checks["accepted"] is True
    assert checks["pairs"] == 40
    assert validated == {"items": 80, "pairs": 40, "conditions": {"known": 40, "unknown": 40}}
    assert all(item["split"] == "diagnostic" for item in face["episodes"])


def test_diagnostic_item_validator_rejects_holdout_relabeling_and_duplicate_ids() -> None:
    excluded_names, excluded_words = _excluded()
    face = builder.build_face(excluded_names, excluded_words)
    holdout = [dict(item) for item in face["episodes"]]
    holdout[0]["split"] = "holdout"
    with pytest.raises(ValueError, match="diagnostic split required"):
        evaluator.validate_diagnostic_items(holdout)

    duplicate = [dict(item) for item in face["episodes"]]
    duplicate[1]["id"] = duplicate[0]["id"]
    with pytest.raises(ValueError, match="80 unique diagnostic items"):
        evaluator.validate_diagnostic_items(duplicate)


def test_measure_rejects_non_diagnostic_split_before_touching_the_model() -> None:
    class UntouchableModel:
        def __getattr__(self, _name: str):
            raise AssertionError("model methods must not run before the split guard")

    item = {"split": "holdout", "answer": "abcdef"}
    with pytest.raises(ValueError, match="diagnostic split and frozen position arm"):
        evaluator.measure_diagnostic(UntouchableModel(), item, "position", "safety-probe")


def test_measure_diagnostic_scores_without_relabiling_the_split(monkeypatch) -> None:
    class FakeModel:
        def teacher_logits(self, _tape, _session, _question, target):
            return torch.zeros((len(target) + 1, 257)), None

        def generate_trace(self, _tape, _session, _question, _limit):
            return SimpleNamespace(output=b"abcdef", raw=b"abcdef", stop_reason="boundary", generated_steps=7)

    paired = evaluator.frozen_eval.n132.e11.scorer.paired
    monkeypatch.setattr(paired, "project", lambda item: SimpleNamespace(question=item["question"]))
    monkeypatch.setattr(paired, "observe", lambda _model, _data, _session: None)
    monkeypatch.setattr(paired, "evidence", lambda _model, _session, _arm: object())
    item = {
        "id": "v25-pair-000-known",
        "pair_id": "v25-pair-000",
        "pair_condition": "known",
        "kind": "lookup",
        "split": "diagnostic",
        "target_start_index": 0,
        "paired_answer_length": 6,
        "answer_length": 6,
        "question": b"material of target?",
        "answer": "abcdef",
    }

    row = evaluator.measure_diagnostic(FakeModel(), item, "position", "diagnostic-probe")

    assert row["split"] == "diagnostic"
    assert row["free_exact"] is True
    assert row["output_hex"] == b"abcdef".hex()
    assert row["pair_id"] == item["pair_id"]


def test_source_union_excludes_the_entire_consumed_v24_face() -> None:
    entities, words, source_hashes, summary = builder._collect_source_union()

    assert len(entities) == 38_122
    assert len(words) == 15_288
    assert summary["n1_147_entity_exclusion_count"] == 36_802
    assert summary["n1_147_word_exclusion_count"] == 13_968
    assert summary["consumed_v24_entity_count"] == 1_320
    assert summary["consumed_v24_material_count"] == 1_320
    assert summary["v24_scored_rows"] == 0
    assert summary["v24_query_counter"] == 1
    assert summary["entity_overlap_with_previous_union"] == 0
    assert summary["word_overlap_with_previous_union"] == 0
    assert builder.n147_builder.n144_builder._verify_hashes(source_hashes)


def test_frozen_n1_148_plan_hash_and_endpoint_registry() -> None:
    assert builder.sha256(builder.PLAN) == builder.PLAN_SHA256
    specs = evaluator._endpoint_specs()
    assert len(specs) == 12
    assert {spec["source"] for spec in specs} == {"N1-132", "N1-140", "N1-141"}
