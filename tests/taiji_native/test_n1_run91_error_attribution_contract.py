"""PLAN-N1-91 attribution instrument contract.

The instrument reads one sealed report and one face manifest, so every risk here is a
silent mis-parse that would publish an attribution no one can re-derive. Both sides of
each guard are walked: the four §1 bindings must answer True on the real face and must
each go red on a one-nudge mutation, and the six §2 shape labels must be produced by
handcrafted rows, not inferred from the real distribution.

Nothing is written in the repo: the single-attempt refusal is walked against a tmp file.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import attribute_taiji_n1_confirmation_errors_run89 as attr  # noqa: E402


def row(**overrides) -> dict:
    base = {
        "id": "v5-confirm-000",
        "kind": "lookup",
        "split": "holdout",
        "target_symbols": [99, 115, 99, 97, 117, 120, 256],
        "teacher_symbols": [99, 115, 99, 97, 117, 120, 256],
        "teacher_tokens_correct": 7,
        "teacher_exact": True,
        "teacher_ce": 0.0001,
        "raw_hex": "637363617578",
        "output_hex": "637363617578",
        "stop_reason": "boundary",
        "generated_steps": 7,
        "free_exact": True,
        "seed": 11,
        "arm": "position",
    }
    base.update(overrides)
    return base


def shape_of(**overrides) -> tuple:
    return attr.shape_class(row(**overrides))


def test_shape_labels_walk_all_six_classes() -> None:
    assert shape_of() == ("equal_output_not_exact", False)
    assert shape_of(free_exact=False) == ("equal_output_not_exact", False)
    assert shape_of(output_hex="", generated_steps=1, free_exact=False) == (
        "empty_output",
        False,
    )
    assert shape_of(output_hex="62", free_exact=False) == ("first_symbol_wrong", False)
    assert shape_of(output_hex="6362", free_exact=False) == ("prefix_then_diverge", False)
    assert shape_of(output_hex="63", free_exact=False) == ("truncated_prefix", False)
    assert shape_of(output_hex="637363617578ff", free_exact=False) == ("overshoot", False)
    assert shape_of(stop_reason="cap", generated_steps=64, free_exact=False) == (
        "equal_output_not_exact",
        True,
    )


def test_shape_classes_cannot_be_confused_with_the_capped_column() -> None:
    label, capped = shape_of(output_hex="62", stop_reason="cap", generated_steps=64)
    assert (label, capped) == ("first_symbol_wrong", True)


def test_agreement_class_is_exhaustive_and_named() -> None:
    assert attr.agreement_class([True, True, True]) == "all_three_right"
    assert attr.agreement_class([False, False, False]) == "all_three_wrong"
    assert attr.agreement_class([True, False, False]) == "exactly_one_right"
    assert attr.agreement_class([True, True, False]) == "exactly_two_right"


@pytest.fixture(scope="module")
def sealed() -> tuple[dict, dict]:
    return attr.load()


def test_guards_answer_true_on_the_sealed_face(sealed) -> None:
    report, face = sealed
    guarded = attr.guards(report, face)
    assert guarded["B1_shape"]["rows"] == 288
    assert guarded["B2_face_seal"]["matched_convention"]
    assert guarded["B3_aggregates"]["per_arm_free_exact"] == report["aggregate_exact"]
    assert guarded["B4_field_semantics"]["boundary_rows_incoherent"] == 0


def _mutate(report: dict, path: list, value) -> dict:
    clone = copy.deepcopy(report)
    node = clone
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return clone


def test_b1_goes_red_when_a_row_is_dropped(sealed) -> None:
    report, face = sealed
    broken = _mutate(report, ["rows"], report["rows"][:-1])
    with pytest.raises(attr.Refuse, match="B1 rows must be 288"):
        attr.guards(broken, face)


def test_b1_goes_red_when_row_ids_stop_matching_the_face(sealed) -> None:
    report, face = sealed
    broken = copy.deepcopy(report)
    broken["rows"][0]["id"] = "v5-confirm-999"
    with pytest.raises(attr.Refuse, match="B1 row ids differ"):
        attr.guards(broken, face)


def test_b2_goes_red_when_the_face_bytes_change(sealed, tmp_path: Path, monkeypatch) -> None:
    report, face = sealed
    moved = tmp_path / "n1_fresh_confirmation_v5_20261016.json"
    moved.write_bytes(attr.FACE.read_bytes().replace(b"cscaux", b"cscauy"))
    monkeypatch.setattr(attr, "FACE", moved)
    assert moved == attr.FACE, "the patched face must be the one the guard opens"
    with pytest.raises(attr.Refuse, match="B2 face digest matches neither convention"):
        attr.guards(report, face)


def test_b3_goes_red_when_one_free_exact_is_flipped(sealed) -> None:
    report, face = sealed
    broken = copy.deepcopy(report)
    for entry in broken["rows"]:
        if entry["free_exact"]:
            entry["free_exact"] = False
            break
    with pytest.raises(attr.Refuse, match="B3 recomputed per-arm totals differ"):
        attr.guards(broken, face)


def test_b4_goes_red_when_a_boundary_row_loses_its_step_identity(sealed) -> None:
    report, face = sealed
    broken = copy.deepcopy(report)
    for entry in broken["rows"]:
        if entry["stop_reason"] == "boundary":
            entry["generated_steps"] += 1
            break
    with pytest.raises(attr.Refuse, match="B4 .* boundary rows break"):
        attr.guards(broken, face)


def test_target_without_a_boundary_symbol_is_refused(sealed) -> None:
    report, _face = sealed
    broken = copy.deepcopy(report)
    sample = attr.output_symbols(broken["rows"][0])
    broken["rows"][0]["target_symbols"] = sample
    broken["rows"][0]["free_exact"] = False
    with pytest.raises(attr.Refuse, match="does not end with boundary"):
        attr.attribute(broken)


def test_attribution_totals_are_self_consistent_on_the_real_rows(sealed) -> None:
    report, face = sealed
    attr.guards(report, face)
    reading = attr.attribute(report)
    assert reading["axis1_groups_total"] == 96
    assert reading["axis2_unclassified"] == 0
    assert (
        reading["axis3_compounding_only"] + reading["axis3_selection_side"]
        == reading["axis2_free_wrong_rows"]
    )
    assert reading["axis5_all_wrong_items"] and len(reading["axis5_all_wrong_items"]) == (
        reading["axis5_all_three_wrong_groups"]
    )
    assert reading["axis1_agreement"]["all_three_wrong"] == len(reading["axis5_all_wrong_items"])
    for seed in attr.SEEDS:
        by_seed = reading["axis1_agreement_by_seed"][str(seed)]
        assert sum(by_seed.values()) == 32, "each seed must group into 32 triples"
    assert {
        arm: sum(1 for entry in report["rows"] if entry["arm"] == arm and entry["free_exact"])
        for arm in attr.ARMS
    } == report["aggregate_exact"]


def test_publish_refuses_a_second_attempt_file(tmp_path: Path) -> None:
    target = tmp_path / "attribution.json"
    target.write_text("{}", encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises(attr.Refuse, match="attribution output already exists"):
        attr.build(target)
    assert target.read_bytes() == before, "refusal must not rewrite the sealed output"


def test_readiness_and_approved_are_exclusive() -> None:
    assert attr.main(["--readiness"]) == 0
    assert attr.main([]) == 2
    assert attr.main(["--readiness", "--approved"]) == 2


def test_approved_run_publishes_one_report_that_reparses(tmp_path: Path) -> None:
    target = tmp_path / "attribution.json"
    assert attr.main(["--approved", "--out-report", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["accepted"] is True
    assert payload["status"] == "published"
    assert payload["not_a_capability_reading"] is True
    assert payload["confirmation_queries"] == 0
    assert payload["claims_forbidden_by_preregistration"]
