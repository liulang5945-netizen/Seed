"""Contract guard for the A-prime instrument (paired held-out mean-surprise).

Why these particular tests: the instrument exists because the CAP-0 behavioural surface is at a
capability floor (3/20, 1/16, 0/14) and cannot resolve the effect the frozen criteria ask about.
That makes the **reading rule** the dangerous part -- a continuous number is easy to over-read.  So
the verdict arithmetic is pinned in both directions here, and the disjointness of the evaluation set
is pinned against the arms' own committed manifests rather than a hand-typed row number.

The module under test is read-only: it calls ``Seed.score_bytes`` (documented as non-mutating),
trains nothing, and never writes under ``checkpoints/``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
INSTRUMENT = REPO / "scripts" / "training" / "eval_p3b_heldout_surprise.py"
ARM_MANIFESTS = (
    REPO / "plans" / "manifests" / "p3b_dialogue_fresh_manifest.json",
    REPO / "plans" / "manifests" / "p3b_all_fresh_manifest.json",
)


def _load(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    for entry in (str(path.parent), str(REPO)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument() -> Any:
    return _load("_p3b_heldout_under_test", INSTRUMENT)


def _state(whole: float, first: float, second: float) -> dict[str, float]:
    return {
        "whole_mean_surprise": whole,
        "slice_a_mean_surprise": first,
        "slice_b_mean_surprise": second,
    }


# --------------------------------------------------------------------------- #
# The evaluation set: disjoint from both arms, stable, and refused otherwise
# --------------------------------------------------------------------------- #


def test_the_start_row_is_derived_from_the_arm_manifests_not_typed(
    instrument: Any,
) -> None:
    expected = max(
        int(json.loads(path.read_text(encoding="utf-8"))["last_emitted_row"])
        for path in ARM_MANIFESTS
    )
    assert instrument.MIN_START_ROW == expected, "the arms moved their window; so must the holdout"
    assert instrument.EVAL_BYTES == 65_536


def test_the_derived_bytes_are_stable_bounded_and_recorded(instrument: Any) -> None:
    first, meta_first = instrument.derive_eval_rows(instrument.MIN_START_ROW + 1)
    second, meta_second = instrument.derive_eval_rows(instrument.MIN_START_ROW + 1)
    assert first == second, "the same rows must produce the same bytes"
    assert meta_first["sha256"] == meta_second["sha256"]
    assert 0 < len(first) <= instrument.EVAL_BYTES
    assert meta_first["rows_used"] >= 2, "a split-half needs at least two rows"
    assert meta_first["total_bytes"] == len(b"".join(first))
    assert meta_first["first_row_index"] > instrument.MIN_START_ROW
    assert meta_first["last_row_index"] >= meta_first["first_row_index"]
    joined = b"".join(first)
    assert len(joined) == len(joined.decode("utf-8").encode("utf-8")), "UTF-8 end to end"


def test_a_heldout_set_that_overlaps_the_arms_is_refused(instrument: Any) -> None:
    with pytest.raises(SystemExit, match="overlaps the arms"):
        instrument.derive_eval_rows(instrument.MIN_START_ROW)
    with pytest.raises(SystemExit, match="overlaps the arms"):
        instrument.derive_eval_rows(0)


def test_a_moved_evaluation_set_is_refused_before_any_model_loads(
    instrument: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "p3b_heldout_eval_manifest.json"
    manifest.write_text(
        json.dumps({"format": "taiji-p3b-heldout-eval-v1", "sha256": "0" * 64}), encoding="utf-8"
    )
    monkeypatch.setattr(instrument, "MANIFEST", manifest)
    monkeypatch.setattr(
        instrument, "STATES", {"start": tmp_path / "nope.pt"}
    )  # must never be reached
    with pytest.raises(SystemExit, match="refusing to score against a moved evaluation set"):
        instrument.run(instrument.MIN_START_ROW + 1, tmp_path / "report.json")


# --------------------------------------------------------------------------- #
# The reading rule: a continuous number is exactly what needs a pre-declared floor
# --------------------------------------------------------------------------- #


def test_a_difference_smaller_than_the_split_half_noise_is_not_resolved(
    instrument: Any,
) -> None:
    per_state = {
        "start": _state(3.000, 3.020, 2.980),  # halves disagree by 0.040 => that is the noise
        "treatment": _state(2.990, 3.000, 2.980),
        "control": _state(2.995, 3.010, 2.980),
    }
    verdict = instrument.paired_verdict(per_state)
    assert verdict["split_half_noise"] == pytest.approx(0.04)
    assert abs(verdict["treatment_minus_control_whole"]) < verdict["split_half_noise"]
    assert verdict["verdict_label"] == "not_resolved"


def test_a_difference_beyond_the_noise_with_both_halves_agreeing_is_resolved(
    instrument: Any,
) -> None:
    per_state = {
        "start": _state(3.000, 3.010, 2.990),
        "treatment": _state(2.900, 2.910, 2.890),
        "control": _state(3.000, 3.005, 2.995),
    }
    verdict = instrument.paired_verdict(per_state)
    assert verdict["both_halves_same_sign"] is True
    assert abs(verdict["treatment_minus_control_whole"]) > verdict["split_half_noise"]
    assert verdict["verdict_label"] == "resolved"
    assert verdict["treatment_minus_control_whole"] < 0, "treatment is the lower-surprise arm here"


def test_a_difference_that_flips_between_halves_is_not_resolved_even_if_large(
    instrument: Any,
) -> None:
    per_state = {
        "start": _state(3.000, 3.000, 3.000),
        "treatment": _state(3.000, 2.800, 3.200),
        "control": _state(3.000, 3.000, 3.000),
    }
    verdict = instrument.paired_verdict(per_state)
    assert verdict["both_halves_same_sign"] is False
    assert verdict["verdict_label"] == "not_resolved"


def test_not_resolved_is_never_worded_as_no_effect(instrument: Any) -> None:
    per_state = {
        "start": _state(3.0, 3.0, 3.0),
        "treatment": _state(3.0, 3.0, 3.0),
        "control": _state(3.0, 3.0, 3.0),
    }
    meaning = instrument.paired_verdict(per_state)["meaning_of_not_resolved"]
    assert "NOT evidence" in meaning
    for banned in ("no effect", "irrelevant", "无关"):
        assert banned not in meaning.lower()
