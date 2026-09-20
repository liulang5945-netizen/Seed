"""Contract guard for the readout-retrain step-0 cost calibration instrument.

Why this exists: the contract draft
(``plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md`` §4) authorises exactly one
pre-approval action -- a few-dozen-tick trial calibration whose whole purpose is to put *measured*
wall/tick and peak memory into §4.1 instead of a guessed number.  That instrument is only worth
trusting if two things are true, and neither is self-evident from reading the script:

* the three arms really are **mutually exclusive write surfaces**.  The A arm claims to train the
  readout with the motor side frozen; the C arm claims to train nothing.  If A also moved the
  motor, or C moved anything, the measured cost would belong to a configuration the contract does
  not describe -- so the instrument must refuse, not report.  This file pins that property against
  the real substrate, in both directions (A changes the readout and *not* the motor; C changes
  neither).
* the instrument **cannot be used as an unauthorised training run**: it has a hard tick ceiling, it
  never overwrites a report, it refuses a corpus whose lineage-derived skip point is unknown, and
  it refuses to replay the prefix its checkpoint has already consumed.

The shipped calibration report is pinned too, but only structurally (status, write surfaces, sha of
the substrate it was taken on) -- timing numbers are machine noise and asserting them would be
asserting nothing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "calibrate_taiji_r2_readout_retrain.py"
REPORT = REPO / "reports" / "taiji_r2_readout_retrain_step0_calibration.json"
LINEAGE_MANIFEST = REPO / "plans" / "manifests" / "p3b_all_fresh_manifest.json"

#: The substrate the calibration must have been taken on.  Taken from the product constant rather
#: than typed, so changing the default substrate turns this red instead of leaving a stale number
#: in the contract.
from api.seed_runtime import DEFAULT_CHECKPOINT as PRODUCT_DEFAULT  # noqa: E402


def _load_module() -> Any:
    name = "_readout_retrain_step0_under_test"
    if name in sys.modules:
        return sys.modules[name]
    for entry in (str(SCRIPT.parent), str(REPO)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument() -> Any:
    return _load_module()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_corpus(path: Path, texts: list[str]) -> None:
    # Mirrors ``build_p3b_arm_corpus``'s emission: one ``{"text": ...}`` row per line.
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def _write_manifest(path: Path, corpus: Path, skip_symbols: int) -> None:
    path.write_text(
        json.dumps(
            {
                "output": str(corpus),
                "skip_symbols": skip_symbols,
                "first_emitted_row": 0,
                "replay_symbols_from_seen_region": 0,
                "skip_derivation": {"method": "test fixture"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# The property the whole instrument rests on: arms do not share a write surface
# --------------------------------------------------------------------------- #


def _surface_flags(instrument: Any, name: str, symbols: list[int]) -> dict[str, bool]:
    """Run one arm briefly on the real substrate and report which surfaces moved."""

    import torch

    envelope = torch.load(PRODUCT_DEFAULT, map_location="cpu", weights_only=False)
    report = instrument.run_arm(
        checkpoint=envelope,
        name=name,
        symbols=symbols,
        warmup=2,
        device="cpu",
    )
    return {
        surface: report["digests_before"][surface] != report["digests_after"][surface]
        for surface in ("motor", "predictive_readout")
    }


@pytest.mark.skipif(not PRODUCT_DEFAULT.is_file(), reason="product default substrate absent")
def test_arm_a_trains_the_readout_and_leaves_the_motor_alone(instrument: Any) -> None:
    flags = _surface_flags(instrument, "A", [200 + index for index in range(6)])
    assert flags == {"motor": False, "predictive_readout": True}


@pytest.mark.skipif(not PRODUCT_DEFAULT.is_file(), reason="product default substrate absent")
def test_arm_b_trains_the_motor_and_leaves_the_readout_alone(instrument: Any) -> None:
    flags = _surface_flags(instrument, "B", [200 + index for index in range(6)])
    assert flags == {"motor": True, "predictive_readout": False}


@pytest.mark.skipif(not PRODUCT_DEFAULT.is_file(), reason="product default substrate absent")
def test_arm_c_trains_nothing(instrument: Any) -> None:
    """The reverse pin: 'two frozen sites' has to mean *no* surface moves.

    Without this line, the A test above would still pass if ``learn=False`` quietly wrote
    everywhere -- and then the C arm's cost would not be the drift control it is used as.
    """

    flags = _surface_flags(instrument, "C", [200 + index for index in range(6)])
    assert flags == {"motor": False, "predictive_readout": False}


def test_the_arm_specs_and_their_expected_surfaces_agree(instrument: Any) -> None:
    """The declared expectation must be the one the run checks against."""

    assert {name: spec["write_surface"] for name, spec in instrument.ARMS.items()} == {
        "A": ("predictive_readout",),
        "B": ("motor",),
        "C": (),
    }
    # A arm: the readout path only.  ``learn_motor=False`` is not consulted on that path; it is
    # there so the spec says out loud which side is meant to stay frozen.
    assert instrument.ARMS["A"]["observe_kwargs"]["readout"] == "predictive"
    assert instrument.ARMS["A"]["observe_kwargs"]["learn_motor"] is False
    assert instrument.ARMS["A"]["observe_kwargs"]["learn_fabric"] is False
    assert instrument.ARMS["A"]["observe_kwargs"]["learn_predictive_readout"] is True
    assert instrument.ARMS["B"]["observe_kwargs"]["readout"] == "action"
    assert instrument.ARMS["C"]["observe_kwargs"]["learn"] is False


# --------------------------------------------------------------------------- #
# It cannot be turned into a training run by accident
# --------------------------------------------------------------------------- #


def _expect_refusal(instrument: Any, capsys: Any, phrase: str) -> None:
    """``parser.error`` exits with code 2 and puts the reason on stderr, not in the exception."""

    with pytest.raises(SystemExit) as info:
        instrument.main()
    assert info.value.code not in (0, None)
    stderr = capsys.readouterr().err
    assert phrase in stderr, stderr


def test_tick_ceiling_refuses_a_training_sized_run(
    instrument: Any, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["calibrate", "--ticks", str(instrument.CALIBRATION_TICK_CAP + 1)],
    )
    _expect_refusal(instrument, capsys, "--ticks must be within")


def test_refuses_to_overwrite_an_existing_report(
    instrument: Any, tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    existing = tmp_path / "already-here.json"
    existing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["calibrate", "--out-report", str(existing)])
    _expect_refusal(instrument, capsys, "never overwrites")
    assert existing.read_text(encoding="utf-8") == "{}"


def test_refuses_a_report_path_under_the_weights_directory(
    instrument: Any, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(sys, "argv", ["calibrate", "--out-report", "checkpoints/step0.json"])
    _expect_refusal(instrument, capsys, "must not live under")


def test_unknown_arm_fails_closed(instrument: Any, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(sys, "argv", ["calibrate", "--arms", "A,Z"])
    _expect_refusal(instrument, capsys, "unknown arm")


def test_refuses_a_corpus_with_no_lineage_derived_skip_point(
    instrument: Any, tmp_path: Path
) -> None:
    """A hand-sliced corpus has no derivation, and guessing one would be replay by accident."""

    corpus = tmp_path / "handmade.jsonl"
    _write_corpus(corpus, ["甲乙丙丁"])
    other = tmp_path / "other.jsonl"
    _write_manifest(other, tmp_path / "different.jsonl", 10)
    with pytest.raises(SystemExit, match="no lineage-derived skip"):
        instrument.load_lineage_skip(other, corpus)


def test_refuses_to_slice_inside_the_seen_prefix(
    instrument: Any, tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    corpus = tmp_path / "arm.jsonl"
    _write_corpus(corpus, ["甲乙丙丁戊己庚辛壬癸"] * 4)
    manifest = tmp_path / "arm.json"
    _write_manifest(manifest, corpus, 12)
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--checkpoint",
            str(PRODUCT_DEFAULT),
            "--corpus",
            str(corpus),
            "--lineage-manifest",
            str(manifest),
            "--out-report",
            str(report),
            "--skip-symbols",
            "0",
        ],
    )
    _expect_refusal(instrument, capsys, "seen prefix")
    assert not report.exists()


# --------------------------------------------------------------------------- #
# End to end on a synthetic corpus: the guards around the real substrate hold
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not PRODUCT_DEFAULT.is_file(), reason="product default substrate absent")
def test_end_to_end_run_leaves_the_substrate_byte_identical(
    instrument: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    corpus = tmp_path / "arm.jsonl"
    _write_corpus(corpus, ["甲乙丙丁戊己庚辛壬癸"] * 4)
    manifest = tmp_path / "arm.json"
    _write_manifest(manifest, corpus, 6)
    report = tmp_path / "step0.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--checkpoint",
            str(PRODUCT_DEFAULT),
            "--corpus",
            str(corpus),
            "--lineage-manifest",
            str(manifest),
            "--ticks",
            "3",
            "--warmup-ticks",
            "1",
            "--arms",
            "A,B,C",
            "--out-report",
            str(report),
        ],
    )
    before = _sha256(PRODUCT_DEFAULT)
    assert instrument.main() == 0
    assert _sha256(PRODUCT_DEFAULT) == before

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["checkpoint_unchanged"] is True
    assert payload["corpus"]["replay_symbols_in_window"] == 0
    for name, arm in payload["arms"].items():
        assert arm["write_surface_ok"] is True, name
        assert arm["seconds_per_tick"] > 0.0
    assert payload["peak_working_set_status"] in {"measured", "unavailable"}
    if payload["peak_working_set_status"] == "unavailable":
        assert payload["peak_working_set_bytes"] is None


# --------------------------------------------------------------------------- #
# The shipped report, pinned structurally
# --------------------------------------------------------------------------- #


def test_the_shipped_calibration_is_internally_consistent() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["step"].startswith("step-0")
    assert payload["checkpoint_unchanged"] is True
    assert payload["checkpoint"]["sha256"] == payload["checkpoint_sha256_after"]
    assert set(payload["arms"]) == {"A", "B", "C"}
    for name, arm in payload["arms"].items():
        assert arm["write_surface_ok"] is True, name
        assert arm["realised_write_surface"] == arm["expected_write_surface"]
        assert arm["timed_ticks"] == payload["timed_ticks"]
    assert payload["timed_ticks"] <= 200, "the shipped run must be a calibration, not a training"


def test_the_shipped_calibration_was_taken_on_the_current_default_substrate() -> None:
    """A calibration against some other substrate is a stale reference, not a budget basis."""

    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert Path(payload["checkpoint"]["path"]) == PRODUCT_DEFAULT
    assert payload["checkpoint"]["sha256"] == _sha256(PRODUCT_DEFAULT)


def test_the_shipped_calibration_used_the_lineage_derived_skip() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    manifest = json.loads(LINEAGE_MANIFEST.read_text(encoding="utf-8"))
    assert payload["corpus"]["skip_symbols"] == manifest["skip_symbols"]
    assert payload["corpus"]["replay_symbols_in_window"] == 0
    assert payload["corpus"]["lineage"]["manifest"] == (
        "plans/manifests/p3b_all_fresh_manifest.json"
    )
