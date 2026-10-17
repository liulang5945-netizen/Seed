"""PLAN-N1-92 probe contract: the anchor must be able to fail, and the exits must be reachable.

The probe reimplements only the oracle-address extraction, because the pinned instrument's
own extractor refuses non-train items and its bytes may not be edited. That makes the
bit-identity anchor the load-bearing guard of this whole gate, so the first test walks it on
the real train face and the second proves it goes red when the rewritten math differs.
Nothing here loads a checkpoint or queries the v5 face.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import (  # noqa: E402
    audit_taiji_n1_address_intervention as oracle,
)
from scripts.training import (  # noqa: E402
    probe_taiji_n1_oracle_address_run92 as probe,
)


class FakeTape:
    def __init__(self, positions):
        self.positions = list(positions)


def item(**overrides) -> dict:
    base = {
        "id": "v5-x",
        "kind": "lookup",
        "split": "holdout",
        "observations": ["jyfhkghvmc uses cscaux.", "wfubrppzti uses mzpwll."],
        "question": "material of jyfhkghvmc?",
        "answer": "cscaux",
    }
    base.update(overrides)
    return base


def tape_for(item: dict) -> FakeTape:
    positions = []
    for index, text in enumerate(item["observations"]):
        source = f"source-{index + 1}"
        for offset, _char in enumerate(text.encode()):
            positions.append((source, offset))
    return FakeTape(positions)


def test_extractor_matches_the_pinned_one_on_a_constructed_pair() -> None:
    sample = item()
    tape = tape_for(sample)
    train_sample = copy.deepcopy(sample)
    train_sample["split"] = "train"
    expected = oracle.addresses(train_sample, tape)
    assert probe.oracle_addresses(sample, tape) == expected
    start = sample["observations"][0].encode().rfind(b"cscaux")
    assert start > 0
    assert expected[1] == [tape.positions.index(("source-1", start + step)) for step in range(6)]


def test_guarded_kinds_are_classified_out_with_reason_codes() -> None:
    ok, reason = probe.classify(item(kind="transition", answer="alpha"))
    assert (ok, reason) == (
        False,
        "no_unique_uses_answer_observation",
    )
    ok, reason = probe.classify(item(kind="unknown", answer="unknown"))
    assert (ok, reason) == (
        False,
        "answer_is_the_literal_unknown_with_no_source",
    )
    ok, reason = probe.classify(item())
    assert (ok, reason) == (True, "addressable")


def test_anchor_proves_equal_and_can_go_red(monkeypatch) -> None:
    reading = probe.anchor()
    assert reading["checked_equal"] > 0, "the anchor walked an empty face"
    assert reading["checked_unequal"] == 0

    exact = probe.oracle_addresses

    def drifted(sample, tape):
        source, positions = exact(sample, tape)
        return source, positions[1:] + positions[:1]

    monkeypatch.setattr(probe, "oracle_addresses", drifted)
    with pytest.raises(probe.Refuse, match=f"diverges on {reading['checked_equal']}"):
        probe.anchor()


def test_coverage_face_shape_and_its_own_guard(tmp_path, monkeypatch) -> None:
    covered = probe.coverage()
    assert covered["addressable"] == 16
    assert covered["items_total"] == 32
    assert sum(covered["excluded_reasons"].values()) == 16

    face = probe.read_json(probe.V5)
    promoted = 0
    for episode in face["episodes"]:
        if episode["kind"] == "transition" and promoted < 2:
            episode["kind"] = "lookup"
            episode["observations"] = [
                f"{episode['entities'][0]} uses cscaux.",
                "wfubrppzti uses mzpwll.",
            ]
            episode["answer"] = "cscaux"
            promoted += 1
    widened = tmp_path / "widened_v5.json"
    widened.write_text(json.dumps(face, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(probe, "V5", widened)
    monkeypatch.setattr(probe, "V5_SHA", probe.sha(widened))
    with pytest.raises(probe.Refuse, match="the addressable set is"):
        probe.coverage()


def test_modes_adjudicate_and_the_four_exits_stay_exclusive() -> None:
    def entry(seed, arm, native, position_six, position_seven, gains, losses):
        return {
            "seed": seed,
            "arm": arm,
            "modes": {
                "native": {"six_exact": native, "seven_exact": native},
                "source": {"six_exact": 0, "seven_exact": 0},
                "position": {"six_exact": position_six, "seven_exact": position_seven},
            },
            "position_paired_gains": gains,
            "position_paired_losses": losses,
            "ruler_usable": native <= probe.NATIVE_CEILING,
        }

    lifted = [
        entry(
            seed, arm, native=10, position_six=15, position_seven=12, gains=6 + seed - 11, losses=1
        )
        for seed in probe.SEEDS
        for arm in probe.ARM_ORDER
    ]
    reading = probe.adjudicate(lifted, 16)
    assert reading["endpoints_usable"] == 9
    assert reading["oracle_lift_supported"] is True
    assert reading["arm_indistinguishable"] is True
    assert reading["exit_named"].startswith("⑵")

    by_arm = [
        entry(
            seed,
            arm,
            native=10,
            position_six=15 + 2 * (arm == "cut"),
            position_seven=12,
            gains=6,
            losses=1,
        )
        for seed in probe.SEEDS
        for arm in probe.ARM_ORDER
    ]
    reading = probe.adjudicate(by_arm, 16)
    assert reading["arm_indistinguishable"] is False
    assert reading["exit_named"].startswith("⑴")

    flat = [
        entry(seed, arm, native=10, position_six=11, position_seven=11, gains=1, losses=1)
        for seed in probe.SEEDS
        for arm in probe.ARM_ORDER
    ]
    reading = probe.adjudicate(flat, 16)
    assert reading["oracle_lift_supported"] is False
    assert reading["exit_named"].startswith("⑶")

    saturated = [
        entry(seed, arm, native=13, position_six=14, position_seven=14, gains=6, losses=0)
        for seed in probe.SEEDS
        for arm in probe.ARM_ORDER
    ]
    reading = probe.adjudicate(saturated, 16)
    assert reading["ruler_unusable"] is True
    assert reading["oracle_lift_supported"] is False
    assert reading["exit_named"].startswith("⑷")


def test_cli_modes_are_exclusive_and_refuse_an_existing_output(tmp_path: Path, monkeypatch) -> None:
    assert probe.main([]) == 2
    assert probe.main(["--readiness", "--execute-approved"]) == 2

    sealed = tmp_path / "probe.json"
    sealed.write_text("{}", encoding="utf-8")
    before = sealed.read_bytes()
    monkeypatch.setattr(probe, "REPORT", sealed)
    assert probe.main(["--execute-approved"]) == 2
    assert sealed.read_bytes() == before, "refusal must not rewrite the sealed output"


def test_native_anchor_decision_sums_and_can_produce_a_stop() -> None:
    assert probe.anchor_decision([]) == (0, 0)
    clean = [{"anchor_equal": 16, "anchor_unequal": 0} for _ in probe.SEEDS]
    assert probe.anchor_decision(clean) == (48, 0)
    dirty = clean[:1] + [{"anchor_equal": 15, "anchor_unequal": 1}]
    checked, unequal = probe.anchor_decision(dirty)
    assert (checked, unequal) == (32, 1), "a diverged pair must be counted, not dropped"
