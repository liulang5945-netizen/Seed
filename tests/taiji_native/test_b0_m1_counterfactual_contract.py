"""Contract guard for the M1/M4 counterfactual measurement (decision D5).

The measurement itself trains the frozen members and runs six counterfactual
variants (about 50 seconds), so it is a scripted instrument.  What is pinned here
is the integrity of the method and the load-bearing results:

* the counterfactual is the frozen executor plus a **documented, unique** set of
  replacements -- if the frozen rule changes, the anchors stop being unique and
  the build must fail loudly rather than silently measure something else;
* ``frozen._member_episode`` is never rebound, so the mechanism under test is
  untouched and the counterfactual cannot leak into a gate;
* the candidate surface is **satisfiable and has a mandatory order**, without which
  every negative composition result would be uninterpretable;
* the six variants' verdicts are pinned, including the one that works.

The measured payload lives in
``reports/taiji_b0_m1_counterfactual_20260913.json`` and is summarised in
``plans/reference/M5_B0_M1_COUNTERFACTUAL_RESULT_20260913.md``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE_SCRIPT = REPO / "scripts" / "training" / "probe_taiji_b0_m1_counterfactual.py"
MODULE_REPORT = REPO / "reports" / "taiji_b0_m1_counterfactual_20260913.json"
FROZEN_GATE = REPO / "scripts" / "training" / "eval_taiji_p5_2b_group_causal_corpora_gate.py"


def _load(name: str, path: Path):
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
def counterfactual():
    return _load("_b0_m1_counterfactual_under_test", MODULE_SCRIPT)


@pytest.fixture(scope="module")
def frozen(counterfactual):
    return counterfactual.load_frozen()


# --------------------------------------------------------------------------- #
# Method integrity
# --------------------------------------------------------------------------- #


def test_every_anchor_is_unique_in_the_frozen_source(counterfactual, frozen):
    """A non-unique anchor means the frozen rule changed; the build must abort."""

    source = counterfactual.inspect.getsource(frozen._member_episode)
    assert source.count(counterfactual.FROZEN_SELECTION) == 1
    assert source.count(counterfactual.CUE_ANCHOR) == 1

    for variant, pairs in counterfactual.VARIANTS.items():
        assert pairs, f"{variant} has no replacements"
        for anchor, replacement in pairs:
            assert anchor != replacement
            assert anchor.endswith("\n")


def test_counterfactual_never_rebinds_the_frozen_episode(counterfactual, frozen):
    """The mechanism under test must stay untouched, or the probe would be circular."""

    before = frozen._member_episode
    episode_fn, delta = counterfactual.build_counterfactual(frozen, "m4_failure_handoff")

    assert frozen._member_episode is before
    assert episode_fn is not before
    assert delta["frozen_attribute_unchanged"] is True
    assert delta["replacement_count"] == 2


def test_counterfactual_unknown_variant_is_rejected(counterfactual, frozen):
    with pytest.raises(ValueError):
        counterfactual.build_counterfactual(frozen, "not-a-variant")


def test_counterfactual_is_read_only_and_not_wired_into_any_gate(counterfactual):
    source = MODULE_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("torch.save", "save_file", "from_checkpoint(", "optimizer"):
        assert forbidden not in source, f"the counterfactual must not contain {forbidden!r}"

    # No gate or runner may import the counterfactual.
    for name in (
        "eval_taiji_p5_2b_group_causal_corpora_gate.py",
        "eval_taiji_p5_2a_predictive_execution_gate.py",
        "eval_taiji_p5_2c_double_prime_unseen_combination_transfer_gate.py",
        "eval_taiji_p5_2c_triple_prime_representation_repair_gate.py",
    ):
        gate = REPO / "scripts" / "training" / name
        assert "m1_counterfactual" not in gate.read_text(encoding="utf-8"), name


# --------------------------------------------------------------------------- #
# The candidate surface must be interpretable
# --------------------------------------------------------------------------- #


def test_candidate_surface_is_satisfiable_and_order_mandatory(counterfactual):
    """Without satisfiability every negative result would be void."""

    payload = json.loads(MODULE_REPORT.read_text(encoding="utf-8"))
    sat = payload["candidate_satisfiability"]

    assert sat["satisfiable"] is True
    assert sat["unreachable"] == []
    assert sat["order_is_mandatory"] is True
    assert len(sat["mandatory_order_tasks"]) == 4

    for row in sat["rows"]:
        assert row["forward_reaches_goal"] is True
        assert row["reversed_reaches_goal"] is False
        # Forward order creates the file before the override; reversed does not.
        forward = [step["kind"] for step in row["forward"]["steps"]]
        assert forward.index("workspace.create") < forward.index("editor.set_language")
        # The override binds but fails to execute while the file is missing.
        reversed_steps = row["reversed"]["steps"]
        assert reversed_steps[0]["kind"] == "editor.set_language"
        assert reversed_steps[0]["bound"] is True
        assert reversed_steps[0]["success"] is False


def test_member_repertoire_is_read_from_train_evidence(counterfactual):
    payload = json.loads(MODULE_REPORT.read_text(encoding="utf-8"))
    assert payload["member_repertoire"] == {
        "member-a": 3,
        "member-b": 3,
        "member-c": 3,
        "member-d": 4,
    }


# --------------------------------------------------------------------------- #
# Variant verdicts
# --------------------------------------------------------------------------- #


def _variants() -> dict[str, Any]:
    payload = json.loads(MODULE_REPORT.read_text(encoding="utf-8"))
    return {item["delta"]["variant"]: item for item in payload["variants"]}


def test_six_variants_were_measured_and_none_touched_the_frozen_episode():
    variants = _variants()
    assert set(variants) == {
        "m1a_no_progress",
        "m1b_tick_rotation",
        "m2_per_member_progress",
        "m2a_progress_plus_handoff",
        "m3_repertoire_aware",
        "m4_failure_handoff",
    }
    for item in variants.values():
        assert item["delta"]["frozen_attribute_unchanged"] is True


def test_five_of_six_variants_fail_to_produce_a_handoff():
    """The negative half of the result: rule tuning alone does not work."""

    variants = _variants()
    for name in (
        "m1a_no_progress",
        "m1b_tick_rotation",
        "m2_per_member_progress",
        "m2a_progress_plus_handoff",
        "m3_repertoire_aware",
    ):
        effect = variants[name]["effect"]
        assert effect["positive_same_reference_gain"] is False, name
        for order in effect["by_order"].values():
            assert order["interleaved_contexts"] == 0, name


def test_m1a_is_safe_but_inert_and_m1b_is_inadmissible():
    """The cheapest proxy rules: one changes nothing, the other breaks the baseline."""

    variants = _variants()

    m1a = variants["m1a_no_progress"]
    assert m1a["regression"]["no_regression"] is True
    assert m1a["regression"]["regressions"] == []
    assert m1a["regression"]["improvements"] == []

    m1b = variants["m1b_tick_rotation"]
    assert m1b["regression"]["no_regression"] is False
    assert len(m1b["regression"]["regressions"]) == 3
    assert len(m1b["regression"]["improvements"]) == 1


def test_m4_is_the_measured_working_configuration():
    """Per-member progress + failure-driven handoff: no regression, positive gain."""

    m4 = _variants()["m4_failure_handoff"]

    # It does not regress the frozen surface, and improves two cells.
    assert m4["regression"]["no_regression"] is True
    assert m4["regression"]["regressions"] == []
    improved = {row["cell"]: row["delta"] for row in m4["regression"]["improvements"]}
    assert improved == {"member-a-member-b": 0.25, "member-a-member-c": 0.25}

    # On the candidate surface every context now yields an interleaved trajectory,
    # and a pair reaches the escape-route ceiling (success - B) * k / n = 2.0.
    effect = m4["effect"]
    assert effect["positive_same_reference_gain"] is True
    for order, item in effect["by_order"].items():
        assert item["interleaved_contexts"] == 4, order
        assert item["best_pair_gain"] == 2.0, order
        assert all(rate == 0.0 for rate in item["singleton_success_rates"].values())
        # Every candidate reference becomes feasible, including the strictest.
        assert all(row["feasible"] is True for row in item["reference_requirements"])

    assert effect["by_order"]["frozen_order"]["best_pair"] == "member-a+member-c"
    assert effect["by_order"]["reversed_order"]["best_pair"] == "member-c+member-d"


def test_m4_beats_the_strictest_candidate_reference():
    """+2.0 clears the all-singleton-oracle bar of 1.65, which no prior round did."""

    m4 = _variants()["m4_failure_handoff"]
    for item in m4["effect"]["by_order"].values():
        strictest = max(row["required"] for row in item["reference_requirements"])
        assert item["best_pair_gain"] > strictest


def test_m4_delta_is_two_documented_replacements():
    """The working configuration must stay reviewable: two anchors, 21 added lines."""

    delta = _variants()["m4_failure_handoff"]["delta"]
    assert delta["replacement_count"] == 2
    anchors = [item["anchor"].strip() for item in delta["replacements"]]
    assert anchors == [
        "chosen = bindable[0]",
        "cues = tuple([cue] * (len(steps) + 1))",
    ]
    assert delta["added_lines"] == 21
    assert delta["injected_names"] == []


def test_frozen_gate_still_uses_priority_fallback(frozen):
    """If the frozen rule ever changes, these anchors stop matching and this fails."""

    source = FROZEN_GATE.read_text(encoding="utf-8")
    assert "chosen = bindable[0]" in source
    assert "cues = tuple([cue] * (len(steps) + 1))" in source
