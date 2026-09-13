"""Contract guard for the M4 artifact audit.

The audit itself trains members for four seed offsets and runs five surface
measurements (about 65 seconds), so it is a scripted instrument.  What is pinned
here is the audit's *verdict* and the reasoning behind it, because those are what
decision D5 rests on:

* M4's positive gain must stay **specific** to the surface designed to require a
  handoff -- a gain appearing anywhere else would indicate an artifact;
* the frozen intervention-reality gate must keep reporting
  ``interventions_happened = true`` under M4;
* the gain must vanish under the frozen rule and every singleton must fail, so it
  cannot be attributed to one member or to a trivially solvable task;
* the two frozen-surface improvements must stay explained by a **handoff trace**,
  not by an unexplained cell flip.

The measured payload lives in ``reports/taiji_b0_m4_artifact_audit_20260913.json``
and is summarised in ``plans/reference/M5_B0_M4_ARTIFACT_AUDIT_20260913.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AUDIT_REPORT = REPO / "reports" / "taiji_b0_m4_artifact_audit_20260913.json"
AUDIT_SCRIPT = REPO / "scripts" / "training" / "audit_taiji_b0_m4_artifact.py"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))


def test_audit_is_read_only_and_rule_is_still_counterfactual(payload):
    assert payload["status"] == "draft_for_review"
    assert payload["rule_under_audit"] == "m4_failure_handoff"
    assert payload["frozen_attribute_intact"] is True
    assert payload["rule_delta"]["frozen_attribute_unchanged"] is True

    source = AUDIT_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("torch.save", "save_file", "from_checkpoint(", "optimizer"):
        assert forbidden not in source


def test_specificity_positive_gain_only_on_the_handoff_surface(payload):
    specificity = payload["specificity"]
    assert specificity["specific"] is True
    assert specificity["unexpected_positive_surfaces"] == []
    assert specificity["surfaces_with_positive_gain"] == {"create_and_override": 2.0}


def test_other_surfaces_do_not_gain(payload):
    """The rule must not manufacture a gain where no handoff is required."""

    surfaces = payload["surfaces"]
    assert surfaces["frozen_validation"]["best_pair_gain"] == -0.5
    assert surfaces["frozen_validation"]["positive_same_reference_gain"] is False
    assert surfaces["dual_requirement"]["best_pair_gain"] == -2.0
    assert surfaces["dual_requirement"]["positive_same_reference_gain"] is False
    assert surfaces["create_then_patch"]["best_pair_gain"] == 0.0
    assert surfaces["create_then_patch"]["positive_same_reference_gain"] is False


def test_intervention_reality_holds_on_every_surface(payload):
    """No inert non-baseline cell under M4 -- the P5.2b defect must not return."""

    for label, surface in payload["surfaces"].items():
        reality = surface["intervention_reality"]
        assert reality["interventions_happened"] is True, label
        assert reality["zero_step_intervention_episodes"] == 0, label


def test_gain_requires_both_the_handoff_and_both_members(payload):
    lesion = payload["lesion"]
    assert lesion["gain_vanishes_without_handoff"] is True
    assert lesion["member_lesion"]["all_singletons_fail"] is True
    assert lesion["baseline_rule"]["interleaved_contexts"] == 0
    assert lesion["baseline_rule"]["best_pair_gain"] <= 0.0
    assert lesion["audited_rule"]["interleaved_contexts"] == 4
    assert lesion["audited_rule"]["best_pair_gain"] == 2.0


def test_the_gain_clears_the_strictest_candidate_reference(payload):
    surface = payload["surfaces"]["create_and_override"]
    strictest = max(row["required"] for row in surface["reference_requirements"])
    assert strictest == 1.65
    assert surface["best_pair_gain"] > strictest
    assert all(row["feasible"] is True for row in surface["reference_requirements"])


def test_reversed_order_also_produces_the_gain(payload):
    """Robustness: the win is not an artifact of the frozen pair priority."""

    reversed_surface = payload["surfaces"]["create_and_override_reversed"]
    assert reversed_surface["best_pair_gain"] == 2.0
    assert reversed_surface["interleaved_contexts"] == 4
    assert reversed_surface["best_pair"] == "member-c+member-d"
    assert payload["surfaces"]["create_and_override"]["best_pair"] == "member-a+member-c"


def test_frozen_surface_improvements_are_explained_by_a_handoff_trace(payload):
    """An unexplained cell flip would be exactly the P5.2b failure mode again."""

    forensics = payload["improvement_forensics"]
    assert forensics["changed_count"] == 2
    changed = {row["cell"]: row for row in forensics["changed_cells"]}
    assert set(changed) == {"member-a+member-b", "member-a+member-c"}
    for row in changed.values():
        assert row["delta"] == 0.25

    # member-a+member-b: the baseline starves member-b of any turn; M4 hands off.
    ab = changed["member-a+member-b"]
    flips = []
    for item in ab["per_context"]:
        baseline = item["baseline"]["episodes"][0]
        audited = item["audited"]["episodes"][0]
        if not baseline["success"] and audited["success"]:
            flips.append((baseline, audited))
    assert flips, "expected at least one fail -> success flip"
    for baseline, audited in flips:
        assert baseline["stop_reason"] == "all_members_exhausted"
        assert audited["stop_reason"] == "goal_reached"
        # The baseline never let member-b act at all.
        assert all(step["chosen"] in (None, "member-a") for step in baseline["trace"])
        # M4 does: member-b reads and patches.
        chosen = [step["chosen"] for step in audited["trace"]]
        assert "member-b" in chosen
        kinds = [step["kind"] for step in audited["trace"]]
        assert "workspace.apply_patch" in kinds

    # member-a+member-c: the baseline starves member-c; M4 interleaves a and c.
    ac = changed["member-a+member-c"]
    ac_flips = [
        (item["baseline"]["episodes"][0], item["audited"]["episodes"][0])
        for item in ac["per_context"]
        if not item["baseline"]["episodes"][0]["success"]
        and item["audited"]["episodes"][0]["success"]
    ]
    assert ac_flips, "expected at least one fail -> success flip"
    for baseline, audited in ac_flips:
        assert baseline["stop_reason"] == "all_members_exhausted"
        assert audited["stop_reason"] == "goal_reached"
        assert all(step["chosen"] in (None, "member-a") for step in baseline["trace"])
        chosen = [step["chosen"] for step in audited["trace"]]
        assert "member-c" in chosen
        # A genuine interleaving: control alternates between the two members.
        assert len({member for member in chosen if member}) == 2


def test_seed_robustness_holds_across_offsets(payload):
    robustness = payload["seed_robustness"]
    assert robustness["offsets"] == [0, 101, 202]
    assert robustness["all_seeds_positive_on_candidate"] is True
    assert robustness["all_seeds_no_gain_on_frozen"] is True
    assert robustness["min_candidate_gain"] == 2.0
    assert robustness["max_candidate_gain"] == 2.0
    for row in robustness["rows"]:
        assert row["create_and_override"]["best_pair_gain"] == 2.0
        assert row["create_and_override"]["all_singletons_fail"] is True
        assert row["frozen_validation"]["positive_same_reference_gain"] is False


def test_new_stop_reason_is_disclosed(payload):
    """``all_members_blocked`` is new semantics and must stay visible."""

    reasons = payload["surfaces"]["create_and_override"]["stop_reasons"]
    assert "all_members_blocked" in reasons
    # It is distinct from the pre-existing reason.
    assert reasons["all_members_blocked"] > 0
    assert reasons.get("all_members_exhausted", 0) > 0
