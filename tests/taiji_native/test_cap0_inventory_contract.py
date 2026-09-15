"""Contract guard for the CAP-0 whole-model inventory.

CAP-0 (``plans/active/roadmap/07_MINI_MODEL_DELIVERY.md`` section 5.1) is an
inventory, and its findings are the kind that quietly rot: a default checkpoint
swap, a migration added without a note, or a probe turning into a scored baseline
would each change the meaning of the result without changing its shape.

So the load-bearing facts are asserted, not described:

* the default entry serves an **untrained** state (tick 2) while the most trained
  checkpoint on record is orders of magnitude ahead;
* the trained checkpoint is **not loadable** through the current chain, and the
  older files carry no current-format component block;
* the default chain's probe output collapses to **one template signature**;
* the unmeasured dimensions stay marked ``untested`` -- never 0, never a pass.

The inventory itself runs a fresh-process load plus a seven-turn probe (~15 s), so
this file reads the committed report and never re-runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports" / "taiji_cap0_inventory_20260915.json"
DELIVERY_PLAN = REPO / "plans" / "active" / "roadmap" / "07_MINI_MODEL_DELIVERY.md"
RUNNER = REPO / "scripts" / "training" / "eval_taiji_cap0_inventory.py"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_delivery_plan_and_runner_exist(report):
    assert DELIVERY_PLAN.is_file()
    assert RUNNER.is_file()
    assert report["delivery_plan"].endswith("07_MINI_MODEL_DELIVERY.md")
    assert report["status"] == "completed"
    assert report.get("error") is None


def test_default_entry_serves_an_untrained_state(report):
    """The wiring defect: the default path is not the trained path."""

    reality = report["model_reality"]
    assert reality["default_checkpoint"] == "seed_corpus.pt"
    assert reality["default_tick"] == 2
    assert reality["most_trained_checkpoint"] == "seed_beta.pt"
    assert reality["most_trained_tick"] == 16_000_000
    assert reality["wiring_defect"] is True
    assert "tick=2" in reality["wiring_defect_statement"]


def test_trained_state_is_stranded_not_merely_unwired(report):
    """A second, separate defect: the trained checkpoint cannot load at all."""

    entry = report["raw_output_inventory"]["most_trained_entry"]
    assert entry["probed"] is True
    assert entry["load_ok"] is False
    assert "identity organ checkpoint payload is missing" in entry["load_error"]
    assert entry["turns_answered"] == 0

    inventory = {row["filename"]: row for row in report["checkpoint_inventory"]}
    # The current-format envelope is the one the default entry uses.
    assert inventory["seed_corpus.pt"]["has_metadata"] is True
    assert inventory["seed_corpus.pt"]["tick"] == 2
    assert inventory["seed_beta.pt"]["tick"] == 16_000_000
    assert inventory["resumed_seed_corpus.pt"]["tick"] == 4_800_200
    assert inventory["seed_corpus_prev_20260823.pt"]["tick"] is None


def test_model_format_is_read_from_the_substrate_key(report):
    """The decisive evidence for the fix: the stranded files are a *supported* legacy format.

    ``taiji/model.py`` reads the model format from the ``substrate`` payload and
    accepts ``taiji-native-v8``/``v9`` via ``LEGACY_CHECKPOINT_FORMATS``.  The sibling
    ``taiji`` key is the Seed adapter envelope with an unrelated version string, so a
    check that read that key would misreport the format.
    """

    reality = report["model_reality"]
    assert reality["default_model_format"] == "taiji-native-v10"
    assert reality["most_trained_model_format"] == "taiji-native-v8"
    assert reality["stranded_is_legacy_format"] is True

    inventory = {row["filename"]: row for row in report["checkpoint_inventory"]}
    for name in ("seed_beta.pt", "resumed_seed_corpus.pt", "seed_corpus_prev_20260823.pt"):
        assert inventory[name]["model_format"] == "taiji-native-v8", name
        assert inventory[name]["has_taiji_adapter_block"] is False, name
    # The default is current-format and *does* carry the adapter block: the two keys
    # must not be conflated.
    assert inventory["seed_corpus.pt"]["has_taiji_adapter_block"] is True

    diagnosis = reality["stranded_defect_diagnosis"]
    assert "LEGACY_CHECKPOINT_FORMATS" in diagnosis
    assert "is_legacy_checkpoint" in diagnosis
    assert "2726-2732" in diagnosis
    assert "policy decision" in diagnosis


def test_probe_output_collapses_to_one_template(report):
    """07 section 3.B: fixed phrasing does not count as content ability."""

    default = report["raw_output_inventory"]["default_entry"]
    assert default["turns_answered"] == 7
    assert default["turns_errored"] == 0
    signature = default["template_signature"]
    assert signature["templated"] is True
    assert signature["distinct_signatures"] == 1
    assert "<PROMPT>" in signature["signature"]
    assert "已收到你的问题" in signature["signature"]


def test_raw_outputs_are_verbatim_and_prompt_echoing(report):
    turns = report["raw_output_inventory"]["default_turns"]
    assert len(turns) == 7
    for turn in turns:
        assert "raw_output" in turn, turn["label"]
        assert turn["prompt"] in turn["raw_output"]
        assert turn["output_bytes"] > 0


def test_dimension_a_measured_and_everything_else_untested(report):
    dimensions = report["dimensions"]
    assert dimensions["A_model_reality"].startswith("measured")
    for key in (
        "B_basic_dialogue",
        "C_basic_qa",
        "D_context",
        "E_instruction_reasoning",
        "G_uncertainty_safety",
        "H_performance_stability",
        "I_continual_adaptation",
    ):
        assert dimensions[key].startswith("untested"), key
    # F must not be upgraded to a whole-model claim by a selector-surface result.
    assert dimensions["F_representative_capability"].startswith("partial")
    assert "not whole-model conversation ability" in dimensions["F_representative_capability"]


def test_missing_checkpoint_is_rejected_deterministically(report):
    reality = report["model_reality"]
    assert reality["missing_checkpoint_rejected"] is True
    assert "FileNotFoundError" in reality["missing_checkpoint_error"]
    assert reality["fresh_process"] is True
    assert reality["external_provider_used"] is False


def test_probe_is_labelled_diagnostic_not_a_scored_baseline(report):
    """07 section 4.1: the official set must be frozen before any score is seen."""

    inventory = report["raw_output_inventory"]
    assert inventory["probe_kind"].startswith("diagnostic")
    assert "not the frozen B-H evaluation set" in inventory["probe_kind"]
    assert "freeze the CAP evaluation set" in report["next_decision"]
    assert "BEFORE any candidate score is seen" in report["next_decision"]
    # The inventory must state its own restraint.
    joined = " ".join(report["does_not_do"])
    assert "trains nothing" in joined
    assert "does not freeze or score" in joined


def test_gap_statement_separates_the_three_facts(report):
    gap = report["gap_statement"]
    assert "WIRING" in gap
    assert "STRANDED STATE" in gap
    assert "CAPABILITY" in gap
    assert "UNTESTED rather than failed" in gap
    assert "no trained state is currently reachable" in gap
