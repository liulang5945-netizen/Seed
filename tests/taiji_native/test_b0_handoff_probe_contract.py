"""Contract guard for the route-B handoff feasibility probe.

The probe itself trains the frozen members and executes ~500 episodes (about
25 seconds), so it is a scripted instrument rather than a unit test.  What *is*
pinned here is everything that would silently rot around it:

* the candidate surfaces keep the structural properties that make them
  informative -- ``create_and_override`` must require two capabilities no single
  policy covers, and ``create_then_patch`` must stay single-member solvable so the
  report keeps documenting *why* that shape is unusable;
* the probe must **reuse** the frozen composition rule instead of re-implementing
  it, because a private copy could drift from the mechanism under test;
* the pure helpers must round-trip, so a reported number can be recomputed by
  hand from the same inputs.

The measured results live in
``reports/taiji_b0_handoff_feasibility_probe_20260913.json`` and are summarised in
``plans/reference/M5_B0_HANDOFF_PROBE_RESULT_20260913.md``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = REPO / "scripts" / "training" / "probe_taiji_b0_handoff_feasibility.py"
PROBE_REPORT = REPO / "reports" / "taiji_b0_handoff_feasibility_probe_20260913.json"
FROZEN_GATE = REPO / "scripts" / "training" / "eval_taiji_p5_2b_group_causal_corpora_gate.py"


def _load(name: str, path: Path):
    scripts = str(path.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load("_b0_handoff_probe_under_test", PROBE_SCRIPT)


@pytest.fixture(scope="module")
def frozen(probe):
    return probe.load_frozen()


# --------------------------------------------------------------------------- #
# The probe must reuse the frozen mechanism, never copy it
# --------------------------------------------------------------------------- #


def test_probe_does_not_reimplement_the_frozen_composition_rule(probe):
    """A private copy of the rule could drift from the mechanism under test."""

    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert "def _member_episode" not in source
    assert "def _train_members" not in source
    assert "frozen._member_episode(" in source
    assert "frozen._execute_matrix(" in source
    assert "frozen._train_members(" in source


def test_probe_is_read_only_and_registers_no_task(probe):
    """It must not write a checkpoint, register a task, or touch a gate."""

    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("torch.save", "from_checkpoint(", "optimizer", "save_file"):
        assert forbidden not in source
    # candidate tasks live only here, and only for the probe partition
    assert 'partition="probe"' in source
    assert "b0probe-" in source


# --------------------------------------------------------------------------- #
# Candidate surfaces keep the properties that make them informative
# --------------------------------------------------------------------------- #


def test_create_and_override_requires_two_capabilities_no_member_covers(probe, frozen):
    """The shape that produced ``combination_only = 4/4`` must stay intact."""

    tasks = probe.create_and_override_tasks(frozen, count=2)
    assert len(tasks) == 2
    for task in tasks:
        # Nothing exists yet, so a patch cannot bind and create is mandatory ...
        assert task.initial_files == {}
        assert set(task.goal_files) == {task.main_path}
        # ... and the goal additionally demands a recorded explicit override,
        # which ``workspace.create`` never provides.
        assert task.goal_language == {task.main_path: "python"}
        assert task.requires_explicit_language_override is True
        assert task.template == "create_and_override"

    frozen.p52a._assert_nontrivial_goals(tasks, partition="probe")


def test_create_then_patch_stays_single_member_solvable(probe, frozen):
    """This shape is documented as unusable, and the reason must stay visible.

    ``p52a._bind`` binds ``content = task.goal_files[main_path]`` for
    ``workspace.create``, so creating already lands on the goal.  If that ever
    changes, the report's conclusion about this candidate becomes stale.
    """

    tasks = probe.create_then_patch_tasks(frozen, count=1)
    task = tasks[0]
    assert task.initial_files == {}
    assert set(task.goal_files) == {task.main_path}
    assert task.goal_language == {}

    # Re-derive the structural reason from the frozen binder, not from prose.
    params, _provenance, failure = frozen.p52a._bind(
        "workspace.create", task, {"root": Path("."), "last_undo_token": None}
    )
    assert failure is None
    assert params["content"] == task.goal_files[task.main_path]


def test_dual_requirement_is_expressible_on_a_single_main_path(probe, frozen):
    """``_bind`` targets ``main_path`` for every action, so both goals share a file."""

    tasks = probe.dual_requirement_tasks(frozen, count=1)
    task = tasks[0]
    assert set(task.initial_files) == {task.main_path}
    assert set(task.goal_files) == {task.main_path}
    assert task.goal_language == {task.main_path: "python"}
    assert task.requires_explicit_language_override is True

    override_params, _p, override_failure = frozen.p52a._bind(
        "editor.set_language", task, {"root": Path("."), "last_undo_token": None}
    )
    assert override_failure is None
    assert override_params["path"] == task.main_path
    assert override_params["user_override"] is True


def test_every_candidate_builder_is_registered_for_the_probe(probe):
    """A new candidate that is not in the registry would never be measured."""

    assert set(probe.CANDIDATE_BUILDERS) == {
        "dual_requirement",
        "create_then_patch",
        "create_and_override",
    }
    for builder in probe.CANDIDATE_BUILDERS.values():
        assert callable(builder)


# --------------------------------------------------------------------------- #
# Pure helpers round-trip
# --------------------------------------------------------------------------- #


def _episode(task_id: str, members: tuple[str, ...], success: bool, repeat: int) -> dict[str, Any]:
    return {
        "episode_id": f"e:{task_id}:r{repeat}:{'-'.join(members) or 'none'}",
        "task_id": task_id,
        "active_members": list(members),
        "success": success,
    }


def test_outcome_by_cell_encodes_success_as_plus_one(probe):
    episodes = [
        _episode("t0", (), False, 0),
        _episode("t0", (), False, 1),
        _episode("t0", ("member-a",), True, 0),
        _episode("t0", ("member-a",), False, 1),
        _episode("t1", ("member-a",), True, 0),
        _episode("t1", ("member-a",), True, 1),
    ]
    outcomes = probe.outcome_by_cell(episodes)

    assert outcomes[()]["mean_outcome"] == -1.0
    # member-a: +1, -1, +1, +1 -> mean 0.5, success rate 0.75
    assert outcomes[("member-a",)]["mean_outcome"] == 0.5
    assert outcomes[("member-a",)]["success_rate"] == 0.75
    assert outcomes[("member-a",)]["ctx:t1"] == 1.0


def test_table_from_outcomes_rebuilds_a_context_table(probe):
    episodes = [
        _episode("t0", (), False, 0),
        _episode("t0", ("member-a",), True, 0),
        _episode("t0", ("member-b",), False, 0),
        _episode("t0", ("member-a", "member-b"), True, 0),
    ]
    outcomes = probe.outcome_by_cell(episodes)
    table = probe.table_from_outcomes(outcomes, ["t0"], ["member-a", "member-b"])

    assert table.contexts == ("t0",)
    assert table.blank("t0") == -1.0
    assert table.singleton("t0", "member-a") == 1.0
    assert table.combination("t0", ("member-a", "member-b")) == 1.0
    assert table.pair_cells() == (("member-a", "member-b"),)

    precheck = probe.load_precheck()
    # member-a already solves t0, so the context is not combination-only solvable.
    assert precheck.combination_only_solvable_contexts(table) == []
    assert precheck.arbitration_ceiling(table, ("member-a", "member-b")) == 0.0


# --------------------------------------------------------------------------- #
# The frozen report the probe is validated against must not drift
# --------------------------------------------------------------------------- #


def test_probe_report_matches_the_measured_findings():
    """Pin the load-bearing numbers so a silent re-run cannot rewrite history."""

    payload = json.loads(PROBE_REPORT.read_text(encoding="utf-8"))
    assert payload["status"] == "draft_for_review"
    assert payload["ground_truth"]["reproduced"] is True
    assert payload["ground_truth"]["mismatches"] == []
    assert payload["ground_truth"]["cells_compared"] == 11

    surfaces = {surface["label"]: surface for surface in payload["surfaces"]}
    assert set(surfaces) == {
        "dual_requirement",
        "create_then_patch",
        "create_and_override",
    }

    # Every surface: the frozen rule never produced an interleaved trajectory.
    for surface in surfaces.values():
        assert surface["condition_2"]["interleaved_contexts"] == 0
        assert surface["condition_2"]["satisfied"] is False

    # The two single-member-solvable shapes: some singleton wins outright.
    for label, winner in (("dual_requirement", "member-d"), ("create_then_patch", "member-c")):
        rates = surfaces[label]["singleton_success_rates"]
        assert rates[winner] == 1.0
        assert all(rate == 0.0 for member, rate in rates.items() if member != winner)
        assert surfaces[label]["combination_only_contexts"] == []

    # The shape that fixes the task layer: no singleton wins, every context is
    # combination-only solvable, and every candidate reference becomes feasible.
    hard = surfaces["create_and_override"]
    assert all(rate == 0.0 for rate in hard["singleton_success_rates"].values())
    assert len(hard["combination_only_contexts"]) == 4
    assert hard["combination_only_gain_potential"] == 2.0
    assert all(row["feasible"] is True for row in hard["reference_requirements"])
    assert all(row["max_clearable_reference"] == 1.85 for row in hard["reference_requirements"])
    # ... and yet the mechanism still cannot do it: the layers are separated.
    assert hard["condition_2"]["satisfied"] is False
    assert hard["order_sensitivity"]["mean_order_headroom"] == 0.0


def test_gate_reports_which_composition_rule_it_ships(frozen):
    """This file's analysis is about rule_revision 0; the gate declares what it runs.

    Landing HANDOFF-M4 replaces ``chosen = bindable[0]``, so "the old line is still
    there" is not a durable guard.  What is durable: exactly one of the two rules is
    present, and ``RULE_REVISION`` names it.  The cell structure below is rule-independent
    and stays pinned.
    """

    source = FROZEN_GATE.read_text(encoding="utf-8")
    body = source.split("def _member_episode", 1)[1].split("\ndef ", 1)[0]
    revision_0 = "chosen = bindable[0]" in body
    revision_1 = 'return finish("all_members_blocked")' in body
    assert revision_0 != revision_1, "exactly one composition rule may be present"
    assert getattr(frozen, "RULE_REVISION", None) == (0 if revision_0 else 1)
    assert frozen.CELL_MEMBER_SETS[0] == ()
    assert len(frozen.CELL_MEMBER_SETS) == 11
