"""Contract guard for the N1 structure-space probe.

N1 asks whether ``m4_failure_handoff`` gains because of the one composition structure
measured so far (create + explicit override) or because of something general about
failure handoffs.  The probe answers it by enumerating every structure the frozen goal
predicate can express and measuring each one, so the assertions pinned here are the ones
the answer leans on, in **both directions**:

* the grid must be the whole expressible space minus the trivial cell, and every cell it
  keeps must be measurable **under the contract** -- a cell no member is allowed to
  finish would turn a zero gain into a fake structural finding, which is exactly the
  mistake the first iteration of this probe made;
* the jointly-required prediction must be checked against measurement, and a
  contradiction must stay visible rather than be smoothed away;
* every cell M4 changes must be attributed to a second acting member, and the
  attribution classifier must really refuse a flip with no mechanism behind it;
* the three section-5.2 candidates must stay reported as inexpressible, because that is
  the boundary on what decision D5 can claim from this evidence.

Measured payloads: ``reports/taiji_b0_structure_space_probe_20260913.json`` (the archived
run every structural claim above is made from) and
``reports/taiji_b0_structure_space_probe_m4landed_20260915.json`` (the same probe after
HANDOFF-M4 shipped, pinned at the bottom of this file so "the counterfactual predicted it
and the shipped rule reproduced it" is a checked statement, not a recollection).
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE_SCRIPT = REPO / "scripts" / "training" / "probe_taiji_b0_structure_space.py"
MODULE_REPORT = REPO / "reports" / "taiji_b0_structure_space_probe_20260913.json"

#: The structure the hardening pass measured, and the row N1 has to widen it to.
DESIGNED_CELL = "create__override"
CREATE_ROW = ["create__mismatch", "create__observation", "create__override"]
GRID_CELLS = 11


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
def payload() -> dict:
    return json.loads(MODULE_REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probe():
    return _load("_b0_structure_space_under_test", MODULE_SCRIPT)


@pytest.fixture(scope="module")
def counterfactual(probe):
    return probe.load_counterfactual()


@pytest.fixture(scope="module")
def frozen(counterfactual):
    return counterfactual.load_frozen()


# --------------------------------------------------------------------------- #
# Read-only discipline
# --------------------------------------------------------------------------- #


def test_probe_is_read_only_and_the_archived_report_states_its_scope(payload):
    """Scoped to the sealed revision-0 artifact: it describes the rule as of 2026-09-13."""

    assert payload["status"] == "draft_for_review"
    assert payload["rule_under_audit"] == "m4_failure_handoff"
    assert payload["frozen_attribute_intact"] is True

    source = MODULE_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("torch.save", "save_file", "from_checkpoint(", "optimizer"):
        assert forbidden not in source


def test_module_docstring_states_the_question_and_the_bound(probe):
    """The probe must not be re-badged later as a mechanism implementation."""

    doc = probe.__doc__ or ""
    assert "surface" in doc and "structure" in doc
    assert "cannot express" in doc
    assert "Read-only" in doc


# --------------------------------------------------------------------------- #
# The grid is the expressible space, not a selection from it
# --------------------------------------------------------------------------- #


def test_grid_is_the_full_route_product_minus_the_trivial_cell(probe):
    cells = probe.grid()
    assert len(cells) == GRID_CELLS
    assert len({cell["label"] for cell in cells}) == GRID_CELLS
    assert [cell["ordinal"] for cell in cells] == list(range(GRID_CELLS))
    assert {cell["content_route"] for cell in cells} == set(probe.CONTENT_ROUTES)
    assert {cell["language_route"] for cell in cells} == set(probe.LANGUAGE_ROUTES)
    assert "none__none" not in {cell["label"] for cell in cells}


def test_grid_axes_come_from_the_goal_clauses_the_binder_can_ask_for(payload):
    surface = payload["binder_expression_surface"]
    assert surface["cells_in_grid"] == GRID_CELLS
    assert surface["cells_skipped_as_trivial"] == ["none__none"]
    assert surface["binder_targets_main_path_only"] is True
    assert set(surface["goal_clause_present"]) == {
        "task.goal_files",
        "task.goal_language",
        "requires_explicit_language_override",
    }
    assert len(surface["bindable_kinds"]) == surface["bindable_kind_count"] == 10
    assert {"workspace.create", "editor.set_language", "workspace.apply_patch"} <= set(
        surface["bindable_kinds"]
    )


def test_combination_only_prediction_is_the_existence_precondition(probe):
    """Predicted jointly-required == create route with a language clause, exactly.

    The negative half matters as much as the positive one: the ``patch`` row is *not*
    predicted jointly-required, because ``member-d`` patches and selects inside one policy.
    """

    predicted = {
        (content, language)
        for content in probe.CONTENT_ROUTES
        for language in probe.LANGUAGE_ROUTES
        if probe.cell_is_combination_only(content, language)
    }
    assert predicted == {("create", route) for route in ("observation", "override", "mismatch")}
    assert probe.cell_is_combination_only("patch", "mismatch") is False
    assert probe.cell_is_combination_only("create", "none") is False
    assert probe.cell_is_combination_only("none", "override") is False


class _Record:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _stub_frozen() -> Any:
    class StubFrozen:
        class p52:
            ScriptedStep = _Record

            @staticmethod
            def _patch_ops(base: str, old: str, new: str) -> tuple[str, str, str]:
                return (base, old, new)

        class p52a:
            Task = _Record

    return StubFrozen()


def _task_index(task: Any) -> int:
    return int(task.kwargs["task_id"].rsplit("-", 1)[1])


def test_each_cell_wires_only_the_routes_it_claims_to_probe(probe):
    """One axis moves at a time, so a verdict is about structure and not about wording."""

    frozen = _stub_frozen()
    tasks = {
        cell["label"]: probe.build_cell_tasks(frozen, cell)["tasks"][0] for cell in probe.grid()
    }

    for label, task in tasks.items():
        content_route, language_route = label.split("__")
        kwargs = task.kwargs
        index = _task_index(task)
        name = next(iter(kwargs["goal_files"]))
        kinds = [step.args[0] for step in kwargs["reference_steps"]]

        assert (kwargs["initial_files"] == {}) is (content_route == "create")
        expected_goal = (
            probe.PATCHED_TEMPLATE.format(i=index)
            if content_route == "patch"
            else probe.BASE_TEMPLATE.format(i=index)
        )
        assert kwargs["goal_files"] == {name: expected_goal}

        has_resolve = "workspace.programming_language.resolve" in kinds
        has_set_language = "editor.set_language" in kinds
        assert has_resolve is (language_route != "none")
        assert has_set_language is (language_route in ("override", "mismatch"))
        if has_set_language:
            step = next(s for s in kwargs["reference_steps"] if s.args[0] == "editor.set_language")
            assert step.args[1]["user_override"] is True
        assert kwargs["requires_explicit_language_override"] is (
            language_route in ("override", "mismatch")
        )

        expected_language = {
            "none": None,
            "observation": probe.INFERABLE_LANGUAGE,
            "override": probe.INFERABLE_LANGUAGE,
            "mismatch": probe.NON_INFERABLE_LANGUAGE,
        }[language_route]
        assert kwargs["goal_language"] == ({name: expected_language} if expected_language else {})

        assert kinds.count("workspace.apply_patch") == (1 if content_route == "patch" else 0)
        assert kinds.count("workspace.create") == (1 if content_route == "create" else 0)


def test_content_carries_the_evidence_the_contract_reads(probe):
    """A bare assignment produced ``language_evidence_ambiguous`` and voided four cells."""

    assert "def " in probe.BASE_TEMPLATE
    assert "def " in probe.PATCHED_TEMPLATE
    assert probe.PATCHED_TEMPLATE != probe.BASE_TEMPLATE
    index = probe.INDEX_BASE
    assert probe.PATCHED_TEMPLATE.format(i=index).strip().endswith("+ 1")


# --------------------------------------------------------------------------- #
# Validity under the contract, not under the executor alone
# --------------------------------------------------------------------------- #


def test_every_grid_cell_is_measurable_under_the_contract(payload):
    rows = payload["validity"]
    assert len(rows) == GRID_CELLS
    assert payload["not_measurable_cells"] == []
    assert sorted(payload["cells_measured"]) == sorted(row["cell"] for row in rows)
    for row in rows:
        assert row["measurable"] is True, row["cell"]
        assert row["satisfiable"] is True
        assert row["nontrivial"] is True
        assert row["contract_admissible"] is True
        assert row["scripted_path_clean"] is True
        assert row["not_measurable_because"] == []
        for item in row["rows"]:
            assert item["forward_reaches_goal"] is True, item["task_id"]
            assert item["contract_blocked_steps"] == []
            assert item["scripted_blocked_steps"] == []


def test_the_validity_gate_runs_the_contract_and_not_just_the_executor(probe, frozen):
    """The check must be able to refuse: a task with no work is not a testable cell."""

    cell = {
        "label": "none__none",
        "ordinal": 90,
        "content_route": "none",
        "language_route": "none",
    }
    row = probe.validity(frozen, cell)
    assert row["nontrivial"] is False
    assert row["measurable"] is False
    assert row["not_measurable_because"] == ["tick0_already_satisfied"]


def test_order_is_mandatory_only_where_a_file_must_exist_first(payload):
    by_cell = {row["cell"]: row for row in payload["validity"]}
    for cell in CREATE_ROW:
        assert by_cell[cell]["order_is_mandatory"] is True, cell
    for cell, row in by_cell.items():
        if not cell.startswith("create__"):
            assert row["order_is_mandatory"] is False, cell


# --------------------------------------------------------------------------- #
# The answer N1 was asked for
# --------------------------------------------------------------------------- #


def test_gain_reproduces_across_the_create_row_and_is_not_confined(payload):
    verdict = payload["verdict"]
    assert verdict["positive_gain_cells"] == CREATE_ROW
    assert DESIGNED_CELL in verdict["positive_gain_cells"]
    assert verdict["gain_confined_to_create_and_override"] is False
    assert verdict["gain_reproduces_on_a_second_structure"] is True
    assert verdict["positive_gain_count"] == len(CREATE_ROW)


def test_gain_is_exactly_escaped_oracle_defeat_on_those_cells(payload):
    rows = {row["cell"]: row for row in payload["rows"]}
    for cell in CREATE_ROW:
        row = rows[cell]
        assert row["best_pair_gain_frozen"] == 0.0
        assert row["best_pair_gain_audited"] == 2.0
        assert row["gain_delta"] == 2.0
        assert row["interleaved_contexts_frozen"] == 0
        assert row["interleaved_contexts_audited"] > 0
        assert row["positive_under_frozen"] is False


def test_the_rest_of_the_grid_is_unmoved_by_m4(payload):
    rows = {row["cell"]: row for row in payload["rows"]}
    others = [row["cell"] for row in payload["rows"] if row["cell"] not in CREATE_ROW]
    assert len(others) == GRID_CELLS - len(CREATE_ROW)
    for cell in others:
        row = rows[cell]
        assert row["gain_delta"] == 0.0, cell
        assert row["positive_under_m4"] is False, cell
        assert row["observed_combination_only"] is False, cell
    # the patch row is strictly worse than the oracle under both rules: pairs cannot win
    # where a singleton already covers every clause
    for cell in ("patch__observation", "patch__override", "patch__mismatch"):
        assert rows[cell]["best_pair_gain_audited"] == -2.0, cell


def test_every_positive_cell_shares_the_existence_precondition(payload):
    shared = payload["verdict"]["shared_by_positive_cells"]
    assert shared == {
        "existence_precondition": True,
        "combination_only": True,
        "requires_second_member": True,
    }
    rows = {row["cell"]: row for row in payload["rows"]}
    for cell in payload["verdict"]["positive_gain_cells"]:
        assert rows[cell]["observed_combination_only"] is True
        assert rows[cell]["content_route"] == "create"
        assert rows[cell]["existence_precondition"] is True


def test_prediction_matches_measurement_on_every_cell(payload):
    verdict = payload["verdict"]
    assert verdict["combination_only_prediction_contradicted"] == []
    assert verdict["predicted_combination_only_cells"] == CREATE_ROW
    assert verdict["observed_combination_only_cells"] == CREATE_ROW
    assert verdict["prediction_contradictions_are_findings"] is True


def test_frozen_rule_gains_nowhere_and_m4_leaves_nothing_unexplained(payload):
    verdict = payload["verdict"]
    assert verdict["m4_regresses_cells"] == []
    assert verdict["cells_with_unexplained_change"] == []
    for row in payload["rows"]:
        assert row["unexplained_changes"] == []
        assert row["positive_under_frozen"] is False
        if row["gain_delta"] > 0.0:
            assert row["handoff_explained_changes"], row["cell"]


def test_handoff_explained_changes_really_show_two_actors(payload):
    """Not a label: every jointly-required context of an explained change has two actors."""

    checked = 0
    for row in payload["rows"]:
        for change in row["handoff_explained_changes"]:
            required = [item for item in change["per_context"] if item["jointly_required"]]
            assert required, change["cell"]
            assert change["delta"] > 0.0
            for item in required:
                assert len(item["acting_members"]) >= 2, item
                checked += 1
    assert checked >= 2 * len(CREATE_ROW)


def test_the_three_positive_cells_span_two_distinct_frozen_failure_shapes(payload):
    """How far the replication really goes, stated as a shape count rather than a cell count.

    All three create-row cells share one outcome fingerprint, so they are one structural
    factor seen through three routes -- not three independent confirmations.  What *is*
    independent is the failure they repair: ``create x observation`` dies under the frozen
    rule on a contract interception (assessing the language of a file that does not exist
    yet), while ``create x override`` and ``create x mismatch`` die on the step cap.
    """

    rows = {row["cell"]: row for row in payload["rows"]}
    intercepted = rows["create__observation"]["stop_reasons_frozen"]
    capped = rows["create__override"]["stop_reasons_frozen"]
    mismatched = rows["create__mismatch"]["stop_reasons_frozen"]

    assert any(key.startswith("contract_intercepted:language_") for key in intercepted)
    assert "step_cap" not in intercepted
    assert capped.get("step_cap", 0) > 0
    assert mismatched.get("step_cap", 0) > 0
    assert not any(key.startswith("contract_intercepted:language_") for key in capped)

    def frozen_shape(cell: str) -> str:
        reasons = rows[cell]["stop_reasons_frozen"]
        if any(key.startswith("contract_intercepted:language_") for key in reasons):
            return "contract_interception"
        if reasons.get("step_cap", 0) > 0:
            return "step_cap"
        return "other"

    assert {frozen_shape(cell) for cell in CREATE_ROW} == {"contract_interception", "step_cap"}

    for cell in CREATE_ROW:
        assert "goal_reached" in rows[cell]["stop_reasons_audited"], cell
        assert rows[cell]["stop_reasons_audited"]["all_members_blocked"] > 0, cell
        assert "all_members_blocked" not in rows[cell]["stop_reasons_frozen"], cell
        assert "goal_reached" not in rows[cell]["stop_reasons_frozen"], cell


def test_fallback_only_changes_move_no_claim(payload):
    """A pair rate can move where a singleton already succeeds; that is not a gain."""

    rows = {row["cell"]: row for row in payload["rows"]}
    recorded = 0
    for row in payload["rows"]:
        for change in row["fallback_only_changes"]:
            assert all(not item["jointly_required"] for item in change["per_context"]), row["cell"]
            recorded += 1
    assert recorded > 0
    assert payload["verdict"]["fallback_only_pair_cells"] == [
        "member-a+member-b",
        "member-a+member-c",
    ]
    assert rows["create__none"]["changed_cells"] > 0
    assert rows["create__none"]["gain_delta"] == 0.0
    assert rows["create__none"]["positive_under_m4"] is False


def test_the_audited_rule_does_not_move_the_comparison_reference(payload):
    """A rule change that lowers any singleton's success would inflate the oracle gain.

    H2's criterion is ``mean(P - max_i S_i)``, so the reference ``S_i`` belongs to the rule
    being tested.  If a handoff rule made singletons worse, ``+2.000`` could be manufactured
    by weakening the comparator instead of by real collaboration.  Verified by hand on the
    widened report and pinned here so any future rule change that lowers the reference fails.
    """

    for row in payload["rows"]:
        assert row["singleton_success_rates_frozen"] == row["singleton_success_rates_audited"], row[
            "cell"
        ]
    # and the three winning cells really do have a zero reference to beat
    for cell in CREATE_ROW:
        rates = {r["cell"]: r for r in payload["rows"]}[cell]["singleton_success_rates_audited"]
        assert set(rates.values()) == {0.0}, cell


def test_seed_sweep_reproduces_every_create_row_gain(payload, probe):
    sweep = payload["seed_sweep"]
    assert sweep["offsets"] == list(probe.SEED_OFFSETS)
    assert sweep["positive_cells_every_seed"] == CREATE_ROW
    assert sweep["positive_cells_any_seed"] == CREATE_ROW
    assert sweep["reality_ok_everywhere"] is True
    for row in sweep["rows"]:
        expected = 2.0 if row["cell"] in CREATE_ROW else 0.0
        if row["cell"] in CREATE_ROW:
            assert row["best_pair_gain"] == expected, row
            assert row["positive"] is True, row


def test_grid_separates_the_cells_it_claims_to_separate(payload):
    distinct = payload["outcome_distinctness"]
    assert distinct["cells_compared"] == GRID_CELLS
    assert distinct["distinct_outcome_fingerprints"] == 3
    assert distinct["grid_separates_cells"] is True


def test_distinctness_diagnostic_is_not_a_constant(probe):
    """Identical fingerprints must read as *no* separation, so the check can fail."""

    def row(cell, gain, combo, interleaved):
        return {
            "cell": cell,
            "best_pair_gain_audited": gain,
            "best_pair_gain_frozen": 0.0,
            "combination_only_contexts": combo,
            "interleaved_contexts_audited": interleaved,
        }

    same = probe.outcome_distinctness([row("a", 2.0, 2, 2), row("b", 2.0, 2, 2)])
    assert same["distinct_outcome_fingerprints"] == 1
    assert same["grid_separates_cells"] is False

    differ = probe.outcome_distinctness([row("a", 2.0, 2, 2), row("b", 0.0, 0, 0)])
    assert differ["distinct_outcome_fingerprints"] == 2
    assert differ["grid_separates_cells"] is True


def test_no_measured_cell_is_inert_under_either_rule(payload):
    """A zero-step cell is what made P5.2b a false success; it must not return."""

    for row in payload["rows"]:
        assert row["intervention_reality_frozen_ok"] is True, row["cell"]
        assert row["intervention_reality_audited_ok"] is True, row["cell"]


# --------------------------------------------------------------------------- #
# The classifier itself, on synthetic surfaces (both directions)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthetic_surface() -> dict:
    """One jointly-required context and one that a singleton already solves."""

    def episode(task_id: str, members: tuple[str, ...], success: bool) -> dict:
        return {"task_id": task_id, "active_members": list(members), "success": success}

    return {
        "contexts": ["c-joined", "c-solo"],
        "singleton_success_rates": {"member-a": 0.5, "member-c": 0.0},
        "episodes_raw": [
            episode("c-joined", (), False),
            episode("c-joined", ("member-a",), False),
            episode("c-joined", ("member-c",), False),
            episode("c-solo", (), False),
            episode("c-solo", ("member-a",), True),
            episode("c-solo", ("member-c",), False),
        ],
    }


def _change(cell: str, context: str, actors: tuple[str, ...]) -> dict:
    return {
        "cell": cell,
        "delta": 1.0,
        "baseline_success_rate": 0.0,
        "audited_success_rate": 1.0,
        "per_context": [
            {
                "baseline": {"task_id": context, "episodes": []},
                "audited": {
                    "task_id": context,
                    "episodes": [
                        {"trace": [{"chosen": actor, "executed": True} for actor in actors]}
                    ],
                },
            }
        ],
    }


def test_jointly_required_context_ids_are_recomputed_per_context(probe, synthetic_surface):
    assert probe.combination_only_context_ids(synthetic_surface) == {"c-joined"}


def test_changed_cells_split_into_explained_fallback_and_unexplained(probe, synthetic_surface):
    forensics = {
        "changed_cells": [
            _change("member-a+member-c", "c-joined", ("member-a", "member-c")),
            _change("member-a+member-b", "c-solo", ("member-a",)),
            _change("member-b+member-c", "c-joined", ("member-c",)),
        ]
    }
    result = probe.classify_changed_cells(forensics, synthetic_surface)
    assert [item["cell"] for item in result["handoff_explained"]] == ["member-a+member-c"]
    assert [item["cell"] for item in result["fallback_only"]] == ["member-a+member-b"]
    assert [item["cell"] for item in result["unexplained"]] == ["member-b+member-c"]
    assert result["jointly_required_contexts"] == ["c-joined"]


def test_a_second_actor_on_a_solved_context_is_not_a_handoff_claim(probe, synthetic_surface):
    """Two actors alone are not enough: the context has to need the combination."""

    forensics = {
        "changed_cells": [_change("member-a+member-c", "c-solo", ("member-a", "member-c"))]
    }
    result = probe.classify_changed_cells(forensics, synthetic_surface)
    assert result["handoff_explained"] == []
    assert [item["cell"] for item in result["fallback_only"]] == ["member-a+member-c"]


def test_an_unexecuted_actor_does_not_count_toward_a_handoff(probe, synthetic_surface):
    forensics = {
        "changed_cells": [
            {
                "cell": "member-a+member-c",
                "delta": 1.0,
                "baseline_success_rate": 0.0,
                "audited_success_rate": 1.0,
                "per_context": [
                    {
                        "baseline": {"task_id": "c-joined", "episodes": []},
                        "audited": {
                            "task_id": "c-joined",
                            "episodes": [
                                {
                                    "trace": [
                                        {"chosen": "member-a", "executed": True},
                                        {"chosen": "member-c", "executed": False},
                                    ]
                                }
                            ],
                        },
                    }
                ],
            }
        ]
    }
    result = probe.classify_changed_cells(forensics, synthetic_surface)
    assert result["handoff_explained"] == []
    assert [item["cell"] for item in result["unexplained"]] == ["member-a+member-c"]


# --------------------------------------------------------------------------- #
# The section 5.2 candidates stay bounded
# --------------------------------------------------------------------------- #


def test_T1_and_T3_remain_inexpressible_under_the_frozen_binder(payload):
    candidates = payload["section_5_2_candidates"]
    t1 = candidates["T1_serial_handoff"]
    assert t1["expressible"] is False
    assert t1["verdict"] == "inexpressible_under_frozen_binder"
    assert t1["goal_reached_by_create_alone"] is True
    assert t1["create_bound_content"] == "x = 2\n"
    assert any(
        step["kind"] == "workspace.apply_patch" and not step["success"]
        for step in t1["scripted_steps"]
    )

    t3 = candidates["T3_dual_goal_two_files"]
    assert t3["expressible"] is False
    assert t3["scripted_goal_reached"] is False
    assert set(t3["bound_paths_for_a_two_path_goal"]) == {
        "workspace.create",
        "editor.set_language",
    }
    assert all(paths[0] == "t3_main.py" for paths in t3["bound_paths_for_a_two_path_goal"].values())


def test_T2_reports_no_evidence_channel_rather_than_a_missing_implementation(payload, probe):
    surface = payload["t2_member_input_surface"]
    assert surface["expressible"] is False
    assert surface["verdict"] == "requires_new_evidence_channel"
    assert surface["arguments_carry_history_length_only"] is True
    assert surface["episodes"] > 0
    # one argument besides self is the whole reason no failure evidence can be handed in
    signature = surface["predict_episode_signature"].replace(" ", "")
    assert signature.startswith("(self,cues:")
    assert "->'tuple[str,...]'" in signature
    assert surface["distinct_cue_values_seen"] <= probe.CONTEXTS_PER_CELL


def test_the_probe_bounds_its_own_conclusion(payload):
    """What stays unprovable has to be written down, not implied by a positive verdict."""

    assert payload["does_not_change"]
    assert any("NOT implemented" in line for line in payload["does_not_change"])
    assert any("growth_admitted=false" in line for line in payload["does_not_change"])
    assert payload["contexts_per_cell"] == 2
    assert payload["binder_expression_surface"]["not_covered"]
    assert "T2" in payload["binder_expression_surface"]["not_covered"]


# --------------------------------------------------------------------------- #
# WP-1.5: scale is a parameter, and the defaults are the archived report
# --------------------------------------------------------------------------- #


def test_the_scale_parameters_default_to_the_archived_constants(probe):
    """A flagless run must still reproduce the 20260913 report byte for byte.

    Widening the sample is only evidence if the archived measurement stays
    reachable from the same instrument, so the defaults are pinned here.
    """

    signatures = {
        name: inspect.signature(getattr(probe, name)).parameters
        for name in ("probe", "validity", "measure_cell", "seed_sweep")
    }
    assert signatures["probe"]["contexts_per_cell"].default == probe.CONTEXTS_PER_CELL == 2
    assert signatures["probe"]["seed_offsets"].default == probe.SEED_OFFSETS == (0, 101, 202)
    assert signatures["probe"]["contexts_per_cell"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signatures["validity"]["contexts_per_cell"].default == probe.CONTEXTS_PER_CELL
    assert signatures["measure_cell"]["contexts_per_cell"].default == probe.CONTEXTS_PER_CELL
    assert signatures["seed_sweep"]["offsets"].default == probe.SEED_OFFSETS
    assert signatures["seed_sweep"]["contexts_per_cell"].default == probe.CONTEXTS_PER_CELL


def test_a_wider_cell_yields_more_contexts_of_the_same_structure(probe):
    frozen = _stub_frozen()
    cell = next(item for item in probe.grid() if item["label"] == DESIGNED_CELL)

    wide = probe.build_cell_tasks(frozen, cell, 6)["tasks"]

    assert len(wide) == 6
    assert len({task.kwargs["task_id"] for task in wide}) == 6
    assert len({next(iter(task.kwargs["goal_files"])) for task in wide}) == 6
    assert len({_task_index(task) for task in wide}) == 6
    for task in wide:
        kinds = [step.args[0] for step in task.kwargs["reference_steps"]]
        assert kinds.count("workspace.create") == 1
        assert "editor.set_language" in kinds
        assert task.kwargs["requires_explicit_language_override"] is True


def test_the_validity_gate_does_not_relax_at_a_wider_scale(probe, frozen):
    """More contexts per cell must not make an untestable cell testable."""

    trivial = {
        "label": "none__none",
        "ordinal": 90,
        "content_route": "none",
        "language_route": "none",
    }
    row = probe.validity(frozen, trivial, contexts_per_cell=4)
    assert len(row["contexts"]) == 4
    assert row["measurable"] is False
    assert row["not_measurable_because"] == ["tick0_already_satisfied"]


def test_main_rejects_a_degenerate_scale(probe, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        probe.main(["--contexts-per-cell", "0", "--output", str(tmp_path / "degenerate.json")])
    assert excinfo.value.code == 2
    assert not (tmp_path / "degenerate.json").exists()


# --------------------------------------------------------------------------- #
# WP-3 exits 2/3: the shipped rule vs the rule it replaced
# --------------------------------------------------------------------------- #

LANDED_REPORT = REPO / "reports" / "taiji_b0_structure_space_probe_m4landed_20260915.json"
#: The sealed rule_revision=0 run at the same scale; it is what the counterfactual predicted.
WIDE_REPORT = REPO / "reports" / "taiji_b0_structure_space_probe_wide_20260915.json"
#: D1 of the frozen route-B preregistration: gain must clear this on the main criterion.
REQUIRED_GAIN = 1.65


@pytest.fixture(scope="module")
def landed_payload() -> dict:
    return json.loads(LANDED_REPORT.read_text(encoding="utf-8"))


def test_the_landed_run_measures_two_distinct_rules(landed_payload):
    """A landed rule turns the probe's baseline into itself unless the arms are resolved.

    The first post-landing run did exactly that: ``frozen`` and ``audited`` were the same
    function, ``gain_delta`` was 0.000 everywhere and "no regressions" was a tautology.  The
    provenance block is what makes that state impossible to read as a result.
    """

    provenance = landed_payload["arm_provenance"]
    assert landed_payload["shipped_rule_revision"] == 1
    assert provenance["arms_are_distinct"] is True
    assert provenance["audited_arm"] == "shipped source, unpatched"
    assert provenance["baseline_arm"] == "shipped source reverted to rule_revision 0"
    assert provenance["baseline_delta"]["direction"] == "revert"
    assert [item["status"] for item in provenance["baseline_delta"]["replacements"]] == [
        "reverted",
        "reverted",
    ]
    assert landed_payload["frozen_attribute_intact"] is True
    assert landed_payload["does_not_change"][0].startswith("HANDOFF-M4 ships")


def test_the_landed_run_reproduces_the_sealed_prediction_field_by_field(landed_payload):
    """Exit 2, in the strong form: landing changed nothing about the measured grid.

    Compared against the sealed revision-0 wide report at the same scale, so the
    reproduction is the same numbers rather than a paraphrase of a copied table.
    """

    wide = json.loads(WIDE_REPORT.read_text(encoding="utf-8"))
    assert landed_payload["contexts_per_cell"] == wide["contexts_per_cell"]
    assert landed_payload["seed_sweep"]["offsets"] == wide["seed_sweep"]["offsets"]

    assert landed_payload["rows"] == wide["rows"]
    assert landed_payload["validity"] == wide["validity"]
    assert landed_payload["verdict"] == wide["verdict"]
    assert landed_payload["outcome_distinctness"] == wide["outcome_distinctness"]
    assert landed_payload["seed_sweep"] == wide["seed_sweep"]


def test_the_shipped_rule_still_beats_the_reconstructed_revision_0(landed_payload):
    """Exit 3, non-vacuously: the gain is a difference between two measured rules."""

    seen = 0
    for row in landed_payload["rows"]:
        if row["cell"] in CREATE_ROW:
            seen += 1
            assert row["best_pair_gain_frozen"] == 0.0, row["cell"]
            assert row["gain_delta"] > 0.0, row["cell"]
            assert row["best_pair_gain_audited"] > REQUIRED_GAIN, row["cell"]
            assert row["interleaved_contexts_audited"] > 0, row["cell"]
            assert row["positive_under_frozen"] is False
            assert row["positive_under_m4"] is True
        else:
            assert row["gain_delta"] == 0.0, row["cell"]
            assert row["unexplained_changes"] == [], row["cell"]
    assert seen == len(CREATE_ROW)
    assert landed_payload["verdict"]["m4_regresses_cells"] == []
    assert landed_payload["verdict"]["cells_with_unexplained_change"] == []
    assert set(landed_payload["verdict"]["positive_gain_cells"]) == set(CREATE_ROW)

    sweep = landed_payload["seed_sweep"]
    assert set(sweep["positive_cells_every_seed"]) == set(CREATE_ROW)
    assert len(sweep["offsets"]) == 7
    for row in sweep["rows"]:
        if row["cell"] in CREATE_ROW:
            assert row["positive"] is True and row["reality_ok"] is True, row


def test_the_landed_run_did_not_move_the_structural_boundary(landed_payload):
    """L1 stays open: shipping a rule cannot manufacture a second independent structure."""

    distinct = landed_payload["outcome_distinctness"]
    assert distinct["distinct_outcome_fingerprints"] == 3
    assert distinct["cells_compared"] == GRID_CELLS
    for name, item in landed_payload["section_5_2_candidates"].items():
        assert item["expressible"] is False, name
