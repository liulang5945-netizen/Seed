"""Regression guard for the P5.2b gate defect found by the P5.2c entry audit.

The P5.2b nine gates passed on a matrix in which every non-baseline
intervention cell in three of twelve contexts executed zero steps.  The old
``cell_completeness`` only counted episodes and ``real_execution`` only asked
whether *some* action, anywhere, had executed with valid provenance, so an
entirely inert matrix was indistinguishable from a working one.

These tests pin the repaired semantics: a cell whose task was already
satisfied before any member acted cannot attribute its outcome to its
members.  Such a cell is tolerated only for the ``(F,F)`` baseline, where
"no member was invoked" is the treatment itself.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

from seed_platform.workbench import WorkbenchEnvironment

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "training" / "eval_taiji_p5_2b_group_causal_corpora_gate.py"


def _load_runner():
    """Import the gate runner by path so the test does not depend on sys.path."""

    scripts = str(RUNNER.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("_p52b_gate_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _episode(task_id: str, active_members: tuple[str, ...], steps: list[dict]) -> dict:
    return {"task_id": task_id, "active_members": list(active_members), "steps": steps}


def _step(executed: bool = True) -> dict:
    return {
        "executed": executed,
        "provenance": {"goal_state": "goal_state"},
        "safety_violation": False,
    }


def test_source_declares_intervention_reality_gate(runner) -> None:
    """The hardening must be present in the runner, not only in this test."""

    source = RUNNER.read_text(encoding="utf-8")
    assert "zero_step_intervention_cells" in source
    assert "interventions_happened" in source
    # both gates must consume the intervention-reality evidence
    assert "and not zero_step_intervention_cells" in source
    assert 'intervention_reality["interventions_happened"]' in source


def test_baseline_zero_step_cells_are_legitimate(runner) -> None:
    """A zero-step (F,F) baseline is the treatment itself and must be allowed."""

    episodes = [
        _episode("ctx-1", (), []),  # baseline, zero steps: legitimate
        _episode("ctx-1", ("member-a",), [_step()]),
        _episode("ctx-1", ("member-a", "member-b"), [_step()]),
    ]
    reality = runner._intervention_reality(episodes)
    assert reality["interventions_happened"] is True
    assert reality["zero_step_intervention_cells"] == {}
    assert reality["zero_step_episodes_total"] == 1


def test_zero_step_intervention_cell_is_detected(runner) -> None:
    """The P5.2b failure mode: an intervention cell that executed nothing."""

    episodes = [
        # lang_confirm-shaped context: goal state equals initial state, so the
        # episode stops on tick 0 but still records an outcome.
        _episode("ctx-100", (), []),
        _episode("ctx-100", ("member-a",), []),
        _episode("ctx-100", ("member-a", "member-b"), []),
    ]
    reality = runner._intervention_reality(episodes)
    assert reality["interventions_happened"] is False
    assert reality["zero_step_intervention_episodes"] == 2
    assert set(reality["zero_step_intervention_cells"]) == {
        "ctx-100:member-a",
        "ctx-100:member-a-member-b",
    }
    # the baseline cell must not be reported as an offender
    assert "ctx-100:none" not in reality["zero_step_intervention_cells"]


def test_single_inert_cell_among_many_is_still_caught(runner) -> None:
    """One bad cell is enough -- the old gates needed only 'some' execution."""

    episodes = [_episode(f"ctx-{i}", (), []) for i in range(12)]
    for i in range(12):
        episodes.append(_episode(f"ctx-{i}", ("member-a",), [_step()]))
        episodes.append(_episode(f"ctx-{i}", ("member-a", "member-b"), [_step()]))
    # exactly one intervention cell is inert
    inert = [
        item
        for item in episodes
        if item["task_id"] == "ctx-7" and item["active_members"] == ["member-a"]
    ][0]
    inert["steps"] = []

    reality = runner._intervention_reality(episodes)
    assert reality["interventions_happened"] is False
    assert reality["zero_step_intervention_cells"] == {"ctx-7:member-a": 1}


def test_fully_executed_matrix_passes(runner) -> None:
    episodes = []
    for i in range(12):
        episodes.append(_episode(f"ctx-{i}", (), []))
        for members in (("member-a",), ("member-b",), ("member-a", "member-b")):
            episodes.append(_episode(f"ctx-{i}", members, [_step()]))

    reality = runner._intervention_reality(episodes)
    assert reality["interventions_happened"] is True
    assert reality["zero_step_intervention_episodes"] == 0
    assert reality["baseline_cells"] == 12


# ---------------------------------------------------------------------------
# Bind-layer guard.
#
# Detecting the inert matrix (above) is necessary but not sufficient.  The
# first repair of the ``lang_confirm`` defect merely *demanded* a language
# override in ``_goal_reached`` while ``_bind`` kept dropping the flag from the
# bound parameters.  The result was the mirror image of the original defect:
# the matrix stopped showing zero-step cells, not because the intervention now
# ran, but because the task had become permanently unsatisfiable for every
# member.  A gate that only counts executed steps cannot tell "fixed" from
# "impossible", so the end-to-end satisfiability must be pinned directly.
# ---------------------------------------------------------------------------


def _p52a_module():
    spec = importlib.util.spec_from_file_location(
        "_p52a_under_test",
        REPO / "scripts" / "training" / "eval_taiji_p5_2a_predictive_execution_gate.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_p52a_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def p52a():
    scripts = str(REPO / "scripts" / "training")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return _p52a_module()


def test_override_task_binds_the_override_flag(p52a) -> None:
    """``_bind`` must propagate ``user_override`` for an override-requiring task."""

    tasks = {task.task_id: task for task in p52a._validation_tasks()}
    task = tasks["p52a-validation-100"]
    assert task.requires_explicit_language_override is True

    params, provenance, failure = p52a._bind(
        "editor.set_language", task, {"root": None, "last_undo_token": None}
    )
    assert failure is None
    assert params.get("user_override") is True, (
        "editor.set_language dropped user_override; the environment then records "
        'selection_source="taiji_selection" and the task can never be satisfied'
    )
    assert provenance.get("user_override") == "goal_state"


def test_non_override_task_does_not_claim_an_override(p52a) -> None:
    """A task needing only a resolvable language must not assert more than that."""

    tasks = {task.task_id: task for task in p52a._validation_tasks()}
    # header_override requires a language but not an explicit user override.
    for task in tasks.values():
        if task.requires_explicit_language_override:
            continue
        if not task.goal_language:
            continue
        params, _provenance, failure = p52a._bind(
            "editor.set_language", task, {"root": None, "last_undo_token": None}
        )
        assert failure is None
        assert "user_override" not in params


def test_every_validation_task_is_nontrivial_and_satisfiable(p52a) -> None:
    """The invariant the P5.2b defect broke, in both directions.

    Each validation task must be *unsatisfied* before any action (otherwise the
    episode fabricates an empty-event success) and *satisfied* after executing
    its own reference steps (otherwise it is impossible and every cell is
    vacuous).  Checking only one of the two would have accepted the broken
    intermediate state.
    """

    offenders: list[str] = []
    for task in p52a._validation_tasks():
        environment = WorkbenchEnvironment(root=Path(tempfile.mkdtemp()))
        for name, content in task.initial_files.items():
            (environment.root / name).write_text(content, encoding="utf-8", newline="")
        state: dict = {"root": environment.root, "last_undo_token": None}

        if p52a._goal_reached(environment, task):
            offenders.append(f"{task.task_id} ({task.template}): satisfied before any action")

        for step in task.reference_steps:
            if step.kind in {"workspace.undo"}:
                continue
            params, _provenance, failure = p52a._bind(step.kind, task, state)
            assert failure is None, f"{task.task_id}: {step.kind} failed to bind ({failure})"
            environment.execute_tool(step.kind, params)

        if not p52a._goal_reached(environment, task):
            offenders.append(f"{task.task_id} ({task.template}): unreachable via own steps")

    assert not offenders, "non-trivial/satisfiable invariant violated:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# Episode-reset guard.
#
# The third layer of the same defect.  ``_member_episode`` resets the world
# between episodes with ``restore_language_state(None)``, but that call is a
# NO-OP (``if not payload: return``), so the previous episode's selection --
# including an explicit ``user_override`` -- survived.  A ``lang_confirm`` goal
# is exactly that override, so every episode after the first started already
# satisfied and recorded an empty-event success.
#
# This was invisible while ``_goal_reached`` only compared the resolved language
# id, because extension resolution re-established it anyway.  It only became
# load-bearing once the goal started requiring the recorded override, which is
# why the guard below pins the *reset behaviour* rather than the goal predicate.
# ---------------------------------------------------------------------------


def test_restore_none_is_a_noop_and_must_not_be_relied_on() -> None:
    """Document the environment semantics the fix depends on."""

    environment = WorkbenchEnvironment(root=Path(tempfile.mkdtemp()))
    environment.restore_language_state(
        {
            "format": "seed-workbench-language-state-v1",
            "version": 1,
            "registry_revision": environment.programming_language_registry.revision,
            "selections": [],
        }
    )
    assert environment.language_state_checkpoint()["selections"] == []

    environment.restore_language_state(None)
    # still empty, but only because nothing was set -- not because None clears
    assert environment.language_state_checkpoint()["selections"] == []


def test_empty_payload_clears_a_prior_selection() -> None:
    """An explicit empty payload is the supported way to clear selections."""

    environment = WorkbenchEnvironment(root=Path(tempfile.mkdtemp()))
    root = environment.root
    (root / "probe.py").write_text("x = 1\n", encoding="utf-8", newline="")

    environment.execute_tool(
        "editor.set_language",
        {"path": "probe.py", "programming_language_id": "python", "user_override": True},
    )
    selected = environment.language_state_checkpoint()["selections"]
    assert selected and selected[0]["selection_state"] == "user_override"

    # naming ``None`` here would leave the override in place
    environment.restore_language_state(None)
    assert environment.language_state_checkpoint()["selections"] != []

    environment.restore_language_state(
        {
            "format": "seed-workbench-language-state-v1",
            "version": 1,
            "registry_revision": environment.programming_language_registry.revision,
            "selections": [],
        }
    )
    assert environment.language_state_checkpoint()["selections"] == []


def test_runner_resets_with_an_explicit_empty_payload(runner) -> None:
    """The runner must not rely on the ``None`` no-op between episodes."""

    source = RUNNER.read_text(encoding="utf-8")
    assert "environment.restore_language_state(None)" not in source
    assert '"selections": []' in source


def test_report_composition_field_matches_the_executed_mechanism(runner) -> None:
    """The report must describe the composition it actually ran.

    The field still said "sequential fallback" after the mechanism was changed
    to dual-predict-select, so the emitted report mis-described its own
    execution.  A report that cannot be trusted about its mechanism cannot be
    trusted about its results.
    """

    source = RUNNER.read_text(encoding="utf-8")
    assert "sequential fallback by opaque-id order" not in source
    # the implemented mechanism is the one the report must name
    assert "dual-predict-select" in source
    assert "lexicographically first member whose prediction binds" in source
