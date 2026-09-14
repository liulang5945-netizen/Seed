"""N1: is M4's gain a property of a *structure*, or of the one structure measured so far?

Why this probe exists
---------------------
The M4 hardening pass swept six variants and every one produced the same outcome
fingerprint, so it could only bound **surface** sensitivity -- and its own report
says so (``effective_sample_size.outcome_distinctness.does_not_bound``).  All six
variants were the same *composition structure*: create a file, then override its
language.  The open question for decision D5 is whether ``m4_failure_handoff``
gains because that structure happens to suit it, or because it fixes something
general about failure handoffs.

The mechanism doc (``M5_B0_MECHANISM_AND_TASK_PRECHECK_20260913.md`` section 5.2)
proposed three candidate structures -- T1 serial handoff, T2 error-correcting
handoff, T3 jointly-required dual goal.  Probing them turned out to be the wrong
first move: **the frozen binder cannot express two of the three.**  ``p52a._bind``
resolves every action against ``task.main_path``, and ``workspace.create`` binds
``content = task.goal_files[main_path]``, so a created file lands directly on its
goal state.  Building T1/T3 anyway would have produced negative results that say
nothing about the mechanism.

So this probe replaces "hand-design three candidates" with "enumerate what the
goal predicate can ask for, and measure every cell".  The goal predicate has two
independent clauses -- file content and language selection -- and each has a fixed
route from the initial world state.  That gives a 3 x 4 grid:

* content route: ``none`` (already at goal) / ``create`` (absent) / ``patch``
  (present at a different content);
* language route: ``none`` / ``observation`` (goal language is inferable from the
  extension, so ``programming_language.resolve`` completes it) / ``override``
  (inferable, but the goal demands the recorded ``user_override`` flag) /
  ``mismatch`` (goal language is *not* inferable, so an intervention is needed for
  the value itself).

Eleven of the twelve cells are non-trivial, and all eleven are certified admissible
*under the contract* before any measurement: a scripted run walks the same path a member
walks (bind -> ``policy_for`` -> approval -> ``execute_tool``), because an interception
such as ``language_evidence_ambiguous`` is raised by the policy layer and never by the
executor.  Certifying through the executor alone is the mistake this file's first
iteration made -- it admitted four cells nobody can legitimately perform, and a zero gain
on such a cell reads as a structural finding when it is only a measurement artifact.  Each
cell then runs through the frozen rule and through M4, and carries a prediction made
*before* measurement from the four trained member policies.

What the probe is for
---------------------
Two questions, answered separately:

1. **Which cells are jointly required?**  A cell is combination-only when every
   singleton fails on it.  The prediction is that this needs an existence
   precondition: on the ``patch`` route ``member-d``'s own policy
   (read -> resolve -> set_language -> apply_patch) already covers both clauses,
   and on the language-free cells a single member covers the only demand.
2. **Where does M4 gain?**  If positive gain appears on more than
   ``create x override``, the mechanism fix is structural and D5 can be decided on
   that basis.  If it appears only there, M4's evidence is one structure wide and
   D5 should be read that way.

How the answer has to be read: cells that share one outcome fingerprint are one
structural factor seen through several routes, not several confirmations -- that is the
lesson the hardening pass drew from six variants with identical fingerprints.  So the
breadth claim is counted in *distinct frozen failure shapes repaired* (``stop_reasons_*``
per cell), and the cell count is reported next to it rather than instead of it.

The three section-5.2 candidates are still reported -- as inexpressible, with
measured witnesses rather than assertions -- because "the plan's next step cannot
be built on the frozen binder" is itself the finding that bounds what D5 can claim.

Read-only: no task is registered, nothing is trained for a candidate, no gate,
runner, rule or frozen artifact is modified, and M4 runs as the documented
counterfactual from ``probe_taiji_b0_m1_counterfactual``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "scripts" / "training"
#: Revision-0 evidence, sealed by sha256 in test_b0_rule_revision_seal_contract.py.
REVISION_0_OUTPUT = PROJECT_ROOT / "reports" / "taiji_b0_structure_space_probe_20260913.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "taiji_b0_structure_space_probe_m4landed_20260915.json"
)

COUNTERFACTUAL_MODULE = TRAINING_DIR / "probe_taiji_b0_m1_counterfactual.py"
HANDOFF_PROBE = TRAINING_DIR / "probe_taiji_b0_handoff_feasibility.py"
ARTIFACT_AUDIT = TRAINING_DIR / "audit_taiji_b0_m4_artifact.py"
FROZEN_GATE = TRAINING_DIR / "eval_taiji_p5_2b_group_causal_corpora_gate.py"
FROZEN_ROUTE_A_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
)

PROBE_FORMAT = "taiji-b0-structure-space-probe-v1"
VERSION = 1

RULE_UNDER_AUDIT = "m4_failure_handoff"

#: The extension is held constant across the whole grid on purpose: the hardening
#: pass already established that extension / language / content shape are surface,
#: not structure.  Only the route from the initial world to the goal varies.
EXTENSION = ".py"
#: The content deliberately looks like Python.  The contract assesses language
#: evidence from the registry's content patterns (``def``/``import``), so a bare
#: assignment in a ``.py`` file gives *no* content evidence and an explicit
#: ``editor.set_language`` without an override is then intercepted as
#: ``language_evidence_ambiguous``.  Holding the evidence constant is what makes a
#: cell's verdict about its structure rather than about its wording.
BASE_TEMPLATE = "def run_{i}():\n    return {i}\n"
PATCHED_TEMPLATE = "def run_{i}():\n    return {i} + 1\n"
INFERABLE_LANGUAGE = "python"
NON_INFERABLE_LANGUAGE = "javascript"

CONTEXTS_PER_CELL = 2
INDEX_BASE = 600

#: Three offsets.  The axis under test here is structure, not seed; five offsets
#: for ``create x override`` are already on record in the hardening report.
SEED_OFFSETS: tuple[int, ...] = (0, 101, 202)

#: At least two jointly-required cells must survive validity checks, or the probe
#: cannot answer its own question and must not publish a verdict.
MIN_MEASURABLE_COMBINATION_ONLY_CELLS = 2

CONTENT_ROUTES = ("none", "create", "patch")
LANGUAGE_ROUTES = ("none", "observation", "override", "mismatch")

CANDIDATES_FOR_REVIEW: dict[str, float] = {
    "all_singleton_oracle": 1.5,
    "best_fixed_singleton": 0.5,
    "best_observed_fixed_pair": 1.0,
}


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    for entry in (str(path.parent), str(PROJECT_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_counterfactual() -> Any:
    return _load("_b0_structure_counterfactual", COUNTERFACTUAL_MODULE)


def load_handoff_probe() -> Any:
    return _load("_b0_structure_handoff_probe", HANDOFF_PROBE)


def load_audit() -> Any:
    return _load("_b0_structure_m4_audit", ARTIFACT_AUDIT)


def load_frozen() -> Any:
    return _load("_b0_structure_frozen_p52b", FROZEN_GATE)


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cell_is_combination_only(content_route: str, language_route: str) -> bool:
    """Prediction, made from the four trained member policies before measuring.

    ``member-a`` = read -> resolve -> set_language, ``member-b`` = read -> patch ->
    undo, ``member-c`` = list -> create -> undo, ``member-d`` = read -> resolve ->
    set_language -> patch.  A cell is single-member solvable whenever one of those
    sequences covers every unsatisfied goal clause:

    * ``patch`` content route: ``member-d`` patches *and* selects a language, so all
      four language routes on that row are covered by one member;
    * language-free cells: one clause, one member (``b``/``d`` patch, ``c`` creates);
    * ``create`` content route with a language clause: no member both creates a file
      and leaves it in place while selecting a language -- ``c`` never selects, and
      the selecting members cannot bring a file into existence.
    """

    if content_route == "create":
        return language_route != "none"
    return False


def grid() -> list[dict[str, Any]]:
    """The 3 x 4 route grid minus the trivial ``none x none`` cell."""

    cells: list[dict[str, Any]] = []
    for content_route in CONTENT_ROUTES:
        for language_route in LANGUAGE_ROUTES:
            if content_route == "none" and language_route == "none":
                continue
            cells.append(
                {
                    "label": f"{content_route}__{language_route}",
                    "ordinal": len(cells),
                    "content_route": content_route,
                    "language_route": language_route,
                    "existence_precondition": content_route == "create",
                    "expect_combination_only": cell_is_combination_only(
                        content_route, language_route
                    ),
                }
            )
    return cells


def build_cell_tasks(
    frozen: Any, cell: Mapping[str, Any], count: int = CONTEXTS_PER_CELL
) -> dict[str, Any]:
    """``count`` contexts of one structure: same routes, different index suffix."""

    p52a, p52 = frozen.p52a, frozen.p52
    content_route = str(cell["content_route"])
    language_route = str(cell["language_route"])
    tasks: list[Any] = []
    owner: dict[str, str] = {}
    for step in range(count):
        i = INDEX_BASE + int(cell["ordinal"]) * 10 + step
        name = f"struct_{content_route}_{language_route}_{i}{EXTENSION}"
        base = BASE_TEMPLATE.format(i=i)
        goal = PATCHED_TEMPLATE.format(i=i) if content_route == "patch" else base
        language = {
            "none": None,
            "observation": INFERABLE_LANGUAGE,
            "override": INFERABLE_LANGUAGE,
            "mismatch": NON_INFERABLE_LANGUAGE,
        }[language_route]
        # A mismatched language needs the override flag too: selecting a language
        # the file gives no evidence for is exactly what the contract refuses
        # without one, so an unflagged mismatch cell is a task nobody can perform
        # rather than a structure the mechanism can fail on.
        requires_override = language_route in ("override", "mismatch")

        steps: list[Any] = []
        if content_route == "create":
            steps.append(p52.ScriptedStep("workspace.list", {"path": "."}))
            steps.append(p52.ScriptedStep("workspace.create", {"path": name, "content": goal}))
        steps.append(p52.ScriptedStep("workspace.read", {"path": name}))
        if language_route != "none":
            steps.append(p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}))
        if language_route in ("override", "mismatch"):
            steps.append(
                p52.ScriptedStep(
                    "editor.set_language",
                    {
                        "path": name,
                        "programming_language_id": language,
                        "user_override": requires_override,
                    },
                )
            )
        if content_route == "patch":
            steps.append(
                p52.ScriptedStep(
                    "workspace.apply_patch",
                    {
                        "path": name,
                        "before_digest": _sha(base),
                        "patch": p52._patch_ops(base, f"return {i}", f"return {i} + 1"),
                        "expected_after_digest": _sha(goal),
                    },
                )
            )

        task_id = f"b0struct-{cell['label']}-{i}"
        tasks.append(
            p52a.Task(
                task_id=task_id,
                goal_text=(
                    f"Work order {task_id}: reach the requested end state on "
                    f"{name} using the workspace contract, and leave every temporary "
                    f"change reverted."
                ),
                initial_files=({} if content_route == "create" else {name: base}),
                goal_files={name: goal},
                goal_language=({name: language} if language else {}),
                main_path=name,
                reference_steps=tuple(steps),
                partition="probe",
                template=f"struct_{cell['label']}",
                requires_explicit_language_override=requires_override,
            )
        )
        owner[task_id] = str(cell["label"])
    return {"tasks": tuple(tasks), "owner": owner}


# --------------------------------------------------------------------------- #
# Validity: expressible, satisfiable, non-trivial -- checked before measuring
# --------------------------------------------------------------------------- #


def _tick0_reached(frozen: Any, task: Any) -> bool:
    """Is the goal satisfied before any action?  The P5.2b defect shape."""

    root = Path(tempfile.mkdtemp(prefix="b0struct-tick0-"))
    try:
        environment = frozen.WorkbenchEnvironment(root)
        environment.restore_language_state(
            {
                "format": "seed-workbench-language-state-v1",
                "version": 1,
                "registry_revision": environment.programming_language_registry.revision,
                "selections": [],
            }
        )
        for name, content in task.initial_files.items():
            (root / name).write_text(content, encoding="utf-8", newline="")
        return bool(frozen.p52a._goal_reached(environment, task))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run_scripted_under_contract(frozen: Any, task: Any, steps: Sequence[Any]) -> dict[str, Any]:
    """Scripted run that walks the *contract* path, not just the executor.

    ``counterfactual._run_scripted`` calls ``execute_tool`` directly, which is the right
    tool for a rule-free satisfiability probe but is blind to the interceptions that
    decide whether a step is admissible at all: ``language_evidence_ambiguous`` and its
    siblings are raised by ``environment.policy_for``, never by the executor.  A cell
    certified admissible through the rule-free path can be a cell no member is allowed
    to finish, and a zero gain measured on such a cell says nothing about any mechanism.
    So validity is judged on the same path ``_member_episode`` walks -- bind -> intent
    -> ``policy_for`` -> (approval) -> ``execute_tool``.
    """

    p52a = frozen.p52a
    root = Path(tempfile.mkdtemp(prefix="b0struct-valid-"))
    try:
        environment = frozen.WorkbenchEnvironment(root)
        environment.restore_language_state(
            {
                "format": "seed-workbench-language-state-v1",
                "version": 1,
                "registry_revision": environment.programming_language_registry.revision,
                "selections": [],
            }
        )
        for name, content in task.initial_files.items():
            (root / name).write_text(content, encoding="utf-8", newline="")
        state: dict[str, Any] = {"root": root, "last_undo_token": None}

        executed: list[dict[str, Any]] = []
        blocked: list[str] = []
        for tick, step in enumerate(steps):
            params, _provenance, failure = p52a._bind(step.kind, task, state)
            if failure is not None:
                reason = f"{step.kind}:bind_refused:{failure}"
                executed.append({"tick": tick, "kind": step.kind, "bound": False, "stop": reason})
                blocked.append(reason)
                continue
            intent = frozen.ActionIntent(
                f"b0struct-valid:{task.task_id}:{tick:04d}",
                step.kind,
                parameters=params,
                confidence=0.9,
                tick=tick,
            )
            request = frozen.WorkbenchActionRequest.from_action_intent(
                intent, snapshot_id=environment.capability_snapshot.snapshot_id
            )
            decision = environment.policy_for(request)
            needs_approval = (
                decision.decision == "ask_user"
                and decision.reason_code == "capability_requires_approval"
            )
            intercepts = decision.decision == "deny" or (
                decision.decision == "ask_user" and not needs_approval
            )
            if intercepts:
                reason = f"{step.kind}:contract_intercepted:{decision.reason_code}"
                executed.append({"tick": tick, "kind": step.kind, "bound": True, "stop": reason})
                blocked.append(reason)
                break
            if needs_approval:
                approval = environment.issue_approval(request)
                tokened = frozen.dataclasses.replace(
                    request, approval_token=approval["approval_token"]
                )
                redecision = environment.policy_for(tokened)
                if redecision.decision != "allow":
                    reason = f"{step.kind}:approval_rejected:{redecision.reason_code}"
                    executed.append(
                        {"tick": tick, "kind": step.kind, "bound": True, "stop": reason}
                    )
                    blocked.append(reason)
                    break
                environment.consume_approval(tokened)
            outcome = environment.execute_tool(step.kind, params)
            last = environment.last_result
            if "transaction" in last and last["transaction"].get("undo_token"):
                state["last_undo_token"] = str(last["transaction"]["undo_token"])
            executed.append(
                {"tick": tick, "kind": step.kind, "bound": True, "success": bool(outcome.success)}
            )
            if not outcome.success:
                blocked.append(f"{step.kind}:execution_failed")
        return {
            "steps": executed,
            "blocked": sorted(set(blocked)),
            "goal_reached": bool(p52a._goal_reached(environment, task)),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def validity(
    frozen: Any, cell: Mapping[str, Any], *, contexts_per_cell: int = CONTEXTS_PER_CELL
) -> dict[str, Any]:
    """Scripted check, under the contract, that a cell's reference steps reach its goal.

    ``measurable`` is the gate this probe must not relax: a cell that the contract
    refuses, that is already satisfied at tick 0, or whose own reference path cannot
    reach its goal is not a structure the mechanism can be tested on.
    """

    rows: list[dict[str, Any]] = []
    for task in build_cell_tasks(frozen, cell, contexts_per_cell)["tasks"]:
        forward = _run_scripted_under_contract(frozen, task, task.reference_steps)
        backward = _run_scripted_under_contract(frozen, task, tuple(reversed(task.reference_steps)))
        rows.append(
            {
                "task_id": task.task_id,
                "forward_reaches_goal": bool(forward["goal_reached"]),
                "reversed_reaches_goal": bool(backward["goal_reached"]),
                "order_is_mandatory": bool(forward["goal_reached"] != backward["goal_reached"]),
                "tick0_already_satisfied": _tick0_reached(frozen, task),
                "contract_blocked_steps": [
                    item for item in forward["blocked"] if "contract_intercepted" in item
                ],
                "scripted_blocked_steps": forward["blocked"],
                "trace": forward["steps"],
            }
        )
    nontrivial_ok = not any(row["tick0_already_satisfied"] for row in rows)
    satisfiable_ok = all(row["forward_reaches_goal"] for row in rows)
    contract_ok = not any(row["contract_blocked_steps"] for row in rows)
    executor_ok = not any(row["scripted_blocked_steps"] for row in rows)
    reasons = [
        reason
        for row in rows
        for reason in row["contract_blocked_steps"] + row["scripted_blocked_steps"]
    ]
    if not satisfiable_ok and not reasons:
        reasons = ["reference_steps_do_not_reach_goal"]
    if not nontrivial_ok and not reasons:
        reasons = ["tick0_already_satisfied"]
    return {
        "cell": cell["label"],
        "contexts": [row["task_id"] for row in rows],
        "rows": rows,
        "satisfiable": satisfiable_ok,
        "nontrivial": nontrivial_ok,
        "contract_admissible": contract_ok,
        "scripted_path_clean": executor_ok,
        "order_is_mandatory": any(row["order_is_mandatory"] for row in rows),
        "measurable": bool(satisfiable_ok and nontrivial_ok and contract_ok),
        "not_measurable_because": sorted(set(reasons)),
    }


# --------------------------------------------------------------------------- #
# The section 5.2 candidates: expressibility witnesses, not assumptions
# --------------------------------------------------------------------------- #


def candidate_witnesses(frozen: Any, counterfactual: Any) -> dict[str, Any]:
    """T1 / T3 / T2 from the mechanism doc, each with a measured expressibility verdict."""

    p52a, p52 = frozen.p52a, frozen.p52

    def steps(*kinds: str) -> tuple[Any, ...]:
        return tuple(p52.ScriptedStep(kind, {"path": "w"}) for kind in kinds)

    # -- T1: serial handoff, create then patch -------------------------------- #
    name = "t1_witness.py"
    t1 = p52a.Task(
        task_id="b0struct-t1-witness",
        goal_text=(
            f"Work order b0struct-t1-witness: reach the requested end state on {name} "
            "using the workspace contract, and leave every temporary change reverted."
        ),
        initial_files={},
        goal_files={name: "x = 2\n"},
        goal_language={},
        main_path=name,
        reference_steps=steps(
            "workspace.list",
            "workspace.create",
            "workspace.read",
            "workspace.apply_patch",
        ),
        partition="probe",
        template="struct_t1_witness",
    )
    t1_run = counterfactual._run_scripted(frozen, t1, t1.reference_steps)
    t1_create_binds = p52a._bind(
        "workspace.create", t1, {"root": Path("."), "last_undo_token": None}
    )[0]

    # -- T3: jointly required, literally two files ---------------------------- #
    main, other = "t3_main.py", "t3_other.py"
    t3 = p52a.Task(
        task_id="b0struct-t3-witness",
        goal_text=(
            f"Work order b0struct-t3-witness: reach the requested end state on {main} "
            "using the workspace contract, and leave every temporary change reverted."
        ),
        initial_files={other: "value 1\n", main: "x = 1\n"},
        goal_files={main: "x = 2\n", other: "value 1\n"},
        goal_language={main: "python", other: "python"},
        main_path=main,
        reference_steps=steps(
            "workspace.read",
            "workspace.apply_patch",
            "workspace.programming_language.resolve",
        ),
        partition="probe",
        template="struct_t3_witness",
    )
    t3_run = counterfactual._run_scripted(frozen, t3, t3.reference_steps)
    probe_state = {"root": Path("."), "last_undo_token": None}
    t3_binds = {
        kind: [
            p52a._bind(kind, t3, probe_state)[0].get("path"),
            p52a._bind(kind, t3, probe_state)[0].get("programming_language_id"),
        ]
        for kind in ("workspace.create", "editor.set_language")
    }

    witness = {
        "T1_serial_handoff": {
            "spec": "create a file, then patch the freshly created content",
            "expressible": False,
            "verdict": "inexpressible_under_frozen_binder",
            "why": (
                "workspace.create binds content = task.goal_files[main_path], so the "
                "created file is already at its goal state and workspace.apply_patch "
                "then refuses with 'file already at goal state'"
            ),
            "scripted_steps": [
                {
                    "kind": step.get("kind"),
                    "bound": bool(step.get("bound")),
                    "success": step.get("success"),
                    "failure": step.get("failure"),
                }
                for step in t1_run["steps"]
            ],
            "goal_reached_by_create_alone": bool(t1_run["goal_reached"]),
            "create_bound_content": t1_create_binds.get("content"),
            "consequence": (
                "the earlier create_then_patch surface is a create-only task, which is "
                "why member-c solves it alone; a serial-handoff structure needs a binder "
                "change, not a composition change"
            ),
        },
        "T3_dual_goal_two_files": {
            "spec": "two independent files, one needing an override and one needing a patch",
            "expressible": False,
            "verdict": "inexpressible_under_frozen_binder",
            "why": (
                "_bind resolves every action kind against task.main_path, so no action "
                "can ever target the second goal path; its content clause can be "
                "pre-satisfied in initial_files but its language clause cannot"
            ),
            "scripted_goal_reached": bool(t3_run["goal_reached"]),
            "bound_paths_for_a_two_path_goal": t3_binds,
            "note": (
                "the expressible reading of 'jointly required' is one file with two "
                "clauses, which is the patch x * row of the grid -- and member-d covers "
                "it alone.  The grid is therefore the closest admissible substitute."
            ),
        },
    }
    return witness


def member_input_surface(
    frozen: Any,
    counterfactual: Any,
    cell: Mapping[str, Any],
    embedder: Any,
) -> dict[str, Any]:
    """T2 (error-correcting handoff) needs a channel.  Does one exist?

    Wraps the members' ``predict_episode`` in a recording proxy for a handful of
    episodes and reports every argument a member is ever given.  T2 is expressible
    only if a member can be *shown* another member's failure; if the only per-call
    variation is the member's own history length, there is no such channel and T2
    is a mechanism change, not a task design.
    """

    class _Recording:
        def __init__(self, learner: Any, member_id: str, sink: list[dict[str, Any]]) -> None:
            self._learner = learner
            self._member_id = member_id
            self._sink = sink

        def predict_episode(self, cues: Sequence[Any]) -> Any:
            prediction = self._learner.predict_episode(cues)
            first = cues[0]
            self._sink.append(
                {
                    "member": self._member_id,
                    "cue_count": len(cues),
                    "distinct_objects_in_cue_tuple": len({id(cue) for cue in cues}),
                    "cue_fingerprint": round(float(first.double().sum()), 6),
                    "predicted": str(prediction[-1]),
                }
            )
            return prediction

        def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
            return getattr(self._learner, item)

    records: list[dict[str, Any]] = []
    tasks = build_cell_tasks(frozen, cell)["tasks"]

    def run(members: Mapping[str, Any]) -> list[dict[str, Any]]:
        return counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, tasks
        )

    real = frozen._train_members(embedder)
    wrapped = {
        member_id: _Recording(learner, member_id, records) for member_id, learner in real.items()
    }
    episodes = run(wrapped)

    per_member: dict[str, dict[str, Any]] = {}
    for member_id in sorted({row["member"] for row in records}):
        rows = [row for row in records if row["member"] == member_id]
        per_member[member_id] = {
            "calls": len(rows),
            "distinct_cue_fingerprints": sorted({row["cue_fingerprint"] for row in rows}),
            "history_lengths_seen": sorted({row["cue_count"] for row in rows}),
            "max_distinct_cue_objects_in_one_call": max(
                row["distinct_objects_in_cue_tuple"] for row in rows
            ),
            "kinds_predicted": sorted({row["predicted"] for row in rows}),
        }

    signature = str(inspect.signature(type(real[frozen.MEMBER_IDS[0]]).predict_episode))
    cue_values = {
        fingerprint
        for row in per_member.values()
        for fingerprint in row["distinct_cue_fingerprints"]
    }
    history_only = all(
        row["max_distinct_cue_objects_in_one_call"] == 1 for row in per_member.values()
    )
    return {
        "cell": cell["label"],
        "episodes": len(episodes),
        "predict_episode_signature": signature,
        "per_member": per_member,
        "distinct_cue_values_seen": len(cue_values),
        "cue_values_per_task": bool(len(cue_values) <= len(tasks)),
        "arguments_carry_history_length_only": history_only,
        "expressible": False,
        "verdict": "requires_new_evidence_channel",
        "why": (
            "predict_episode takes one argument, and every element of the tuple passed to "
            "it is the same tensor object repeated -- so the only per-call variation is the "
            "member's own history length.  No member is ever handed another member's "
            "failure, so 'B reads A's failure evidence' cannot be built out of task design"
        ),
        "consequence": (
            "M4 does carry that failure information, but in the composition layer (it reads "
            "the step list), not in any member -- which is why M4 is a rule change while T2 "
            "would be a capability change"
        ),
    }


# --------------------------------------------------------------------------- #
# What the binder can express at all, read off the source
# --------------------------------------------------------------------------- #

_KIND_RE = re.compile(r'kind == "([a-z_.]+)"|kind in \(([^)]*)\)')


def binder_expression_surface(frozen: Any) -> dict[str, Any]:
    """Enumerate the vocabulary a goal can demand, from ``_bind`` and ``_goal_reached``.

    This is what makes the exhaustiveness claim checkable instead of asserted: the
    grid's axes are derived from the goal predicate's clauses, and the binder's
    action vocabulary is listed by parsing the frozen source.
    """

    bind_source = inspect.getsource(frozen.p52a._bind)
    goal_source = inspect.getsource(frozen.p52a._goal_reached)

    kinds: set[str] = set()
    for direct, group in _KIND_RE.findall(bind_source):
        if direct:
            kinds.add(direct)
        for token in re.findall(r'"([a-z_.]+)"', group or ""):
            kinds.add(token)

    clauses = {
        clause: clause in goal_source
        for clause in (
            "task.goal_files",
            "task.goal_language",
            "requires_explicit_language_override",
        )
    }
    main_only = (
        bool(re.search(r"main\s*=\s*task\.main_path", bind_source))
        and "task.main_path" in bind_source
    )
    return {
        "bindable_kinds": sorted(kinds),
        "bindable_kind_count": len(kinds),
        "goal_clauses": sorted(clauses),
        "goal_clause_present": clauses,
        "binder_targets_main_path_only": main_only,
        "grid_axes": {
            "content_route": list(CONTENT_ROUTES),
            "language_route": list(LANGUAGE_ROUTES),
        },
        "cells_in_grid": len(grid()),
        "cells_skipped_as_trivial": ["none__none"],
        "coverage_argument": (
            "the goal predicate is a conjunction over goal_files content and goal_language "
            "selection (+ optional override flag); each clause has exactly one route from a "
            "given initial world, so the 3 x 4 route grid covers every conjunction the "
            "binder can express on a single path"
        ),
        "not_covered": (
            "anything needing a second action target (T3), a second transformation of one "
            "file (T1), or an inter-member evidence channel (T2) -- each reported with its "
            "own witness above"
        ),
    }


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def combination_only_context_ids(surface: Mapping[str, Any]) -> set[str]:
    """Context ids where no singleton beats the blank, recomputed from raw episodes.

    ``measure_surface`` publishes the *count* of jointly-required contexts; classifying
    an individual changed cell needs to know *which* ones.  The predicate is the same one
    ``audit_taiji_b0_task_reachability_precheck.combination_only_solvable_contexts`` uses
    (``max_i S_i <= B`` on the context), applied per context instead of per surface.
    """

    buckets: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for episode in surface["episodes_raw"]:
        key = (str(episode["task_id"]), tuple(str(m) for m in episode["active_members"]))
        buckets.setdefault(key, []).append(1 if episode["success"] else 0)

    def rate(task_id: str, cell: tuple[str, ...]) -> float | None:
        values = buckets.get((task_id, cell))
        return None if not values else sum(values) / len(values)

    ids: set[str] = set()
    for context_id in surface["contexts"]:
        singletons = [
            value
            for value in (
                rate(context_id, (member,)) for member in sorted(surface["singleton_success_rates"])
            )
            if value is not None
        ]
        blank = rate(context_id, ())
        oracle = max(singletons) if singletons else 0.0
        if oracle <= (blank if blank is not None else 0.0):
            ids.add(context_id)
    return ids


def classify_changed_cells(
    forensics: Mapping[str, Any], audited: Mapping[str, Any]
) -> dict[str, Any]:
    """Sort every cell M4 changes into handoff-explained, fallback-only, unexplained.

    A pair success rate can move for two very different reasons.  On a context where a
    singleton already succeeds, the all-singleton oracle is already at ``success``, so
    the change cannot lift the claim-bearing gain: that is the ordinary fallback effect
    of blocking a dead-end member.  On a jointly-required context the only legitimate
    explanation is that a *second* member acted where the frozen rule never let one
    through -- which the per-step trace says directly, and which is what the counterfactual
    is supposed to do.  Anything else is a flip with no mechanism behind it.
    """

    jointly_required = combination_only_context_ids(audited)
    handoff_explained: list[dict[str, Any]] = []
    fallback_only: list[dict[str, Any]] = []
    unexplained: list[dict[str, Any]] = []
    for change in forensics["changed_cells"]:
        per_context: list[dict[str, Any]] = []
        required: list[dict[str, Any]] = []
        for context in change["per_context"]:
            task_id = str(context["audited"]["task_id"])
            actors = sorted(
                {
                    str(step.get("chosen"))
                    for episode in context["audited"]["episodes"]
                    for step in episode["trace"]
                    if step.get("chosen") and step.get("executed")
                }
            )
            entry = {
                "context": task_id,
                "jointly_required": task_id in jointly_required,
                "acting_members": actors,
            }
            per_context.append(entry)
            if entry["jointly_required"]:
                required.append(entry)
        record = {
            "cell": change["cell"],
            "delta": change["delta"],
            "baseline_success_rate": change["baseline_success_rate"],
            "audited_success_rate": change["audited_success_rate"],
            "per_context": per_context,
        }
        if not required:
            fallback_only.append(record)
        elif all(len(item["acting_members"]) >= 2 for item in required):
            handoff_explained.append(record)
        else:
            unexplained.append(record)
    return {
        "handoff_explained": handoff_explained,
        "fallback_only": fallback_only,
        "unexplained": unexplained,
        "jointly_required_contexts": sorted(jointly_required),
        "reading": (
            "handoff_explained: on every jointly-required context of the cell at least "
            "two members acted under M4; fallback_only: the cell only changed on contexts "
            "a singleton already solves, so the oracle-bound gain is unmoved; "
            "unexplained: a jointly-required context flipped without a second actor"
        ),
    }


def measure_cell(
    frozen: Any,
    counterfactual: Any,
    audit: Any,
    *,
    cell: Mapping[str, Any],
    members: Mapping[str, Any],
    embedder: Any,
    audited_episode: Any,
    frozen_episode: Any,
    margin: float,
    contexts_per_cell: int = CONTEXTS_PER_CELL,
) -> dict[str, Any]:
    """One structural cell under both rules, plus the forensic trace of any change."""

    tasks = build_cell_tasks(frozen, cell, contexts_per_cell)["tasks"]
    baseline = audit.measure_surface(
        frozen,
        counterfactual,
        frozen_episode,
        members,
        embedder,
        tasks,
        label=cell["label"],
        margin=margin,
    )
    audited = audit.measure_surface(
        frozen,
        counterfactual,
        audited_episode,
        members,
        embedder,
        tasks,
        label=cell["label"],
        margin=margin,
    )
    forensics = audit.improvement_forensics(baseline, audited)
    classification = classify_changed_cells(forensics, audited)

    combination_only = audited["combination_only_contexts"] > 0
    row = {
        "cell": cell["label"],
        "content_route": cell["content_route"],
        "language_route": cell["language_route"],
        "existence_precondition": cell["existence_precondition"],
        "expect_combination_only": cell["expect_combination_only"],
        "observed_combination_only": combination_only,
        "combination_only_prediction_matches": combination_only == cell["expect_combination_only"],
        "contexts": audited["contexts"],
        "singleton_success_rates_frozen": baseline["singleton_success_rates"],
        "singleton_success_rates_audited": audited["singleton_success_rates"],
        "best_pair_frozen": baseline["best_pair"],
        "best_pair_gain_frozen": baseline["best_pair_gain"],
        "best_pair_audited": audited["best_pair"],
        "best_pair_gain_audited": audited["best_pair_gain"],
        "gain_delta": audited["best_pair_gain"] - baseline["best_pair_gain"],
        "positive_under_frozen": baseline["positive_same_reference_gain"],
        "positive_under_m4": audited["positive_same_reference_gain"],
        "combination_only_contexts": audited["combination_only_contexts"],
        "interleaved_contexts_frozen": baseline["interleaved_contexts"],
        "interleaved_contexts_audited": audited["interleaved_contexts"],
        "changed_cells": forensics["changed_count"],
        "handoff_explained_changes": classification["handoff_explained"],
        "fallback_only_changes": classification["fallback_only"],
        "unexplained_changes": classification["unexplained"],
        "jointly_required_contexts_audited": classification["jointly_required_contexts"],
        "stop_reasons_frozen": baseline["stop_reasons"],
        "stop_reasons_audited": audited["stop_reasons"],
        "intervention_reality_frozen_ok": bool(
            baseline["intervention_reality"]["interventions_happened"]
        ),
        "intervention_reality_audited_ok": bool(
            audited["intervention_reality"]["interventions_happened"]
        ),
        "reference_requirements_audited": audited["reference_requirements"],
    }
    return row


def outcome_distinctness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Do the cells differ in outcome, or only in name?

    The mirror image of the hardening diagnostic: there, identical fingerprints were
    the bad news (six surfaces, one structure).  Here the cells *are* different
    structures, so identical fingerprints would mean the grid does not separate them.
    """

    fingerprints = {
        row["cell"]: (
            row["best_pair_gain_audited"],
            row["best_pair_gain_frozen"],
            row["combination_only_contexts"],
            row["interleaved_contexts_audited"],
        )
        for row in rows
    }
    distinct = {str(item) for item in fingerprints.values()}
    return {
        "cells_compared": len(fingerprints),
        "distinct_outcome_fingerprints": len(distinct),
        "fingerprints": {cell: str(item) for cell, item in sorted(fingerprints.items())},
        "grid_separates_cells": len(distinct) > 1,
    }


def structure_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["positive_under_m4"]]
    positive_cells = sorted(row["cell"] for row in positive)
    gained = sorted(row["cell"] for row in rows if row["gain_delta"] > 0.0)
    combination_only = sorted(row["cell"] for row in rows if row["observed_combination_only"])
    predicted_only = sorted(row["cell"] for row in rows if row["expect_combination_only"])

    shared = {
        "existence_precondition": all(row["existence_precondition"] for row in positive),
        "combination_only": all(row["observed_combination_only"] for row in positive),
        "requires_second_member": all(
            row["language_route"] != "none" and row["content_route"] != "none" for row in positive
        ),
    }
    return {
        "positive_gain_cells": positive_cells,
        "positive_gain_count": len(positive_cells),
        "gain_delta_positive_cells": gained,
        "observed_combination_only_cells": combination_only,
        "predicted_combination_only_cells": predicted_only,
        "combination_only_prediction_contradicted": sorted(
            set(predicted_only) ^ set(combination_only)
        ),
        "prediction_contradictions_are_findings": True,
        "gain_confined_to_create_and_override": bool(positive_cells)
        and set(positive_cells) == {"create__override"},
        "gain_reproduces_on_a_second_structure": bool(
            len([cell for cell in positive_cells if cell != "create__override"]) >= 1
        ),
        "shared_by_positive_cells": {
            name: (bool(value) if bool(positive) else None) for name, value in shared.items()
        },
        "cells_with_unexplained_change": sorted(
            {row["cell"] for row in rows if row["unexplained_changes"]}
        ),
        "handoff_explained_pair_cells": sorted(
            {item["cell"] for row in rows for item in row["handoff_explained_changes"]}
        ),
        "fallback_only_pair_cells": sorted(
            {item["cell"] for row in rows for item in row["fallback_only_changes"]}
        ),
        "m4_regresses_cells": sorted(row["cell"] for row in rows if row["gain_delta"] < 0.0),
    }


def seed_sweep(
    frozen: Any,
    counterfactual: Any,
    audit: Any,
    *,
    cells: Sequence[Mapping[str, Any]],
    embedder: Any,
    audited_episode: Any,
    margin: float,
    offsets: Sequence[int] = SEED_OFFSETS,
    contexts_per_cell: int = CONTEXTS_PER_CELL,
) -> dict[str, Any]:
    """Is the structural result a seed artifact?  Re-run M4 at each offset."""

    rows: list[dict[str, Any]] = []
    for offset in offsets:
        members = audit.train_members_with_seed(frozen, embedder, offset)
        for cell in cells:
            tasks = build_cell_tasks(frozen, cell, contexts_per_cell)["tasks"]
            measured = audit.measure_surface(
                frozen,
                counterfactual,
                audited_episode,
                members,
                embedder,
                tasks,
                label=cell["label"],
                margin=margin,
            )
            rows.append(
                {
                    "seed_offset": offset,
                    "cell": cell["label"],
                    "best_pair": measured["best_pair"],
                    "best_pair_gain": measured["best_pair_gain"],
                    "positive": measured["positive_same_reference_gain"],
                    "reality_ok": measured["intervention_reality"]["interventions_happened"],
                }
            )
    positive_cells = {row["cell"] for row in rows if row["positive"]}
    always = {
        cell
        for cell in positive_cells
        if all(row["positive"] for row in rows if row["cell"] == cell)
    }
    return {
        "offsets": list(offsets),
        "rows": rows,
        "positive_cells_any_seed": sorted(positive_cells),
        "positive_cells_every_seed": sorted(always),
        "reality_ok_everywhere": all(bool(row["reality_ok"]) for row in rows),
    }


def probe(
    *,
    contexts_per_cell: int = CONTEXTS_PER_CELL,
    seed_offsets: Sequence[int] = SEED_OFFSETS,
) -> dict[str, Any]:
    counterfactual = load_counterfactual()
    audit = load_audit()
    frozen = counterfactual.load_frozen()
    embedder = frozen.DocumentEmbedder()
    margin = float(
        json.loads(FROZEN_ROUTE_A_REPORT.read_text(encoding="utf-8"))["control_summary"]["margin"]
    )

    shipped_episode = frozen._member_episode
    shipped_revision = int(getattr(frozen, "RULE_REVISION", 0))
    if shipped_revision >= 1:
        # The audited rule is what the source ships, so the contrast can only come from
        # undoing it: ``*_frozen`` columns must not start measuring the audited rule.
        audited_episode = shipped_episode
        delta = {
            "variant": RULE_UNDER_AUDIT,
            "direction": "shipped",
            "arm_is_shipped_source": True,
            "shipped_rule_revision": shipped_revision,
        }
        frozen_episode, baseline_delta = counterfactual.build_reverted(frozen, RULE_UNDER_AUDIT)
    else:
        frozen_episode = shipped_episode
        baseline_delta = None
        audited_episode, delta = counterfactual.build_counterfactual(frozen, RULE_UNDER_AUDIT)
    if frozen_episode is audited_episode:
        raise SystemExit(
            "both arms are the same function: the frozen-vs-audited contrast would be a "
            "tautology. Resolve which rule revision the gate ships before measuring."
        )
    assert frozen._member_episode is shipped_episode

    cells = grid()
    surfaces = {
        cell["label"]: build_cell_tasks(frozen, cell, contexts_per_cell) for cell in cells
    }
    all_tasks = [task for surface in surfaces.values() for task in surface["tasks"]]
    frozen.p52a._assert_nontrivial_goals(all_tasks, partition="probe")

    validity_rows = [
        validity(frozen, cell, contexts_per_cell=contexts_per_cell) for cell in cells
    ]
    not_measurable = [
        {"cell": row["cell"], "because": row["not_measurable_because"]}
        for row in validity_rows
        if not row["measurable"]
    ]
    measurable = [
        cell for cell in cells if cell["label"] not in {item["cell"] for item in not_measurable}
    ]
    combination_only_measurable = [
        cell
        for cell in measurable
        if cell_is_combination_only(cell["content_route"], cell["language_route"])
    ]
    if len(combination_only_measurable) < MIN_MEASURABLE_COMBINATION_ONLY_CELLS:
        print(
            f"only {len(combination_only_measurable)} predicted jointly-required cells are "
            f"measurable (need {MIN_MEASURABLE_COMBINATION_ONLY_CELLS}); not publishing a "
            "structural verdict",
            file=sys.stderr,
        )
        raise SystemExit(2)

    members = audit.train_members_with_seed(frozen, embedder, 0)
    rows = [
        measure_cell(
            frozen,
            counterfactual,
            audit,
            cell=cell,
            members=members,
            embedder=embedder,
            audited_episode=audited_episode,
            frozen_episode=frozen_episode,
            margin=margin,
            contexts_per_cell=contexts_per_cell,
        )
        for cell in measurable
    ]

    inert = [row["cell"] for row in rows if not row["intervention_reality_audited_ok"]]
    if inert:
        print(
            f"intervention-reality gate failed on cells {inert}: an inert non-baseline "
            "cell would fabricate the gain",
            file=sys.stderr,
        )
        raise SystemExit(2)

    verdict = structure_verdict(rows)
    sweep = seed_sweep(
        frozen,
        counterfactual,
        audit,
        cells=measurable,
        embedder=embedder,
        audited_episode=audited_episode,
        margin=margin,
        offsets=seed_offsets,
        contexts_per_cell=contexts_per_cell,
    )

    return {
        "format": PROBE_FORMAT,
        "version": VERSION,
        "status": "draft_for_review",
        "question": (
            "does the M4 gain hold on more than one composition structure, or only on "
            "create + explicit override?"
        ),
        "rule_under_audit": RULE_UNDER_AUDIT,
        "shipped_rule_revision": shipped_revision,
        "arm_provenance": {
            "audited_arm": (
                "shipped source, unpatched"
                if shipped_revision >= 1
                else "shipped source + forward patch"
            ),
            "baseline_arm": (
                "shipped source reverted to rule_revision 0"
                if shipped_revision >= 1
                else "shipped source as-is (rule_revision 0)"
            ),
            "arms_are_distinct": frozen_episode is not audited_episode,
            "forward_delta": delta,
            "baseline_delta": baseline_delta,
        },
        "rule_delta": delta,
        "does_not_change": [
            (
                "HANDOFF-M4 ships in this gate script as rule_revision=1; no product "
                "mechanism adopts it and no runner uses this script"
                if shipped_revision >= 1
                else "M4 is NOT implemented; no gate, runner, rule or frozen artifact changes"
            ),
            "no task is registered and nothing is trained for a candidate cell",
            "growth_admitted=false and can_promote=false remain in force",
            "a positive structural result is still not a claim of achieved collaboration",
        ],
        "frozen_attribute_intact": frozen._member_episode is shipped_episode,
        "margin": margin,
        "contexts_per_cell": contexts_per_cell,
        "binder_expression_surface": binder_expression_surface(frozen),
        "validity": validity_rows,
        "not_measurable_cells": not_measurable,
        "cells_measured": [row["cell"] for row in rows],
        "rows": rows,
        "outcome_distinctness": outcome_distinctness(rows),
        "verdict": verdict,
        "seed_sweep": sweep,
        "section_5_2_candidates": candidate_witnesses(frozen, counterfactual),
        "t2_member_input_surface": member_input_surface(
            frozen, counterfactual, combination_only_measurable[0], embedder
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--contexts-per-cell",
        type=int,
        default=CONTEXTS_PER_CELL,
        help="contexts per structural cell (default reproduces the archived 20260913 report)",
    )
    parser.add_argument(
        "--seed-offsets",
        type=int,
        nargs="+",
        default=list(SEED_OFFSETS),
        help="seed offsets for the seed sweep (default reproduces the archived report)",
    )
    args = parser.parse_args(argv)

    if args.contexts_per_cell < 1:
        parser.error("--contexts-per-cell must be >= 1")

    payload = probe(
        contexts_per_cell=args.contexts_per_cell, seed_offsets=tuple(args.seed_offsets)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"scale: {payload['contexts_per_cell']} contexts/cell, "
        f"{len(payload['seed_sweep']['offsets'])} seed offsets {payload['seed_sweep']['offsets']}"
    )
    surface = payload["binder_expression_surface"]
    print(
        f"binder vocabulary: {surface['bindable_kind_count']} kinds; "
        f"main-path-only binder: {surface['binder_targets_main_path_only']}"
    )
    print(
        f"cells: {len(payload['validity'])} in grid, "
        f"{len(payload['cells_measured'])} measured, "
        f"not measurable: {[item['cell'] for item in payload['not_measurable_cells']] or 'none'}"
    )
    for item in payload["not_measurable_cells"]:
        print(f"  dropped {item['cell']}: {item['because']}")
    provenance = payload["arm_provenance"]
    print(
        f"arms: baseline={provenance['baseline_arm']}; audited={provenance['audited_arm']}; "
        f"gate rule_revision={payload['shipped_rule_revision']}"
    )
    print(f"{'cell':<24} {'combo_only':<11} {'frozen':<8} {'m4':<8} {'delta':<8} interleave")
    for row in payload["rows"]:
        print(
            f"{row['cell']:<24} "
            f"{str(row['observed_combination_only']):<11} "
            f"{row['best_pair_gain_frozen']:<+8.3f} "
            f"{row['best_pair_gain_audited']:<+8.3f} "
            f"{row['gain_delta']:<+8.3f} "
            f"{row['interleaved_contexts_frozen']}->{row['interleaved_contexts_audited']}"
        )
    distinct = payload["outcome_distinctness"]
    print(
        f"grid separation: {distinct['distinct_outcome_fingerprints']} distinct fingerprints "
        f"over {distinct['cells_compared']} cells"
    )
    verdict = payload["verdict"]
    print(f"positive under M4: {verdict['positive_gain_cells']}")
    print(f"gain confined to create__override: {verdict['gain_confined_to_create_and_override']}")
    print(
        f"gain reproduces on a second structure: "
        f"{verdict['gain_reproduces_on_a_second_structure']}"
    )
    print(f"m4 regresses cells: {verdict['m4_regresses_cells'] or 'none'}")
    print(
        f"combination-only prediction contradictions: "
        f"{verdict['combination_only_prediction_contradicted'] or 'none'}"
    )
    sweep = payload["seed_sweep"]
    print(f"positive every seed: {sweep['positive_cells_every_seed']}")
    print(f"unexplained changes: {verdict['cells_with_unexplained_change'] or 'none'}")
    print("section 5.2 candidates:")
    for name, item in payload["section_5_2_candidates"].items():
        print(f"  {name:<26} expressible={item['expressible']} verdict={item['verdict']}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
