"""Route-B handoff feasibility probe -- B0 design package section 5.3, condition 2.

Why this probe exists
---------------------
The B0 pre-check can prove that *arbitration* cannot clear the gate
(``max(S_i, S_j) <= max_i S_i``), but it cannot answer the next question:
**can the frozen composition rule actually produce an interleaved trajectory at
all?**  The oracle has no access to the binding-failure pattern, and that pattern
is exactly what decides whether the second member ever gets to act.

So condition 2 of section 5.3 requires a **scripted, untrained probe**: run the
frozen mechanism over a candidate task surface and read the trajectories.  This
module does that, reusing three frozen artefacts verbatim:

* the composition rule -- ``p5_2b._member_episode`` (priority fallback:
  ``chosen = bindable[0]``);
* the four member learners -- ``p5_2b._train_members``, trained only on the four
  existing train template families, so a candidate task is evaluated against
  members that never saw it;
* the factorial cell enumeration and repeat structure -- ``p5_2b.CELL_MEMBER_SETS``
  and ``p5_2b.REPEATS``.

Nothing is trained for a candidate task, nothing is registered, and no gate is
modified.  ``p5_2b`` is imported rather than re-implemented on purpose: the
frozen constants *are* the specification here, so inheriting them is correct
(unlike the P5.2c gates, which had their own datasets and therefore had to
re-implement the helpers they delegate to).

Ground truth
------------
Before any candidate is probed, the probe re-runs the **frozen** validation
surface through ``p5_2b._execute_matrix`` and asserts the per-cell outcomes match
the frozen route-C report.  A probe that cannot reproduce known history is not
allowed to report anything about the future.

What the candidates test
------------------------
``p52a._bind`` targets ``task.main_path`` for *every* action kind, and
``workspace.create`` binds ``content = task.goal_files[main_path]`` directly.
Two consequences the probe measures rather than assumes:

* ``dual_requirement`` -- one file whose goal needs **both** an explicit language
  override and a content patch.  Expressible, and by construction no single member
  can satisfy it: ``member-a`` only overrides, ``member-b`` only patches.
* ``create_then_patch`` -- a file that must be created and then patched.  Expected
  to be **single-member solvable** (``create`` already lands on the goal content),
  which is the evidence that this shape needs a ``_bind`` change rather than a
  representation change.

The probe also reports **order sensitivity**: the frozen rule always lets the
first member in ``active_members`` order act first.  If a handoff succeeds under
one order and fails under the other, the blocker is the frozen *order* -- a much
smaller change than a new mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "scripts" / "training"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_b0_handoff_feasibility_probe_20260913.json"

FROZEN_GATE = TRAINING_DIR / "eval_taiji_p5_2b_group_causal_corpora_gate.py"
DICTIONARY_MODULE = TRAINING_DIR / "audit_taiji_b0_measurement_reachability.py"
PRECHECK_MODULE = TRAINING_DIR / "audit_taiji_b0_task_reachability_precheck.py"
FROZEN_ROUTE_C_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json"
)

PROBE_FORMAT = "taiji-b0-handoff-feasibility-probe-v1"
VERSION = 1

#: Outcome encoding, frozen with the measurement dictionary.
SUCCESS_OUTCOME = 1.0
FAILURE_OUTCOME = -1.0

#: Candidate contexts per surface.  Four mirrors the current design's holdout
#: size, so the required ``k`` from the pre-check is directly comparable.
CANDIDATE_CONTEXT_COUNT = 4

#: Candidate context index base -- deliberately far from the 100..111 range used
#: by the frozen validation surface so an id can never be confused across runs.
CANDIDATE_INDEX_BASE = 200


def _load(name: str, path: Path):
    """Load a module by path, caching it under ``name`` in ``sys.modules``."""

    if name in sys.modules:
        return sys.modules[name]
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_frozen() -> Any:
    """The frozen P5.2b gate; its constants and helpers are the specification."""

    return _load("_b0_probe_frozen_p52b", FROZEN_GATE)


def load_dictionary() -> Any:
    return _load("_b0_probe_dictionary", DICTIONARY_MODULE)


def load_precheck() -> Any:
    return _load("_b0_probe_precheck", PRECHECK_MODULE)


# --------------------------------------------------------------------------- #
# Candidate task surfaces (probe-only; NOT registered, NOT frozen)
# --------------------------------------------------------------------------- #


def _goal_text(tag: str, path: str) -> str:
    """Goal wording in the frozen P5.2a register, so the cue stays in-distribution."""

    return (
        f"Work order {tag}: reach the requested end state on "
        f"{path} using the workspace contract, and leave every "
        f"temporary change reverted."
    )


def dual_requirement_tasks(frozen: Any, count: int = CANDIDATE_CONTEXT_COUNT) -> tuple[Any, ...]:
    """One file whose goal needs an explicit override *and* a content patch.

    Expressible under ``_bind`` because both actions target ``main_path``.
    ``member-a`` can only override, ``member-b`` can only patch, so no singleton
    can satisfy the goal -- the shape the current task set lacks.
    """

    p52a, p52 = frozen.p52a, frozen.p52
    tasks = []
    for offset in range(count):
        index = CANDIDATE_INDEX_BASE + offset
        name = f"dual_{index:03d}.py"
        base = f"def run_{index}():\n    return {index}\n"
        goal = f"def run_{index}():\n    return {index} + 1\n"
        steps = (
            p52.ScriptedStep("workspace.read", {"path": name}),
            p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}),
            p52.ScriptedStep(
                "editor.set_language",
                {"path": name, "programming_language_id": "python", "user_override": True},
            ),
            p52.ScriptedStep("workspace.read", {"path": name}),
            p52.ScriptedStep(
                "workspace.apply_patch",
                {
                    "path": name,
                    "before_digest": {"$digest_of": name},
                    "patch": p52._patch_ops(base, f"return {index}", f"return {index} + 1"),
                    "expected_after_digest": hashlib.sha256(goal.encode("utf-8")).hexdigest(),
                },
            ),
        )
        tasks.append(
            p52a.Task(
                task_id=f"b0probe-dual-{index}",
                goal_text=_goal_text(f"b0probe-dual-{index}", name),
                initial_files={name: base},
                goal_files={name: goal},
                goal_language={name: "python"},
                main_path=name,
                reference_steps=steps,
                partition="probe",
                template="dual_requirement",
                requires_explicit_language_override=True,
            )
        )
    return tuple(tasks)


def create_then_patch_tasks(frozen: Any, count: int = CANDIDATE_CONTEXT_COUNT) -> tuple[Any, ...]:
    """A file that must be created and then patched.

    Expected to be single-member solvable: ``workspace.create`` binds
    ``content = task.goal_files[main_path]``, so creating already lands on the
    goal.  Measuring that (instead of asserting it) is the point.
    """

    p52a, p52 = frozen.p52a, frozen.p52
    tasks = []
    for offset in range(count):
        index = CANDIDATE_INDEX_BASE + offset
        name = f"ctp_{index:03d}.txt"
        content = f"created {index}\n"
        steps = (
            p52.ScriptedStep("workspace.list", {"path": "."}),
            p52.ScriptedStep("workspace.create", {"path": name, "content": content}),
        )
        tasks.append(
            p52a.Task(
                task_id=f"b0probe-ctp-{index}",
                goal_text=_goal_text(f"b0probe-ctp-{index}", name),
                initial_files={},
                goal_files={name: content},
                goal_language={},
                main_path=name,
                reference_steps=steps,
                partition="probe",
                template="create_then_patch",
            )
        )
    return tuple(tasks)


def create_and_override_tasks(frozen: Any, count: int = CANDIDATE_CONTEXT_COUNT) -> tuple[Any, ...]:
    """A file that must be **created** *and* carry an explicit language override.

    Motivation from the first probe round: ``dual_requirement`` (override +
    patch) is already single-member solvable, because ``member-d``'s
    ``header_override`` policy is itself ``read -> resolve -> set_language ->
    apply_patch`` -- the existing task set already contains a dual-requirement
    shape, just assigned to one member.  Splitting the requirements does not help
    while one member's policy covers both.

    ``create`` + ``override`` is the combination no single policy covers:
    ``member-c`` creates but never overrides, ``member-a`` overrides but never
    creates, and ``member-d``'s patch step cannot bind while the file is absent.
    """

    p52a, p52 = frozen.p52a, frozen.p52
    tasks = []
    for offset in range(count):
        index = CANDIDATE_INDEX_BASE + offset
        name = f"crgover_{index:03d}.py"
        content = f"def run_{index}():\n    return {index}\n"
        steps = (
            p52.ScriptedStep("workspace.list", {"path": "."}),
            p52.ScriptedStep("workspace.create", {"path": name, "content": content}),
            p52.ScriptedStep("workspace.read", {"path": name}),
            p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}),
            p52.ScriptedStep(
                "editor.set_language",
                {"path": name, "programming_language_id": "python", "user_override": True},
            ),
        )
        tasks.append(
            p52a.Task(
                task_id=f"b0probe-crgover-{index}",
                goal_text=_goal_text(f"b0probe-crgover-{index}", name),
                initial_files={},
                goal_files={name: content},
                goal_language={name: "python"},
                main_path=name,
                reference_steps=steps,
                partition="probe",
                template="create_and_override",
                requires_explicit_language_override=True,
            )
        )
    return tuple(tasks)


CANDIDATE_BUILDERS: dict[str, Any] = {
    "dual_requirement": dual_requirement_tasks,
    "create_then_patch": create_then_patch_tasks,
    "create_and_override": create_and_override_tasks,
}


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def execute_surface(
    frozen: Any,
    members: Mapping[str, Any],
    embedder: Any,
    tasks: Sequence[Any],
    *,
    cell_member_sets: Sequence[tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Run one surface through the frozen episode function.

    ``cell_member_sets`` defaults to the frozen ``CELL_MEMBER_SETS``; passing a
    different enumeration is only used for the order-sensitivity diagnostic and
    still calls the frozen ``_member_episode`` for every episode.
    """

    cells = tuple(cell_member_sets) if cell_member_sets is not None else frozen.CELL_MEMBER_SETS
    root = Path(tempfile.mkdtemp(prefix="b0probe-"))
    cue_by_task = {task.task_id: embedder.embed([task.goal_text])[0] for task in tasks}
    episodes: list[dict[str, Any]] = []
    try:
        environment = frozen.WorkbenchEnvironment(root)
        for repeat in range(frozen.REPEATS):
            for task in tasks:
                cue = cue_by_task[task.task_id]
                for active in cells:
                    episode_id = (
                        f"b0probe-episode:{task.task_id}:r{repeat}:"
                        f"{'-'.join(active) or 'none'}"
                    )
                    episodes.append(
                        frozen._member_episode(
                            environment, task, cue, active, dict(members), episode_id
                        )
                    )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return episodes


def outcome_by_cell(episodes: Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], dict[str, float]]:
    """Mean outcome per cell and context, plus the pooled mean."""

    per_cell: dict[tuple[str, ...], dict[str, list[float]]] = {}
    for episode in episodes:
        key = tuple(episode["active_members"])
        value = SUCCESS_OUTCOME if episode["success"] else FAILURE_OUTCOME
        per_cell.setdefault(key, {}).setdefault(str(episode["task_id"]), []).append(value)

    result: dict[tuple[str, ...], dict[str, float]] = {}
    for key, by_context in per_cell.items():
        pooled = [value for values in by_context.values() for value in values]
        result[key] = {
            "mean_outcome": sum(pooled) / len(pooled),
            "success_rate": sum(1 for value in pooled if value > 0) / len(pooled),
            **{f"ctx:{context_id}": sum(values) / len(values) for context_id, values in
               sorted(by_context.items())},
        }
    return result


def table_from_outcomes(
    outcomes: Mapping[tuple[str, ...], Mapping[str, float]],
    context_ids: Sequence[str],
    member_ids: Sequence[str],
) -> Any:
    """Build a measurement-dictionary ``ContextTable`` from probe outcomes."""

    dictionary = load_dictionary()

    def value(key: tuple[str, ...], context_id: str) -> float:
        return float(outcomes[key][f"ctx:{context_id}"])

    baseline = {context_id: value((), context_id) for context_id in context_ids}
    singletons = {
        member: {context_id: value((member,), context_id) for context_id in context_ids}
        for member in member_ids
    }
    combinations = {
        tuple(sorted(key)): {context_id: value(key, context_id) for context_id in context_ids}
        for key in outcomes
        if len(key) == 2
    }
    return dictionary.ContextTable(
        contexts=tuple(context_ids),
        baseline=baseline,
        singletons=singletons,
        combinations=combinations,
    )


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def ground_truth(frozen: Any, members: Mapping[str, Any], embedder: Any) -> dict[str, Any]:
    """Reproduce the frozen validation matrix with the frozen executor verbatim."""

    contexts = frozen.p52a._validation_tasks()[: frozen.CONTEXT_COUNT]
    root = Path(tempfile.mkdtemp(prefix="b0probe-truth-"))
    try:
        cue_by_task = {
            task.task_id: embedder.embed([task.goal_text])[0] for task in contexts
        }
        episodes = frozen._execute_matrix(root, contexts, dict(members), cue_by_task)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    outcomes = outcome_by_cell(episodes)
    frozen_report = json.loads(FROZEN_ROUTE_C_REPORT.read_text(encoding="utf-8"))
    frozen_rows = {row["cell"]: row for row in frozen_report["success_matrix"]["rows"]}

    mismatches: list[dict[str, Any]] = []
    for key, values in outcomes.items():
        name = "-".join(key) if key else "none"
        row = frozen_rows.get(name)
        if row is None:
            mismatches.append({"cell": name, "reason": "cell absent from frozen report"})
            continue
        attempts = int(row["total_attempts"])
        successes = int(row["total_successes"])
        frozen_rate = successes / attempts if attempts else 0.0
        if abs(values["success_rate"] - frozen_rate) > 1e-9:
            mismatches.append(
                {
                    "cell": name,
                    "probe_success_rate": values["success_rate"],
                    "frozen_success_rate": frozen_rate,
                }
            )

    return {
        "contexts": [task.task_id for task in contexts],
        "episodes": len(episodes),
        "cells_compared": len(outcomes),
        "reproduced": not mismatches,
        "mismatches": mismatches,
        "note": (
            "the probe must reproduce known history before it is allowed to say "
            "anything about a candidate surface"
        ),
    }


# --------------------------------------------------------------------------- #
# Candidate probing
# --------------------------------------------------------------------------- #


def probe_surface(
    frozen: Any,
    members: Mapping[str, Any],
    embedder: Any,
    tasks: Sequence[Any],
    *,
    label: str,
    template: str,
    margin: float,
    candidates: Mapping[str, float],
) -> dict[str, Any]:
    """Probe one candidate surface and answer section 5.3 condition 2."""

    precheck = load_precheck()
    episodes = execute_surface(frozen, members, embedder, tasks)
    outcomes = outcome_by_cell(episodes)
    context_ids = [task.task_id for task in tasks]
    table = table_from_outcomes(outcomes, context_ids, frozen.MEMBER_IDS)

    trajectories = [
        precheck.classify_pair_trajectory(table, pair) for pair in table.pair_cells()
    ]
    interleaved_total = sum(item["contexts_interleaved"] for item in trajectories)
    combination_only = precheck.combination_only_solvable_contexts(table)

    # Order sensitivity: reverse each pair's priority and re-run the pairs only.
    reversed_cells = tuple(
        tuple(reversed(cell)) for cell in frozen.CELL_MEMBER_SETS if len(cell) == 2
    )
    reversed_episodes = execute_surface(
        frozen, members, embedder, tasks, cell_member_sets=reversed_cells
    )
    reversed_outcomes = outcome_by_cell(reversed_episodes)
    order_effects: list[dict[str, Any]] = []
    for key in outcomes:
        if len(key) != 2:
            continue
        forward = outcomes[key]["mean_outcome"]
        backward = reversed_outcomes[tuple(reversed(key))]["mean_outcome"]
        order_effects.append(
            {
                "pair": "+".join(key),
                "forward_outcome": forward,
                "reversed_outcome": backward,
                "best_order_outcome": max(forward, backward),
                "order_headroom": max(forward, backward) - forward,
                "order_sensitive": bool(abs(forward - backward) > 1e-9),
            }
        )

    # Headroom a better arbitration could recover on this surface, without any
    # new task and without any representation change.
    order_headroom = (
        sum(item["order_headroom"] for item in order_effects) / len(order_effects)
        if order_effects
        else 0.0
    )
    # Ceiling of a combination that picks the best member per context: the
    # within-pair oracle, which the pre-check already proved is dominated.
    pair_oracle_gain = (
        sum(
            max(
                table.singleton(context_id, key[0]),
                table.singleton(context_id, key[1]),
            )
            - max(table.singleton(context_id, m) for m in table.singletons)
            for key in table.pair_cells()
            for context_id in table.contexts
        )
        / (len(table.pair_cells()) * len(table.contexts))
        if table.pair_cells()
        else 0.0
    )

    return {
        "label": label,
        "template": template,
        "contexts": context_ids,
        "episodes": len(episodes),
        "cells": [
            {
                "cell": "+".join(key) if key else "none",
                "mean_outcome": values["mean_outcome"],
                "success_rate": values["success_rate"],
            }
            for key, values in sorted(outcomes.items(), key=lambda item: (len(item[0]), item[0]))
        ],
        "singleton_success_rates": {
            member: outcomes[(member,)]["success_rate"] for member in frozen.MEMBER_IDS
        },
        "trajectories": trajectories,
        "combination_only_contexts": combination_only,
        "combination_only_gain_potential": precheck.combination_only_gain_potential(table),
        "reference_requirements": precheck.reference_requirements(
            table, margin=margin, candidates=candidates
        ),
        "condition_2": {
            "requirement": "the frozen mechanism must produce an interleaved trajectory",
            "interleaved_contexts": interleaved_total,
            "satisfied": bool(interleaved_total > 0),
            "reading": (
                "the frozen priority-fallback rule produced an interleaved "
                "trajectory, so a handoff is reachable without a mechanism change"
                if interleaved_total > 0
                else "no interleaved trajectory: under the frozen rule the pair "
                "still reproduces a single member's outcome, so this task shape "
                "needs a mechanism change, not a representation change"
            ),
        },
        "order_sensitivity": {
            "pairs": order_effects,
            "any_order_sensitive": any(item["order_sensitive"] for item in order_effects),
            "mean_order_headroom": order_headroom,
            "within_pair_oracle_gain": pair_oracle_gain,
            "reading": (
                "reversing the priority changes an outcome, so the frozen order "
                "-- not the task or the representation -- is what blocks the handoff"
                if any(item["order_sensitive"] for item in order_effects)
                else "priority order does not change any outcome on this surface"
            ),
            "arbitration_ceiling_note": (
                "mean_order_headroom is recoverable by a better arbitration, but "
                "within_pair_oracle_gain is the ceiling of ANY member-arbitration "
                "mechanism and is <= 0 by construction -- so fixing the order alone "
                "still cannot produce a collaboration claim"
            ),
        },
    }


def run_probe(candidates: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Train the frozen members once and probe every candidate surface."""

    frozen = load_frozen()
    embedder = frozen.DocumentEmbedder()
    members = frozen._train_members(embedder)

    route_a = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
        ).read_text(encoding="utf-8")
    )
    margin = float(route_a["control_summary"]["margin"])
    if candidates is None:
        candidates = {
            "all_singleton_oracle": 1.5,
            "best_fixed_singleton": 0.5,
            "best_observed_fixed_pair": 1.0,
        }

    truth = ground_truth(frozen, members, embedder)
    surfaces = []
    if truth["reproduced"]:
        for label, builder in CANDIDATE_BUILDERS.items():
            tasks = builder(frozen)
            frozen.p52a._assert_nontrivial_goals(tasks, partition="probe")
            surfaces.append(
                probe_surface(
                    frozen,
                    members,
                    embedder,
                    tasks,
                    label=label,
                    template=tasks[0].template,
                    margin=margin,
                    candidates=candidates,
                )
            )

    return {
        "format": PROBE_FORMAT,
        "version": VERSION,
        "status": "draft_for_review",
        "does_not_change": [
            "no gate, preregistration, frozen report or threshold is modified",
            "no candidate task is registered; the definitions live only in this probe",
            "no model is trained for a candidate task -- the frozen members are reused",
            "growth_admitted=false and can_promote=false remain in force",
        ],
        "frozen_artifacts_reused": {
            "gate": str(FROZEN_GATE.relative_to(PROJECT_ROOT)),
            "composition_rule": "p5_2b._member_episode (priority fallback)",
            "members": "p5_2b._train_members (four train template families only)",
            "cells": "p5_2b.CELL_MEMBER_SETS",
            "repeats": "p5_2b.REPEATS",
        },
        "margin": margin,
        "candidates_for_review": candidates,
        "ground_truth": truth,
        "surfaces": surfaces,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    truth = payload["ground_truth"]
    print(f"ground truth reproduced: {truth['reproduced']} ({truth['cells_compared']} cells)")
    for mismatch in truth["mismatches"]:
        print(f"  MISMATCH {mismatch}")
    for surface in payload["surfaces"]:
        condition = surface["condition_2"]
        print(
            f"  {surface['label']:<20} interleaved={condition['interleaved_contexts']} "
            f"satisfied={condition['satisfied']} "
            f"combination_only={len(surface['combination_only_contexts'])} "
            f"order_sensitive={surface['order_sensitivity']['any_order_sensitive']}"
        )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
