"""Counterfactual measurement of the proposed M1 handoff rule (decision D5).

Why a counterfactual, and why this way
--------------------------------------
The handoff probe established that the task layer is solved (``create_and_override``:
every singleton fails, ``k = 4/4``, all three candidate references feasible) while the
mechanism still produces ``interleaved = 0``.  Decision D5 asks whether to change the
composition rule.  That decision needs one more measurement: **would the proposed rule
actually work, and would it regress the frozen surface?**

Answering it must not change the mechanism under test.  So the counterfactual is built
by taking ``inspect.getsource(p5_2b._member_episode)`` and applying **exactly one**
textual replacement::

    frozen:          chosen = bindable[0]
    M1-a:            the first bindable member whose action kind has not already been
                     executed in this episode; if every bindable member would repeat,
                     stop with ``no_progress``

The transformed source is executed in a **copy** of the frozen module namespace, so
``frozen._member_episode`` itself is never rebound.  The result is therefore literally
the frozen executor plus one documented delta -- drift is impossible by construction,
and the delta is pinned by a regression test.

What this module is not
-----------------------
It is **not** an implementation of M1.  No gate is modified, no rule is frozen, no
result is claimed as collaboration.  It reports what the proposed rule *would* do, so
that D5 can be decided on measurement instead of prose.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "scripts" / "training"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_b0_m1_counterfactual_20260913.json"

FROZEN_GATE = TRAINING_DIR / "eval_taiji_p5_2b_group_causal_corpora_gate.py"
HANDOFF_PROBE = TRAINING_DIR / "probe_taiji_b0_handoff_feasibility.py"
FROZEN_ROUTE_C_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json"
)

COUNTERFACTUAL_FORMAT = "taiji-b0-m1-counterfactual-v1"
VERSION = 1

#: The single anchor the counterfactual is allowed to touch.  A test asserts it is
#: unique in the frozen source, so a silent change to the frozen rule fails loudly.
FROZEN_SELECTION = "        chosen = bindable[0]\n"

#: M1-a: hand off when the current member would only repeat an already-executed
#: action kind, instead of waiting for a binding failure that never comes.
M1A_SELECTION = (
    '        executed_kinds = [s.get("kind") for s in steps if s.get("executed")]\n'
    '        chosen = next((c for c in bindable if c["kind"] not in executed_kinds), None)\n'
    "        if chosen is None:\n"
    "            steps.append(\n"
    "                {\n"
    '                    "tick": tick,\n'
    '                    "called": [c["member"] for c in calls],\n'
    '                    "executed": False,\n'
    '                    "stop": "no_progress",\n'
    "                }\n"
    "            )\n"
    '            return finish("no_progress")\n'
)

#: M1-b: rotate the executing member by the number of already-executed actions, so
#: every active member gets the floor in turn instead of the first one monopolising.
M1B_SELECTION = (
    '        executed_count = sum(1 for s in steps if s.get("executed"))\n'
    "        chosen = bindable[executed_count % len(bindable)]\n"
)

#: M2 anchor: the cue sequence length is the **episode's** step count, so a member
#: that joins at episode-step 3 is queried at its own position 4 -- outside the
#: repertoire it was trained on.  A member's own progress is never tracked, which is
#: why a late-joining member cannot contribute at all.
CUE_ANCHOR = "            cues = tuple([cue] * (len(steps) + 1))\n"

#: M2: query each member at **its own** step count instead of the episode's.
M2_CUE = (
    "            own_steps = sum(\n"
    '                1 for s in steps if s.get("executed") and s.get("chosen") == member_id\n'
    "            )\n"
    "            cues = tuple([cue] * (own_steps + 1))\n"
)

#: M3: hand off when the current member has already executed as many actions as its
#: training repertoire has steps.  The frozen rule has no way to learn this, because
#: ``ProceduralSequenceLearner.predict_episode`` always returns an action kind -- the
#: action vocabulary has no end-of-repertoire term, so a member cannot say "done".
#: The repertoire length therefore has to come from the member's train evidence.
M3_SELECTION = (
    "        done_by_member = {}\n"
    "        for step in steps:\n"
    '            if step.get("executed") and step.get("chosen"):\n'
    '                done_by_member[step["chosen"]] = (\n'
    '                    done_by_member.get(step["chosen"], 0) + 1\n'
    "                )\n"
    "        chosen = next(\n"
    "            (\n"
    "                cand\n"
    "                for cand in bindable\n"
    '                if done_by_member.get(cand["member"], 0)\n'
    '                < REPERTOIRE.get(cand["member"], 0)\n'
    "            ),\n"
    "            None,\n"
    "        )\n"
    "        if chosen is None:\n"
    "            steps.append(\n"
    "                {\n"
    '                    "tick": tick,\n'
    '                    "called": [c["member"] for c in calls],\n'
    '                    "executed": False,\n'
    '                    "stop": "repertoire_exhausted",\n'
    "                }\n"
    "            )\n"
    '            return finish("repertoire_exhausted")\n'
)

#: M4: a failed execution neither consumes the member's turn nor triggers a handoff,
#: so the first member retries a failing action until ``STEP_CAP``.  Block a member
#: whose most recent attempt failed, and clear the block as soon as anyone succeeds
#: (the world changed, so the previously failing action may now take effect).
M4_SELECTION = (
    "        last_success_index = max(\n"
    '            (i for i, s in enumerate(steps) if s.get("executed")), default=-1\n'
    "        )\n"
    "        blocked = {\n"
    '            s.get("chosen")\n'
    "            for i, s in enumerate(steps)\n"
    '            if s.get("chosen") and not s.get("executed") and i > last_success_index\n'
    "        }\n"
    '        chosen = next((c for c in bindable if c["member"] not in blocked), None)\n'
    "        if chosen is None:\n"
    "            steps.append(\n"
    "                {\n"
    '                    "tick": tick,\n'
    '                    "called": [c["member"] for c in calls],\n'
    '                    "executed": False,\n'
    '                    "stop": "all_members_blocked",\n'
    "                }\n"
    "            )\n"
    '            return finish("all_members_blocked")\n'
)

#: Each variant is a list of ``(anchor, replacement)`` pairs.  Every anchor must be
#: unique in the frozen source, so a silent change to the frozen rule fails loudly.
VARIANTS: dict[str, tuple[tuple[str, str], ...]] = {
    "m1a_no_progress": ((FROZEN_SELECTION, M1A_SELECTION),),
    "m1b_tick_rotation": ((FROZEN_SELECTION, M1B_SELECTION),),
    "m2_per_member_progress": ((CUE_ANCHOR, M2_CUE),),
    "m2a_progress_plus_handoff": (
        (FROZEN_SELECTION, M1A_SELECTION),
        (CUE_ANCHOR, M2_CUE),
    ),
    "m3_repertoire_aware": (
        (FROZEN_SELECTION, M3_SELECTION),
        (CUE_ANCHOR, M2_CUE),
    ),
    "m4_failure_handoff": (
        (FROZEN_SELECTION, M4_SELECTION),
        (CUE_ANCHOR, M2_CUE),
    ),
}

#: Variants that need an extra name injected into the counterfactual namespace.
NEEDS_REPERTOIRE: frozenset[str] = frozenset({"m3_repertoire_aware"})

#: Priority orders the effect check is run under.  The frozen cell enumeration puts
#: ``member-a`` first in every pair; reversing it tests whether the *order* rather
#: than the *rule* is what blocks the handoff.
ORDERS: tuple[str, ...] = ("frozen_order", "reversed_order")


def _load(name: str, path: Path):
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
    return _load("_b0_m1_frozen_p52b", FROZEN_GATE)


def load_handoff_probe() -> Any:
    return _load("_b0_m1_handoff_probe", HANDOFF_PROBE)


# --------------------------------------------------------------------------- #
# Building the counterfactual executor
# --------------------------------------------------------------------------- #


def member_repertoire(frozen: Any) -> dict[str, int]:
    """Per-member trained repertoire length, from the member's own train family.

    This is train-only evidence (the reference step counts of the family the member
    was consolidated on), so using it in the composition does not leak a test-side
    quantity -- but it *is* information the frozen rule never consumed, which is
    exactly why M3 is a design change rather than a tuning change.
    """

    families = frozen._family_tasks()
    return {
        member: max(len(task.reference_steps) for task in tasks)
        for member, tasks in families.items()
    }


def build_counterfactual(
    frozen: Any,
    variant: str = "m1a_no_progress",
    *,
    extra_names: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(episode_fn, delta)``: the frozen executor plus documented replacements.

    ``frozen._member_episode`` is captured before and re-checked after, so a caller
    can assert the frozen mechanism was not rebound.
    """

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")

    original = inspect.getsource(frozen._member_episode)
    transformed = original
    replacements: list[dict[str, Any]] = []
    for anchor, replacement in VARIANTS[variant]:
        occurrences = transformed.count(anchor)
        if occurrences != 1:
            raise SystemExit(
                f"anchor appears {occurrences} times in the frozen source, expected "
                f"exactly 1: {anchor.strip()!r}. The frozen rule changed and the "
                "counterfactual must be re-derived."
            )
        transformed = transformed.replace(anchor, replacement)
        replacements.append(
            {
                "anchor": anchor.rstrip("\n"),
                "replacement_lines": replacement.rstrip("\n").splitlines(),
            }
        )

    # Execute into a COPY of the module namespace: rebinding the frozen module's own
    # attribute would change the mechanism under test.
    namespace = dict(vars(frozen))
    if extra_names:
        namespace.update(extra_names)
    exec(compile(transformed, f"<counterfactual:{variant}>", "exec"), namespace)
    episode_fn = namespace["_member_episode"]

    delta = {
        "variant": variant,
        "replacements": replacements,
        "replacement_count": len(replacements),
        "injected_names": sorted(extra_names) if extra_names else [],
        "frozen_source_lines": len(original.splitlines()),
        "counterfactual_source_lines": len(transformed.splitlines()),
        "added_lines": len(transformed.splitlines()) - len(original.splitlines()),
        "frozen_attribute_unchanged": frozen._member_episode is not episode_fn,
        "rule_text": " + ".join(item["anchor"].strip() for item in replacements),
    }
    return episode_fn, delta


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def execute_surface(
    frozen: Any,
    episode_fn: Any,
    members: Mapping[str, Any],
    embedder: Any,
    tasks: Sequence[Any],
    *,
    cell_member_sets: Sequence[tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Run one surface through ``episode_fn`` (frozen or counterfactual)."""

    cells = tuple(cell_member_sets) if cell_member_sets is not None else frozen.CELL_MEMBER_SETS
    root = Path(tempfile.mkdtemp(prefix="b0m1-"))
    cue_by_task = {task.task_id: embedder.embed([task.goal_text])[0] for task in tasks}
    episodes: list[dict[str, Any]] = []
    try:
        environment = frozen.WorkbenchEnvironment(root)
        for repeat in range(frozen.REPEATS):
            for task in tasks:
                cue = cue_by_task[task.task_id]
                for active in cells:
                    episode_id = (
                        f"b0m1-episode:{task.task_id}:r{repeat}:{'-'.join(active) or 'none'}"
                    )
                    episodes.append(
                        episode_fn(environment, task, cue, active, dict(members), episode_id)
                    )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return episodes


def stop_reason_counts(episodes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        reason = str(episode.get("stop_reason", "?"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _frozen_rates() -> dict[str, float]:
    report = json.loads(FROZEN_ROUTE_C_REPORT.read_text(encoding="utf-8"))
    rates: dict[str, float] = {}
    for row in report["success_matrix"]["rows"]:
        attempts = int(row["total_attempts"])
        rates[row["cell"]] = (int(row["total_successes"]) / attempts) if attempts else 0.0
    return rates


def regression_check(
    frozen: Any, episode_fn: Any, members: Mapping[str, Any], embedder: Any
) -> dict[str, Any]:
    """Does the proposed rule change any outcome on the frozen validation surface?"""

    contexts = frozen.p52a._validation_tasks()[: frozen.CONTEXT_COUNT]
    episodes = execute_surface(frozen, episode_fn, members, embedder, contexts)
    probe = load_handoff_probe()
    outcomes = probe.outcome_by_cell(episodes)
    frozen_rates = _frozen_rates()

    rows: list[dict[str, Any]] = []
    for key, values in sorted(outcomes.items(), key=lambda item: (len(item[0]), item[0])):
        name = "-".join(key) if key else "none"
        reference = frozen_rates.get(name)
        rows.append(
            {
                "cell": name,
                "counterfactual_success_rate": values["success_rate"],
                "frozen_success_rate": reference,
                "delta": (None if reference is None else values["success_rate"] - reference),
            }
        )

    regressions = [row for row in rows if row["delta"] is not None and row["delta"] < 0]
    improvements = [row for row in rows if row["delta"] is not None and row["delta"] > 0]
    return {
        "contexts": [task.task_id for task in contexts],
        "episodes": len(episodes),
        "rows": rows,
        "regressions": regressions,
        "improvements": improvements,
        "no_regression": not regressions,
        "stop_reasons": stop_reason_counts(episodes),
        "reading": (
            "the proposed rule reproduces every frozen outcome"
            if not regressions and not improvements
            else (
                f"{len(regressions)} cell(s) regress and {len(improvements)} improve under "
                "the proposed rule; the frozen results stay valid only for the frozen rule"
            )
        ),
    }


def effect_check(
    frozen: Any,
    episode_fn: Any,
    members: Mapping[str, Any],
    embedder: Any,
    *,
    margin: float,
    candidates: Mapping[str, float],
) -> dict[str, Any]:
    """Does the proposed rule produce a handoff on the task shape that needs one?

    Run under both priority orders: the handoff probe showed the frozen order can
    suppress a capable member, so a rule that fails one way may succeed the other.
    """

    probe = load_handoff_probe()
    precheck = probe.load_precheck()
    dictionary = probe.load_dictionary()
    tasks = probe.create_and_override_tasks(frozen)
    context_ids = [task.task_id for task in tasks]

    by_order: dict[str, Any] = {}
    for order in ORDERS:
        cells = None
        if order == "reversed_order":
            # Keep the baseline and singletons; only the pair priority is reversed,
            # so the surface stays comparable cell for cell.
            cells = tuple(
                tuple(reversed(cell)) if len(cell) == 2 else cell
                for cell in frozen.CELL_MEMBER_SETS
            )
        episodes = execute_surface(
            frozen, episode_fn, members, embedder, tasks, cell_member_sets=cells
        )
        outcomes = probe.outcome_by_cell(episodes)
        table = probe.table_from_outcomes(outcomes, context_ids, frozen.MEMBER_IDS)
        trajectories = [
            precheck.classify_pair_trajectory(table, pair) for pair in table.pair_cells()
        ]
        oracle_gain = {
            "+".join(pair): dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair)
            for pair in table.pair_cells()
        }
        best_pair = max(oracle_gain.items(), key=lambda item: item[1]) if oracle_gain else ("", 0.0)
        by_order[order] = {
            "contexts": context_ids,
            "episodes": len(episodes),
            "singleton_success_rates": {
                member: outcomes[(member,)]["success_rate"] for member in frozen.MEMBER_IDS
            },
            "pair_success_rates": {
                "+".join(key): values["success_rate"]
                for key, values in outcomes.items()
                if len(key) == 2
            },
            "trajectories": trajectories,
            "interleaved_contexts": sum(item["contexts_interleaved"] for item in trajectories),
            "gain_vs_all_singleton_oracle": oracle_gain,
            "best_pair": best_pair[0],
            "best_pair_gain": best_pair[1],
            "positive_same_reference_gain": bool(best_pair[1] > 0.0),
            "reference_requirements": precheck.reference_requirements(
                table, margin=margin, candidates=candidates
            ),
            "stop_reasons": stop_reason_counts(episodes),
        }

    any_positive = any(item["positive_same_reference_gain"] for item in by_order.values())
    best = max(by_order.values(), key=lambda item: item["best_pair_gain"])
    return {
        "surface": "create_and_override",
        "by_order": by_order,
        "positive_same_reference_gain": any_positive,
        "best_order": next(name for name, item in by_order.items() if item is best),
        "best_pair": best["best_pair"],
        "best_pair_gain": best["best_pair_gain"],
        "reading": (
            "the proposed rule produced an interleaved trajectory and a positive "
            "same-reference gain, so a handoff is reachable under this rule"
            if any_positive
            else "the proposed rule produced no positive same-reference gain on this "
            "task shape under either priority order"
        ),
    }


def satisfiability_check(frozen: Any, tasks: Sequence[Any]) -> dict[str, Any]:
    """Scripted oracle: can the task's own reference steps reach the goal at all?

    Section 5.3 requires a candidate to be *satisfiable*.  A task whose scripted
    reference sequence cannot reach the goal is invalid regardless of what any
    composition does -- and it would make every negative result uninterpretable.
    This is the check that separates "the mechanism failed" from "the task is
    impossible".

    The reference order is tried both ways.  If only one order works, the task has a
    **mandatory order**, and a composition that fixes its priority the other way
    round cannot succeed no matter how it arbitrates.
    """

    rows: list[dict[str, Any]] = []
    for task in tasks:
        forward = _run_scripted(frozen, task, task.reference_steps)
        reversed_order = _run_scripted(frozen, task, tuple(reversed(task.reference_steps)))
        rows.append(
            {
                "task_id": task.task_id,
                "forward": forward,
                "reversed": reversed_order,
                "forward_reaches_goal": forward["goal_reached"],
                "reversed_reaches_goal": reversed_order["goal_reached"],
                "order_is_mandatory": bool(
                    forward["goal_reached"] != reversed_order["goal_reached"]
                ),
            }
        )

    unreachable = [row["task_id"] for row in rows if not row["forward_reaches_goal"]]
    mandatory = [row["task_id"] for row in rows if row["order_is_mandatory"]]
    return {
        "tasks": len(rows),
        "unreachable": unreachable,
        "satisfiable": not unreachable,
        "mandatory_order_tasks": mandatory,
        "order_is_mandatory": bool(mandatory),
        "rows": rows,
        "reading": (
            "the scripted reference steps reach the goal, so a negative composition "
            "result is interpretable"
            if not unreachable
            else "the scripted reference steps do NOT reach the goal: the task itself "
            "is unsatisfiable and every composition result on it is void"
        ),
    }


def _run_scripted(frozen: Any, task: Any, steps: Sequence[Any]) -> dict[str, Any]:
    """Execute one scripted step sequence against a clean world and report the goal."""

    root = Path(tempfile.mkdtemp(prefix="b0m1-sat-"))
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
        for step in steps:
            params, _provenance, failure = frozen.p52a._bind(step.kind, task, state)
            if failure is not None:
                executed.append({"kind": step.kind, "bound": False, "failure": failure})
                continue
            outcome = environment.execute_tool(step.kind, params)
            last = environment.last_result
            if "transaction" in last and last["transaction"].get("undo_token"):
                state["last_undo_token"] = str(last["transaction"]["undo_token"])
            executed.append({"kind": step.kind, "bound": True, "success": bool(outcome.success)})
        return {
            "steps": executed,
            "goal_reached": bool(frozen.p52a._goal_reached(environment, task)),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def evaluate() -> dict[str, Any]:
    frozen = load_frozen()
    embedder = frozen.DocumentEmbedder()
    members = frozen._train_members(embedder)
    frozen_episode = frozen._member_episode

    route_a = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
        ).read_text(encoding="utf-8")
    )
    margin = float(route_a["control_summary"]["margin"])
    candidates = {
        "all_singleton_oracle": 1.5,
        "best_fixed_singleton": 0.5,
        "best_observed_fixed_pair": 1.0,
    }

    repertoire = member_repertoire(frozen)
    candidate_tasks = load_handoff_probe().create_and_override_tasks(frozen)
    satisfiability = satisfiability_check(frozen, candidate_tasks)

    variants: list[dict[str, Any]] = []
    for name in VARIANTS:
        extra = {"REPERTOIRE": repertoire} if name in NEEDS_REPERTOIRE else None
        episode_fn, delta = build_counterfactual(frozen, name, extra_names=extra)
        variants.append(
            {
                "delta": delta,
                "regression": regression_check(frozen, episode_fn, members, embedder),
                "effect": effect_check(
                    frozen,
                    episode_fn,
                    members,
                    embedder,
                    margin=margin,
                    candidates=candidates,
                ),
            }
        )

    return {
        "format": COUNTERFACTUAL_FORMAT,
        "version": VERSION,
        "status": "draft_for_review",
        "does_not_change": [
            "M1 is NOT implemented: no gate, runner or rule is modified",
            "the counterfactual runs in a copy of the module namespace; "
            "frozen._member_episode is never rebound",
            "no result here is claimed as collaboration; it is a counterfactual measurement",
            "growth_admitted=false and can_promote=false remain in force",
        ],
        "frozen_attribute_intact": frozen._member_episode is frozen_episode,
        "member_repertoire": member_repertoire(frozen),
        "candidate_satisfiability": satisfiability,
        "margin": margin,
        "candidates_for_review": candidates,
        "variants": variants,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"frozen attribute intact: {payload['frozen_attribute_intact']}")
    print(f"member repertoire: {payload['member_repertoire']}")
    sat = payload["candidate_satisfiability"]
    print(
        f"candidate satisfiable: {sat['satisfiable']} unreachable={sat['unreachable']} "
        f"order_mandatory={sat['order_is_mandatory']}"
    )
    for row in sat["rows"]:
        print(
            f"  {row['task_id']}: forward={row['forward_reaches_goal']} "
            f"reversed={row['reversed_reaches_goal']}"
        )
        print(f"    forward steps: {[s['kind'] for s in row['forward']['steps']]}")
        print(f"    reversed steps: {[s['kind'] for s in row['reversed']['steps']]}")
    for variant in payload["variants"]:
        delta = variant["delta"]
        regression = variant["regression"]
        effect = variant["effect"]
        print(
            f"  variant={delta['variant']} added_lines={delta['added_lines']} "
            f"no_regression={regression['no_regression']} "
            f"regressions={len(regression['regressions'])} "
            f"improvements={len(regression['improvements'])}"
        )
        print(
            f"    effect: best_order={effect['best_order']} "
            f"best_pair={effect['best_pair']} gain={effect['best_pair_gain']:+.3f} "
            f"positive={effect['positive_same_reference_gain']}"
        )
        for order, item in effect["by_order"].items():
            print(
                f"      {order:<15} interleaved={item['interleaved_contexts']} "
                f"best={item['best_pair']} gain={item['best_pair_gain']:+.3f} "
                f"positive={item['positive_same_reference_gain']}"
            )
            print(f"        stop reasons: {item['stop_reasons']}")
        print(f"    regression stop reasons: {regression['stop_reasons']}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
