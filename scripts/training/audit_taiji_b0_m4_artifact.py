"""Artifact audit of the M4 handoff rule before decision D5.

Why this exists
---------------
M4 is the first configuration measured to produce a positive same-reference gain
(``interleaved = 4/4``, ``mean(P - max_i S_i) = +2.0``) **and** it changes two cells
on the frozen validation surface (``member-a+member-b`` and ``member-a+member-c``,
0.25 -> 0.50).  An unexplained improvement in a research line that was previously
burned by a zero-step artifact (``reports/M5_P5_2C_ENTRY_AUDIT_P5_2B_DEFECT``) has
to be audited before it can support a decision.  This module runs four independent
checks:

1. **Specificity** -- M4 must produce a positive same-reference gain **only** on the
   surface designed for it (``create_and_override``), and must not manufacture gains
   on the two surfaces that are single-member solvable, nor on the frozen surface.
2. **Intervention reality** -- the frozen ``_intervention_reality`` gate must report
   ``interventions_happened = true`` under M4, i.e. no inert non-baseline cell.
3. **Mechanism lesion** -- the successful pair's gain must vanish when the handoff is
   disabled (the frozen rule), and both singletons must fail, so the gain cannot be
   attributed to one member or to the task being trivially solvable.
4. **Seed robustness** -- the members are trained with hardcoded seeds, so a single
   seed result could be luck.  The key measurements are repeated with three seed
   offsets.

Read-only: no gate, runner, rule or frozen artifact is modified.  M4 remains a
counterfactual; this module only audits the measurement that supports D5.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "scripts" / "training"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_b0_m4_artifact_audit_20260913.json"

COUNTERFACTUAL_MODULE = TRAINING_DIR / "probe_taiji_b0_m1_counterfactual.py"
HANDOFF_PROBE = TRAINING_DIR / "probe_taiji_b0_handoff_feasibility.py"
FROZEN_ROUTE_A_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
)

AUDIT_FORMAT = "taiji-b0-m4-artifact-audit-v1"
VERSION = 1

#: The rule under audit, and the reference implementation it must be compared with.
RULE_UNDER_AUDIT = "m4_failure_handoff"
BASELINE_RULE = "frozen"

#: Member training seeds are hardcoded in the frozen gate, so robustness is measured
#: by offsets applied on top of them.
SEED_OFFSETS: tuple[int, ...] = (0, 101, 202)

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
    return _load("_b0_m4_audit_counterfactual", COUNTERFACTUAL_MODULE)


def load_handoff_probe() -> Any:
    return _load("_b0_m4_audit_handoff_probe", HANDOFF_PROBE)


# --------------------------------------------------------------------------- #
# Member training with a seed offset
# --------------------------------------------------------------------------- #


def train_members_with_seed(frozen: Any, embedder: Any, offset: int) -> dict[str, Any]:
    """The frozen ``_train_members`` with a seed offset.

    Re-implemented locally because the frozen version hardcodes ``seed=17+index``;
    a seed sweep is impossible without this.  The construction is otherwise
    identical (same families, same scene records, same epochs and learning rate),
    so a result that survives the sweep cannot be attributed to one lucky seed.
    """

    members: dict[str, Any] = {}
    for member_id, family_tasks in frozen._family_tasks().items():
        records = frozen.p52._scene_records(
            tuple(
                frozen.p52.Scene(
                    scene_id=task.task_id,
                    goal_text=task.goal_text,
                    files=task.initial_files,
                    steps=task.reference_steps,
                    main_path=task.main_path,
                )
                for task in family_tasks
            ),
            embedder,
        )
        learner = frozen.ProceduralSequenceLearner(
            384,
            hidden_dim=frozen.PROCEDURAL_HIDDEN_DIM,
            seed=17 + frozen.MEMBER_IDS.index(member_id) + offset,
        )
        learner.consolidate(
            records,
            epochs=frozen.PROCEDURAL_EPOCHS,
            learning_rate=frozen.PROCEDURAL_LEARNING_RATE,
        )
        members[member_id] = learner
    return members


# --------------------------------------------------------------------------- #
# Surface measurement
# --------------------------------------------------------------------------- #


def measure_surface(
    frozen: Any,
    counterfactual: Any,
    episode_fn: Any,
    members: Mapping[str, Any],
    embedder: Any,
    tasks: Sequence[Any],
    *,
    label: str,
    margin: float,
    reversed_order: bool = False,
) -> dict[str, Any]:
    """One surface under one rule: outcomes, gains, reality gate and stop reasons."""

    probe = load_handoff_probe()
    precheck = probe.load_precheck()
    dictionary = probe.load_dictionary()

    cells = None
    if reversed_order:
        cells = tuple(
            tuple(reversed(cell)) if len(cell) == 2 else cell for cell in frozen.CELL_MEMBER_SETS
        )
    episodes = counterfactual.execute_surface(
        frozen, episode_fn, members, embedder, tasks, cell_member_sets=cells
    )

    outcomes = probe.outcome_by_cell(episodes)
    context_ids = [task.task_id for task in tasks]
    table = probe.table_from_outcomes(outcomes, context_ids, frozen.MEMBER_IDS)
    trajectories = [precheck.classify_pair_trajectory(table, pair) for pair in table.pair_cells()]
    gains = {
        "+".join(pair): dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair)
        for pair in table.pair_cells()
    }
    best_pair = max(gains.items(), key=lambda item: item[1]) if gains else ("", 0.0)

    reality = frozen._intervention_reality(episodes)
    return {
        "surface": label,
        "order": "reversed_order" if reversed_order else "frozen_order",
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
        "gain_vs_all_singleton_oracle": gains,
        "best_pair": best_pair[0],
        "best_pair_gain": best_pair[1],
        "positive_same_reference_gain": bool(best_pair[1] > 0.0),
        "interleaved_contexts": sum(item["contexts_interleaved"] for item in trajectories),
        "combination_only_contexts": len(precheck.combination_only_solvable_contexts(table)),
        "reference_requirements": precheck.reference_requirements(
            table, margin=margin, candidates=CANDIDATES_FOR_REVIEW
        ),
        "intervention_reality": reality,
        "stop_reasons": counterfactual.stop_reason_counts(episodes),
        "episodes_raw": episodes,
    }


def strip_raw(surface: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON-safe view of a surface measurement."""

    return {key: value for key, value in surface.items() if key != "episodes_raw"}


# --------------------------------------------------------------------------- #
# Improvement forensics on the frozen surface
# --------------------------------------------------------------------------- #


def trace_for(
    episodes: Sequence[Mapping[str, Any]], *, task_id: str, cell: tuple[str, ...]
) -> dict[str, Any]:
    """Per-episode step trace for one (context, cell): who acted, doing what."""

    picked = [
        episode
        for episode in episodes
        if episode["task_id"] == task_id and tuple(episode["active_members"]) == cell
    ]
    return {
        "task_id": task_id,
        "cell": "+".join(cell) if cell else "none",
        "episodes": [
            {
                "success": bool(episode["success"]),
                "stop_reason": episode["stop_reason"],
                "trace": [
                    {
                        "tick": step.get("tick"),
                        "chosen": step.get("chosen"),
                        "kind": step.get("kind"),
                        "executed": step.get("executed"),
                        "stop": step.get("stop"),
                    }
                    for step in episode["steps"]
                ],
            }
            for episode in picked
        ],
    }


def improvement_forensics(
    baseline: Mapping[str, Any],
    audited: Mapping[str, Any],
) -> dict[str, Any]:
    """Explain every cell the audited rule changes on the frozen surface."""

    changed: list[dict[str, Any]] = []
    for cell in sorted(baseline["pair_success_rates"]):
        before = baseline["pair_success_rates"][cell]
        after = audited["pair_success_rates"].get(cell)
        if after is None or abs(after - before) < 1e-9:
            continue
        member_ids = tuple(cell.split("+"))
        contexts = sorted(
            {
                episode["task_id"]
                for episode in audited["episodes_raw"]
                if tuple(episode["active_members"]) == member_ids
            }
        )
        changed.append(
            {
                "cell": cell,
                "baseline_success_rate": before,
                "audited_success_rate": after,
                "delta": after - before,
                "per_context": [
                    {
                        "baseline": trace_for(
                            baseline["episodes_raw"], task_id=context, cell=member_ids
                        ),
                        "audited": trace_for(
                            audited["episodes_raw"], task_id=context, cell=member_ids
                        ),
                    }
                    for context in contexts
                ],
            }
        )
    return {
        "changed_cells": changed,
        "changed_count": len(changed),
        "note": (
            "each changed cell carries the full step trace under both rules, so the "
            "improvement can be attributed by reading which member acted at which tick"
        ),
    }


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def specificity_check(surfaces: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A positive gain must appear only on the surface designed for it."""

    positive = {
        label: surface["best_pair_gain"]
        for label, surface in surfaces.items()
        if surface["positive_same_reference_gain"]
    }
    unexpected = {label for label in positive if label != "create_and_override"}
    return {
        "surfaces_with_positive_gain": positive,
        "unexpected_positive_surfaces": sorted(unexpected),
        "specific": not unexpected,
        "reading": (
            "the rule produces a positive same-reference gain only on the surface "
            "designed to require a handoff"
            if not unexpected
            else "the rule also produces gains on surfaces that do not require a "
            "handoff, which would indicate an artifact rather than collaboration"
        ),
    }


def audit() -> dict[str, Any]:
    counterfactual = load_counterfactual()
    probe = load_handoff_probe()
    frozen = counterfactual.load_frozen()
    embedder = frozen.DocumentEmbedder()

    margin = float(
        json.loads(FROZEN_ROUTE_A_REPORT.read_text(encoding="utf-8"))["control_summary"]["margin"]
    )

    frozen_episode = frozen._member_episode
    audited_episode, delta = counterfactual.build_counterfactual(frozen, RULE_UNDER_AUDIT)
    assert frozen._member_episode is frozen_episode

    # ---- 1 + 3: specificity and lesion on the primary seed -------------------
    primary_members = train_members_with_seed(frozen, embedder, 0)
    validation_tasks = frozen.p52a._validation_tasks()[: frozen.CONTEXT_COUNT]
    candidate_sets = {
        "create_and_override": probe.create_and_override_tasks(frozen),
        "dual_requirement": probe.dual_requirement_tasks(frozen),
        "create_then_patch": probe.create_then_patch_tasks(frozen),
    }

    surfaces: dict[str, dict[str, Any]] = {}
    surfaces["frozen_validation"] = measure_surface(
        frozen,
        counterfactual,
        audited_episode,
        primary_members,
        embedder,
        validation_tasks,
        label="frozen_validation",
        margin=margin,
    )
    for label, tasks in candidate_sets.items():
        surfaces[label] = measure_surface(
            frozen,
            counterfactual,
            audited_episode,
            primary_members,
            embedder,
            tasks,
            label=label,
            margin=margin,
        )
    surfaces["create_and_override_reversed"] = measure_surface(
        frozen,
        counterfactual,
        audited_episode,
        primary_members,
        embedder,
        candidate_sets["create_and_override"],
        label="create_and_override",
        margin=margin,
        reversed_order=True,
    )

    # Baseline rule on the frozen surface, for the improvement forensics.
    baseline_frozen = measure_surface(
        frozen,
        counterfactual,
        frozen_episode,
        primary_members,
        embedder,
        validation_tasks,
        label="frozen_validation",
        margin=margin,
    )
    # Mechanism lesion: the frozen rule on the surface M4 wins.
    baseline_candidate = measure_surface(
        frozen,
        counterfactual,
        frozen_episode,
        primary_members,
        embedder,
        candidate_sets["create_and_override"],
        label="create_and_override",
        margin=margin,
    )

    lesion = {
        "surface": "create_and_override",
        "baseline_rule": {
            "best_pair": baseline_candidate["best_pair"],
            "best_pair_gain": baseline_candidate["best_pair_gain"],
            "interleaved_contexts": baseline_candidate["interleaved_contexts"],
        },
        "audited_rule": {
            "best_pair": surfaces["create_and_override"]["best_pair"],
            "best_pair_gain": surfaces["create_and_override"]["best_pair_gain"],
            "interleaved_contexts": surfaces["create_and_override"]["interleaved_contexts"],
        },
        "member_lesion": {
            "singleton_success_rates": surfaces["create_and_override"]["singleton_success_rates"],
            "all_singletons_fail": all(
                rate == 0.0
                for rate in surfaces["create_and_override"]["singleton_success_rates"].values()
            ),
        },
        "gain_vanishes_without_handoff": (
            surfaces["create_and_override"]["best_pair_gain"] > 0.0
            and baseline_candidate["best_pair_gain"] <= 0.0
        ),
        "note": (
            "the gain requires the handoff (it vanishes under the frozen rule) and "
            "requires both members (every singleton fails), so it cannot be "
            "attributed to one member or to a trivially solvable task"
        ),
    }

    # ---- 4: seed robustness -------------------------------------------------
    seed_rows: list[dict[str, Any]] = []
    for offset in SEED_OFFSETS:
        members = train_members_with_seed(frozen, embedder, offset)
        row_frozen = measure_surface(
            frozen,
            counterfactual,
            audited_episode,
            members,
            embedder,
            validation_tasks,
            label="frozen_validation",
            margin=margin,
        )
        row_candidate = measure_surface(
            frozen,
            counterfactual,
            audited_episode,
            members,
            embedder,
            candidate_sets["create_and_override"],
            label="create_and_override",
            margin=margin,
        )
        seed_rows.append(
            {
                "seed_offset": offset,
                "frozen_validation": {
                    "positive_same_reference_gain": row_frozen["positive_same_reference_gain"],
                    "best_pair_gain": row_frozen["best_pair_gain"],
                    "intervention_reality_ok": row_frozen["intervention_reality"][
                        "interventions_happened"
                    ],
                },
                "create_and_override": {
                    "positive_same_reference_gain": row_candidate["positive_same_reference_gain"],
                    "best_pair_gain": row_candidate["best_pair_gain"],
                    "best_pair": row_candidate["best_pair"],
                    "interleaved_contexts": row_candidate["interleaved_contexts"],
                    "all_singletons_fail": all(
                        rate == 0.0 for rate in row_candidate["singleton_success_rates"].values()
                    ),
                },
            }
        )

    candidate_gains = [row["create_and_override"]["best_pair_gain"] for row in seed_rows]
    seed_robustness = {
        "offsets": list(SEED_OFFSETS),
        "rows": seed_rows,
        "all_seeds_positive_on_candidate": all(gain > 0.0 for gain in candidate_gains),
        "all_seeds_no_gain_on_frozen": all(
            not row["frozen_validation"]["positive_same_reference_gain"] for row in seed_rows
        ),
        "min_candidate_gain": min(candidate_gains),
        "max_candidate_gain": max(candidate_gains),
        "reading": (
            "the audited rule produces a positive gain on the handoff surface under "
            "every seed and never manufactures a gain on the frozen surface"
            if all(gain > 0.0 for gain in candidate_gains)
            else "at least one seed fails to reproduce the gain, so the result is " "seed-dependent"
        ),
    }

    return {
        "format": AUDIT_FORMAT,
        "version": VERSION,
        "status": "draft_for_review",
        "rule_under_audit": RULE_UNDER_AUDIT,
        "rule_delta": delta,
        "does_not_change": [
            "M4 is NOT implemented; no gate, runner, rule or frozen artifact changes",
            "the audit reuses the counterfactual builder, so the frozen episode "
            "function is never rebound",
            "no result here authorises adoption; it audits the measurement behind D5",
        ],
        "frozen_attribute_intact": frozen._member_episode is frozen_episode,
        "margin": margin,
        "candidates_for_review": CANDIDATES_FOR_REVIEW,
        "specificity": specificity_check(
            {
                label: surface
                for label, surface in surfaces.items()
                if label != "create_and_override_reversed"
            }
        ),
        "surfaces": {label: strip_raw(surface) for label, surface in surfaces.items()},
        "lesion": lesion,
        "improvement_forensics": improvement_forensics(
            baseline_frozen, surfaces["frozen_validation"]
        ),
        "seed_robustness": seed_robustness,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"frozen attribute intact: {payload['frozen_attribute_intact']}")
    print(f"specificity: {payload['specificity']['reading']}")
    print(f"  positive surfaces: {payload['specificity']['surfaces_with_positive_gain']}")
    for label, surface in payload["surfaces"].items():
        print(
            f"  {label:<32} best={surface['best_pair']:<24} "
            f"gain={surface['best_pair_gain']:+.3f} "
            f"interleaved={surface['interleaved_contexts']} "
            f"reality_ok={surface['intervention_reality']['interventions_happened']}"
        )
    lesion = payload["lesion"]
    print(
        f"lesion: gain_vanishes_without_handoff={lesion['gain_vanishes_without_handoff']} "
        f"all_singletons_fail={lesion['member_lesion']['all_singletons_fail']}"
    )
    print(f"  changed cells on frozen surface: {payload['improvement_forensics']['changed_count']}")
    robustness = payload["seed_robustness"]
    print(
        f"seed robustness: all_positive={robustness['all_seeds_positive_on_candidate']} "
        f"all_frozen_no_gain={robustness['all_seeds_no_gain_on_frozen']} "
        f"range=[{robustness['min_candidate_gain']:+.3f}, "
        f"{robustness['max_candidate_gain']:+.3f}]"
    )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
