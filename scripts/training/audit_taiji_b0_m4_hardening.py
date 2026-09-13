"""Hardening pass for the M4 evidence, closing three of the audit's residual risks.

The M4 artifact audit passed, but it left five residual risks open.  Three of them
are addressable without any decision:

**Risk 3 -- scale, and a self-critique.**  The audited surface used four contexts
(``dual_200``..``dual_203``) that differ only by an index suffix.  Under the
project's own rule ("multiple instances of one structural fact are not independent
samples") that is **one** structural instance measured four times, not four.  This
module replaces it with **six structurally distinct variants** -- the extension, the
goal language, and the content shape all vary -- two contexts each, i.e. twelve
contexts, which is the count the audit asked for.  It reports the effective sample
size as the number of distinct variants as well as the number of contexts, because
only the variant count is a count of independent structural facts.

**Risk 4 -- seeds.**  The audit swept three offsets; this module sweeps five.

**Risk 2 -- new stop-reason semantics.**  M4 introduces ``all_members_blocked``,
which is distinct from the frozen ``all_members_exhausted``.  This module scans the
repository for every consumer of ``stop_reason`` and reports which reasons each one
compares against, so the gate-semantics review has a concrete list instead of a
guess.  The scan is read-only.  It also attributes stop reasons to each variant, so
"where did the gain come from" is answerable per variant rather than only pooled.

Each variant carries a ``expect_contract_ok`` prediction made from the language
registry, and the module checks it against a **rule-independent** scripted
execution of that variant's reference steps.  A prediction that the evidence
contradicts stops the run rather than being quietly kept: a sweep whose
expectations do not match observation cannot support a robustness claim.

Risks 1 (freeze the priority definition) and 5 (re-run if the member families
change) are preregistration matters and are deliberately left to D5.

Read-only: M4 is still a counterfactual; no gate, runner, rule or frozen artifact
is modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "scripts" / "training"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_b0_m4_hardening_20260913.json"

COUNTERFACTUAL_MODULE = TRAINING_DIR / "probe_taiji_b0_m1_counterfactual.py"
HANDOFF_PROBE = TRAINING_DIR / "probe_taiji_b0_handoff_feasibility.py"
FROZEN_ROUTE_A_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
)

HARDENING_FORMAT = "taiji-b0-m4-hardening-v1"
VERSION = 1

RULE_UNDER_AUDIT = "m4_failure_handoff"

#: Five seed offsets (risk 4: the audit used three).
SEED_OFFSETS: tuple[int, ...] = (0, 101, 202, 303, 404)

#: Contexts per structural variant.  Two keeps a repeat without inflating the
#: apparent sample size -- the variant, not the context, is the unit of variation.
CONTEXTS_PER_VARIANT = 2

#: Files scanned for ``stop_reason`` consumers (risk 2).
SCAN_ROOTS: tuple[str, ...] = ("scripts", "seed_platform", "taiji", "tests")

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
    return _load("_b0_harden_counterfactual", COUNTERFACTUAL_MODULE)


def load_handoff_probe() -> Any:
    return _load("_b0_harden_handoff_probe", HANDOFF_PROBE)


# --------------------------------------------------------------------------- #
# Structurally distinct variants (risk 3)
# --------------------------------------------------------------------------- #

#: Each entry varies the structural knobs that matter: the file extension, the
#: override language, and the content shape.  ``expect_contract_ok`` records whether
#: the variant is predicted to survive the contract layer, judged from the language
#: registry (an unambiguous extension plus an explicit ``user_override``).
#:
#: This field was originally written with ``.h``/``cpp`` predicted to be intercepted,
#: by analogy with the frozen block-3 family's ``language_evidence_ambiguous``.  The
#: scripted check below refutes that: block-3's ambiguity comes from *absent* language
#: evidence, whereas every variant here supplies an explicit override, so all six are
#: admissible.  ``.h`` stays in the sweep because that corrected prediction -- made,
#: checked, and overturned -- is more informative than one that was never tested.
VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "label": "py_python_fn",
        "extension": ".py",
        "language": "python",
        "base": "def run_{i}():\n    return {i}\n",
        "goal": "def run_{i}():\n    return {i} + 1\n",
        "expect_contract_ok": True,
    },
    {
        "label": "txt_python_plain",
        "extension": ".txt",
        "language": "python",
        "base": "value {i}\n",
        "goal": "value {i} + 1\n",
        "expect_contract_ok": True,
    },
    {
        "label": "py_python_single_line",
        "extension": ".py",
        "language": "python",
        "base": "x = {i}\n",
        "goal": "x = {i} + 1\n",
        "expect_contract_ok": True,
    },
    {
        "label": "h_cpp_header",
        "extension": ".h",
        "language": "cpp",
        "base": "int value_{i} = {i};\n",
        "goal": "int value_{i} = {i} + 1;\n",
        "expect_contract_ok": True,
    },
    {
        "label": "py_python_two_lines",
        "extension": ".py",
        "language": "python",
        "base": "def run_{i}():\n    return {i}\n\n# note {i}\n",
        "goal": "def run_{i}():\n    return {i} + 1\n\n# note {i}\n",
        "expect_contract_ok": True,
    },
    {
        "label": "js_javascript_const",
        "extension": ".js",
        "language": "javascript",
        "base": "const value_{i} = {i};\n",
        "goal": "const value_{i} = {i} + 1;\n",
        "expect_contract_ok": True,
    },
)


def variant_tasks(frozen: Any, spec: Mapping[str, Any], index: int) -> Any:
    """One context of one structural variant: create the file *and* override language."""

    p52a, p52 = frozen.p52a, frozen.p52
    name = f"hard_{spec['label']}_{index}{spec['extension']}"
    content = spec["base"].format(i=index)
    goal = spec["goal"].format(i=index)
    steps = (
        p52.ScriptedStep("workspace.list", {"path": "."}),
        p52.ScriptedStep("workspace.create", {"path": name, "content": content}),
        p52.ScriptedStep("workspace.read", {"path": name}),
        p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}),
        p52.ScriptedStep(
            "editor.set_language",
            {
                "path": name,
                "programming_language_id": spec["language"],
                "user_override": True,
            },
        ),
    )
    return p52a.Task(
        task_id=f"b0harden-{spec['label']}-{index}",
        goal_text=(
            f"Work order b0harden-{spec['label']}-{index}: reach the requested end state on "
            f"{name} using the workspace contract, and leave every temporary change reverted."
        ),
        initial_files={},
        goal_files={name: goal},
        goal_language={name: spec["language"]},
        main_path=name,
        reference_steps=steps,
        partition="probe",
        template=f"harden_{spec['label']}",
        requires_explicit_language_override=True,
    )


def build_variant_surface(frozen: Any, index_base: int = 400) -> dict[str, Any]:
    """All variants, each with ``CONTEXTS_PER_VARIANT`` contexts."""

    tasks: list[Any] = []
    owner: dict[str, str] = {}
    for offset, spec in enumerate(VARIANT_SPECS):
        for step in range(CONTEXTS_PER_VARIANT):
            index = index_base + offset * 10 + step
            task = variant_tasks(frozen, spec, index)
            tasks.append(task)
            owner[task.task_id] = spec["label"]
    return {"tasks": tuple(tasks), "owner": owner}


# --------------------------------------------------------------------------- #
# stop_reason consumers (risk 2)
# --------------------------------------------------------------------------- #

_REASON_TOKEN = re.compile(r"[\"']([a-z_]+(?::[a-zA-Z_]+)?)[\"']")


def stop_reason_consumers() -> dict[str, Any]:
    """Every file that reads ``stop_reason`` and the reasons it compares against."""

    rows: list[dict[str, Any]] = []
    for root in SCAN_ROOTS:
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in {"__pycache__", "archive"} for part in path.parts):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if "stop_reason" not in source:
                continue
            reasons: set[str] = set()
            for line in source.splitlines():
                if "stop_reason" not in line and "stop" not in line:
                    continue
                reasons.update(_REASON_TOKEN.findall(line))
            rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "literals_on_stop_lines": sorted(reasons),
                }
            )

    all_reasons = sorted({reason for row in rows for reason in row["literals_on_stop_lines"]})
    return {
        "files": rows,
        "file_count": len(rows),
        "reasons_seen": all_reasons,
        "new_reason_under_review": "all_members_blocked",
        "new_reason_already_referenced": any(
            "all_members_blocked" in row["literals_on_stop_lines"] for row in rows
        ),
        "note": (
            "the scan lists literals appearing on lines that mention stop reasons; it "
            "bounds the review surface, it does not prove a gate ignores the new reason"
        ),
    }


CONTRACT_INTERCEPT_TOKEN = "contract_intercepted"


def stop_reasons_by_variant(
    episodes: Sequence[Mapping[str, Any]], owner: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    """Attribute each episode's stop reason to the structural variant that owns it."""

    rows: dict[str, dict[str, int]] = {}
    for episode in episodes:
        label = owner.get(str(episode.get("task_id", "")))
        if label is None:
            continue
        counts = rows.setdefault(label, {})
        reason = str(episode.get("stop_reason", "?"))
        counts[reason] = counts.get(reason, 0) + 1
    return {label: dict(sorted(counts.items())) for label, counts in sorted(rows.items())}


def was_contract_intercepted(stop_reasons: Mapping[str, int]) -> bool:
    return any(CONTRACT_INTERCEPT_TOKEN in reason for reason in stop_reasons)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def measure_variants(
    frozen: Any,
    counterfactual: Any,
    episode_fn: Any,
    members: Mapping[str, Any],
    embedder: Any,
    surface: Mapping[str, Any],
    *,
    margin: float,
) -> dict[str, Any]:
    """Per-variant and pooled measurement of one rule on the varied surface."""

    probe = load_handoff_probe()
    precheck = probe.load_precheck()
    dictionary = probe.load_dictionary()
    tasks = surface["tasks"]
    owner = surface["owner"]

    episodes = counterfactual.execute_surface(frozen, episode_fn, members, embedder, tasks)
    outcomes = probe.outcome_by_cell(episodes)
    context_ids = [task.task_id for task in tasks]
    table = probe.table_from_outcomes(outcomes, context_ids, frozen.MEMBER_IDS)

    by_variant_reasons = stop_reasons_by_variant(episodes, owner)

    per_variant: list[dict[str, Any]] = []
    for spec in VARIANT_SPECS:
        label = spec["label"]
        owned = [task_id for task_id in context_ids if owner[task_id] == label]
        variant_table = probe.table_from_outcomes(
            {
                key: {
                    k: v
                    for k, v in values.items()
                    if k == "mean_outcome" or k.removeprefix("ctx:") in owned
                }
                for key, values in outcomes.items()
            },
            owned,
            frozen.MEMBER_IDS,
        )
        gains = {
            "+".join(pair): dictionary.policy_mean_gain_vs_all_singleton_oracle(variant_table, pair)
            for pair in variant_table.pair_cells()
        }
        best_pair = max(gains.items(), key=lambda item: item[1]) if gains else ("", 0.0)
        singleton_rates = {
            member: outcomes[(member,)].get(f"ctx:{owned[0]}", outcomes[(member,)]["mean_outcome"])
            for member in frozen.MEMBER_IDS
        }
        per_variant.append(
            {
                "variant": label,
                "extension": spec["extension"],
                "language": spec["language"],
                "expect_contract_ok": spec["expect_contract_ok"],
                "stop_reasons": by_variant_reasons.get(label, {}),
                "observed_contract_interception": was_contract_intercepted(
                    by_variant_reasons.get(label, {})
                ),
                "contexts": owned,
                "singleton_outcomes": {
                    member: outcomes[(member,)]["mean_outcome"] for member in frozen.MEMBER_IDS
                },
                "singleton_success_rates_ctx0": singleton_rates,
                "best_pair": best_pair[0],
                "best_pair_gain": best_pair[1],
                "positive_same_reference_gain": bool(best_pair[1] > 0.0),
                "combination_only_contexts": len(
                    precheck.combination_only_solvable_contexts(variant_table)
                ),
                "interleaved_contexts": sum(
                    precheck.classify_pair_trajectory(variant_table, pair)["contexts_interleaved"]
                    for pair in variant_table.pair_cells()
                ),
            }
        )

    gains = {
        "+".join(pair): dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair)
        for pair in table.pair_cells()
    }
    best_pair = max(gains.items(), key=lambda item: item[1]) if gains else ("", 0.0)
    reality = frozen._intervention_reality(episodes)

    positive_variants = [
        row["variant"] for row in per_variant if row["positive_same_reference_gain"]
    ]
    return {
        "contexts": context_ids,
        "context_count": len(context_ids),
        "distinct_variants": len(VARIANT_SPECS),
        "episodes": len(episodes),
        "singleton_outcomes": {
            member: outcomes[(member,)]["mean_outcome"] for member in frozen.MEMBER_IDS
        },
        "pooled_best_pair": best_pair[0],
        "pooled_best_pair_gain": best_pair[1],
        "pooled_positive": bool(best_pair[1] > 0.0),
        "interleaved_contexts": sum(
            precheck.classify_pair_trajectory(table, pair)["contexts_interleaved"]
            for pair in table.pair_cells()
        ),
        "reference_requirements": precheck.reference_requirements(
            table, margin=margin, candidates=CANDIDATES_FOR_REVIEW
        ),
        "intervention_reality": reality,
        "stop_reasons": counterfactual.stop_reason_counts(episodes),
        "per_variant": per_variant,
        "positive_variants": positive_variants,
        "positive_variant_count": len(positive_variants),
    }


def variant_outcome_distinctness(audited: Mapping[str, Any]) -> dict[str, Any]:
    """Are the variants distinguishable by outcome, or only by surface?

    Each variant changes the extension, goal language and content shape, but every
    one keeps the *same* composition structure (create a file, then override its
    language).  Identical outcome vectors therefore mean the sweep bounds surface
    sensitivity, not structural sensitivity -- six copies of one structural fact are
    still one structural fact, which is the very error the risk-3 self-critique made
    about the four index-suffixed contexts.
    """

    fingerprints = {
        row["variant"]: (
            row["best_pair"],
            round(row["best_pair_gain"], 12),
            row["combination_only_contexts"],
            row["interleaved_contexts"],
            tuple(sorted(row["stop_reasons"].items())),
        )
        for row in audited["per_variant"]
    }
    distinct = {str(item) for item in fingerprints.values()}
    return {
        "variants_compared": len(fingerprints),
        "distinct_outcome_fingerprints": len(distinct),
        "variants_indistinguishable_in_outcome": len(distinct) <= 1,
        "bounds": (
            "surface sensitivity (extension, goal language, content shape)"
            if len(distinct) <= 1
            else "surface and outcome sensitivity"
        ),
        "does_not_bound": (
            "all six variants share one composition structure (create + override), so "
            "this sweep cannot rule out that the gain is specific to that structure; "
            "structurally distinct demands (B0 continuation T1/T2/T3) remain open"
        ),
    }


def hardening() -> dict[str, Any]:
    counterfactual = load_counterfactual()
    frozen = counterfactual.load_frozen()
    embedder = frozen.DocumentEmbedder()
    margin = float(
        json.loads(FROZEN_ROUTE_A_REPORT.read_text(encoding="utf-8"))["control_summary"]["margin"]
    )

    frozen_episode = frozen._member_episode
    audited_episode, delta = counterfactual.build_counterfactual(frozen, RULE_UNDER_AUDIT)
    assert frozen._member_episode is frozen_episode

    surface = build_variant_surface(frozen)
    for task in surface["tasks"]:
        frozen.p52a._assert_nontrivial_goals((task,), partition="probe")

    members = _train_members(frozen, embedder, 0)

    baseline = measure_variants(
        frozen, counterfactual, frozen_episode, members, embedder, surface, margin=margin
    )
    audited = measure_variants(
        frozen, counterfactual, audited_episode, members, embedder, surface, margin=margin
    )

    # Satisfiability per variant: without it a negative result is uninterpretable.
    # The scripted run is rule-independent, so it is what ``expect_contract_ok``
    # is actually a prediction about (would the contract layer admit these steps?).
    satisfiability: list[dict[str, Any]] = []
    for task in surface["tasks"]:
        forward = counterfactual._run_scripted(frozen, task, task.reference_steps)
        satisfiability.append(
            {
                "task_id": task.task_id,
                "variant": surface["owner"][task.task_id],
                "forward_reaches_goal": forward["goal_reached"],
                "scripted_blocked_steps": sorted(
                    {
                        str(step.get("kind", "?"))
                        for step in forward["steps"]
                        if not (step.get("bound") and step.get("success", True))
                    }
                ),
            }
        )

    expectation_rows: list[dict[str, Any]] = []
    for spec in VARIANT_SPECS:
        label = spec["label"]
        owned_rows = [row for row in satisfiability if row["variant"] == label]
        admissible = bool(owned_rows) and all(
            row["forward_reaches_goal"] and not row["scripted_blocked_steps"] for row in owned_rows
        )
        expectation_rows.append(
            {
                "variant": label,
                "expect_contract_ok": bool(spec["expect_contract_ok"]),
                "observed_scripted_admissible": admissible,
                "matches": admissible == bool(spec["expect_contract_ok"]),
            }
        )
    contract_expectation_audit = {
        "rows": expectation_rows,
        "all_match": all(row["matches"] for row in expectation_rows),
        "contradicted": [row["variant"] for row in expectation_rows if not row["matches"]],
    }
    if not contract_expectation_audit["all_match"]:
        print(
            f"contract expectations contradicted by scripted evidence: "
            f"{contract_expectation_audit['contradicted']}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # Seed sweep (risk 4: five offsets).
    seed_rows: list[dict[str, Any]] = []
    for offset in SEED_OFFSETS:
        seed_members = _train_members(frozen, embedder, offset)
        row = measure_variants(
            frozen,
            counterfactual,
            audited_episode,
            seed_members,
            embedder,
            surface,
            margin=margin,
        )
        seed_rows.append(
            {
                "seed_offset": offset,
                "pooled_best_pair_gain": row["pooled_best_pair_gain"],
                "pooled_positive": row["pooled_positive"],
                "positive_variant_count": row["positive_variant_count"],
                "distinct_variants": row["distinct_variants"],
                "intervention_reality_ok": row["intervention_reality"]["interventions_happened"],
            }
        )

    return {
        "format": HARDENING_FORMAT,
        "version": VERSION,
        "status": "draft_for_review",
        "rule_under_audit": RULE_UNDER_AUDIT,
        "rule_delta": delta,
        "does_not_change": [
            "M4 is NOT implemented; no gate, runner, rule or frozen artifact changes",
            "the sweep is read-only and reuses the counterfactual builder",
            "no result here authorises adoption",
        ],
        "frozen_attribute_intact": frozen._member_episode is frozen_episode,
        "margin": margin,
        "effective_sample_size": {
            "contexts": audited["context_count"],
            "distinct_variants": audited["distinct_variants"],
            "self_critique": (
                "the earlier audited surface used four contexts differing only by an "
                "index suffix, i.e. one structural instance measured four times; this "
                "sweep replaces it with structurally distinct variants so the unit of "
                "variation is the variant, not the context"
            ),
            "outcome_distinctness": variant_outcome_distinctness(audited),
        },
        "satisfiability": satisfiability,
        "contract_expectation_audit": contract_expectation_audit,
        "variants": {
            spec["label"]: {
                "extension": spec["extension"],
                "language": spec["language"],
                "expect_contract_ok": spec["expect_contract_ok"],
            }
            for spec in VARIANT_SPECS
        },
        "baseline_rule": baseline,
        "audited_rule": audited,
        "seed_sweep": {
            "offsets": list(SEED_OFFSETS),
            "rows": seed_rows,
            "all_seeds_pooled_positive": all(row["pooled_positive"] for row in seed_rows),
            "min_pooled_gain": min(row["pooled_best_pair_gain"] for row in seed_rows),
            "max_pooled_gain": max(row["pooled_best_pair_gain"] for row in seed_rows),
            "min_positive_variant_count": min(row["positive_variant_count"] for row in seed_rows),
        },
        "stop_reason_consumers": stop_reason_consumers(),
    }


def _train_members(frozen: Any, embedder: Any, offset: int) -> dict[str, Any]:
    """Delegates to the audit module's seed-offset trainer (single implementation)."""

    audit = _load(
        "_b0_harden_m4_audit",
        TRAINING_DIR / "audit_taiji_b0_m4_artifact.py",
    )
    return audit.train_members_with_seed(frozen, embedder, offset)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = hardening()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"frozen attribute intact: {payload['frozen_attribute_intact']}")
    size = payload["effective_sample_size"]
    print(
        f"effective sample: {size['contexts']} contexts / "
        f"{size['distinct_variants']} distinct variants"
    )
    distinct = size["outcome_distinctness"]
    print(
        f"variant outcome distinctness: {distinct['distinct_outcome_fingerprints']} distinct "
        f"of {distinct['variants_compared']} variants -> bounds {distinct['bounds']}"
    )
    sat = payload["satisfiability"]
    ok = sum(1 for row in sat if row["forward_reaches_goal"])
    print(f"satisfiable variants: {ok} / {len(sat)}")
    expect = payload["contract_expectation_audit"]
    print(
        f"contract expectations vs scripted evidence: all_match={expect['all_match']} "
        f"contradicted={expect['contradicted'] or 'none'}"
    )
    for rule in ("baseline_rule", "audited_rule"):
        surface = payload[rule]
        print(
            f"  {rule:<14} pooled_best={surface['pooled_best_pair']:<24} "
            f"gain={surface['pooled_best_pair_gain']:+.3f} "
            f"positive_variants={surface['positive_variant_count']}/{surface['distinct_variants']} "
            f"interleaved={surface['interleaved_contexts']} "
            f"reality_ok={surface['intervention_reality']['interventions_happened']}"
        )
    for row in payload["audited_rule"]["per_variant"]:
        print(
            f"    {row['variant']:<24} gain={row['best_pair_gain']:+.3f} "
            f"positive={row['positive_same_reference_gain']} "
            f"combo_only={row['combination_only_contexts']} "
            f"interleaved={row['interleaved_contexts']}"
        )
    sweep = payload["seed_sweep"]
    print(
        f"seed sweep: all_positive={sweep['all_seeds_pooled_positive']} "
        f"range=[{sweep['min_pooled_gain']:+.3f}, {sweep['max_pooled_gain']:+.3f}] "
        f"min_positive_variants={sweep['min_positive_variant_count']}"
    )
    consumers = payload["stop_reason_consumers"]
    print(
        f"stop_reason consumers: {consumers['file_count']} files; "
        f"new reason referenced={consumers['new_reason_already_referenced']}"
    )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
