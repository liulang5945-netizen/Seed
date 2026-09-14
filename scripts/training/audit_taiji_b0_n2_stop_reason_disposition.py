"""N2 stop-reason disposition: the review surface for ``all_members_blocked``.

Why this module exists
----------------------
M4 introduces a stop reason the frozen rule never produced::

    frozen:  all_members_exhausted   -- no member can bind
    M4:      all_members_blocked     -- every bindable member just failed

Before M4 can be implemented, every consumer of ``stop_reason`` must be shown to
treat the new value safely.  The hardening round produced a **file list** and said
plainly that the list "only bounds the review surface -- it does not prove a gate
ignores the new reason".

Two problems with leaving it there:

1. **The list was already stale.**  The hardening report lists 11 files.  A scan of
   the current tree finds **14**: the N1 probe and two contract-test files were
   written after that report was generated.  A live scan committed as evidence
   silently rots as soon as anything new references ``stop_reason``.
2. **A list is not a disposition.**  "File X reads stop_reason" does not say whether
   X *judges* on it.  The distinction is the whole question: recording is harmless,
   judging may not be.

So this module does three things:

* re-derives the file list from the live tree and **fails loudly when it grows**
  (``expected_consumers`` is frozen; a new consumer must be dispositioned);
* records the reviewed **judgement sites** and checks their source markers;
  marker presence is only a drift hint, not proof that a predicate is unchanged;
* classifies every remaining file as record-only, with the reason.

It does **not** decide N2.  The preregistration still requires the user's D5=落地;
this is the review surface that decision needs, produced read-only.

Read-only: no gate, runner, rule or frozen artifact is modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "scripts" / "training"
#: The 2026-09-14 review snapshot, sealed by sha256 in test_b0_rule_revision_seal_contract.py.
REVISION_0_OUTPUT = PROJECT_ROOT / "reports" / "taiji_b0_n2_stop_reason_disposition_20260914.json"
#: The living inventory guard is re-scanned every round; it must never default onto the
#: sealed snapshot above.
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_b0_n2_stop_reason_disposition_20260915.json"

HARDENING_MODULE = TRAINING_DIR / "audit_taiji_b0_m4_hardening.py"
DISPOSITION_FORMAT = "taiji-b0-n2-stop-reason-disposition-v1"
VERSION = 2

#: The reason M4 introduces, and the frozen reason it must not be confused with.
NEW_REASON = "all_members_blocked"
FROZEN_COUNTERPART = "all_members_exhausted"

#: Token used by the reviewed interception checks.  P5.2a checks a prefix;
#: the hardening helper checks substring membership.  These are not equivalent.
INTERCEPTION_PREFIX = "contract_intercepted"

N2_SCRIPT_PATH = "scripts/training/audit_taiji_b0_n2_stop_reason_disposition.py"
N2_TEST_PATH = "tests/taiji_native/test_b0_n2_stop_reason_disposition_contract.py"
SELF_AUDIT_PATHS: frozenset[str] = frozenset({N2_SCRIPT_PATH, N2_TEST_PATH})

#: Frozen consumer list.  A live scan that returns anything else is a drift and
#: must be reviewed rather than silently accepted.  11 entries came from the
#: hardening report; the three marked ``added_after_hardening_report`` were written
#: afterwards and are the reason that report's count is stale.
EXPECTED_CONSUMERS: tuple[dict[str, str], ...] = (
    {
        "path": "scripts/training/audit_taiji_b0_m4_artifact.py",
        "class": "record_only",
        "why": "writes stop_reason into the audit payload and aggregates it; no comparison",
    },
    {
        "path": "scripts/training/audit_taiji_b0_m4_hardening.py",
        "class": "record_only + prefix_judgement",
        "why": "aggregates per variant; its was_contract_intercepted() keys on the "
        "contract_intercepted prefix, which the new reason does not carry",
    },
    {
        "path": "scripts/training/audit_taiji_b0_measurement_reachability.py",
        "class": "record_only",
        "why": "reads the frozen route-A report field block3_audit.stop_reason_classes; "
        "it does not scan a live tree",
    },
    {
        "path": "scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py",
        "class": "judgement (replica equality + interception prefix)",
        "why": "see J1 and J2 below",
    },
    {
        "path": "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
        "class": "judgement (replica equality)",
        "why": "see J3 below; it is also the only file M4 actually edits",
    },
    {
        "path": "scripts/training/eval_taiji_p5_2c_double_prime_unseen_combination_transfer_gate.py",
        "class": "record_only",
        "why": "aggregates by reason.split(':')[0] into the report; no comparison",
    },
    {
        "path": "scripts/training/eval_taiji_p5_2c_prime_unseen_combination_transfer_gate.py",
        "class": "record_only",
        "why": "same class aggregation as the other P5.2c gates",
    },
    {
        "path": "scripts/training/eval_taiji_p5_2c_triple_prime_representation_repair_gate.py",
        "class": "record_only",
        "why": "same class aggregation as the other P5.2c gates",
    },
    {
        "path": "scripts/training/eval_taiji_p5_2c_unseen_combination_transfer_gate.py",
        "class": "record_only",
        "why": "same class aggregation as the other P5.2c gates",
    },
    {
        "path": "scripts/training/probe_taiji_b0_m1_counterfactual.py",
        "class": "record_only",
        "why": "the counterfactual itself: produces and aggregates stop reasons",
    },
    {
        "path": "scripts/training/probe_taiji_b0_structure_space.py",
        "class": "record_only",
        "why": "N1 probe: aggregates frozen/audited reason counters per cell",
        "added_after_hardening_report": "yes",
    },
    {
        "path": "tests/taiji_native/test_b0_m4_artifact_audit_contract.py",
        "class": "judgement (test assertion)",
        "why": "see J4 below",
    },
    {
        "path": "tests/taiji_native/test_b0_m4_hardening_contract.py",
        "class": "judgement (test assertion)",
        "why": "see J4 below",
        "added_after_hardening_report": "yes",
    },
    {
        "path": "tests/taiji_native/test_b0_structure_space_contract.py",
        "class": "judgement (test assertion)",
        "why": "see J4 below",
        "added_after_hardening_report": "yes",
    },
    {
        "path": "tests/taiji_native/test_b0_n2_stop_reason_semantics_contract.py",
        "class": "judgement (test assertion)",
        "why": "see J8 below: asserts the frozen N2 semantics against archived reports",
        "added_after_hardening_report": "yes",
    },
    {
        "path": "tests/taiji_native/test_b0_rule_revision_seal_contract.py",
        "class": "judgement (test assertion)",
        "why": "see J9 below: pins the rule_revision=0 evidence seal and the shipped rule",
        "added_after_hardening_report": "yes",
    },
    {
        "path": "tests/taiji_native/test_b0_m1_counterfactual_contract.py",
        "class": "judgement (test assertion)",
        "why": "see J10 below: compares stop_reasons between the sealed and the landed reports",
        "added_after_hardening_report": "yes",
    },
)

#: Every place a stop reason participates in a *decision*.  Each marker must still
#: appear in its file; the ``safe_because`` field is the review conclusion.
JUDGEMENT_SITES: tuple[dict[str, str], ...] = (
    {
        "id": "J1",
        "path": "scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py",
        "marker": 'item["stop_reason"],',
        "kind": "replica_equality",
        "safe_because": (
            "the marker sits in deterministic_surface(), which compares the matrix run "
            "with a replica run under the same rule. A new reason must remain in the "
            "comparison; equality still requires an actual replica test"
        ),
    },
    {
        "id": "J2",
        "path": "scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py",
        "marker": 'startswith("contract_intercepted")',
        "kind": "prefix_predicate",
        "safe_because": (
            "it counts safety stops from the step-level 'stop' field by the "
            "contract_intercepted prefix; all_members_blocked does not carry that "
            "prefix, so it is never miscounted as a safety stop"
        ),
    },
    {
        "id": "J3",
        "path": "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
        "marker": '(item["episode_id"], item["success"], item["stop_reason"], item["resource_cost"])',
        "kind": "replica_equality",
        "safe_because": (
            "a matrix-vs-replica comparison under the same rule. The reason remains "
            "part of the equality; changing the rule does not prove determinism"
        ),
    },
    {
        "id": "J4",
        "path": "tests/taiji_native/test_b0_m4_artifact_audit_contract.py",
        "marker": 'baseline["stop_reason"] == "all_members_exhausted"',
        "kind": "test_assertion",
        "safe_because": (
            "pins the FROZEN-rule baseline inside an already-generated report; it is "
            "historical evidence, not a live gate.  WP-3 must re-review it because the "
            "counterfactual is expected to degenerate to identity once M4 lands"
        ),
    },
    {
        "id": "J5",
        "path": "scripts/training/audit_taiji_b0_m4_hardening.py",
        "marker": "CONTRACT_INTERCEPT_TOKEN in reason",
        "kind": "substring_predicate",
        "safe_because": (
            "tests substring membership, not startswith. The new reason does not "
            "contain contract_intercepted; this does not certify policy compliance"
        ),
    },
    {
        "id": "J6",
        "path": "tests/taiji_native/test_b0_m4_hardening_contract.py",
        "marker": 'assert "all_members_blocked" not in baseline',
        "kind": "test_assertion",
        "safe_because": "pins distinct reason counts in the historical two-rule report",
    },
    {
        "id": "J7",
        "path": "tests/taiji_native/test_b0_structure_space_contract.py",
        "marker": 'assert "all_members_blocked" not in rows[cell]["stop_reasons_frozen"]',
        "kind": "test_assertion",
        "safe_because": "pins the frozen versus counterfactual report, not a live admission gate",
    },
    {
        "id": "J8",
        "path": "tests/taiji_native/test_b0_n2_stop_reason_semantics_contract.py",
        "marker": "assert GOAL_REASON not in block",
        "kind": "test_assertion",
        "safe_because": (
            "the WP-2 two-direction guard itself: it pins the frozen semantics (terminal, "
            "not goal_reached, not an interception, absent from the frozen rule) against "
            "archived reports. It is an admission guard for WP-3, not a runtime gate"
        ),
    },
    {
        "id": "J9",
        "path": "tests/taiji_native/test_b0_rule_revision_seal_contract.py",
        "marker": 'assert delta["variant_is_identity"] is True',
        "kind": "test_assertion",
        "safe_because": (
            "the WP-3 exit-5 seal guard: it pins the rule_revision=0 report digests, the "
            "version labels on the interpreting documents, and which single composition "
            "rule ships. It reads reason names only to assert their immutability; it is "
            "an evidence-integrity guard, not a runtime gate"
        ),
    },
    {
        "id": "J10",
        "path": "tests/taiji_native/test_b0_m1_counterfactual_contract.py",
        "marker": '"interleaved_contexts", "best_pair", "best_pair_gain", "stop_reasons"',
        "kind": "test_assertion",
        "safe_because": (
            "the WP-3 exit-2 guard: it compares the frozen-surface and candidate-surface "
            "stop_reasons of the sealed revision-0 report with the landed report, field by "
            "field, so 'landing changed no outcome' is checked rather than recited. It reads "
            "reason names only to assert immutability across rule revisions; it is an "
            "evidence-integrity guard, not a runtime gate"
        ),
    },
)


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


def load_hardening() -> Any:
    """Reuse the hardening scan so there is exactly one implementation of it."""

    return _load("_b0_n2_hardening_scan", HARDENING_MODULE)


def live_consumers() -> set[str]:
    """The file list from a scan of the CURRENT tree (single implementation)."""

    return {
        row["path"].replace("\\", "/")
        for row in load_hardening().stop_reason_consumers()["files"]
        if row["path"].replace("\\", "/") not in SELF_AUDIT_PATHS
    }


def disposition() -> dict[str, Any]:
    expected = {row["path"] for row in EXPECTED_CONSUMERS}
    live = live_consumers()

    added = sorted(live - expected)
    removed = sorted(expected - live)

    site_checks: list[dict[str, Any]] = []
    for site in JUDGEMENT_SITES:
        path = PROJECT_ROOT / site["path"]
        source = path.read_text(encoding="utf-8") if path.exists() else ""
        site_checks.append(
            {
                "id": site["id"],
                "path": site["path"],
                "kind": site["kind"],
                "marker_present": site["marker"] in source,
                "safe_because": site["safe_because"],
            }
        )

    missing_sites = [row["id"] for row in site_checks if not row["marker_present"]]

    return {
        "format": DISPOSITION_FORMAT,
        "version": VERSION,
        "status": "inventory_guard_for_frozen_preregistration",
        "does_not_decide_n2": (
            "the N2 semantics are frozen in "
            "plans/reference/M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md; "
            "this scan only verifies that the consumer inventory and the judgement-site "
            "markers behind it are still complete and undisplaced"
        ),
        "new_reason": NEW_REASON,
        "frozen_counterpart": FROZEN_COUNTERPART,
        "interception_prefix": INTERCEPTION_PREFIX,
        "new_reason_carries_interception_prefix": NEW_REASON.startswith(INTERCEPTION_PREFIX),
        "self_audit_paths_excluded": sorted(SELF_AUDIT_PATHS),
        "consumer_count": len(live),
        "consumer_count_in_hardening_report": 11,
        "consumer_count_now": len(live),
        "count_was_stale": bool(len(live) != 11),
        "added_since_hardening_report": [
            row["path"] for row in EXPECTED_CONSUMERS if row.get("added_after_hardening_report")
        ],
        "drift": {
            "added": added,
            "removed": removed,
            "clean": not added and not removed,
            "meaning": (
                "a non-empty 'added' list means a new file consumes stop_reason and must "
                "be dispositioned before M4 lands; this check exists because the "
                "hardening report's own list went stale within one round"
            ),
        },
        "judgement_sites": site_checks,
        "missing_judgement_sites": missing_sites,
        "judgement_sites_intact": not missing_sites,
        "review_checks_passed": not missing_sites and not added and not removed,
        "consumers": list(EXPECTED_CONSUMERS),
        "classification_counts": {
            "record_only_files": sum(row["class"] == "record_only" for row in EXPECTED_CONSUMERS),
            "judgement_or_mixed_files": sum(
                row["class"] != "record_only" for row in EXPECTED_CONSUMERS
            ),
        },
        "scope_limit": (
            "text scan and source markers only; not an exhaustive data-flow proof. "
            "Marker presence cannot establish an unchanged predicate or deterministic replicas"
        ),
        "terminal_semantics": {
            "all_members_blocked": "terminal task non-completion, not an intermediate successful handoff",
            "implies_goal_reached": False,
            "implies_global_member_inability": False,
            "safety_status": "requires execution and policy evidence; cannot be inferred from reason alone",
        },
        "conclusion": (
            "the reviewed sites record reasons, compare replicas, count interception "
            "prefixes/substrings, or assert historical reports. all_members_blocked is "
            "not a success label and by itself proves neither safety nor member incapability. "
            "The N2 semantics are frozen in the WP-2 preregistration; runtime changes "
            "still require WP-3's five exits"
            if not missing_sites and not added and not removed
            else "the review surface changed; re-disposition before landing"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = disposition()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"new reason: {payload['new_reason']} (frozen counterpart: {payload['frozen_counterpart']})"
    )
    print(
        f"consumers: report said {payload['consumer_count_in_hardening_report']}, "
        f"live scan finds {payload['consumer_count_now']} "
        f"(stale={payload['count_was_stale']})"
    )
    for row in payload["added_since_hardening_report"]:
        print(f"  added after report: {row}")
    drift = payload["drift"]
    print(f"drift clean: {drift['clean']}")
    for row in drift["added"]:
        print(f"  + {row}")
    for row in drift["removed"]:
        print(f"  - {row}")
    print(
        f"judgement sites intact: {payload['judgement_sites_intact']} "
        f"({len(payload['judgement_sites'])} sites)"
    )
    for row in payload["judgement_sites"]:
        print(f"  {row['id']} {row['kind']:<18} present={row['marker_present']}")
    print(f"conclusion: {payload['conclusion']}")
    print(f"written: {args.output}")
    # Persist failed audits as evidence, but never return success on drift.
    return 0 if payload["review_checks_passed"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
