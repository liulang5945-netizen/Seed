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
    {
        "path": "scripts/training/eval_taiji_b0_b2v4_collaboration_gate.py",
        "class": "record_only",
        "why": "aggregates per-face episode stop-reason counts via _stop_reason_counts; it "
        "records frequencies and makes no safety or capability judgement",
        "added_after_hardening_report": "yes",
    },
    {
        "path": "scripts/training/audit_taiji_r2_h3_7_attribution.py",
        "class": "judgement (assertion on aggregate stop reasons)",
        "why": "asserts the frozen aggregate reason appears in the R2 decision; "
        "see the H3.7 attribution ledger",
        "added_after_hardening_report": "yes",
    },
    # 2026-09-26 补登记：下面六处把被审计面从 19 撑到 25。三条来自 09-19 的
    # HANDOFF-M4／统一入口结案，三条来自 09-19～09-22 的 R2 训练器（后三类**只共享字段名**，
    # 词表与规则侧完全不相交——理由逐条写明，不整批豁免）。
    {
        "path": "scripts/training/run_taiji_collab_handoff_entry_evidence.py",
        "class": "judgement (membership on both reason names)",
        "why": "its evidence booleans key on membership of BOTH the frozen and the new "
        "name (all_members_blocked_demonstrated / all_members_exhausted_demonstrated) plus "
        "an equality against goal_reached; see J11 below",
        "added_after_hardening_report": "yes",
    },
    {
        "path": "scripts/training/run_taiji_unified_entry_evidence.py",
        "class": "judgement (writes the rule's reason literals itself)",
        "why": "a hand-rolled replica of the loop: it emits goal_reached / "
        "all_members_exhausted / contract_intercepted:<code> / approval_rejected / step_cap "
        "as literals. See J12 below.",
        "added_after_hardening_report": "yes",
    },
    {
        "path": "taiji/collab_handoff.py",
        "class": "record_only",
        "why": "the rule-side producer for the handoff episode: it copies "
        "`decision.stop` verbatim into the episode payload and events; its only comparison on "
        "the value is `is None` (the step_cap fallback), which no new reason can satisfy by "
        "accident",
        "added_after_hardening_report": "yes",
    },
)

ADDITIONAL_CONSUMERS: tuple[dict[str, str], ...] = tuple(
    {"path": path, "class": kind, "why": why}
    for path, kind, why in (
        (
            "scripts/training/_taiji_n1_plan118_lossless_receipt.py",
            "judgement (boundary)",
            '2026-10-23 live scan: and row["stop_reason"] == "boundary"',
        ),
        (
            "scripts/training/_taiji_n1_plan120_lossless_receipt.py",
            "judgement (boundary)",
            '2026-10-23 live scan: and row["stop_reason"] == "boundary"',
        ),
        (
            "scripts/training/_taiji_n1_transition_factorial_training_receipt.py",
            "judgement (boundary)",
            '2026-10-23 live scan: and row["stop_reason"] == "boundary"',
        ),
        (
            "scripts/training/analyze_taiji_n1_plan161_question_surface_itemwise_attribution.py",
            "judgement (boundary)",
            '2026-10-23 live scan: if raw == target and row.get("stop_reason") == "boundary":',
        ),
        (
            "scripts/training/analyze_taiji_n1_plan162b_train_only_itemwise_attribution.py",
            "record_only",
            '2026-10-23 live scan: "control_stop_reason": control.get("stop_reason"),',
        ),
        (
            "scripts/training/analyze_taiji_n1_plan162d_transition_output_type_attribution.py",
            "judgement (boundary)",
            '2026-10-23 live scan: return "correct_exact" if stop_reason == "boundary" else "target_bytes_bad_stop"',
        ),
        (
            "scripts/training/analyze_taiji_n1_position_weight_history_crossover_session_scope.py",
            "record_only",
            '2026-10-23 live scan: "stop_reason",',
        ),
        (
            "scripts/training/analyze_taiji_n1_transition_operator_error_attribution.py",
            "judgement (boundary)",
            '2026-10-23 live scan: return "correct" if row.get("stop_reason") == "boundary" else "expected_text_non_boundary"',
        ),
        (
            "scripts/training/attribute_taiji_n1_confirmation_errors_run89.py",
            "judgement (boundary)",
            '2026-10-23 live scan: row["stop_reason"] == "boundary"',
        ),
        (
            "scripts/training/audit_taiji_n1_entity_binding.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: for k in ("raw_hex", "stop_reason", "teacher_tokens_correct")',
        ),
        (
            "scripts/training/audit_taiji_n1_fidelity_content_read.py",
            "judgement (branch/comparison)",
            '2026-10-23 live scan: if (prediction["raw_hex"], prediction["stop_reason"]) != (',
        ),
        (
            "scripts/training/audit_taiji_n1_iterative_content_read.py",
            "judgement (branch/comparison)",
            '2026-10-23 live scan: if (prediction["raw_hex"], prediction["stop_reason"]) != (',
        ),
        (
            "scripts/training/audit_taiji_n1_native_evidence_train_fit.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "free_exact": trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/audit_taiji_n1_plan162f_transcript_integrity.py",
            "record_only",
            '2026-10-23 live scan: str(row["stop_reason"]),',
        ),
        (
            "scripts/training/audit_taiji_n1_question_fusion.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: all(row[k] == old[k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))',
        ),
        (
            "scripts/training/audit_taiji_n1_relation_seed_drift.py",
            "judgement (branch/comparison)",
            '2026-10-23 live scan: stop_reasons=dict(Counter(r["stop_reason"] for r in subset)),',
        ),
        (
            "scripts/training/audit_taiji_n1_train_content_read.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "exact": trace.raw == item["answer"].encode() and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/audit_taiji_n1_transfer_binding.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: for k in ("raw_hex", "stop_reason", "teacher_tokens_correct")',
        ),
        (
            "scripts/training/build_taiji_n1_v15_after_n132.py",
            "record_only",
            '2026-10-23 live scan: "prior_stop_reasons_zh": [',
        ),
        (
            "scripts/training/diagnose_taiji_n1_fresh_training_logs.py",
            "record_only",
            "2026-10-23 live scan: confusion[f\"{item['answer']}->{label}/{row['stop_reason']}\"] += 1",
        ),
        (
            "scripts/training/diagnose_taiji_n1_fresh_transition_factorial.py",
            "record_only",
            '2026-10-23 live scan: "stop_reason",',
        ),
        (
            "scripts/training/diagnose_taiji_n1_position_weight_history_crossover.py",
            "record_only",
            '2026-10-23 live scan: "stop_reason",',
        ),
        (
            "scripts/training/eval_taiji_n1_fresh_frozen_confirmation.py",
            "judgement (boundary)",
            '2026-10-23 live scan: free_exact=trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/eval_taiji_n1_frozen_v4_confirmation.py",
            "judgement (boundary)",
            '2026-10-23 live scan: free_exact=trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/eval_taiji_n1_input_factorial_diagnostic.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "free_exact": trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/eval_taiji_n1_transition_factorial_confirmation.py",
            "judgement (boundary)",
            '2026-10-23 live scan: free_exact=trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/eval_taiji_n1_v13_v14_full_context_question_binding_mask.py",
            "judgement (boundary)",
            '2026-10-23 live scan: free_exact=trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/eval_taiji_n1_v13_v14_full_context_source_mask.py",
            "judgement (boundary)",
            '2026-10-23 live scan: audit["boundary_native"] = trace.stop_reason == "boundary"',
        ),
        (
            "scripts/training/eval_taiji_n1_v17_matched_context_length.py",
            "record_only",
            '2026-10-23 live scan: "stop_reason": row["stop_reason"],',
        ),
        (
            "scripts/training/eval_taiji_n1_v18_target_position_context_order.py",
            "record_only",
            '2026-10-23 live scan: "stop_reason": row["stop_reason"],',
        ),
        (
            "scripts/training/eval_taiji_n1_v19_neighbor_distance_context_interference.py",
            "record_only",
            '2026-10-23 live scan: "stop_reason": row["stop_reason"],',
        ),
        (
            "scripts/training/eval_taiji_n1_v20_neighbor_distance_dose_curve.py",
            "record_only",
            '2026-10-23 live scan: "stop_reason": row["stop_reason"],',
        ),
        (
            "scripts/training/eval_taiji_n1_v25_abstention_after_n147.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "free_exact": trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/eval_taiji_n1_v26_abstention_length_after_n148.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "free_exact": trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/run_taiji_n1_address_consistency.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: all(r[k] == old[r["id"]][k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))',
        ),
        (
            "scripts/training/run_taiji_n1_bound_consistency.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: all(r[k] == old[r["id"]][k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))',
        ),
        (
            "scripts/training/run_taiji_n1_contextual_question.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: all(r[k] == old[r["id"]][k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))',
        ),
        (
            "scripts/training/run_taiji_n1_local_question.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: all(r[k] == old[r["id"]][k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))',
        ),
        (
            "scripts/training/run_taiji_n1_local_relation.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: all(r[k] == old[r["id"]][k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))',
        ),
        (
            "scripts/training/run_taiji_n1_native_evidence_paired.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "exact": trace.raw == item["answer"].encode() and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/run_taiji_n1_plan153_v28_fresh_context_length_confirmation.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "free_exact": trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/run_taiji_n1_plan155_short_unknown_context_length_training.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "free_exact": trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/run_taiji_n1_plan162f_transition_template_crossover_diagnostic.py",
            "record_only",
            '2026-10-23 live scan: "stop_reason": row["stop_reason"],',
        ),
        (
            "scripts/training/run_taiji_n1_reader_transfer.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: all(r[k] == old[r["id"]][k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))',
        ),
        (
            "scripts/training/run_taiji_n1_relative_continuation.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "free_exact": trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/run_taiji_n1_relative_margin.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: all(r[k] == old[r["id"]][k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))',
        ),
        (
            "scripts/training/run_taiji_n1_relative_paired.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "free_exact": trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "scripts/training/run_taiji_n1_renaming.py",
            "judgement (raw+stop identity)",
            '2026-10-23 live scan: all(r[k] == old[r["id"]][k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))',
        ),
        (
            "scripts/training/run_taiji_n1_v27_length_robust_training_after_n149.py",
            "judgement (boundary)",
            '2026-10-23 live scan: "free_exact": trace.raw == target and trace.stop_reason == "boundary",',
        ),
        (
            "taiji/position_evidence_reference.py",
            "record_only",
            "2026-10-23 live scan: stop_reason: str",
        ),
        (
            "tests/taiji_native/test_n1_plan135_matched_context_length_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: "stop_reason": "boundary" if exact else "limit",',
        ),
        (
            "tests/taiji_native/test_n1_plan136_target_position_context_order_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: "stop_reason": "boundary" if exact else "limit",',
        ),
        (
            "tests/taiji_native/test_n1_plan137_neighbor_distance_context_interference_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: "stop_reason": "boundary" if exact else "limit",',
        ),
        (
            "tests/taiji_native/test_n1_plan138_neighbor_distance_dose_curve_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: "stop_reason": "boundary" if exact else "limit",',
        ),
        (
            "tests/taiji_native/test_n1_plan148_v25_abstention_scorer_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: output=b"abcdef", raw=b"abcdef", stop_reason="boundary", generated_steps=7',
        ),
        (
            "tests/taiji_native/test_n1_plan161_question_surface_itemwise_attribution_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: row = {"raw_hex": b"oldval".hex(), "stop_reason": "boundary", "context_length": 8}',
        ),
        (
            "tests/taiji_native/test_n1_plan162b_train_only_itemwise_attribution_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: "stop_reason": "boundary",',
        ),
        (
            "tests/taiji_native/test_n1_plan162f1_transcript_integrity_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: "stop_reason": "boundary",',
        ),
        (
            "tests/taiji_native/test_n1_plan162f_transition_template_crossover_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: "stop_reason": "boundary",',
        ),
        (
            "tests/taiji_native/test_n1_run91_error_attribution_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: assert shape_of(stop_reason="cap", generated_steps=64, free_exact=False) == (',
        ),
        (
            "tests/taiji_native/test_native_fresh_frozen_confirmation_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: and measured[0]["stop_reason"] == "boundary"',
        ),
        (
            "tests/taiji_native/test_native_fresh_training_log_diagnostic_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: stop_reason="boundary" if free else "cap",',
        ),
        (
            "tests/taiji_native/test_native_fresh_transition_factorial_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: stop_reason="boundary",',
        ),
        (
            "tests/taiji_native/test_native_frozen_v4_confirmation_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: and measured[0]["stop_reason"] == "boundary"',
        ),
        (
            "tests/taiji_native/test_native_iterative_evidence_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: assert trace.raw == b"" and trace.stop_reason == "boundary"',
        ),
        (
            "tests/taiji_native/test_native_question_fusion_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: row = dict(raw_hex="61", stop_reason="boundary", teacher_tokens_correct=2, teacher_ce=0.5)',
        ),
        (
            "tests/taiji_native/test_native_relation_seed_drift_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: stop_reason="boundary",',
        ),
        (
            "tests/taiji_native/test_native_transition_factorial_calibration_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: stop_reason="boundary",',
        ),
        (
            "tests/taiji_native/test_native_transition_factorial_confirmation_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: and measured[0]["stop_reason"] == "boundary"',
        ),
        (
            "tests/taiji_native/test_native_transition_factorial_paired_training_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: stop_reason="boundary",',
        ),
        (
            "tests/taiji_native/test_native_transition_factorial_training_receipt_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: stop_reason="boundary",',
        ),
        (
            "tests/taiji_native/test_position_evidence_reference_contract.py",
            "judgement (test assertion)",
            '2026-10-23 live scan: assert trace.stop_reason == "boundary" and trace.generated_steps == 1',
        ),
    )
)
EXPECTED_CONSUMERS = EXPECTED_CONSUMERS + ADDITIONAL_CONSUMERS

PENDING_FRONTIER_CONSUMERS: dict[str, str] = {
    "scripts/training/evaluate_taiji_r2_daily_dialogue_recovery_v2.py": "2026-10-23 R2/new-front consumer; leave to that line: f\"**{labels[endpoint_id]}**（{output['stop_reason']}，{output['generated_bytes_length']}字节）\",",
    "scripts/training/run_taiji_r2_daily_dialogue_continuation.py": '2026-10-23 R2/new-front consumer; leave to that line: "stop_reason": stop,',
    "scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py": '2026-10-23 R2/new-front consumer; leave to that line: if stopped.get("stop_reason") != "throughput_projection_exceeds_remaining_budget":',
    "scripts/training/run_taiji_r2_daily_dialogue_full_turn_pretraining.py": "2026-10-23 R2/new-front consumer; leave to that line: text, utf8_valid, no_replacement, boundary, stop_reason = trainer._generated_text(raw)",
    "scripts/training/run_taiji_r2_daily_dialogue_material_comparison.py": "2026-10-23 R2/new-front consumer; leave to that line: text, valid_utf8, no_replacement, boundary, stop_reason = trainer._generated_text(raw)",
    "tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py": '2026-10-23 R2/new-front consumer; leave to that line: "stop_reason": "end_marker",',
    "tests/taiji_native/test_r2_dialogue_continuation_extension_runner_contract.py": '2026-10-23 R2/new-front consumer; leave to that line: assert events[0]["stop_reason"] == "cumulative_cpu_ceiling_exhausted"',
    "tests/taiji_native/test_r2_dialogue_continuation_source_revision_contract.py": '2026-10-23 R2/new-front consumer; leave to that line: "stop_reason": "throughput_projection_exceeds_remaining_budget",',
    "scripts/training/run_taiji_r2_resume.py": '2026-10-23 R2/new-front consumer; leave to that line: self.pending_stop_details = {"stop_reason": "work_unit_refused", ...}',
}

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
    {
        "id": "J11",
        "path": "scripts/training/run_taiji_collab_handoff_entry_evidence.py",
        "marker": '"all_members_blocked_demonstrated": STOP_ALL_MEMBERS_BLOCKED in stops',
        "kind": "membership_predicate",
        "safe_because": (
            "the HANDOFF-M4 entry-evidence runner: it treats the new reason as something that "
            "must be *demonstrated* (membership in the observed stop-reason set), alongside a "
            "separate exhausted check and a goal_reached equality.  It therefore never reads "
            "all_members_blocked as success or as an interception; but its gate does require "
            "the new name to appear, so this file's evidence is only valid post-M4 and must be "
            "re-read if the reason is ever renamed"
        ),
    },
    {
        "id": "J12",
        "path": "scripts/training/run_taiji_unified_entry_evidence.py",
        "marker": '"stop": "all_members_exhausted"})',
        "kind": "rule_replica_literal",
        "safe_because": (
            "this runner hand-writes the rule's terminal labels instead of reading them from "
            "the policy. The literal above sits in the `if not bindable:` branch, which is "
            "exactly the FROZEN rule's exhausted condition, while all_members_blocked arises "
            "from bindable members whose execution failed -- a different branch handled by "
            "policy.select below.  success is computed independently via _goal_reached and "
            "trace_valid only checks event *kinds* (taiji/unified_entry.py), so no new reason "
            "can be silently counted as a success.  The residual exposure is the replica "
            "itself: if the two conditions were ever merged here, this label would need "
            "re-review, and M4's own evidence path does not go through this file"
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
    registered = {row["path"] for row in EXPECTED_CONSUMERS}
    pending = set(PENDING_FRONTIER_CONSUMERS)
    expected = registered | pending
    live = live_consumers()

    added = sorted(live - expected)
    removed = sorted(expected - live)
    pending_absent = sorted(pending - live)

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
        "registered_consumer_count": len(registered),
        "pending_frontier_consumers": [
            {"path": path, "why": PENDING_FRONTIER_CONSUMERS[path]} for path in sorted(pending)
        ],
        "pending_frontier_absent": pending_absent,
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
        "review_checks_passed": (
            not missing_sites and not added and not removed and not pending_absent
        ),
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
            if not missing_sites and not added and not removed and not pending_absent
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
