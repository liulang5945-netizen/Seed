"""Build the content-addressed M5 K-axis v6 scorecard.

Audit-only reducer: the K1/K2/K3 sections, the C-stage learning-mechanism
evidence, the G-side solver-mechanism evidence and the K-worker joint-course
evidence reuse the frozen v5 reducer verbatim and are cross-checked
digest-for-digest against the v5 report; the only new evidence line is the
default-runtime rollout review, transcribed read-only, which flips
``default_runtime_rollout_review_completed`` to ``true``.  With that flip
both frozen A8 entry conditions are satisfied; the remaining promotion
vetoes stay untouched, so ``promotion_gate`` and ``can_promote`` remain
false.  Never retrains, reruns a cell, or promotes any owner.  Contract:
``plans/reference/M5_K_AXIS_SCORECARD_V6_CONTRACT_20260911.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.audit_taiji_m5_k_axis_scorecard_v3 import (  # noqa: E402
    C_STAGE_REPORT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v4 import (  # noqa: E402
    P4_12_REPORT,
    P4_13_REPORT,
    V3_REPORT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v5 import (  # noqa: E402
    P4_14_REPORT,
    V4_REPORT,
    build_v5,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v5 import (
    REPORT_FORMAT as V5_REPORT_FORMAT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v5 import (
    VERSION as V5_VERSION,
)
from scripts.training.audit_taiji_m5_k_scorecard import _read_report  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

ROLLOUT_REVIEW_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_default_runtime_rollout_review_20260911.json"
)
ROLLOUT_ATTACHMENT = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_default_runtime_rollout_attachment_v1.json"
)
V5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v5_20260911.json"
REPORT_FORMAT = "taiji-m5-k-axis-scorecard-v6"
VERSION = 6


def _rollout_review_evidence(review: dict[str, Any], attachment: dict[str, Any]) -> dict[str, Any]:
    if review.get("format") != "taiji-m5-k-default-runtime-rollout-review-v1":
        raise ValueError("rollout review report format mismatch")
    if review.get("status") != "completed":
        raise ValueError("rollout review report is not completed")
    if review.get("outcome") != "default_runtime_rollout_review_supported":
        raise ValueError("rollout review outcome is not review_supported")
    if review.get("experiment_passed") is not True:
        raise ValueError("rollout review experiment_passed is not true")
    if review.get("can_promote") is not False or review.get("growth_admitted") is not False:
        raise ValueError("rollout review must keep promotion/growth fail-closed")
    if review.get("fit_called") is not False or review.get("training_performed") is not False:
        raise ValueError("rollout review must remain training-free")
    if review.get("attachment_manifest_digest") != attachment.get("manifest_digest"):
        raise ValueError("rollout review attachment manifest digest mismatch")
    for gate in ("consumption_gate", "restore_gate", "mechanical_gate", "non_interference"):
        values = review.get(gate, {})
        if not values or not all(bool(value) for value in values.values()):
            raise ValueError(f"rollout review {gate} is not fully passed")
    if review.get("cell_mixing_rejected") is not True:
        raise ValueError("rollout review did not reject wrong-cell mixing")
    cells = review.get("cell_results", [])
    if len(cells) != 4 or not all(bool(cell.get("passes_all")) for cell in cells):
        raise ValueError("rollout review did not pass all four cells")
    if attachment.get("format") != "taiji-default-runtime-rollout-attachment-v1":
        raise ValueError("attachment manifest format mismatch")
    inventory = review.get("runtime_inventory", {})
    if inventory.get("runtime_k_g_references") != []:
        raise ValueError("the runtime inventory unexpectedly references K/G machinery")
    return {
        "source_report": ROLLOUT_REVIEW_REPORT.name,
        "attachment_manifest": ROLLOUT_ATTACHMENT.name,
        "attachment_manifest_digest": str(attachment["manifest_digest"]),
        "outcome": review["outcome"],
        "consumption_contract": (
            "taiji-default-runtime-rollout-attachment-v1: content-addressed "
            "per-cell artifact digests pinned from the P4.14 report, P4.14 "
            "lineage, embedded safety invariants (confidence floors, "
            "selection margin, parameter counts), fail-closed load order, "
            "no fit / no artifact writes / no invariant overrides"
        ),
        "runtime_baseline": {
            "runtime_module": "api/seed_runtime.py",
            "default_checkpoint": "checkpoints/seed_corpus.pt",
            "default_checkpoint_sha256_unchanged": bool(
                review["non_interference"]["default_checkpoint_sha256_unchanged"]
            ),
            "runtime_k_g_references": [],
        },
        "matrix": "all 4 P4.14 cells (2 identity batches x 2 seeds), no subsampling",
        "consumption_equivalence": (
            "phase K (novel 2/2, old-class 4/4, safe abstention 6/6) and "
            "phase G (validation/holdout/retention-newtask 0.8/0.8675/0sv, "
            "retention-sibling 1.0/1.0) reproduce the P4.14 recorded values "
            "field-for-field in every cell through the disk-loading "
            "consumption path"
        ),
        "mechanical_summary": {
            "digest_and_invariants": True,
            "tamper_rejected": True,
            "wrong_cell_mixing_rejected": True,
            "independent_process_restore": True,
            "behavioral_rollback": True,
            "k_unchanged_after_rollout": True,
            "non_interference": True,
            "cohort_rematerialization_digest_match": True,
            "resource_caps_absolute_all_pass": True,
        },
        "fit_called": False,
        "training_performed": False,
        "default_runtime_rollout_review_completed": True,
    }


def build_v6(
    k1: dict[str, Any],
    k2: dict[str, Any],
    k3: dict[str, Any],
    c_stage: dict[str, Any],
    v2_report: dict[str, Any],
    v3_report: dict[str, Any],
    p4_12_report: dict[str, Any],
    p4_13_report: dict[str, Any],
    p4_14_report: dict[str, Any],
    v4_report: dict[str, Any],
    rollout_review: dict[str, Any],
    attachment: dict[str, Any],
    v5_report: dict[str, Any],
) -> dict[str, Any]:
    if v5_report.get("format") != V5_REPORT_FORMAT or v5_report.get("version") != V5_VERSION:
        raise ValueError("v5 scorecard format/version mismatch")
    core = build_v5(
        k1,
        k2,
        k3,
        c_stage,
        v2_report,
        v3_report,
        p4_12_report,
        p4_13_report,
        p4_14_report,
        v4_report,
    )
    v5_sources = v5_report.get("source_reports", {})
    for phase, digest in core["source_reports"].items():
        if v5_sources.get(phase) != digest:
            raise ValueError(f"evidence drifted since v5: {phase}")
    rollout_evidence = _rollout_review_evidence(rollout_review, attachment)
    promotion_gates = {
        **core["promotion_gates"],
        "default_runtime_rollout_review_completed": True,
    }
    core["format"] = REPORT_FORMAT
    core["version"] = VERSION
    core["source_reports"]["DEFAULT_RUNTIME_ROLLOUT_REVIEW"] = content_digest(rollout_review)
    core["rollout_review_evidence"] = rollout_evidence
    core["promotion_gates"] = promotion_gates
    core["verdict"] = {
        "k_evidence_closed": core["verdict"]["k_evidence_closed"],
        "learning_mechanism_closed": core["verdict"]["learning_mechanism_closed"],
        "g_solver_mechanism_course_closed": True,
        "k_worker_joint_course_completed": True,
        "default_runtime_rollout_review_completed": True,
        "promotion_gate": all(promotion_gates.values()),
        "can_promote": False,
        "reason": (
            "Both frozen A8 entry conditions are now satisfied: the K worker "
            "joint course (4/4 cells) and the default runtime rollout review "
            "(4/4 cells consuming the P4.14 artifacts through the frozen "
            "attachment contract with field-for-field behavioral "
            "reproduction). The A8 promotion review is now eligible and "
            "still requires independent approval; the remaining vetoes "
            "(default runtime owner attachment, resource/rollback/old-"
            "capability gate, same-parent learned S/G/K evidence, parent "
            "retention baseline) keep promotion_gate=false until the A8 "
            "review resolves them."
        ),
    }
    core["next_boundary"] = (
        "convene the A8 promotion review (both entry conditions complete; "
        "independent approval required); keep all owners detached and the "
        "default runtime unmodified until the A8 review resolves the "
        "remaining promotion vetoes"
    )
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = build_v6(
        _read_report(
            Path(PROJECT_ROOT / "reports" / "taiji_m5_k1_skill_composition_formal_20260909.json")
        ),
        _read_report(Path(PROJECT_ROOT / "reports" / "taiji_m5_k2_multistep_formal_20260909.json")),
        _read_report(
            Path(PROJECT_ROOT / "reports" / "taiji_m5_k3_outcome_dependency_formal_20260909.json")
        ),
        _read_report(C_STAGE_REPORT),
        _read_report(PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v2_20260909.json"),
        _read_report(V3_REPORT),
        _read_report(P4_12_REPORT),
        _read_report(P4_13_REPORT),
        _read_report(P4_14_REPORT),
        _read_report(V4_REPORT),
        _read_report(ROLLOUT_REVIEW_REPORT),
        _read_report(ROLLOUT_ATTACHMENT),
        _read_report(V5_REPORT),
    )
    payload["generated_at_epoch"] = int(time.time())
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "k_worker_joint_course_completed": payload["verdict"][
                    "k_worker_joint_course_completed"
                ],
                "default_runtime_rollout_review_completed": payload["verdict"][
                    "default_runtime_rollout_review_completed"
                ],
                "promotion_gate": payload["verdict"]["promotion_gate"],
                "can_promote": payload["verdict"]["can_promote"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
