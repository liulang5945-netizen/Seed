"""Build the content-addressed M5 K-axis v7 scorecard.

Audit-only reducer: every evidence line reuses the frozen v6 reducer
verbatim and is cross-checked digest-for-digest against the v6 report; the
only change is the mechanical transcription of the independently approved
A8 promotion review (2026-09-12, outcome ``promotion_review_recommended``):
three evidence vetoes flip (``parent_retention_baseline_present``,
``same_parent_continual_s_g_k_evidence``,
``resource_rollback_old_capability_gate``) and the new
``promotion_review_evidence`` line transcribes the decision read-only.

The two attachment gates stay ``false``: they flip only after the
preregistered default-runtime attachment step executes and passes its own
acceptance gates, so ``promotion_gate`` and ``can_promote`` remain false.
Never retrains, reruns a cell, or promotes any owner.  Contract:
``plans/reference/M5_K_AXIS_SCORECARD_V7_CONTRACT_20260912.md``.
"""

from __future__ import annotations

import argparse
import hashlib
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
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v6 import (
    REPORT_FORMAT as V6_REPORT_FORMAT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v6 import (  # noqa: E402
    ROLLOUT_ATTACHMENT,
    ROLLOUT_REVIEW_REPORT,
    V5_REPORT,
    build_v6,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v6 import (
    VERSION as V6_VERSION,
)
from scripts.training.audit_taiji_m5_k_scorecard import _read_report  # noqa: E402

A8_REVIEW_CONTRACT = PROJECT_ROOT / "plans" / "reference" / "M5_K_A8_PROMOTION_REVIEW_20260912.md"
SGK_CONTRACT = (
    PROJECT_ROOT / "plans" / "reference" / "M4V2_SGK_PROMOTION_COURSE_PREREGISTRATION_20260910.md"
)
V6_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v6_20260911.json"
REPORT_FORMAT = "taiji-m5-k-axis-scorecard-v7"
VERSION = 7

FLIPPED_EVIDENCE_VETOES = (
    "parent_retention_baseline_present",
    "same_parent_continual_s_g_k_evidence",
    "resource_rollback_old_capability_gate",
)
UNCHANGED_ATTACHMENT_VETOES = (
    "default_runtime_owner_attached",
    "learning_mechanism_attached_default_runtime",
)


def _promotion_review_evidence(review_contract: Path, sgk_contract: Path) -> dict[str, Any]:
    text = review_contract.read_text(encoding="utf-8")
    for marker in (
        "promotion_review_recommended",
        "评审人对 §3 四项裁决**全部批准**",
        "遇到需要决策的分叉时，优先选择**上限更高**的选项",
    ):
        if marker not in text:
            raise ValueError(f"A8 review contract is missing the marker: {marker}")
    sgk_text = sgk_contract.read_text(encoding="utf-8")
    for marker in (
        "superseded——被 P4.11–P4.14 求解器机制链 + 默认 runtime rollout review 后继取代",
        "不删除、不改写、不重新授权执行",
    ):
        if marker not in sgk_text:
            raise ValueError(f"SGK v1 contract is missing the marker: {marker}")
    return {
        "source_contract": review_contract.name,
        "source_contract_sha256": hashlib.sha256(review_contract.read_bytes()).hexdigest(),
        "superseded_sgk_contract": sgk_contract.name,
        "superseded_sgk_contract_sha256": hashlib.sha256(sgk_contract.read_bytes()).hexdigest(),
        "decision_date": "2026-09-12",
        "outcome": "promotion_review_recommended",
        "dispositions": {
            "parent_retention_baseline_present": (
                "flipped by evidence adjudication: the P2.4 -> rollout-review "
                "retention chain on the same parent exceeds the v1 literal "
                "requirement (a parent_retention field in the K1/K2 formal "
                "contracts)"
            ),
            "same_parent_continual_s_g_k_evidence": (
                "flipped by evidence adjudication: P4.14 is a same-parent "
                "continuous S/G/K course; S remains the architecturally "
                "frozen control-only evidence organ (P3.1 contract) and that "
                "boundary is recorded verbatim; a learned-S exploration line "
                "is acknowledged as a non-blocking future high-ceiling item"
            ),
            "resource_rollback_old_capability_gate": (
                "flipped by successor-chain mapping: SGK v1 (resource "
                "equivalence + rollback + old-capability non-inferiority) is "
                "marked superseded by the P4.12/P4.13/P4.14 absolute budgets, "
                "the P3.0/P4.13/review rollback gates and the P4.13 zero "
                "backward forgetting + review field-for-field retention "
                "reproduction; its FS learning mechanism was proven a failure "
                "mechanism on this task by P4.10/P4.11"
            ),
            "attachment_authorization": (
                "approved: the default-runtime attachment engineering step "
                "enters preregistration; default_runtime_owner_attached and "
                "learning_mechanism_attached_default_runtime stay false until "
                "that step executes and passes its own acceptance gates"
            ),
        },
        "standing_decision_principle": (
            "at future decision points prefer the higher-ceiling option, "
            "provided the repo's fail-closed discipline and frozen criteria "
            "are respected"
        ),
        "promotion_review_recommended": True,
    }


def build_v7(
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
    v6_report: dict[str, Any],
    review_contract: Path,
    sgk_contract: Path,
) -> dict[str, Any]:
    if v6_report.get("format") != V6_REPORT_FORMAT or v6_report.get("version") != V6_VERSION:
        raise ValueError("v6 scorecard format/version mismatch")
    core = build_v6(
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
        rollout_review,
        attachment,
        v5_report,
    )
    v6_sources = v6_report.get("source_reports", {})
    for phase, digest in core["source_reports"].items():
        if v6_sources.get(phase) != digest:
            raise ValueError(f"evidence drifted since v6: {phase}")
    gates_v6 = v6_report["promotion_gates"]
    for veto in (*FLIPPED_EVIDENCE_VETOES, *UNCHANGED_ATTACHMENT_VETOES):
        if gates_v6.get(veto) is not False:
            raise ValueError(f"v6 veto {veto} is not false; transcription aborted")
    promotion_gates = {
        **core["promotion_gates"],
        **{veto: True for veto in FLIPPED_EVIDENCE_VETOES},
    }
    core["format"] = REPORT_FORMAT
    core["version"] = VERSION
    core["source_reports"]["A8_PROMOTION_REVIEW"] = hashlib.sha256(
        review_contract.read_bytes()
    ).hexdigest()
    core["promotion_review_evidence"] = _promotion_review_evidence(review_contract, sgk_contract)
    core["promotion_gates"] = promotion_gates
    core["verdict"] = {
        "k_evidence_closed": core["verdict"]["k_evidence_closed"],
        "learning_mechanism_closed": core["verdict"]["learning_mechanism_closed"],
        "g_solver_mechanism_course_closed": True,
        "k_worker_joint_course_completed": True,
        "default_runtime_rollout_review_completed": True,
        "promotion_review_recommended": True,
        "promotion_gate": all(promotion_gates.values()),
        "can_promote": False,
        "reason": (
            "The independently approved A8 promotion review flipped the three "
            "evidence vetoes (parent retention baseline, same-parent "
            "continuous S/G/K course with S as the architecturally "
            "control-only evidence organ, and the resource/rollback/"
            "old-capability gate satisfied by the successor chain; SGK v1 is "
            "marked superseded). The two attachment gates "
            "(default_runtime_owner_attached, "
            "learning_mechanism_attached_default_runtime) stay false by "
            "design until the preregistered default-runtime attachment step "
            "executes and passes its own acceptance gates, so "
            "promotion_gate=false and can_promote=false remain mechanical."
        ),
    }
    core["next_boundary"] = (
        "execute the preregistered default-runtime attachment engineering "
        "step (fail-closed contract load inside the runtime, behavioral "
        "equivalence, rollback, non-interference); after it passes its own "
        "acceptance gates the next scorecard flips the two attachment gates, "
        "and only then does can_promote become eligible (still requiring "
        "independent approval)"
    )
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = build_v7(
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
        _read_report(V6_REPORT),
        A8_REVIEW_CONTRACT,
        SGK_CONTRACT,
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
                "promotion_review_recommended": payload["verdict"]["promotion_review_recommended"],
                "promotion_gate": payload["verdict"]["promotion_gate"],
                "can_promote": payload["verdict"]["can_promote"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
