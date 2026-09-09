"""Audit the real R6 entry gates without running training.

The audit is intentionally fail-closed.  It only reads the frozen R4/R5/K
reports and the R6 adapter preflight, then decides whether an R6 formal course
may start.  It cannot promote an arm or attach a shadow owner to runtime.
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

from taiji.internalization import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-r6-admission-audit-v1"
VERSION = 1
R4_FORMAL = PROJECT_ROOT / "reports" / "taiji_m4v2_r4_shadow_formal_20260909.json"
R5_FORMAL = PROJECT_ROOT / "reports" / "taiji_m4v2_r5_conditional_formal_20260909.json"
K_SCORECARD = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v2_20260909.json"
R6_PREFLIGHT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_r6_k_adapter_preflight_20260909.json"
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must be an object: {path}")
    return payload


def _all_true(values: Any) -> bool:
    return isinstance(values, dict) and bool(values) and all(
        value is True for value in values.values()
    )


def _source(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "format": payload.get("format"),
        "status": payload.get("status"),
        "content_digest": content_digest(payload),
    }


def audit(
    *,
    r4: dict[str, Any],
    r5: dict[str, Any],
    k_scorecard: dict[str, Any],
    r6_preflight: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    r4_large_g = r4.get("aggregate", {}).get("pressure_vs_fixed_large", {}).get("G", {})
    r5_retention = r5.get("retention_gate", {})
    k_gates = k_scorecard.get("promotion_gates", {})
    k_verdict = k_scorecard.get("verdict", {})
    r6_boundary = r6_preflight.get("boundary", {})

    gates = {
        "r4_report_technical_evidence_closed": (
            r4.get("format") == "taiji-m4v2-r4-shadow-formal-v1"
            and r4.get("status") == "passed"
            and _all_true(r4.get("technical_gates"))
            and isinstance(
                r4.get("aggregate", {}).get("pressure_candidate_lesion"), dict
            )
        ),
        "r4_structural_growth_admitted": False,
        "r5_report_formal_admitted": (
            r5.get("format") == "taiji-m4v2-r5-conditional-formal-v1"
            and r5.get("status") == "rejected"
            and r5.get("hypothesis_supported") is False
            and r5.get("can_promote") is False
        ),
        "r5_router_admitted": False,
        "r6_parent_preflight_passed": (
            r6_preflight.get("format") == "taiji-m4v2-r6-k-adapter-preflight-v1"
            and r6_preflight.get("status") == "passed"
            and _all_true(r6_preflight.get("checks"))
        ),
        "r6_adapter_shadow_only": (
            r6_boundary.get("training_performed") is False
            and r6_boundary.get("default_runtime_attached") is False
            and r6_boundary.get("candidate_promoted") is False
        ),
        "k_evidence_closed": (
            k_verdict.get("k_evidence_closed") is True
            and k_scorecard.get("evidence_gates", {}).get("k_axis_evidence_closed")
            is True
        ),
        "k_parent_retention_baseline_present": (
            k_gates.get("parent_retention_baseline_present") is True
        ),
        "k_default_runtime_owner_attached": (
            k_gates.get("default_runtime_owner_attached") is True
        ),
        "same_parent_continual_course_ready": (
            k_gates.get("same_parent_continual_s_g_k_evidence") is True
        ),
        "resource_rollback_old_capability_gate_ready": (
            k_gates.get("resource_rollback_old_capability_gate") is True
        ),
        "r5_resource_records_clean": (
            r5.get("resource_caps", {}).get("clean") is True
            and not r5.get("resource_caps", {}).get("violations")
        ),
        "r5_primary_threshold_met": (
            int(r5_retention.get("g_non_worse_cells", -1))
            >= int(r5_retention.get("g_required", 0))
            and int(r5_retention.get("s_non_worse_cells", -1))
            >= int(r5_retention.get("valid_cells", 0))
        ),
    }
    r4_count = int(r4_large_g.get("count", 0))
    r4_non_worse = int(r4_large_g.get("non_worse_count", -1))
    reasons = {
        "r4_structural_growth_admitted": (
            f"R4 G vs fixed-large non-worse={r4_non_worse}/{r4_count}; "
            "the fixed-large structural promotion gate was not met."
        ),
        "r5_router_admitted": (
            "R5 formal rejected the conditional-modularity hypothesis; router "
            "remains a rollback-capable shadow and is not an R6 owner."
        ),
        "k_parent_retention_baseline_present": (
            "K1/K2/K3 formal reports explicitly lack a same-parent retention baseline."
        ),
        "k_default_runtime_owner_attached": (
            "K1/K2/K3 remain standalone shadow evidence; no default Taiji owner is attached."
        ),
        "same_parent_continual_course_ready": (
            "No S→G→K same-parent formal course has run yet."
        ),
        "resource_rollback_old_capability_gate_ready": (
            "The adapter preflight covers checkpoint/rollback mechanics only; the full "
            "resource and old-capability retention course is not available."
        ),
        "r5_primary_threshold_met": (
            "R5 conditional module misses its preregistered G and S retention thresholds."
        ),
    }
    formal_entry_requirements = {
        "r4_structural_growth_admitted": gates["r4_structural_growth_admitted"],
        "r5_router_or_explicit_no_router_addendum": False,
        "r6_parent_preflight_passed": gates["r6_parent_preflight_passed"],
        "k_native_adapter_entry": gates["r6_adapter_shadow_only"],
        "same_parent_baseline_epsilon": gates["k_parent_retention_baseline_present"],
        "all_arm_checkpoint_resource_rollback_preflight": gates[
            "resource_rollback_old_capability_gate_ready"
        ],
        "related_ci_native_ledger_clean": False,
    }
    return {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed",
        "sources": {
            name: _source(source_paths[name], payload)
            for name, payload in (
                ("r4_formal", r4),
                ("r5_formal", r5),
                ("k_scorecard_v2", k_scorecard),
                ("r6_adapter_preflight", r6_preflight),
            )
        },
        "observed_metrics": {
            "r4_g_vs_fixed_large_non_worse": r4_large_g,
            "r5_retention_gate": r5_retention,
            "r5_resource_caps": r5.get("resource_caps", {}),
            "k_promotion_gates": k_gates,
            "r6_preflight_check_count": len(r6_preflight.get("checks", {})),
        },
        "gates": gates,
        "gate_reasons": reasons,
        "formal_entry_requirements": formal_entry_requirements,
        "verdict": {
            "can_start_r6_formal": False,
            "can_promote": False,
            "admission": "blocked_shadow_only",
            "reason": (
                "R6 adapter mechanics are preflight-clean, but R4 structural admission, "
                "R5 router/no-router boundary, same-parent retention baseline, full "
                "S→G→K evidence, and full resource/old-capability Gate are not all ready."
            ),
        },
        "allowed_next_action": (
            "Freeze an R6 fixed-capacity parent admission addendum and baseline protocol "
            "that explicitly treats R4/R5 as non-promoted shadows; do not train or attach "
            "the adapter until that contract is reviewed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4-formal", type=Path, default=R4_FORMAL)
    parser.add_argument("--r5-formal", type=Path, default=R5_FORMAL)
    parser.add_argument("--k-scorecard", type=Path, default=K_SCORECARD)
    parser.add_argument("--r6-preflight", type=Path, default=R6_PREFLIGHT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    paths = {
        "r4_formal": args.r4_formal,
        "r5_formal": args.r5_formal,
        "k_scorecard_v2": args.k_scorecard,
        "r6_adapter_preflight": args.r6_preflight,
    }
    payload = audit(
        r4=_read(args.r4_formal),
        r5=_read(args.r5_formal),
        k_scorecard=_read(args.k_scorecard),
        r6_preflight=_read(args.r6_preflight),
        source_paths=paths,
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
                "status": payload["status"],
                "can_start_r6_formal": payload["verdict"]["can_start_r6_formal"],
                "can_promote": payload["verdict"]["can_promote"],
                "blocked_gates": [
                    name for name, value in payload["formal_entry_requirements"].items()
                    if value is not True
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
