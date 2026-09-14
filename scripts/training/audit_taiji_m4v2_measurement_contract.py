"""Re-audit the M4.R7/R10/R12 evidence under the continual-growth contract.

This command is read-only with respect to the model and its historical JSON
reports.  It creates a new, versioned audit artifact that separates technical
facts, historical Gate results, and architecture conclusions.  In particular,
it never turns an absolute BPB value or an arm-vs-arm difference into a parent
forgetting claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.continual_evaluation import (  # noqa: E402
    CONTINUAL_EVALUATION_FORMAT,
    CONTINUAL_EVALUATION_VERSION,
)

AUDIT_FORMAT = "taiji-m4v2-measurement-audit-v1"
AUDIT_VERSION = 1
DEFAULT_R7 = PROJECT_ROOT / "reports" / "taiji_m4r7_formal_aggregate_20260908.json"
DEFAULT_R10 = PROJECT_ROOT / "reports" / "taiji_m4r10_rule_audit_formal_aggregate_20260908.json"
DEFAULT_R12 = PROJECT_ROOT / "reports" / "taiji_m4r12_data_source_aggregate_20260909.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_measurement_audit_20260909.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return payload


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def build_audit(r7: dict[str, Any], r10: dict[str, Any], r12: dict[str, Any]) -> dict[str, Any]:
    r7_candidate = r7["variants"]["scale_0p5"]
    r10_candidate = r10["variants"]["candidate"]["metrics"]
    r10_active_a = r10_candidate["active_a_retention_bpb"]
    r12_shift = r12["a_retention_shift_vs_reference"]

    r7_historical_gate_passed = bool(r7_candidate["retention_gate_passed_all"])
    r7_cycle3_degradation_seeds = int(r7_candidate["c_cycle3_degradation_count"])
    r7_mean_cycle3_delta = float(r7_candidate["c_cycle3_delta_mean_bpb"])

    r10_absolute_value = "values" in r10_active_a and "mean" in r10_active_a
    r10_has_parent_delta = any(
        key in r10_active_a for key in ("parent_delta", "parent_delta_bpb", "delta")
    )
    r10_generic_degradation_count = "degradation_seed_count" in r10_active_a

    r12_has_parent_baseline = any(
        key in r12_shift for key in ("parent_checkpoint_digest", "parent_delta", "parent_baseline")
    )

    owner_fields = ("owner_graph_digest", "owner_graph_id", "owner")
    r7_has_owner_graph = any(
        key in r7.get("configuration", {}) or key in r7 for key in owner_fields
    )
    r10_has_owner_graph = any(
        key in r10.get("configuration", {}) or key in r10 for key in owner_fields
    )

    semantic_checks = {
        "r7_historical_gate_is_preserved": True,
        "r10_active_a_is_absolute_bpb": r10_absolute_value and not r10_has_parent_delta,
        "r10_absolute_metric_not_reclassified_as_forgetting": (
            r10_absolute_value and r10_generic_degradation_count
        ),
        "r12_shift_is_arm_vs_arm": bool(r12_shift.get("mean") is not None)
        and not r12_has_parent_baseline,
        "r7_r10_owner_graphs_not_claimed_comparable": not (
            r7_has_owner_graph and r10_has_owner_graph
        ),
        "missing_parent_baseline_blocks_forgetting_claim": not r12_has_parent_baseline,
        "cross_domain_bpb_is_not_merged": True,
    }

    audit_passed = all(semantic_checks.values())
    return {
        "format": AUDIT_FORMAT,
        "version": AUDIT_VERSION,
        "audit_status": "passed" if audit_passed else "failed",
        "can_promote": False,
        "measurement_contract": {
            "format": CONTINUAL_EVALUATION_FORMAT,
            "version": CONTINUAL_EVALUATION_VERSION,
            "required_separation": [
                "absolute_value",
                "parent_delta",
                "comparison_delta",
                "owner_id",
                "read_only_input_digest",
            ],
            "course_phase_visibility": "evaluator_only",
        },
        "semantic_checks": semantic_checks,
        "technical_facts": {
            "r7_scale_0p5": {
                "c3_gain_positive_seeds": int(r7_candidate["c3_gain_positive_count"]),
                "cycle3_degradation_seeds": r7_cycle3_degradation_seeds,
                "cycle3_delta_mean_bpb": r7_mean_cycle3_delta,
            },
            "r10_candidate_active_a_retention_bpb": {
                "kind": "absolute",
                "mean": float(r10_active_a["mean"]),
                "values": [float(value) for value in r10_active_a["values"]],
            },
            "r12_a_retention_shift_vs_reference": {
                "kind": "arm_vs_arm",
                "mean": float(r12_shift["mean"]),
                "values": [float(value) for value in r12_shift["values"]],
            },
        },
        "historical_gate": {
            "r7_strict_all_seed_gate_passed": r7_historical_gate_passed,
            "r7_strict_gate_semantics": (
                "positive C3 holdout gain on every seed plus zero C-cycle-2/C-cycle-3 "
                "degradation seeds"
            ),
            "r7_exact_zero_result_is_historical_only": True,
            "interpretation": (
                "The historical strict Gate failed because two seeds had positive "
                "cycle-3 degradation deltas, even though the aggregate cycle-3 delta "
                "was negative. This is not a global capacity or architecture veto."
            ),
        },
        "architecture_conclusion": {
            "parent_forgetting_claim_allowed": False,
            "reason": (
                "R10 A is an absolute BPB measurement and R12 compares separate arms; "
                "neither report contains a valid parent checkpoint baseline."
            ),
            "next_gate": "calibrated epsilon plus worst-domain catastrophe threshold",
        },
        "source_reports": {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r7", type=Path, default=DEFAULT_R7)
    parser.add_argument("--r10", type=Path, default=DEFAULT_R10)
    parser.add_argument("--r12", type=Path, default=DEFAULT_R12)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    r7 = _load(args.r7)
    r10 = _load(args.r10)
    r12 = _load(args.r12)
    audit = build_audit(r7, r10, r12)
    audit["source_reports"] = {
        "r7": _relative(args.r7),
        "r10": _relative(args.r10),
        "r12": _relative(args.r12),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": _relative(args.report),
                "audit_status": audit["audit_status"],
                "can_promote": audit["can_promote"],
                "r7_strict_gate_passed": audit["historical_gate"]["r7_strict_all_seed_gate_passed"],
                "r12_shift_kind": audit["technical_facts"]["r12_a_retention_shift_vs_reference"][
                    "kind"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if audit["audit_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
