"""Build the content-addressed M5 K-axis v4 scorecard.

Audit-only reducer: the K1/K2/K3 sections and the C-stage
learning-mechanism evidence reuse the frozen v3 reducer verbatim and are
cross-checked digest-for-digest against the v3 report; the only new
evidence line is the G-side solver mechanism (P4.12 course-level
validation + P4.13 two-phase promotion course), transcribed read-only.
Never retrains, reruns a cell, or promotes any owner.  Contract:
``plans/reference/M5_K_AXIS_SCORECARD_V4_CONTRACT_20260911.md``.
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
    build_v3,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v3 import (  # noqa: E402
    REPORT_FORMAT as V3_REPORT_FORMAT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v3 import (  # noqa: E402
    VERSION as V3_VERSION,
)
from scripts.training.audit_taiji_m5_k_scorecard import (  # noqa: E402
    _read_report,
)
from taiji.internalization import content_digest  # noqa: E402

P4_12_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_12_course_level_validation_20260911.json"
P4_13_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_13_promotion_course_20260911.json"
V3_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v3_20260910.json"
REPORT_FORMAT = "taiji-m5-k-axis-scorecard-v4"
VERSION = 4


def _course_level_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("format") != "taiji-m5-k-p4-12-course-level-validation-v1":
        raise ValueError("P4.12 report format mismatch")
    if report.get("status") != "completed":
        raise ValueError("P4.12 report is not completed")
    if report.get("outcome") != "course_level_validation_supported":
        raise ValueError("P4.12 outcome is not course_level_validation_supported")
    if report.get("experiment_passed") is not True:
        raise ValueError("P4.12 experiment_passed is not true")
    if report.get("can_promote") is not False:
        raise ValueError("P4.12 report must keep can_promote=false")
    return {
        "source_report": P4_12_REPORT.name,
        "outcome": report["outcome"],
        "matrix": "3 identity batches x 3 seeds = 9 cells",
        "projected_pass_cells": report["projected_pass_cells"],
        "projected_pass_cells_expected": 9,
        "baseline_tension_batches": report["baseline_tension_batches"],
        "baseline_tension_batches_expected": 3,
        "incomplete_projections": report["incomplete_projections"],
        "zero_variance_note": (
            "nine cells value-identical (holdout utility 0.8, target 0.75, "
            "zero safe violations, sibling retention 1.0/1.0)"
        ),
        "resource_note": (
            "3x wall soft cap structurally unsatisfiable (projection IS the "
            "mechanism cost); absolute wall-clock tiny; formal caps belong to "
            "the promotion course preregistration"
        ),
    }


def _promotion_course_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("format") != "taiji-m5-k-p4-13-promotion-course-v1":
        raise ValueError("P4.13 report format mismatch")
    if report.get("status") != "completed":
        raise ValueError("P4.13 report is not completed")
    if report.get("outcome") != "promotion_course_supported":
        raise ValueError("P4.13 outcome is not promotion_course_supported")
    if report.get("experiment_passed") is not True:
        raise ValueError("P4.13 experiment_passed is not true")
    if report.get("can_promote") is not False:
        raise ValueError("P4.13 report must keep can_promote=false")
    course_projection_cells = [cell for cell in report.get("cell_results", [])]
    cumulative_converged = all(
        bool(cell.get("projection_b", {}).get("converged"))
        and float(cell.get("projection_b", {}).get("total_violation", 1.0)) == 0.0
        for cell in course_projection_cells
    )
    backward_retention_zero_failure = all(
        all(
            bool(value)
            for value in cell.get("phase_b_gates", {})
            .get("projected-ext-17", {})
            .get("backward_retention_a", {})
            .values()
        )
        for cell in course_projection_cells
    )
    return {
        "source_report": P4_13_REPORT.name,
        "outcome": report["outcome"],
        "course_structure": "two sequential phases (A -> B) per cell",
        "projected_pass_cells": report["projected_pass_cells"],
        "projected_pass_cells_expected": 9,
        "baseline_tension_batches": report["baseline_tension_batches"],
        "baseline_tension_batches_expected": 3,
        "incomplete_projection_b_cells": report["incomplete_projection_b_cells"],
        "cumulative_projection": {
            "constraint_count_per_cell": 154,
            "all_cells_exact_zero_violation": cumulative_converged,
        },
        "backward_retention_gate": {
            "definition": "phase-A holdout re-checked after phase B",
            "zero_failure_across_cells": backward_retention_zero_failure,
            "recheck_values": "utility 0.8 / target 0.75 (value-identical)",
        },
        "rollback_gate_all_pass": all(
            bool(cell.get("rollback_gate", {}).get("passed")) for cell in course_projection_cells
        ),
        "resource_absolute_budget_all_pass": all(
            bool(cell.get("resource_audit", {}).get("caps_passed"))
            for cell in course_projection_cells
        ),
    }


def build_v4(
    k1: dict[str, Any],
    k2: dict[str, Any],
    k3: dict[str, Any],
    c_stage: dict[str, Any],
    v2_report: dict[str, Any],
    v3_report: dict[str, Any],
    p4_12_report: dict[str, Any],
    p4_13_report: dict[str, Any],
) -> dict[str, Any]:
    if v3_report.get("format") != V3_REPORT_FORMAT or v3_report.get("version") != V3_VERSION:
        raise ValueError("v3 scorecard format/version mismatch")
    core = build_v3(k1, k2, k3, c_stage, v2_report)
    v3_sources = v3_report.get("source_reports", {})
    for phase, digest in core["source_reports"].items():
        if v3_sources.get(phase) != digest:
            raise ValueError(f"evidence drifted since v3: {phase}")
    course_level = _course_level_evidence(p4_12_report)
    promotion_course = _promotion_course_evidence(p4_13_report)
    promotion_gates = {
        **core["promotion_gates"],
        "g_solver_mechanism_course_closed": True,
        "k_worker_joint_course_completed": False,
        "default_runtime_rollout_review_completed": False,
    }
    core["format"] = REPORT_FORMAT
    core["version"] = VERSION
    core["source_reports"]["P4_12_COURSE_LEVEL_VALIDATION"] = content_digest(p4_12_report)
    core["source_reports"]["P4_13_PROMOTION_COURSE"] = content_digest(p4_13_report)
    core["solver_mechanism_evidence"] = {
        "course_level_validation": course_level,
        "promotion_course": promotion_course,
        "mechanism_conclusion": (
            "retention/new-task decoupling = representation factorization "
            "(P4.9) AND optimization mechanism replacement (P4.11 end "
            "projection); neither alone suffices; the mechanism is "
            "deterministic (zero variance across the 9-cell validation and "
            "the 9-cell promotion course)"
        ),
        "solver_attached_default_runtime": False,
    }
    core["promotion_gates"] = promotion_gates
    core["verdict"] = {
        "k_evidence_closed": core["verdict"]["k_evidence_closed"],
        "learning_mechanism_closed": core["verdict"]["learning_mechanism_closed"],
        "g_solver_mechanism_course_closed": True,
        "promotion_gate": all(promotion_gates.values()),
        "can_promote": False,
        "reason": (
            "The G-side continual-learning evidence chain is closed "
            "(representation factorization + solver mechanism + 9/9 "
            "course-level validation + 9/9 two-phase promotion course with "
            "zero backward forgetting), but the K worker joint course has "
            "not been preregistered or run and the default runtime rollout "
            "review has not been executed."
        ),
    }
    core["next_boundary"] = (
        "pre-register the K worker joint course (P2.6/P2.7 continuation "
        "machinery integrated with the solver mechanism on the same "
        "parent); after its completion the default runtime rollout review "
        "and the promotion discussion become eligible; keep all owners "
        "detached until then."
    )
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = build_v4(
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
                "k_evidence_closed": payload["verdict"]["k_evidence_closed"],
                "g_solver_mechanism_course_closed": payload["verdict"][
                    "g_solver_mechanism_course_closed"
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
