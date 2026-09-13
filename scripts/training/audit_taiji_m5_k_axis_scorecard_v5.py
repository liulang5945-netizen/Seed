"""Build the content-addressed M5 K-axis v5 scorecard.

Audit-only reducer: the K1/K2/K3 sections, the C-stage learning-mechanism
evidence and the G-side solver-mechanism evidence reuse the frozen v4
reducer verbatim and are cross-checked digest-for-digest against the v4
report; the only new evidence line is the K-worker joint course (P4.14),
transcribed read-only, which flips ``k_worker_joint_course_completed`` to
``true``.  Never retrains, reruns a cell, or promotes any owner.  Contract:
``plans/reference/M5_K_AXIS_SCORECARD_V5_CONTRACT_20260911.md``.
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
    build_v4,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v4 import (
    REPORT_FORMAT as V4_REPORT_FORMAT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v4 import (
    VERSION as V4_VERSION,
)
from scripts.training.audit_taiji_m5_k_scorecard import _read_report  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

P4_14_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_14_joint_course_20260911.json"
V4_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v4_20260911.json"
REPORT_FORMAT = "taiji-m5-k-axis-scorecard-v5"
VERSION = 5


def _joint_course_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("format") != "taiji-m5-k-p4-14-joint-course-v1":
        raise ValueError("P4.14 report format mismatch")
    if report.get("status") != "completed":
        raise ValueError("P4.14 report is not completed")
    if report.get("outcome") != "joint_course_supported":
        raise ValueError("P4.14 outcome is not joint_course_supported")
    if report.get("experiment_passed") is not True:
        raise ValueError("P4.14 experiment_passed is not true")
    if report.get("can_promote") is not False:
        raise ValueError("P4.14 report must keep can_promote=false")
    expected = {
        "passing_cells": 4,
        "phase_k_failed_cells": 0,
        "phase_g_failed_cells": 0,
        "cross_phase_failed_cells": 0,
        "incomplete_projection_cells": 0,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"P4.14 {key} is not {value}")
    cells = report.get("cell_results", [])
    cross_phase_all_pass = all(
        all(bool(value) for value in cell.get("cross_phase_gates", {}).values())
        for cell in cells
    )
    if not cross_phase_all_pass:
        raise ValueError("P4.14 cross-phase gates are not fully passed")
    resource_all_pass = all(
        bool(cell.get("resource_audit", {}).get("caps_passed")) for cell in cells
    )
    if not resource_all_pass:
        raise ValueError("P4.14 resource caps are not fully passed")
    first_k = cells[0]["phase_k"]
    post_k_digests = first_k["post_k_digests"]
    pre_k = {
        "k1": "12851b4b3cc94181",
        "k2": "411ef5348231f355",
    }
    landscape_shifted = not post_k_digests["k1"].startswith(
        pre_k["k1"]
    ) and not post_k_digests["k2"].startswith(pre_k["k2"])
    if not landscape_shifted:
        raise ValueError("P4.14 post-K landscape did not shift from the pre-K workers")
    return {
        "source_report": P4_14_REPORT.name,
        "outcome": report["outcome"],
        "course_structure": (
            "per cell: Phase K (P2.6 mechanics verbatim: p4-14 novel K2-content "
            "cohort + 50-example P2 rehearsal interleave) -> re-materialize "
            "Phase G cohorts from the post-K workers -> Phase G (P4.11 contract "
            "verbatim: task fit + margin-preservation hinge + terminal joint "
            "projection)"
        ),
        "matrix": "2 identity batches x 2 seeds = 4 cells",
        "passing_cells": report["passing_cells"],
        "phase_k_summary": {
            "novel_k2_content_validation": "2/2",
            "old_class_k1_k2_content": "4/4",
            "safe_abstention": "6/6",
            "parameter_count": 5648,
            "k_checkpoint_independent_restore": True,
        },
        "phase_g_summary": {
            "holdout": "target 0.8 / utility 0.8675 / zero safe violations",
            "retention_sibling": "target 1.0 / utility 1.0",
            "retention_newtask": "target 0.8 / utility 0.8675",
            "projection": "88 constraints, exact zero violation in every cell",
            "birth_equivalence": "exact (zero mismatches, zero score deviation)",
        },
        "cross_phase_gates": {
            "k_unchanged_after_g": True,
            "g_preservation_vs_frozen_parent": True,
            "birth_equivalence": True,
        },
        "landscape_shift_note": (
            "post-K K1/K2 digests moved from the pre-K workers (the "
            "interaction surface is real) and the G solver maintained its "
            "balance on the shifted landscape; post-K workers are "
            "bit-identical across batches because the K feature space does "
            "not encode path/project identity (honest structural note)"
        ),
        "resource_absolute_budget_all_pass": resource_all_pass,
        "k_worker_joint_course_completed": True,
    }


def build_v5(
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
) -> dict[str, Any]:
    if (
        v4_report.get("format") != V4_REPORT_FORMAT
        or v4_report.get("version") != V4_VERSION
    ):
        raise ValueError("v4 scorecard format/version mismatch")
    core = build_v4(
        k1, k2, k3, c_stage, v2_report, v3_report, p4_12_report, p4_13_report
    )
    v4_sources = v4_report.get("source_reports", {})
    for phase, digest in core["source_reports"].items():
        if v4_sources.get(phase) != digest:
            raise ValueError(f"evidence drifted since v4: {phase}")
    joint_course = _joint_course_evidence(p4_14_report)
    promotion_gates = {
        **core["promotion_gates"],
        "k_worker_joint_course_completed": True,
    }
    core["format"] = REPORT_FORMAT
    core["version"] = VERSION
    core["source_reports"]["P4_14_JOINT_COURSE"] = content_digest(p4_14_report)
    core["joint_course_evidence"] = joint_course
    core["promotion_gates"] = promotion_gates
    core["verdict"] = {
        "k_evidence_closed": core["verdict"]["k_evidence_closed"],
        "learning_mechanism_closed": core["verdict"]["learning_mechanism_closed"],
        "g_solver_mechanism_course_closed": True,
        "k_worker_joint_course_completed": True,
        "promotion_gate": all(promotion_gates.values()),
        "can_promote": False,
        "reason": (
            "The K worker joint course is complete: P2.6 continuation "
            "mechanics followed by the P4.11 solver contract on the post-K "
            "landscape of the same parent, 4/4 cells passing every phase and "
            "cross-phase gate. The only remaining entry condition for the A8 "
            "promotion review is the default runtime rollout review; all "
            "other vetoes keep promotion_gate=false."
        ),
    }
    core["next_boundary"] = (
        "pre-register the default runtime rollout review (the last remaining "
        "entry condition for the A8 promotion review); keep all owners "
        "detached and the default runtime untouched until the review is "
        "approved."
    )
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = build_v5(
        _read_report(
            Path(
                PROJECT_ROOT
                / "reports"
                / "taiji_m5_k1_skill_composition_formal_20260909.json"
            )
        ),
        _read_report(
            Path(
                PROJECT_ROOT / "reports" / "taiji_m5_k2_multistep_formal_20260909.json"
            )
        ),
        _read_report(
            Path(
                PROJECT_ROOT
                / "reports"
                / "taiji_m5_k3_outcome_dependency_formal_20260909.json"
            )
        ),
        _read_report(C_STAGE_REPORT),
        _read_report(
            PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v2_20260909.json"
        ),
        _read_report(V3_REPORT),
        _read_report(P4_12_REPORT),
        _read_report(P4_13_REPORT),
        _read_report(P4_14_REPORT),
        _read_report(V4_REPORT),
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
                "promotion_gate": payload["verdict"]["promotion_gate"],
                "can_promote": payload["verdict"]["can_promote"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
