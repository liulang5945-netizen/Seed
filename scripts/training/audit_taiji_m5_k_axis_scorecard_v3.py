"""Build the content-addressed M5 K-axis v3 scorecard.

Audit-only reducer: the K1/K2/K3 sections reuse the frozen v2 reducer
mechanics verbatim and are cross-checked digest-for-digest against the v2
report; the only new evidence line is the C-stage learning-mechanism
conclusion (formal v2 on sealed v5), transcribed read-only.  Never
retrains, reruns a cell, or promotes any owner.  Contract:
``plans/reference/M5_K_AXIS_SCORECARD_V3_CONTRACT_20260910.md``.
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

from scripts.training.audit_taiji_m5_k_axis_scorecard_v2 import (  # noqa: E402
    K1_REPORT,
    K2_REPORT,
    K3_REPORT,
    build_scorecard,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v2 import (
    REPORT_FORMAT as V2_REPORT_FORMAT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v2 import (
    VERSION as V2_VERSION,
)
from scripts.training.audit_taiji_m5_k_scorecard import (  # noqa: E402
    _read_report,
)
from taiji.internalization import content_digest  # noqa: E402

C_STAGE_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_c_stage_formal_v2_20260910.json"
)
V2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v2_20260909.json"
REPORT_FORMAT = "taiji-m5-k-axis-scorecard-v3"
VERSION = 3
GATE_KEYS = (
    "G1_learning_floor",
    "G2_catastrophe_bound",
    "G3_weak_class_primary",
    "G4_overall_non_inferiority",
)


def _learning_mechanism_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("format") != "taiji-m4v2-c-stage-formal-v1":
        raise ValueError("C-stage report format mismatch")
    if report.get("status") != "passed":
        raise ValueError("C-stage formal v2 did not pass")
    if not report.get("verdict", {}).get("formal_passed"):
        raise ValueError("C-stage formal v2 verdict is not formal_passed")
    if report.get("can_promote") is not False:
        raise ValueError("C-stage report must keep can_promote=false")
    gates = report.get("gates", {})
    gate_flags = {key: bool(gates.get(key, {}).get("passed")) for key in GATE_KEYS}
    if not all(gate_flags.values()):
        raise ValueError("C-stage formal v2 gates are not all passed")
    aggregation = report["course_aggregation"]
    input_verification = report["input_verification"]
    epsilon = report["pre_sealed"]["epsilon_derivation"]
    return {
        "source_report": C_STAGE_REPORT.name,
        "formal_status": report["status"],
        "sealed_task_seed": input_verification["sealed_v5_task_seed"],
        "sealed_sha256": input_verification["sealed_v5_sha256"],
        "epsilon_cat": epsilon["epsilon_cat"]["epsilon_cat"],
        "epsilon_ni": epsilon["epsilon_ni"]["epsilon_ni"],
        "gates_passed": gate_flags,
        "fs_mechanism_status": "default-learning-mechanism-candidate",
        "fs_mechanism_status_source": (
            "C-stage formal v2 preregistration section 5 frozen outcome mapping"
        ),
        "descriptive": {
            "statistics_unit": aggregation["statistics_unit"],
            "sealed_course_means": {
                "c": aggregation["sealed_c_course_means"],
                "fs": aggregation["sealed_fs_course_means"],
            },
            "weak_class_course_means": aggregation["weak_class_course_means"],
            "courses_won_by_fs_weak_class": aggregation[
                "courses_won_by_fs_weak_class"
            ],
        },
        "fs_attached_default_runtime": False,
    }


def build_v3(
    k1: dict[str, Any],
    k2: dict[str, Any],
    k3: dict[str, Any],
    c_stage: dict[str, Any],
    v2_report: dict[str, Any],
) -> dict[str, Any]:
    if v2_report.get("format") != V2_REPORT_FORMAT or v2_report.get(
        "version"
    ) != V2_VERSION:
        raise ValueError("v2 scorecard format/version mismatch")
    core = build_scorecard(k1, k2, k3)
    v2_sources = v2_report.get("source_reports", {})
    for phase, digest in core["source_reports"].items():
        if v2_sources.get(phase) != digest:
            raise ValueError(f"K1/K2/K3 evidence drifted since v2: {phase}")
    evidence = _learning_mechanism_evidence(c_stage)
    promotion_gates = {
        **core["promotion_gates"],
        "learning_mechanism_candidate_selected": True,
        "learning_mechanism_attached_default_runtime": False,
    }
    core["format"] = REPORT_FORMAT
    core["version"] = VERSION
    core["source_reports"]["C_STAGE_LEARNING_MECHANISM"] = content_digest(c_stage)
    core["learning_mechanism_evidence"] = evidence
    core["promotion_gates"] = promotion_gates
    core["verdict"] = {
        "k_evidence_closed": core["verdict"]["k_evidence_closed"],
        "learning_mechanism_closed": True,
        "promotion_gate": all(promotion_gates.values()),
        "can_promote": False,
        "reason": (
            "K1/K2/K3 close the standalone evidence line (unchanged since v2, "
            "digest-verified) and C-stage formal v2 selects fast/slow+replay "
            "as the K-phase default learning-mechanism candidate, but the "
            "mechanism is not attached to the default runtime and there is "
            "still no same-parent continual S/G/K, resource, rollback, or "
            "old-capability evidence."
        ),
    }
    core["next_boundary"] = (
        "pre-register a same-parent promotion course whose learning mechanism "
        "is the selected fast/slow+replay candidate, with resource "
        "equivalence, rollback, and old-capability retention gates; keep all "
        "K shadow owners detached until that preregistration is frozen."
    )
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k1-report", type=Path, default=K1_REPORT)
    parser.add_argument("--k2-report", type=Path, default=K2_REPORT)
    parser.add_argument("--k3-report", type=Path, default=K3_REPORT)
    parser.add_argument("--c-stage-report", type=Path, default=C_STAGE_REPORT)
    parser.add_argument("--v2-report", type=Path, default=V2_REPORT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = build_v3(
        _read_report(args.k1_report),
        _read_report(args.k2_report),
        _read_report(args.k3_report),
        _read_report(args.c_stage_report),
        _read_report(args.v2_report),
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
                "learning_mechanism_closed": payload["verdict"][
                    "learning_mechanism_closed"
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
