"""Build the content-addressed M5 K-axis v2 scorecard.

This is an audit-only reducer over the already completed K1/K2/K3 formal
reports.  It never retrains a learner, reruns a cell, or promotes a shadow
owner.  The v2 contract is frozen in
``plans/reference/M5_K_AXIS_SCORECARD_V2_CONTRACT_20260909.md``.
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

from scripts.training.audit_taiji_m5_k_scorecard import (  # noqa: E402
    _read_report,
    _snapshot,
)
from taiji.continual_evaluation import (  # noqa: E402
    ContinualScorecard,
    MetricSpec,
)
from taiji.internalization import content_digest  # noqa: E402

K1_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k1_skill_composition_formal_20260909.json"
K2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k2_multistep_formal_20260909.json"
K3_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k3_outcome_dependency_formal_20260909.json"
REPORT_FORMAT = "taiji-m5-k-axis-scorecard-v2"
VERSION = 2


def _robust(report: dict[str, Any]) -> bool:
    return bool(report.get("verdict", {}).get("robust")) and report.get("status") == "passed"


def build_scorecard(
    k1: dict[str, Any],
    k2: dict[str, Any],
    k3: dict[str, Any],
) -> dict[str, Any]:
    k1_aggregate = k1["aggregate"]
    k2_aggregate = k2["aggregate"]
    k3_aggregate = k3["aggregate"]
    specs = (
        MetricSpec(
            name="k1_single_step_composition_success",
            direction="higher",
            unit="episode_ratio",
            baseline_kind="absolute",
            domain_id="M5.K1",
        ),
        MetricSpec(
            name="k2_multistep_composition_success",
            direction="higher",
            unit="episode_ratio",
            baseline_kind="absolute",
            domain_id="M5.K2",
        ),
        MetricSpec(
            name="k3_outcome_dependency_success",
            direction="higher",
            unit="episode_ratio",
            baseline_kind="absolute",
            domain_id="M5.K3",
        ),
    )
    snapshots = (
        _snapshot(
            report=k1,
            phase_id="M5.K1",
            metric_name="k1_single_step_composition_success",
            absolute_value=float(k1_aggregate["a_unseen"]["mean"]),
        ),
        _snapshot(
            report=k2,
            phase_id="M5.K2",
            metric_name="k2_multistep_composition_success",
            absolute_value=float(k2_aggregate["a_holdout"]["mean"]),
        ),
        _snapshot(
            report=k3,
            phase_id="M5.K3",
            metric_name="k3_outcome_dependency_success",
            absolute_value=float(k3_aggregate["a_holdout"]["mean"]),
        ),
    )
    scorecard = ContinualScorecard(metric_specs=specs, snapshots=snapshots)
    robustness = {
        "M5.K1": _robust(k1),
        "M5.K2": _robust(k2),
        "M5.K3": _robust(k3),
    }
    parent_retention_missing = all(
        report.get("metric_contract", {}).get("parent_retention") is None
        for report in (k1, k2, k3)
    )
    standalone_shadow = all(
        "standalone" in report.get("metric_contract", {}).get("parent_retention_note", "")
        for report in (k1, k2, k3)
    )
    promotion_gates = {
        "k1_formal_robust": robustness["M5.K1"],
        "k2_formal_robust": robustness["M5.K2"],
        "k3_formal_robust": robustness["M5.K3"],
        "scorecard_content_addressed": bool(scorecard.content_digest),
        "parent_retention_baseline_present": not parent_retention_missing,
        "default_runtime_owner_attached": False,
        "same_parent_continual_s_g_k_evidence": False,
        "resource_rollback_old_capability_gate": False,
    }
    return {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "source_reports": {
            "M5.K1": content_digest(k1),
            "M5.K2": content_digest(k2),
            "M5.K3": content_digest(k3),
        },
        "scorecard": scorecard.to_payload(),
        "scorecard_digest": scorecard.content_digest,
        "absolute_capabilities": {
            "M5.K1": k1_aggregate["a_unseen"],
            "M5.K2": k2_aggregate["a_holdout"],
            "M5.K3": k3_aggregate["a_holdout"],
        },
        "comparison_evidence": {
            "M5.K1": {
                "A_minus_B": k1_aggregate["a_minus_b"],
                "A_minus_C": k1_aggregate["a_minus_c"],
            },
            "M5.K2": {
                "A_minus_B": k2_aggregate["a_minus_b"],
                "A_minus_C": k2_aggregate["a_minus_c"],
            },
            "M5.K3": {
                "A_minus_B": k3_aggregate["a_minus_b"],
                "A_minus_C": k3_aggregate["a_minus_c"],
                "probe_admission_cells_1p0": k3_aggregate[
                    "probe_admission_cells_1p0"
                ],
                "lineage_admission_cells_1p0": k3_aggregate[
                    "lineage_admission_cells_1p0"
                ],
                "feedback_reward_variance": k3_aggregate[
                    "feedback_reward_variance"
                ],
            },
        },
        "evidence_gates": {
            "k1_evidence_closed": robustness["M5.K1"],
            "k2_evidence_closed": robustness["M5.K2"],
            "k3_evidence_closed": robustness["M5.K3"],
            "k_axis_evidence_closed": all(robustness.values()),
            "parent_retention_missing": parent_retention_missing,
            "standalone_shadow": standalone_shadow,
        },
        "promotion_gates": promotion_gates,
        "verdict": {
            "k_evidence_closed": all(robustness.values()),
            "promotion_gate": all(promotion_gates.values()),
            "can_promote": False,
            "reason": (
                "K1/K2/K3 close the standalone composition and outcome-dependency "
                "evidence line, but all three lack same-parent retention and "
                "default-runtime ownership."
            ),
        },
        "next_boundary": (
            "pre-register a same-parent A8/R6 promotion course with resource, "
            "rollback, and old-capability retention gates; keep K1/K2/K3 shadow "
            "owners detached."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k1-report", type=Path, default=K1_REPORT)
    parser.add_argument("--k2-report", type=Path, default=K2_REPORT)
    parser.add_argument("--k3-report", type=Path, default=K3_REPORT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = build_scorecard(
        _read_report(args.k1_report),
        _read_report(args.k2_report),
        _read_report(args.k3_report),
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
                "promotion_gate": payload["verdict"]["promotion_gate"],
                "can_promote": payload["verdict"]["can_promote"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

