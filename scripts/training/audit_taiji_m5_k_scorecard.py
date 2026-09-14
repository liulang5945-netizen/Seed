"""Audit the closed M5 K1/K2 evidence line without promoting architecture."""

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

from taiji.continual_evaluation import (
    ContinualEvaluationSnapshot,
    ContinualScorecard,
    MetricObservation,
    MetricSpec,
)
from taiji.internalization import content_digest

K1_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k1_skill_composition_formal_20260909.json"
K2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k2_multistep_formal_20260909.json"
REPORT_FORMAT = "taiji-m5-k-axis-scorecard-v1"
VERSION = 1


def _read_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must be an object: {path}")
    return payload


def _snapshot(
    *,
    report: dict[str, Any],
    phase_id: str,
    metric_name: str,
    absolute_value: float,
) -> ContinualEvaluationSnapshot:
    report_digest = content_digest(report)
    checkpoint_digest = content_digest({"source_report": report_digest, "phase_id": phase_id})
    input_digest = content_digest({"source_report": report_digest, "metric": metric_name})
    observation = MetricObservation(
        metric_name=metric_name,
        domain_id=phase_id,
        checkpoint_digest=checkpoint_digest,
        absolute_value=float(absolute_value),
        owner_id=f"m5:{phase_id}:standalone-shadow",
        read_only_input_digest=input_digest,
    )
    return ContinualEvaluationSnapshot(
        checkpoint_digest=checkpoint_digest,
        phase_id=phase_id,
        owner_graph_digest=content_digest({"phase_id": phase_id, "owner": "standalone-shadow"}),
        read_only_input_digest=input_digest,
        observations=(observation,),
        parent_checkpoint_digest=None,
        resource_cost=1.0,
    )


def build_scorecard(k1: dict[str, Any], k2: dict[str, Any]) -> dict[str, Any]:
    k1_aggregate = k1["aggregate"]
    k2_aggregate = k2["aggregate"]
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
    )
    scorecard = ContinualScorecard(metric_specs=specs, snapshots=snapshots)
    k1_robust = bool(k1.get("verdict", {}).get("robust")) and k1.get("status") == "passed"
    k2_robust = bool(k2.get("verdict", {}).get("robust")) and k2.get("status") == "passed"
    k_evidence_closed = k1_robust and k2_robust
    parent_retention_missing = (
        all(k1.get("metric_contract", {}).get("parent_retention") is None for _ in (0,))
        and k2.get("metric_contract", {}).get("parent_retention") is None
    )
    standalone_shadow = "standalone" in k1.get("metric_contract", {}).get(
        "parent_retention_note", ""
    ) and "standalone" in k2.get("metric_contract", {}).get("parent_retention_note", "")
    promotion_gates = {
        "k1_formal_robust": k1_robust,
        "k2_formal_robust": k2_robust,
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
        },
        "scorecard": scorecard.to_payload(),
        "scorecard_digest": scorecard.content_digest,
        "absolute_capabilities": {
            "M5.K1": k1_aggregate["a_unseen"],
            "M5.K2": k2_aggregate["a_holdout"],
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
        },
        "evidence_gates": {
            "k1_evidence_closed": k1_robust,
            "k2_evidence_closed": k2_robust,
            "k_axis_evidence_closed": k_evidence_closed,
            "parent_retention_missing": parent_retention_missing,
            "standalone_shadow": standalone_shadow,
        },
        "promotion_gates": promotion_gates,
        "verdict": {
            "k_evidence_closed": k_evidence_closed,
            "promotion_gate": all(promotion_gates.values()),
            "can_promote": False,
            "reason": (
                "K1/K2 close the standalone composition evidence line, but both lack a "
                "same-parent retention baseline and default-runtime ownership."
            ),
        },
        "next_boundary": (
            "pre-register either K3 outcome-to-world/task-dependency evidence or a same-parent "
            "A8/R6 S-G-K promotion course; keep K2 shadow detached."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k1-report", type=Path, default=K1_REPORT)
    parser.add_argument("--k2-report", type=Path, default=K2_REPORT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = build_scorecard(
        _read_report(args.k1_report),
        _read_report(args.k2_report),
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
                "can_promote": payload["verdict"]["can_promote"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
