"""Aggregate the M4.R3 fixed-capacity attribution smoke reports."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (11, 29, 47)
EXPECTED_ARMS = ("readout_only", "context_only", "joint")
FORMAT = "taiji-m4r3-update-interference-aggregate-v1"


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summary(reports: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    metrics = [report["variants"][arm]["metrics"] for report in reports]
    gains = [float(item["c3_holdout_gain_bpb"]) for item in metrics]
    cycle2 = [float(item["c_cycle2_delta_bpb"]) for item in metrics]
    cycle3 = [float(item["c_cycle3_delta_bpb"]) for item in metrics]
    c2_cycle3 = [float(item["c2_cycle3_delta_bpb"]) for item in metrics]
    return {
        "c3_gain_mean_bpb": _mean(gains),
        "c3_gain_min_bpb": min(gains),
        "c3_gain_max_bpb": max(gains),
        "c3_gain_positive_count": sum(value > 0.0 for value in gains),
        "c_cycle2_delta_mean_bpb": _mean(cycle2),
        "c_cycle2_degradation_count": sum(value > 0.0 for value in cycle2),
        "c_cycle3_delta_mean_bpb": _mean(cycle3),
        "c_cycle3_degradation_count": sum(value > 0.0 for value in cycle3),
        "c2_cycle3_delta_mean_bpb": _mean(c2_cycle3),
        "c2_cycle3_degradation_count": sum(value > 0.0 for value in c2_cycle3),
        "retention_gate_passed_all": all(
            value <= 0.0 for value in (*cycle2, *cycle3)
        ),
        "seed_metrics": {
            str(report["seed"]): {
                "c3_gain_bpb": float(report["variants"][arm]["metrics"]["c3_holdout_gain_bpb"]),
                "c_cycle2_delta_bpb": float(
                    report["variants"][arm]["metrics"]["c_cycle2_delta_bpb"]
                ),
                "c_cycle3_delta_bpb": float(
                    report["variants"][arm]["metrics"]["c_cycle3_delta_bpb"]
                ),
            }
            for report in reports
        },
    }


def aggregate(reports: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    if tuple(sorted(int(report["seed"]) for report in reports)) != EXPECTED_SEEDS:
        raise ValueError("aggregate requires exactly seed11, seed29, and seed47 reports")
    checks = {
        f"seed{report['seed']}_technical_gate": bool(
            report["technical_gate_all_passed"]
        )
        for report in reports
    }
    checks["all_seed_reports_not_promoted"] = all(
        report["can_promote"] is False for report in reports
    )
    checks["all_seed_chains_record_disjoint"] = all(
        bool(report["checks"]["record_disjoint_chain"]) for report in reports
    )
    checks["same_arm_set"] = all(
        set(report["variants"]) == set(EXPECTED_ARMS) for report in reports
    )
    arms = {arm: _summary(reports, arm) for arm in EXPECTED_ARMS}
    joint = arms["joint"]
    joint_attribution_supported = bool(
        joint["c3_gain_positive_count"] == len(EXPECTED_SEEDS)
        and joint["retention_gate_passed_all"]
    )
    report = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": time.time(),
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "seeds": list(EXPECTED_SEEDS),
        "source_reports": [str(item.get("report_path", "")) for item in reports],
        "arms": arms,
        "checks": checks,
        "technical_gate_all_passed": all(checks.values()),
        "diagnosis": {
            "joint_attribution_supported": joint_attribution_supported,
            "interpretation": (
                "joint fixed-capacity updates reproduced positive C3 gain and no C "
                "cycle2/cycle3 degradation in all three smoke seeds; proceed to a "
                "larger-budget formal attribution comparison, without promotion"
                if joint_attribution_supported
                else "three-seed smoke does not support joint attribution; return to "
                "data, course, or update-rule diagnosis before formalization"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        nargs=3,
        type=Path,
        default=[
            PROJECT_ROOT / "reports" / "taiji_m4r3_update_interference_canary_seed11_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r3_update_interference_canary_seed29_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r3_update_interference_canary_seed47_20260908.json",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_m4r3_update_interference_aggregate_20260908.json",
    )
    args = parser.parse_args(argv)
    reports: list[dict[str, Any]] = []
    for path in args.reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["report_path"] = str(path)
        reports.append(payload)
    report = aggregate(reports, args.output)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "status": report["status"],
                "technical_gate_all_passed": report["technical_gate_all_passed"],
                "joint_attribution_supported": report["diagnosis"][
                    "joint_attribution_supported"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
