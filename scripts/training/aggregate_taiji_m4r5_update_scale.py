"""Aggregate the M4.R5 three-seed predictive update-scale pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (11, 29, 47)
EXPECTED_VARIANTS = ("scale_0p0", "scale_0p5", "scale_1p0")
FORMAT = "taiji-m4r5-update-scale-aggregate-v1"


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summary(reports: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    metrics = [report["variants"][variant]["metrics"] for report in reports]
    gains = [float(item["c3_holdout_gain_bpb"]) for item in metrics]
    cycle2 = [float(item["c_cycle2_delta_bpb"]) for item in metrics]
    cycle3 = [float(item["c_cycle3_delta_bpb"]) for item in metrics]
    return {
        "c3_gain_mean_bpb": _mean(gains),
        "c3_gain_min_bpb": min(gains),
        "c3_gain_max_bpb": max(gains),
        "c3_gain_positive_count": sum(value > 0.0 for value in gains),
        "c_cycle2_delta_mean_bpb": _mean(cycle2),
        "c_cycle2_degradation_count": sum(value > 0.0 for value in cycle2),
        "c_cycle3_delta_mean_bpb": _mean(cycle3),
        "c_cycle3_degradation_count": sum(value > 0.0 for value in cycle3),
        "retention_gate_passed_all": all(value <= 0.0 for value in (*cycle2, *cycle3)),
        "seed_metrics": {
            str(report["seed"]): {
                "c3_gain_bpb": float(report["variants"][variant]["metrics"]["c3_holdout_gain_bpb"]),
                "c_cycle2_delta_bpb": float(
                    report["variants"][variant]["metrics"]["c_cycle2_delta_bpb"]
                ),
                "c_cycle3_delta_bpb": float(
                    report["variants"][variant]["metrics"]["c_cycle3_delta_bpb"]
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
    checks["all_reports_not_promoted"] = all(
        report["can_promote"] is False for report in reports
    )
    checks["same_variant_set"] = all(
        set(report["variants"]) == set(EXPECTED_VARIANTS) for report in reports
    )
    checks["all_scale_zero_controls_frozen"] = all(
        report["checks"]["scale_zero_context_unchanged"]
        and report["checks"]["scale_zero_readout_unchanged"]
        for report in reports
    )
    variants = {variant: _summary(reports, variant) for variant in EXPECTED_VARIANTS}
    half = variants["scale_0p5"]
    legacy = variants["scale_1p0"]
    half_scale_supported = bool(
        half["c3_gain_positive_count"] == len(EXPECTED_SEEDS)
        and half["retention_gate_passed_all"]
    )
    report = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": time.time(),
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "seeds": list(EXPECTED_SEEDS),
        "source_reports": [str(item.get("report_path", "")) for item in reports],
        "variants": variants,
        "checks": checks,
        "technical_gate_all_passed": all(checks.values()),
        "diagnosis": {
            "half_scale_supported": half_scale_supported,
            "half_scale_improves_mean_gain": (
                half["c3_gain_mean_bpb"] > legacy["c3_gain_mean_bpb"]
            ),
            "half_scale_reduces_c_cycle2_degradation": (
                half["c_cycle2_degradation_count"]
                < legacy["c_cycle2_degradation_count"]
            ),
            "interpretation": (
                "half-rate improves the cohort mean and reduces C cycle2 degradation, "
                "but it does not satisfy the strict all-seed retention Gate; do not promote"
                if not half_scale_supported
                else "half-rate satisfies the pilot Gate; larger-budget formal validation remains required"
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
            PROJECT_ROOT / "reports/taiji_m4r5_update_scale_pilot_seed11_20260908.json",
            PROJECT_ROOT / "reports/taiji_m4r5_update_scale_pilot_seed29_20260908.json",
            PROJECT_ROOT / "reports/taiji_m4r5_update_scale_pilot_seed47_20260908.json",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/taiji_m4r5_update_scale_pilot_aggregate_20260908.json",
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
                "half_scale_supported": report["diagnosis"]["half_scale_supported"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
