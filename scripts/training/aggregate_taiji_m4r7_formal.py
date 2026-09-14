"""Aggregate the bounded M4.R7 formal predictive update-scale canary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (11, 29, 47)
EXPECTED_VARIANTS = ("scale_0p0", "scale_0p5", "scale_1p0")
EXPECTED_SCALE_VALUES = (0.0, 0.5, 1.0)
FORMAL_PROFILE = "foundation"
FORMAL_TRAIN_BYTES = 65_536
FORMAL_EVAL_BYTES = 16_384
FORMAT = "taiji-m4r7-formal-aggregate-v1"


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _summary(reports: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    metrics = [report["variants"][variant]["metrics"] for report in reports]
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
                "c2_cycle3_delta_bpb": float(
                    report["variants"][variant]["metrics"]["c2_cycle3_delta_bpb"]
                ),
            }
            for report in reports
        },
    }


def aggregate(
    reports: list[dict[str, Any]],
    preflight: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    if tuple(sorted(int(report["seed"]) for report in reports)) != EXPECTED_SEEDS:
        raise ValueError("M4.R7 aggregate requires seed11, seed29, and seed47")
    checks = {
        f"seed{report['seed']}_technical_gate": bool(report["technical_gate_all_passed"])
        for report in reports
    }
    checks["all_reports_not_promoted"] = all(report["can_promote"] is False for report in reports)
    checks["same_variant_set"] = all(
        set(report["variants"]) == set(EXPECTED_VARIANTS) for report in reports
    )
    checks["same_formal_budget"] = all(
        report["configuration"]["profile"] == FORMAL_PROFILE
        and report["configuration"]["train_bytes"] == FORMAL_TRAIN_BYTES
        and report["configuration"]["eval_bytes"] == FORMAL_EVAL_BYTES
        and report["configuration"]["fixed_capacity"] is True
        and report["configuration"]["learn_fabric"] is False
        for report in reports
    )
    checks["all_scale_zero_controls_frozen"] = all(
        report["checks"]["scale_zero_context_unchanged"]
        and report["checks"]["scale_zero_readout_unchanged"]
        and report["checks"]["scale_zero_fabric_unchanged"]
        and report["checks"]["scale_zero_memory_unchanged"]
        for report in reports
    )
    checks["all_variants_checkpoint_round_trip"] = all(
        all(
            report["checks"][f"{variant}_checkpoint_round_trip"]
            and report["checks"][f"{variant}_artifact_source_digest"]
            for variant in EXPECTED_VARIANTS
        )
        for report in reports
    )
    checks["preflight_passed"] = bool(
        preflight["status"] == "passed"
        and preflight["formal_allowed"] is True
        and preflight["technical_gate_all_passed"] is True
    )
    variants = {variant: _summary(reports, variant) for variant in EXPECTED_VARIANTS}
    half = variants["scale_0p5"]
    formal_gate_passed = bool(
        all(checks.values())
        and half["c3_gain_positive_count"] == len(EXPECTED_SEEDS)
        and half["retention_gate_passed_all"]
    )
    report = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": time.time(),
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "formal_gate_passed": formal_gate_passed,
        "seeds": list(EXPECTED_SEEDS),
        "source_reports": [str(item.get("report_path", "")) for item in reports],
        "preflight_report": str(preflight.get("report_path", "")),
        "configuration": {
            "profile": FORMAL_PROFILE,
            "train_bytes": FORMAL_TRAIN_BYTES,
            "eval_bytes": FORMAL_EVAL_BYTES,
            "scales": list(EXPECTED_SCALE_VALUES),
            "fixed_capacity": True,
        },
        "variants": variants,
        "checks": checks,
        "technical_gate_all_passed": all(checks.values()),
        "diagnosis": {
            "half_scale_formal_candidate": formal_gate_passed,
            "half_scale_improves_mean_gain": (
                half["c3_gain_mean_bpb"] > variants["scale_1p0"]["c3_gain_mean_bpb"]
            ),
            "half_scale_retention_gate_passed": half["retention_gate_passed_all"],
            "interpretation": (
                "scale 0.5 passes the bounded formal Gate; keep the result "
                "unpromoted until a review decision"
                if formal_gate_passed
                else "bounded formal Gate failed; do not promote scale 0.5 "
                "and return to course/update-rule diagnosis"
            ),
        },
        "decision_boundary": {
            "promotion_allowed": False,
            "next_action": (
                "review formal evidence before any promotion"
                if formal_gate_passed
                else "withdraw formal candidate and diagnose the failed Gate"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        nargs=3,
        type=Path,
        default=[
            PROJECT_ROOT / "reports" / "taiji_m4r7_formal_seed11_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r7_formal_seed29_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r7_formal_seed47_20260908.json",
        ],
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r7_formal_preflight_20260908.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r7_formal_aggregate_20260908.json",
    )
    args = parser.parse_args(argv)
    reports: list[dict[str, Any]] = []
    for path in args.reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["report_path"] = str(path)
        reports.append(payload)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    preflight["report_path"] = str(args.preflight)
    report = aggregate(reports, preflight, args.output)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "status": report["status"],
                "technical_gate_all_passed": report["technical_gate_all_passed"],
                "formal_gate_passed": report["formal_gate_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
