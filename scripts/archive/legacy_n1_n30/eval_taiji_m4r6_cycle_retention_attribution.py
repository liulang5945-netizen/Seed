"""Attribute M4.R5 retention changes by seed, update scale, and cycle.

This is a read-only report over the already completed M4.R5 three-seed pilot.
It deliberately does not load checkpoints or train a model.  The purpose is
to distinguish a single phase-boundary fluctuation from a systematic
update-scale failure before any formalization decision.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (11, 29, 47)
EXPECTED_VARIANTS = ("scale_0p0", "scale_0p5", "scale_1p0")
EXPECTED_CYCLES = ("cycle1", "cycle2", "cycle3")
FORMAT = "taiji-m4r6-cycle-retention-attribution-v1"
R4_AUDIT_FORMAT = "taiji-m4r4-course-shift-audit-v1"


def _float(value: Any) -> float:
    return float(value)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return payload


def _cycle_row(
    report: Mapping[str, Any],
    variant: str,
    *,
    frozen_difficulty: Mapping[str, float],
    owner_update: Mapping[str, Any] | None,
) -> dict[str, Any]:
    seed = int(report["seed"])
    payload = report["variants"][variant]
    scores = payload["scores"]
    if len(scores) != 3:
        raise ValueError(f"seed{seed} {variant} must contain three cycle scores")
    c = [_float(score["c_bpb"]) for score in scores]
    c2 = [_float(score["c2_bpb"]) for score in scores]
    c3 = [_float(score["c3_bpb"]) for score in scores]
    row = {
        "seed": seed,
        "variant": variant,
        "scale": _float(payload["scale"]),
        "c_holdout": {
            "cycle1_bpb": c[0],
            "cycle2_bpb": c[1],
            "cycle3_bpb": c[2],
        },
        "c2_holdout": {
            "cycle2_bpb": c2[1],
            "cycle3_bpb": c2[2],
        },
        "c3_holdout": {
            "cycle1_bpb": c3[0],
            "cycle2_bpb": c3[1],
            "cycle3_bpb": c3[2],
        },
        "c3_gain_bpb": _float(payload["metrics"]["c3_holdout_gain_bpb"]),
        "c_cycle2_delta_bpb": c[1] - c[0],
        "c_cycle3_delta_bpb": c[2] - c[0],
        "c2_cycle3_delta_bpb": c2[2] - c2[1],
        "c_cycle2_degraded": c[1] > c[0],
        "c_cycle3_degraded": c[2] > c[0],
        "c2_cycle3_degraded": c2[2] > c2[1],
        "frozen_difficulty": dict(frozen_difficulty),
        "owner_update_reference": dict(owner_update or {}),
    }
    return row


def _owner_update_by_seed(
    audit: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    if audit.get("format") != R4_AUDIT_FORMAT:
        raise ValueError("R4 audit format does not match M4.R6 contract")
    result: dict[int, dict[str, Any]] = {}
    for item in audit["reports"]:
        seed = int(item["seed"])
        result[seed] = dict(item["owner_updates"]["joint"])
    if tuple(sorted(result)) != EXPECTED_SEEDS:
        raise ValueError("R4 audit must contain seed11, seed29, and seed47")
    return result


def attribute(
    reports: list[dict[str, Any]],
    audit: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    if tuple(sorted(int(report["seed"]) for report in reports)) != EXPECTED_SEEDS:
        raise ValueError("M4.R6 requires exactly seed11, seed29, and seed47 reports")

    owner_updates = _owner_update_by_seed(audit)
    rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for report in reports:
        seed = int(report["seed"])
        checks[f"seed{seed}_technical_gate"] = bool(report["technical_gate_all_passed"])
        checks[f"seed{seed}_not_promoted"] = report["can_promote"] is False
        checks[f"seed{seed}_variant_set"] = set(report["variants"]) == set(EXPECTED_VARIANTS)
        checks[f"seed{seed}_scale_zero_frozen"] = (
            report["variants"]["scale_0p0"]["metrics"]["c_cycle2_delta_bpb"] == 0.0
            and report["variants"]["scale_0p0"]["metrics"]["c_cycle3_delta_bpb"] == 0.0
            and report["variants"]["scale_0p0"]["owner_before"]
            == report["variants"]["scale_0p0"]["owner_after"]
        )
        frozen = report["variants"]["scale_0p0"]["scores"]
        frozen_difficulty = {
            "c_cycle1_bpb": _float(frozen[0]["c_bpb"]),
            "c2_cycle2_bpb": _float(frozen[0]["c2_bpb"]),
            "c3_cycle3_bpb": _float(frozen[0]["c3_bpb"]),
        }
        for variant in EXPECTED_VARIANTS:
            rows.append(
                _cycle_row(
                    report,
                    variant,
                    frozen_difficulty=frozen_difficulty,
                    owner_update=owner_updates[seed],
                )
            )

    half_rows = [row for row in rows if row["variant"] == "scale_0p5"]
    legacy_rows = [row for row in rows if row["variant"] == "scale_1p0"]
    half_c2_failures = [row for row in half_rows if row["c_cycle2_degraded"]]
    half_c3_failures = [row for row in half_rows if row["c_cycle3_degraded"]]
    legacy_c2_failures = [row for row in legacy_rows if row["c_cycle2_degraded"]]
    half_owner_l2 = {
        str(row["seed"]): _float(
            row["owner_update_reference"]["predictive_context"]["relative_l2_delta"]
        )
        for row in half_rows
    }
    half_owner_median = statistics.median(half_owner_l2.values())
    seed47_owner_ratio = half_owner_l2["47"] / max(half_owner_median, 1e-12)
    diagnosis = {
        "half_scale_c_cycle2_failure_count": len(half_c2_failures),
        "half_scale_c_cycle2_failure_seeds": [int(row["seed"]) for row in half_c2_failures],
        "half_scale_c3_failure_count": len(half_c3_failures),
        "legacy_scale_c_cycle2_failure_count": len(legacy_c2_failures),
        "half_scale_c_cycle2_failure_isolated": (
            len(half_c2_failures) == 1 and not half_c3_failures
        ),
        "seed47_owner_update_relative_l2_ratio_to_median": seed47_owner_ratio,
        "seed47_owner_update_outlier": seed47_owner_ratio > 1.10,
        "interpretation": (
            "scale 0.5 has one isolated C cycle2 boundary failure, no C cycle3 "
            "failure, and no corresponding seed47 owner-update outlier; retain "
            "half-rate as a pre-registered formal candidate, but do not promote"
            if len(half_c2_failures) == 1 and not half_c3_failures
            else "scale 0.5 retention degradation is not isolated to one C boundary; "
            "withdraw the candidate and return to update-rule/course diagnosis"
        ),
    }
    checks["r4_audit_technical_gate"] = bool(audit["technical_gate_all_passed"])
    checks["same_cycle_count"] = all(
        len(report["variants"][variant]["scores"]) == 3
        for report in reports
        for variant in EXPECTED_VARIANTS
    )
    checks["same_pilot_budget"] = all(
        report["configuration"]["train_bytes"] == 16384
        and report["configuration"]["eval_bytes"] == 4096
        and report["configuration"]["fixed_capacity"] is True
        for report in reports
    )
    report = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": time.time(),
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "seeds": list(EXPECTED_SEEDS),
        "source_reports": [str(item.get("report_path", "")) for item in reports],
        "source_audit": str(audit.get("report_path", "")),
        "matrix": rows,
        "checks": checks,
        "technical_gate_all_passed": all(checks.values()),
        "diagnosis": diagnosis,
        "decision_boundary": {
            "formal_candidate": bool(
                diagnosis["half_scale_c_cycle2_failure_isolated"]
                and not diagnosis["seed47_owner_update_outlier"]
            ),
            "promotion_allowed": False,
            "next_required_evidence": (
                "pre-registered larger-budget formal retention Gate for scale 0.5"
                if diagnosis["half_scale_c_cycle2_failure_isolated"]
                else "new course/update-rule diagnosis before any formal run"
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
            PROJECT_ROOT / "reports" / "taiji_m4r5_update_scale_pilot_seed11_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r5_update_scale_pilot_seed29_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r5_update_scale_pilot_seed47_20260908.json",
        ],
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r4_course_shift_audit_20260908.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r6_cycle_retention_attribution_20260908.json",
    )
    args = parser.parse_args(argv)
    reports: list[dict[str, Any]] = []
    for path in args.reports:
        payload = _load_json(path)
        payload["report_path"] = str(path)
        reports.append(payload)
    audit = _load_json(args.audit)
    audit["report_path"] = str(args.audit)
    report = attribute(reports, audit, args.output)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "status": report["status"],
                "technical_gate_all_passed": report["technical_gate_all_passed"],
                "formal_candidate": report["decision_boundary"]["formal_candidate"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
