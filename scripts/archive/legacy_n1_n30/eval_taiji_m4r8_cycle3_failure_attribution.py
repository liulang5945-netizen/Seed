"""Attribute M4.R7 C-cycle3 failure from saved formal artifacts.

This evaluator is read-only.  It loads the inherited source checkpoint and
the saved M4.R7 scale artifacts, measures actual owner deltas, and joins them
with the existing C/C2/C3 metrics.  It does not train, mutate, or create a
new model candidate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (11, 29, 47)
EXPECTED_VARIANTS = ("scale_0p5", "scale_1p0")
FORMAT = "taiji-m4r8-cycle3-failure-attribution-v1"
R6_FORMAT = "taiji-m4r6-cycle-retention-attribution-v1"
R4_FORMAT = "taiji-m4r4-course-shift-audit-v1"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4r3_update_interference_canary import (  # noqa: E402
    _load_checkpoint,
)
from taiji import Taiji  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return payload


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _owner_tensors(model: Taiji) -> dict[str, tuple[torch.Tensor, ...]]:
    return {
        "predictive_context": (model.predictive_context.recurrent.edge_weight,),
        "predictive_readout": (
            model.predictive_readout.synapses.edge_weight,
            model.predictive_readout.bias,
        ),
    }


def _owner_delta(
    source: tuple[torch.Tensor, ...],
    candidate: tuple[torch.Tensor, ...],
) -> dict[str, float | int]:
    if len(source) != len(candidate):
        raise ValueError("owner tensor count changed")
    squared = 0.0
    source_squared = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    for left, right in zip(source, candidate, strict=True):
        if left.shape != right.shape:
            raise ValueError("owner tensor shape changed")
        difference = (right.detach().cpu() - left.detach().cpu()).float()
        squared += float(torch.sum(difference * difference))
        source_squared += float(torch.sum(left.detach().cpu().float() ** 2))
        max_abs = max(max_abs, float(torch.max(torch.abs(difference))))
        changed += int(torch.count_nonzero(difference).item())
        total += difference.numel()
    l2_delta = squared**0.5
    return {
        "l2_delta": l2_delta,
        "relative_l2_delta": l2_delta / max(source_squared**0.5, 1e-12),
        "max_abs_delta": max_abs,
        "changed_scalars": changed,
        "total_scalars": total,
    }


def _owner_summary(
    rows: list[dict[str, Any]],
    owner: str,
) -> dict[str, Any]:
    values = [float(row["owner_updates"][owner]["relative_l2_delta"]) for row in rows]
    failures = [
        float(row["owner_updates"][owner]["relative_l2_delta"])
        for row in rows
        if row["c_cycle3_degraded"]
    ]
    passes = [
        float(row["owner_updates"][owner]["relative_l2_delta"])
        for row in rows
        if not row["c_cycle3_degraded"]
    ]
    pass_median = statistics.median(passes) if passes else None
    return {
        "relative_l2_mean": sum(values) / max(1, len(values)),
        "relative_l2_min": min(values),
        "relative_l2_max": max(values),
        "failure_mean": sum(failures) / max(1, len(failures)),
        "pass_mean": sum(passes) / max(1, len(passes)),
        "failure_outlier_against_pass": bool(
            pass_median is not None and any(value > pass_median * 1.10 for value in failures)
        ),
    }


def attribute(
    formal_reports: list[dict[str, Any]],
    r6_report: dict[str, Any],
    r4_report: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    seeds = tuple(sorted(int(report["seed"]) for report in formal_reports))
    if seeds != EXPECTED_SEEDS:
        raise ValueError("M4.R8 requires formal reports for seed11, seed29, and seed47")
    if r6_report.get("format") != R6_FORMAT:
        raise ValueError("M4.R6 report format mismatch")
    if r4_report.get("format") != R4_FORMAT:
        raise ValueError("M4.R4 report format mismatch")
    r4_by_seed = {int(item["seed"]): item for item in r4_report["reports"]}
    if tuple(sorted(r4_by_seed)) != EXPECTED_SEEDS:
        raise ValueError("M4.R4 audit is missing a cohort seed")

    checks = {
        "r6_technical_gate": bool(r6_report["technical_gate_all_passed"]),
        "r6_not_promoted": r6_report["can_promote"] is False,
        "r4_technical_gate": bool(r4_report["technical_gate_all_passed"]),
        "formal_reports_not_promoted": all(
            report["can_promote"] is False for report in formal_reports
        ),
        "formal_reports_technical": all(
            report["technical_gate_all_passed"] for report in formal_reports
        ),
        "same_formal_variant_set": all(
            set(report["variants"]) >= set(EXPECTED_VARIANTS) for report in formal_reports
        ),
    }
    rows: list[dict[str, Any]] = []
    for report in formal_reports:
        seed = int(report["seed"])
        checkpoint = _resolve_path(str(report["source_checkpoint"]))
        source_payload, source_model = _load_checkpoint(
            checkpoint,
            expected_seed=seed,
        )
        source_digest = str(report["source_checkpoint_digest"])
        for variant in EXPECTED_VARIANTS:
            variant_payload = report["variants"][variant]
            artifact_path = _resolve_path(str(report["artifacts"][variant]))
            artifact = torch.load(
                artifact_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(artifact, Mapping):
                raise ValueError(f"seed{seed} {variant} artifact is not a mapping")
            if artifact.get("source_checkpoint_digest") != source_digest:
                raise ValueError(f"seed{seed} {variant} source digest mismatch")
            candidate = Taiji.from_checkpoint(artifact["model"])
            source_owners = _owner_tensors(source_model)
            candidate_owners = _owner_tensors(candidate)
            metrics = variant_payload["metrics"]
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "scale": float(variant_payload["scale"]),
                    "artifact": str(artifact_path),
                    "c_cycle2_delta_bpb": float(metrics["c_cycle2_delta_bpb"]),
                    "c_cycle3_delta_bpb": float(metrics["c_cycle3_delta_bpb"]),
                    "c2_cycle3_delta_bpb": float(metrics["c2_cycle3_delta_bpb"]),
                    "c3_gain_bpb": float(metrics["c3_holdout_gain_bpb"]),
                    "c_cycle3_degraded": float(metrics["c_cycle3_delta_bpb"]) > 0.0,
                    "owner_updates": {
                        owner: _owner_delta(source_owners[owner], candidate_owners[owner])
                        for owner in ("predictive_context", "predictive_readout")
                    },
                    "r4_joint_owner_reference": r4_by_seed[seed]["owner_updates"]["joint"],
                }
            )
    checks["artifact_owner_delta_loaded"] = len(rows) == 6
    checks["source_digests_match"] = True
    half_rows = [row for row in rows if row["variant"] == "scale_0p5"]
    legacy_rows = [row for row in rows if row["variant"] == "scale_1p0"]
    half_failures = [row for row in half_rows if row["c_cycle3_degraded"]]
    half_passes = [row for row in half_rows if not row["c_cycle3_degraded"]]
    owner_summaries = {
        variant: {
            owner: _owner_summary(
                [row for row in rows if row["variant"] == variant],
                owner,
            )
            for owner in ("predictive_context", "predictive_readout")
        }
        for variant in EXPECTED_VARIANTS
    }
    diagnosis = {
        "half_scale_c_cycle3_failure_count": len(half_failures),
        "half_scale_c_cycle3_failure_seeds": [int(row["seed"]) for row in half_failures],
        "half_scale_c_cycle3_pass_seeds": [int(row["seed"]) for row in half_passes],
        "legacy_scale_c_cycle3_failure_count": sum(row["c_cycle3_degraded"] for row in legacy_rows),
        "failure_isolated": len(half_failures) == 1,
        "owner_update_outlier_by_variant": {
            variant: any(
                owner_summaries[variant][owner]["failure_outlier_against_pass"]
                for owner in ("predictive_context", "predictive_readout")
            )
            for variant in EXPECTED_VARIANTS
        },
        "interpretation": (
            "half-rate C cycle3 failure repeats in at least two seeds without "
            "a corresponding owner-update outlier; withdraw scale as a formal "
            "candidate and diagnose the course boundary or update rule"
            if len(half_failures) >= 2
            and not any(
                owner_summaries["scale_0p5"][owner]["failure_outlier_against_pass"]
                for owner in ("predictive_context", "predictive_readout")
            )
            else "cycle3 failure is not yet separated from owner update magnitude; "
            "keep the candidate withdrawn and require a rule-level diagnosis"
        ),
    }
    report = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": time.time(),
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "formal_candidate": False,
        "source_formal_reports": [str(item.get("report_path", "")) for item in formal_reports],
        "source_r6_report": str(r6_report.get("report_path", "")),
        "source_r4_report": str(r4_report.get("report_path", "")),
        "checks": checks,
        "technical_gate_all_passed": all(checks.values()),
        "matrix": rows,
        "owner_summaries": owner_summaries,
        "diagnosis": diagnosis,
        "decision_boundary": {
            "training_allowed": False,
            "promotion_allowed": False,
            "next_action": (
                "course/update-rule diagnosis before any new formal run"
                if all(checks.values())
                else "repair attribution input Gate before interpretation"
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
        "--formal-reports",
        nargs=3,
        type=Path,
        default=[
            PROJECT_ROOT / "reports" / "taiji_m4r7_formal_seed11_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r7_formal_seed29_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r7_formal_seed47_20260908.json",
        ],
    )
    parser.add_argument(
        "--r6",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r6_cycle_retention_attribution_20260908.json",
    )
    parser.add_argument(
        "--r4",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r4_course_shift_audit_20260908.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r8_cycle3_failure_attribution_20260908.json",
    )
    args = parser.parse_args(argv)
    formal_reports: list[dict[str, Any]] = []
    for path in args.formal_reports:
        payload = _load_json(path)
        payload["report_path"] = str(path)
        formal_reports.append(payload)
    r6_report = _load_json(args.r6)
    r6_report["report_path"] = str(args.r6)
    r4_report = _load_json(args.r4)
    r4_report["report_path"] = str(args.r4)
    report = attribute(formal_reports, r6_report, r4_report, args.output)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "status": report["status"],
                "technical_gate_all_passed": report["technical_gate_all_passed"],
                "formal_candidate": report["formal_candidate"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
