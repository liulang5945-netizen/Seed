"""Audit the M4.R7 foundation C->C2->C3 course without training."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (11, 29, 47)
FORMAL_TRAIN_BYTES = 65_536
FORMAL_EVAL_BYTES = 16_384
FORMAT = "taiji-m4r9-foundation-course-audit-v1"
R7_PREFLIGHT_FORMAT = "taiji-m4r7-formal-preflight-v1"
R7_AGGREGATE_FORMAT = "taiji-m4r7-formal-aggregate-v1"
R8_FORMAT = "taiji-m4r8-cycle3-failure-attribution-v1"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    build_disjoint_phase_chain,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return payload


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _probabilities(data: bytes) -> list[float]:
    counts = Counter(data)
    total = max(1, len(data))
    return [counts.get(index, 0) / total for index in range(256)]


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        value * math.log2(value) for value in probabilities if value > 0.0
    )


def _js_divergence(first: Sequence[float], second: Sequence[float]) -> float:
    midpoint = [(left + right) * 0.5 for left, right in zip(first, second, strict=True)]

    def _kl(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / middle)
            for value, middle in zip(values, midpoint, strict=True)
            if value > 0.0 and middle > 0.0
        )

    return 0.5 * (_kl(first) + _kl(second))


def _byte_stats(data: bytes) -> dict[str, float | int]:
    probabilities = _probabilities(data)
    return {
        "bytes": len(data),
        "unique_bytes": sum(value > 0.0 for value in probabilities),
        "unique_ratio": sum(value > 0.0 for value in probabilities) / 256.0,
        "entropy_bits": _entropy(probabilities),
        "top_byte_probability": max(probabilities),
    }


def _phase_row(name: str, dataset: Any) -> dict[str, Any]:
    train_probabilities = _probabilities(dataset.train)
    holdout_probabilities = _probabilities(dataset.holdout)
    return {
        "phase": name,
        "dataset_digest": dataset.digest,
        "partition_seed": int(dataset.partition_seed),
        "selected_record_count": len(dataset.selected_record_digests),
        "train": _byte_stats(dataset.train),
        "holdout": _byte_stats(dataset.holdout),
        "train_holdout_js_nats": _js_divergence(
            train_probabilities,
            holdout_probabilities,
        ),
    }


def _frozen_difficulty(formal_report: dict[str, Any]) -> dict[str, float]:
    scores = formal_report["variants"]["scale_0p0"]["scores"]
    return {
        "c_cycle1_bpb": float(scores[0]["c_bpb"]),
        "c2_cycle2_bpb": float(scores[1]["c2_bpb"]),
        "c3_cycle3_bpb": float(scores[2]["c3_bpb"]),
    }


def audit(
    preflight: dict[str, Any],
    formal_aggregate: dict[str, Any],
    formal_reports: list[dict[str, Any]],
    r8_report: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    if preflight.get("format") != R7_PREFLIGHT_FORMAT:
        raise ValueError("M4.R7 preflight format mismatch")
    if formal_aggregate.get("format") != R7_AGGREGATE_FORMAT:
        raise ValueError("M4.R7 aggregate format mismatch")
    if r8_report.get("format") != R8_FORMAT:
        raise ValueError("M4.R8 report format mismatch")
    seeds = tuple(sorted(int(report["seed"]) for report in formal_reports))
    if seeds != EXPECTED_SEEDS:
        raise ValueError("M4.R9 requires formal reports for seed11, seed29, and seed47")

    corpus = _resolve_path(str(preflight["corpus"]))
    chain = build_disjoint_phase_chain(
        [corpus],
        cohort_seeds=EXPECTED_SEEDS,
        profile="foundation",
    )
    phase_datasets = (
        ("c", chain.phase_c),
        ("c2", chain.phase_c2),
        ("c3", chain.phase_c3),
    )
    phases = [_phase_row(name, dataset) for name, dataset in phase_datasets]
    for previous, current in zip(phases, phases[1:], strict=False):
        current["train_vs_previous_train_js_nats"] = _js_divergence(
            _probabilities(chain.phase_c.train)
            if previous["phase"] == "c"
            else _probabilities(
                chain.phase_c2.train
                if previous["phase"] == "c2"
                else chain.phase_c3.train
            ),
            _probabilities(
                chain.phase_c2.train
                if current["phase"] == "c2"
                else chain.phase_c3.train
            ),
        )
        current["holdout_vs_previous_holdout_js_nats"] = _js_divergence(
            _probabilities(chain.phase_c.holdout)
            if previous["phase"] == "c"
            else _probabilities(
                chain.phase_c2.holdout
                if previous["phase"] == "c2"
                else chain.phase_c3.holdout
            ),
            _probabilities(
                chain.phase_c2.holdout
                if current["phase"] == "c2"
                else chain.phase_c3.holdout
            ),
        )

    difficulty_rows: list[dict[str, Any]] = []
    for formal_report in formal_reports:
        seed = int(formal_report["seed"])
        half_metrics = formal_report["variants"]["scale_0p5"]["metrics"]
        difficulty_rows.append(
            {
                "seed": seed,
                "frozen_difficulty": _frozen_difficulty(formal_report),
                "c_cycle2_delta_bpb": float(half_metrics["c_cycle2_delta_bpb"]),
                "c_cycle3_delta_bpb": float(half_metrics["c_cycle3_delta_bpb"]),
                "c2_cycle3_delta_bpb": float(half_metrics["c2_cycle3_delta_bpb"]),
                "c_cycle3_degraded": float(half_metrics["c_cycle3_delta_bpb"]) > 0.0,
            }
        )
    c3_degraded = [
        row for row in difficulty_rows if row["c_cycle3_degraded"]
    ]
    c3_phase = phases[2]
    c2_phase = phases[1]
    c3_train_js = float(c3_phase["train_vs_previous_train_js_nats"])
    c3_holdout_js = float(c3_phase["holdout_vs_previous_holdout_js_nats"])
    c2_train_js = float(c2_phase["train_vs_previous_train_js_nats"])
    c2_holdout_js = float(c2_phase["holdout_vs_previous_holdout_js_nats"])
    js_ratio = max(c3_train_js, c3_holdout_js) / max(
        max(c2_train_js, c2_holdout_js),
        1e-12,
    )
    checks = {
        "preflight_passed": (
            preflight["status"] == "passed"
            and preflight["formal_allowed"] is True
        ),
        "formal_aggregate_technical": bool(
            formal_aggregate["technical_gate_all_passed"]
        ),
        "formal_aggregate_not_promoted": formal_aggregate["can_promote"] is False,
        "r8_technical": bool(r8_report["technical_gate_all_passed"]),
        "record_disjoint_chain": all(
            value == 0 for value in chain.overlap_counts.values()
        ),
        "formal_budget_available": all(
            len(dataset.train) >= FORMAL_TRAIN_BYTES
            and len(dataset.holdout) >= FORMAL_EVAL_BYTES
            for _name, dataset in phase_datasets
        ),
        "distinct_phase_digests": len({phase["dataset_digest"] for phase in phases}) == 3,
    }
    report = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": time.time(),
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "training_allowed": False,
        "corpus": str(corpus),
        "configuration": {
            "profile": "foundation",
            "train_bytes": FORMAL_TRAIN_BYTES,
            "eval_bytes": FORMAL_EVAL_BYTES,
            "cohort_seeds": list(EXPECTED_SEEDS),
        },
        "phases": phases,
        "overlap_counts": chain.overlap_counts,
        "formal_seed_alignment": difficulty_rows,
        "checks": checks,
        "technical_gate_all_passed": all(checks.values()),
        "diagnosis": {
            "c3_degraded_seed_count": len(c3_degraded),
            "c3_degraded_seeds": [int(row["seed"]) for row in c3_degraded],
            "c2_boundary_max_js_nats": max(c2_train_js, c2_holdout_js),
            "c3_boundary_max_js_nats": max(c3_train_js, c3_holdout_js),
            "c3_to_c2_js_ratio": js_ratio,
            "c3_boundary_js_outlier": js_ratio > 2.0,
            "interpretation": (
                "C3 has a materially larger distribution boundary than C2; "
                "repair the course/data contract before any update-rule run"
                if js_ratio > 2.0
                else "C3 is not an obvious JS outlier; keep the course contract "
                "under review and design one update/consolidation rule audit"
            ),
        },
        "decision_boundary": {
            "next_action": (
                "repair or re-audit the C3 course boundary"
                if js_ratio > 2.0
                else "audit one update/consolidation rule against the stable course"
            ),
            "promotion_allowed": False,
            "training_allowed": False,
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
        "--preflight",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r7_formal_preflight_20260908.json",
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r7_formal_aggregate_20260908.json",
    )
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
        "--r8",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r8_cycle3_failure_attribution_20260908.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r9_foundation_course_audit_20260908.json",
    )
    args = parser.parse_args(argv)
    preflight = _load_json(args.preflight)
    aggregate_report = _load_json(args.aggregate)
    formal_reports = [_load_json(path) for path in args.formal_reports]
    r8_report = _load_json(args.r8)
    report = audit(
        preflight,
        aggregate_report,
        formal_reports,
        r8_report,
        args.output,
    )
    print(
        json.dumps(
            {
                "report": str(args.output),
                "status": report["status"],
                "technical_gate_all_passed": report["technical_gate_all_passed"],
                "c3_boundary_js_outlier": report["diagnosis"][
                    "c3_boundary_js_outlier"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
