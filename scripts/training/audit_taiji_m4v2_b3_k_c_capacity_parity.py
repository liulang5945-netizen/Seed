"""Audit the C-entry formal comparison and freeze a capacity-parity contract.

This is a read-only audit. It does not train, mutate checkpoints, or attach a
candidate to the default runtime. Its purpose is to distinguish genuine
candidate learning evidence from a strong-control comparison confounded by
different capacity and training budgets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPORT_FORMAT = "taiji-m4v2-b3-k-c-capacity-parity-audit-v1"
FORMAL_REPORT_FORMAT = "taiji-m4v2-b3-k-c-formal-v1"
FIXED_LARGE_REPORT_FORMAT = "taiji-m4v2-b3-k-c-fixed-large-build-v1"
PARAMETER_TOLERANCE_RATIO = 0.01
DEFAULT_FORMAL_REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_c_formal_20260910.json"
)
DEFAULT_FIXED_LARGE_REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_c_fixed_large_build_20260910.json"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_c_capacity_parity_audit_20260910.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: Any, *, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _int(value: Any, *, name: str) -> int:
    number = _number(value, name=name)
    if int(number) != number:
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _ratio(left: int | float, right: int | float) -> float:
    denominator = max(abs(float(left)), abs(float(right)), 1.0)
    return abs(float(left) - float(right)) / denominator


def _resource_number(resource: dict[str, Any], key: str) -> int | float:
    return _number(resource.get(key), name=f"resource.{key}")


def _resource_int(resource: dict[str, Any], key: str) -> int:
    return _int(resource.get(key), name=f"resource.{key}")


def _checkpoint_count(resource: dict[str, Any]) -> int:
    paths = resource.get("checkpoint_write_paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("resource.checkpoint_write_paths must be a list of strings")
    if len(paths) != len(set(paths)):
        raise ValueError("resource.checkpoint_write_paths contains duplicates")
    return len(paths)


def _fixed_cell(
    fixed_report: dict[str, Any],
    *,
    model_seed: int,
    course_seed: int,
) -> dict[str, Any]:
    for cell in fixed_report.get("cells", []):
        if (
            cell.get("model_seed") == model_seed
            and cell.get("course_seed") == course_seed
        ):
            if not isinstance(cell, dict):
                break
            return cell
    raise ValueError(f"missing fixed-large cell model={model_seed} course={course_seed}")


def _audit_cell(
    formal_cell: dict[str, Any],
    fixed_build_cell: dict[str, Any],
) -> dict[str, Any]:
    candidate_training = formal_cell["resource"]["candidate"]["training"]
    fixed_training = formal_cell["resource"]["fixed_large"]["training"]
    candidate_inference = formal_cell["resource"]["candidate"]["inference"]
    fixed_inference = formal_cell["resource"]["fixed_large"]["inference"]

    candidate_parameter_bytes = _resource_int(
        candidate_training, "parameter_bytes"
    )
    fixed_parameter_bytes = _resource_int(fixed_training, "parameter_bytes")
    candidate_update_steps = _resource_int(
        candidate_training, "training_update_steps"
    )
    fixed_update_steps = _resource_int(fixed_training, "training_update_steps")
    candidate_checkpoint_bytes = _resource_int(
        candidate_training, "checkpoint_write_bytes"
    )
    fixed_checkpoint_bytes = _resource_int(
        fixed_training, "checkpoint_write_bytes"
    )
    candidate_checkpoint_count = _checkpoint_count(candidate_training)
    fixed_checkpoint_count = _checkpoint_count(fixed_training)
    candidate_trace_count = _resource_int(
        candidate_inference, "inference_trace_count"
    )
    fixed_trace_count = _resource_int(fixed_inference, "inference_trace_count")

    fixed_build_resource = fixed_build_cell["resource"]
    fixed_build_steps = _resource_int(fixed_build_resource, "training_update_steps")
    fixed_build_parameter_bytes = _resource_int(
        fixed_build_resource, "parameter_bytes"
    )
    fixed_build_episode_indexes = fixed_build_cell.get("train_episode_indexes")
    if not isinstance(fixed_build_episode_indexes, list):
        raise ValueError("fixed-large train_episode_indexes must be a list")

    candidate_training_block = formal_cell["candidate_training"]
    candidate_episode_indexes = candidate_training_block.get("train_episode_indexes")
    if not isinstance(candidate_episode_indexes, list):
        raise ValueError("candidate train_episode_indexes must be a list")

    parameter_ratio = _ratio(candidate_parameter_bytes, fixed_parameter_bytes)
    update_ratio = _ratio(candidate_update_steps, fixed_update_steps)
    checkpoint_count_equal = candidate_checkpoint_count == fixed_checkpoint_count
    checkpoint_bytes_ratio = _ratio(
        candidate_checkpoint_bytes, fixed_checkpoint_bytes
    )

    return {
        "model_seed": formal_cell["model_seed"],
        "course_seed": formal_cell["course_seed"],
        "candidate": {
            "parameter_bytes": candidate_parameter_bytes,
            "training_update_steps": candidate_update_steps,
            "checkpoint_write_bytes": candidate_checkpoint_bytes,
            "checkpoint_write_count": candidate_checkpoint_count,
            "training_wall_clock_seconds": _resource_number(
                candidate_training, "training_wall_clock_seconds"
            ),
            "inference_trace_count": candidate_trace_count,
        },
        "fixed_large": {
            "parameter_bytes": fixed_parameter_bytes,
            "training_update_steps": fixed_update_steps,
            "checkpoint_write_bytes": fixed_checkpoint_bytes,
            "checkpoint_write_count": fixed_checkpoint_count,
            "training_wall_clock_seconds": _resource_number(
                fixed_training, "training_wall_clock_seconds"
            ),
            "inference_trace_count": fixed_trace_count,
        },
        "training_examples": {
            "candidate_episode_indexes": candidate_episode_indexes,
            "fixed_large_episode_indexes": fixed_build_episode_indexes,
            "same_episode_indexes": candidate_episode_indexes
            == fixed_build_episode_indexes,
        },
        "cross_report_consistency": {
            "fixed_large_parameter_bytes_matches_build": (
                fixed_parameter_bytes == fixed_build_parameter_bytes
            ),
            "fixed_large_update_steps_matches_build": (
                fixed_update_steps == fixed_build_steps
            ),
        },
        "parity": {
            "parameter_bytes_ratio": parameter_ratio,
            "parameter_bytes_within_tolerance": (
                parameter_ratio <= PARAMETER_TOLERANCE_RATIO
            ),
            "training_update_steps_ratio": update_ratio,
            "training_update_steps_equal": (
                candidate_update_steps == fixed_update_steps
            ),
            "checkpoint_write_count_equal": checkpoint_count_equal,
            "checkpoint_write_bytes_ratio": checkpoint_bytes_ratio,
            "inference_trace_count_equal": (
                candidate_trace_count == fixed_trace_count
            ),
        },
    }


def build_audit(
    formal_report: dict[str, Any],
    fixed_large_report: dict[str, Any],
    *,
    formal_report_path: Path,
    fixed_large_report_path: Path,
) -> dict[str, Any]:
    if formal_report.get("report_format") != FORMAL_REPORT_FORMAT:
        raise ValueError("formal report format mismatch")
    if fixed_large_report.get("report_format") != FIXED_LARGE_REPORT_FORMAT:
        raise ValueError("fixed-large report format mismatch")

    formal_cells = formal_report.get("cells")
    if not isinstance(formal_cells, list) or not formal_cells:
        raise ValueError("formal report must contain cells")

    audited_cells = []
    for formal_cell in formal_cells:
        if not isinstance(formal_cell, dict):
            raise ValueError("formal cell must be an object")
        fixed_cell = _fixed_cell(
            fixed_large_report,
            model_seed=_int(formal_cell.get("model_seed"), name="model_seed"),
            course_seed=_int(formal_cell.get("course_seed"), name="course_seed"),
        )
        audited_cells.append(_audit_cell(formal_cell, fixed_cell))

    parity_values = [cell["parity"] for cell in audited_cells]
    consistency_values = [cell["cross_report_consistency"] for cell in audited_cells]
    training_values = [cell["training_examples"] for cell in audited_cells]
    aggregate = {
        "parameter_bytes_within_tolerance_all": all(
            value["parameter_bytes_within_tolerance"] for value in parity_values
        ),
        "training_update_steps_equal_all": all(
            value["training_update_steps_equal"] for value in parity_values
        ),
        "checkpoint_write_count_equal_all": all(
            value["checkpoint_write_count_equal"] for value in parity_values
        ),
        "inference_trace_count_equal_all": all(
            value["inference_trace_count_equal"] for value in parity_values
        ),
        "same_training_episode_indexes_all": all(
            value["same_episode_indexes"] for value in training_values
        ),
        "fixed_large_cross_report_consistent_all": all(
            value["fixed_large_parameter_bytes_matches_build"]
            and value["fixed_large_update_steps_matches_build"]
            for value in consistency_values
        ),
        "candidate_parameter_bytes": sorted(
            {cell["candidate"]["parameter_bytes"] for cell in audited_cells}
        ),
        "fixed_large_parameter_bytes": sorted(
            {cell["fixed_large"]["parameter_bytes"] for cell in audited_cells}
        ),
        "candidate_training_update_steps": sorted(
            {cell["candidate"]["training_update_steps"] for cell in audited_cells}
        ),
        "fixed_large_training_update_steps": sorted(
            {cell["fixed_large"]["training_update_steps"] for cell in audited_cells}
        ),
        "candidate_checkpoint_write_counts": sorted(
            {cell["candidate"]["checkpoint_write_count"] for cell in audited_cells}
        ),
        "fixed_large_checkpoint_write_counts": sorted(
            {cell["fixed_large"]["checkpoint_write_count"] for cell in audited_cells}
        ),
    }

    comparison_valid = (
        aggregate["parameter_bytes_within_tolerance_all"]
        and aggregate["training_update_steps_equal_all"]
        and aggregate["same_training_episode_indexes_all"]
        and aggregate["fixed_large_cross_report_consistent_all"]
    )
    reason_codes = []
    if not aggregate["parameter_bytes_within_tolerance_all"]:
        reason_codes.append("parameter_bytes_mismatch")
    if not aggregate["training_update_steps_equal_all"]:
        reason_codes.append("training_update_steps_mismatch")
    if not aggregate["same_training_episode_indexes_all"]:
        reason_codes.append("training_episode_contract_mismatch")
    if not aggregate["fixed_large_cross_report_consistent_all"]:
        reason_codes.append("fixed_large_report_mismatch")

    strong_control_victory = bool(
        formal_report.get("strong_fixed_large_comparison_passed", False)
    )
    preregistration = {
        "format": "taiji-m4v2-b3-k-capacity-parity-preregistration-v1",
        "status": "pre-registered-blocked-until-parity",
        "comparison_unit": {
            "cell_matrix": "same 3 model seeds x 3 course seeds = 9 cells",
            "parent": "same frozen parent checkpoint digest per cell",
            "course": "same C-entry course digest and target-aware target multiset",
            "evaluation": "same validation and sealed inputs; sealed remains unread until formal scoring",
        },
        "hard_gates": {
            "parameter_bytes": {
                "rule": "candidate and fixed-large parameter_bytes ratio <= 1%",
                "tolerance_ratio": PARAMETER_TOLERANCE_RATIO,
            },
            "training_examples": {
                "rule": "same episode indexes and same train/holdout disjointness",
            },
            "training_update_steps": {
                "rule": "exact equality per cell; count actual parameter-update events",
            },
            "checkpoint_policy": {
                "rule": "same logical checkpoint emission count; report byte totals separately",
                "byte_total": "measurement only; do not treat format-specific bytes as a learning-rule advantage",
            },
            "inference_trace": {
                "rule": "same validation/sealed trace counts and input digests",
            },
            "restoration": {
                "rule": "fresh restore and rollback gates remain mandatory for both routes",
            },
        },
        "resource_interpretation": {
            "training_wall_clock": "report, not a scientific parity gate; it is hardware-dependent",
            "peak_working_set": "report as the existing RSS lower bound, not exact allocator usage",
            "inference_wall_clock": "report only after identical trace contract is satisfied",
            "promotion": "requires all technical/resource/quality gates, parity gates, and 9/9 sealed strong-control victory",
        },
        "current_formal_result": {
            "strong_control_victory": strong_control_victory,
            "comparison_valid_for_learning_rule_claim": comparison_valid,
            "promotion_precondition_passed": comparison_valid
            and strong_control_victory,
            "reason_codes": reason_codes,
        },
        "stop_rule": "do not add training, tune learning rate, revive R5/structure growth, or attach default runtime before the parity contract is satisfied",
    }

    return {
        "report_format": REPORT_FORMAT,
        "version": 1,
        "run_kind": "c-entry-capacity-parity-audit",
        "status": "blocked-current-comparison-confounded"
        if not comparison_valid
        else "parity-ready",
        "formal_report_path": str(formal_report_path.resolve()),
        "fixed_large_report_path": str(fixed_large_report_path.resolve()),
        "formal_report_status": formal_report.get("status"),
        "cell_count": len(audited_cells),
        "model_seeds": formal_report.get("model_seeds"),
        "course_seeds": formal_report.get("course_seeds"),
        "aggregate": aggregate,
        "cells": audited_cells,
        "current_verdict": {
            "candidate_learning_evidence_valid": bool(
                formal_report.get("candidate_quality_gate_passed", False)
            ),
            "comparison_valid_for_learning_rule_claim": comparison_valid,
            "strong_control_victory": strong_control_victory,
            "can_promote": False,
            "blocking_reason": (
                "current candidate/fixed-large result is capacity- and update-budget-confounded"
                if not comparison_valid
                else "capacity parity is ready, but strong-control victory is still required"
            ),
            "reason_codes": reason_codes,
        },
        "preregistration": preregistration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-report", type=Path, default=DEFAULT_FORMAL_REPORT)
    parser.add_argument(
        "--fixed-large-report", type=Path, default=DEFAULT_FIXED_LARGE_REPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_audit(
        _load_json(args.formal_report),
        _load_json(args.fixed_large_report),
        formal_report_path=args.formal_report,
        fixed_large_report_path=args.fixed_large_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
