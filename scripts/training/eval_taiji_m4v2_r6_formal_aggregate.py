"""Aggregate the preregistered M4.V2.R6 nine-cell execution ledger.

The aggregate is read-only with respect to Taiji runtime state.  It re-reads
all cell reports, validates content-addressed lineage and the five-arm
resource/causal contract, and never turns a blocked aggregate into promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r6_formal_input_manifest_preflight import (  # noqa: E402
    ARM_IDS,
    COURSE_SEEDS,
    DEFAULT_FIXED_LARGE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_PARENT_DIR,
    DEFAULT_WORKER_DIR,
    MODEL_SEEDS,
    _manifest_digest,
    _resolve_repo_path,
)
from scripts.training.eval_taiji_m4v2_r6_formal_single_cell import (  # noqa: E402
    _resource_gate_valid,
)
from taiji import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-r6-formal-aggregate-v1"
VERSION = 1
DEFAULT_EXECUTION_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_r6_formal_execution_20260909.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r6_formal_aggregate_20260909.json"
STUDENT_T_CRITICAL_95_DF8 = 1.8595480375228424
EXPECTED_CELL_ORDER = tuple(
    (int(model_seed), int(course_seed))
    for model_seed in MODEL_SEEDS
    for course_seed in COURSE_SEEDS
)
RESOURCE_NUMERIC_FIELDS = (
    "wall_clock_seconds",
    "peak_working_set_bytes",
    "training_update_steps",
    "worker_parameter_count",
    "worker_parameter_bytes",
    "candidate_parameter_bytes",
    "checkpoint_write_bytes",
    "inference_trace_count",
)


def _load_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"JSON report must be an object: {path}")
    return {str(key): value for key, value in raw.items()}


def _record_failure(
    failures: list[dict[str, Any]],
    *,
    category: str,
    message: str,
    cell: tuple[int, int] | None = None,
    arm: str | None = None,
    metric: str | None = None,
) -> None:
    failures.append(
        {
            "category": category,
            "cell": (
                None
                if cell is None
                else {"model_seed": int(cell[0]), "course_seed": int(cell[1])}
            ),
            "arm": arm,
            "metric": metric,
            "message": message,
        }
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_resource(resource: Any) -> bool:
    if not isinstance(resource, Mapping) or not _resource_gate_valid(resource):
        return False
    return all(_finite_number(resource.get(field)) for field in RESOURCE_NUMERIC_FIELDS)


def _mean_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot aggregate an empty metric")
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    lower_bound = mean - STUDENT_T_CRITICAL_95_DF8 * sample_std / math.sqrt(len(values))
    return {
        "n": len(values),
        "degrees_of_freedom": len(values) - 1,
        "mean": float(mean),
        "min": float(min(values)),
        "max": float(max(values)),
        "sample_std": float(sample_std),
        "one_sided_95_student_t_lower_bound": float(lower_bound),
    }


def _index_registry(
    manifest: Mapping[str, Any],
    key: str,
    failures: list[dict[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    raw = manifest.get(key)
    if not isinstance(raw, list):
        _record_failure(failures, category="lineage", message=f"manifest {key} is not a list")
        return {}
    indexed: dict[int, Mapping[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            _record_failure(
                failures,
                category="lineage",
                message=f"manifest {key} has a non-object entry",
            )
            continue
        try:
            model_seed = int(entry["model_seed"])
        except (KeyError, TypeError, ValueError):
            _record_failure(
                failures,
                category="lineage",
                message=f"manifest {key} entry has no model_seed",
            )
            continue
        if model_seed in indexed:
            _record_failure(
                failures,
                category="lineage",
                message=f"manifest {key} duplicates model_seed={model_seed}",
            )
            continue
        indexed[model_seed] = entry
    return indexed


def _resource_digest_for_arm(
    arm: str,
    *,
    parent: Mapping[str, Any],
    worker: Mapping[str, Any],
    fixed_large: Mapping[str, Any],
) -> str:
    if arm in {"frozen-parent", "matched-fixed-capacity"}:
        return str(parent["resource_manifest_digest"])
    if arm in {"candidate-continuation", "lesion"}:
        return str(worker["resource_manifest_digest"])
    if arm == "fixed-large":
        return str(fixed_large["resource_manifest_digest"])
    raise ValueError(f"unknown arm: {arm}")


def _validate_arm(
    arm: Mapping[str, Any],
    *,
    cell: tuple[int, int],
    expected_parent_digest: str,
    expected_worker_digest: str,
    expected_resource_digest: str,
    expected_device: str,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    arm_id = str(arm.get("arm", ""))
    resource = arm.get("resource")
    resource_valid = _valid_resource(resource)
    if not resource_valid:
        _record_failure(
            failures,
            category="resource_gate",
            message="arm resource object fails the frozen contract",
            cell=cell,
            arm=arm_id,
        )
    if isinstance(resource, Mapping):
        if resource.get("device") != expected_device:
            _record_failure(
                failures,
                category="resource_gate",
                message=f"resource device is {resource.get('device')!r}, expected {expected_device!r}",
                cell=cell,
                arm=arm_id,
                metric="device",
            )
        if str(resource.get("resource_manifest_digest", "")) != expected_resource_digest:
            _record_failure(
                failures,
                category="lineage",
                message="resource manifest digest does not match the preregistered owner",
                cell=cell,
                arm=arm_id,
                metric="resource_manifest_digest",
            )
    if arm.get("failure") is not None:
        _record_failure(
            failures,
            category="execution",
            message="cell arm contains a failure attribution",
            cell=cell,
            arm=arm_id,
        )
    if arm.get("old_capability_retention") != {"G": True, "S": True}:
        _record_failure(
            failures,
            category="retention",
            message="old S/G capability retention is not exactly true/true",
            cell=cell,
            arm=arm_id,
        )
    side_effects = arm.get("side_effects")
    if not isinstance(side_effects, Mapping):
        _record_failure(
            failures,
            category="side_effect",
            message="arm has no side_effects object",
            cell=cell,
            arm=arm_id,
        )
    else:
        isolated = (
            side_effects.get("default_runtime_attached") is False
            and side_effects.get("external_integrations_attached") is False
            and side_effects.get("parent_namespace_unchanged") is True
        )
        if not isolated:
            _record_failure(
                failures,
                category="side_effect",
                message="arm side-effect contract is not isolated",
                cell=cell,
                arm=arm_id,
            )
        expected_namespace_write = arm_id in {"candidate-continuation", "fixed-large"}
        if side_effects.get("candidate_namespace_written") is not expected_namespace_write:
            _record_failure(
                failures,
                category="side_effect",
                message="candidate namespace write flag does not match arm contract",
                cell=cell,
                arm=arm_id,
            )
    if arm.get("parent_checkpoint_digest") != expected_parent_digest:
        _record_failure(
            failures,
            category="lineage",
            message="arm parent checkpoint digest does not match the parent registry",
            cell=cell,
            arm=arm_id,
            metric="parent_checkpoint_digest",
        )
    if arm_id in {"candidate-continuation", "lesion"} and arm.get(
        "worker_bundle_digest"
    ) != expected_worker_digest:
        _record_failure(
            failures,
            category="lineage",
            message="arm worker bundle digest does not match the worker registry",
            cell=cell,
            arm=arm_id,
            metric="worker_bundle_digest",
        )
    capability = arm.get("new_capability")
    rate = capability.get("task_success_rate") if isinstance(capability, Mapping) else None
    return {
        "arm": arm_id,
        "resource_valid": resource_valid,
        "task_success_rate": rate,
        "resource": resource,
    }


def _validate_cell(
    cell_report: Mapping[str, Any],
    *,
    execution_row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    registries: dict[str, dict[int, Mapping[str, Any]]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    cell_payload = cell_report.get("cell")
    if not isinstance(cell_payload, Mapping):
        raise ValueError("cell report has no cell identity")
    cell = (int(cell_payload["model_seed"]), int(cell_payload["course_seed"]))
    parent = registries["parent"][cell[0]]
    worker = registries["worker"][cell[0]]
    fixed_large = registries["fixed_large"][cell[0]]
    expected_parent_digest = str(parent["checkpoint_digest"])
    expected_worker_digest = str(worker["bundle_digest"])
    expected_fixed_digest = str(fixed_large["artifact_digest"])
    expected_fixed_ensemble_digest = str(fixed_large["ensemble_checkpoint_digest"])

    if cell_report.get("status") != "passed" or cell_report.get("failures"):
        _record_failure(
            failures,
            category="execution",
            message="cell report is not a clean passed report",
            cell=cell,
        )
    if execution_row.get("status") != "executed_passed":
        _record_failure(
            failures,
            category="execution",
            message="execution ledger row is not executed_passed",
            cell=cell,
        )
    for key, expected in (
        ("parent_checkpoint_digest", expected_parent_digest),
        ("worker_bundle_digest", expected_worker_digest),
        ("fixed_large_artifact_digest", expected_fixed_digest),
        ("fixed_large_ensemble_checkpoint_digest", expected_fixed_ensemble_digest),
    ):
        if str(cell_report.get(key, "")) != expected:
            _record_failure(
                failures,
                category="lineage",
                message=f"cell {key} does not match its registry entry",
                cell=cell,
                metric=key,
            )
    if execution_row.get("execution_contract_digest") != cell_report.get(
        "execution_contract_digest"
    ):
        _record_failure(
            failures,
            category="lineage",
            message="ledger and cell report execution contract digests differ",
            cell=cell,
            metric="execution_contract_digest",
        )
    raw_arms = cell_report.get("arms")
    if not isinstance(raw_arms, Mapping) or set(raw_arms) != set(ARM_IDS):
        _record_failure(
            failures,
            category="input_contract",
            message="cell arm set differs from the preregistered five-arm set",
            cell=cell,
        )
        return {
            "cell": {"model_seed": cell[0], "course_seed": cell[1]},
            "status": "blocked",
            "capability": {},
            "resources": {},
        }
    ledger_arms = execution_row.get("arms")
    if not isinstance(ledger_arms, list) or [arm.get("arm") for arm in ledger_arms] != list(
        ARM_IDS
    ):
        _record_failure(
            failures,
            category="lineage",
            message="execution ledger arm snapshot has the wrong arm order",
            cell=cell,
            metric="arm_snapshot",
        )
    else:
        for ledger_arm in ledger_arms:
            arm_id = str(ledger_arm["arm"])
            arm_snapshot = raw_arms[arm_id]
            if not isinstance(arm_snapshot, Mapping) or any(
                ledger_arm.get(key) != arm_snapshot.get(key)
                for key in ledger_arm
                if key != "arm"
            ):
                _record_failure(
                    failures,
                    category="lineage",
                    message="execution ledger arm snapshot differs from cell report",
                    cell=cell,
                    arm=arm_id,
                    metric="arm_snapshot",
                )
    if cell_report.get("resource_contract") != manifest.get("resource_contract"):
        _record_failure(
            failures,
            category="resource_gate",
            message="cell resource contract differs from the input manifest",
            cell=cell,
        )
    if cell_report.get("resource_gate") != {arm: True for arm in ARM_IDS}:
        _record_failure(
            failures,
            category="resource_gate",
            message="cell resource_gate is not true for every arm",
            cell=cell,
        )

    expected_device = str(manifest["resource_contract"]["device"])
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        arm_summaries[arm_id] = _validate_arm(
            raw_arms[arm_id],
            cell=cell,
            expected_parent_digest=expected_parent_digest,
            expected_worker_digest=expected_worker_digest,
            expected_resource_digest=_resource_digest_for_arm(
                arm_id, parent=parent, worker=worker, fixed_large=fixed_large
            ),
            expected_device=expected_device,
            failures=failures,
        )

    candidate_rate = arm_summaries["candidate-continuation"]["task_success_rate"]
    fixed_large_rate = arm_summaries["fixed-large"]["task_success_rate"]
    lesion_rate = arm_summaries["lesion"]["task_success_rate"]
    for arm_id, rate in (
        ("candidate-continuation", candidate_rate),
        ("fixed-large", fixed_large_rate),
        ("lesion", lesion_rate),
    ):
        if not _finite_number(rate):
            _record_failure(
                failures,
                category="causal",
                message="observed K arm has no finite task_success_rate",
                cell=cell,
                arm=arm_id,
                metric="task_success_rate",
            )
    candidate_floor_pass = _finite_number(candidate_rate) and float(candidate_rate) >= 0.75
    if not candidate_floor_pass:
        _record_failure(
            failures,
            category="capability_gate",
            message="candidate K holdout success is below the preregistered 0.75 floor",
            cell=cell,
            arm="candidate-continuation",
            metric="task_success_rate",
        )
    lesion_breaks_gain = (
        _finite_number(candidate_rate)
        and _finite_number(lesion_rate)
        and float(candidate_rate) > float(lesion_rate)
    )
    if not lesion_breaks_gain:
        _record_failure(
            failures,
            category="causal",
            message="K3 lesion does not break the observed candidate gain",
            cell=cell,
            metric="candidate_minus_lesion_task_success_rate",
        )

    candidate_resource = arm_summaries["candidate-continuation"]["resource"]
    matched_resource = arm_summaries["matched-fixed-capacity"]["resource"]
    fixed_resource = arm_summaries["fixed-large"]["resource"]
    resources: dict[str, Any] = {}
    if all(
        isinstance(item, Mapping)
        for item in (candidate_resource, matched_resource, fixed_resource)
    ):
        candidate_wall = float(candidate_resource["wall_clock_seconds"])
        matched_wall = float(matched_resource["wall_clock_seconds"])
        candidate_peak = float(candidate_resource["peak_working_set_bytes"])
        matched_peak = float(matched_resource["peak_working_set_bytes"])
        wall_ratio = math.inf if matched_wall == 0.0 else candidate_wall / matched_wall
        peak_ratio = math.inf if matched_peak == 0.0 else candidate_peak / matched_peak
        wall_cap = float(manifest["resource_contract"]["wall_clock_multiplier_cap"])
        peak_cap = float(manifest["resource_contract"]["peak_working_set_multiplier_cap"])
        wall_pass = math.isfinite(wall_ratio) and wall_ratio <= wall_cap
        peak_pass = math.isfinite(peak_ratio) and peak_ratio <= peak_cap
        if not wall_pass:
            _record_failure(
                failures,
                category="resource_budget",
                message=(
                    f"candidate/matched wall multiplier {wall_ratio:.6f} "
                    f"exceeds cap {wall_cap:.6f}"
                ),
                cell=cell,
                arm="candidate-continuation",
                metric="wall_clock_multiplier_candidate_over_matched",
            )
        if not peak_pass:
            _record_failure(
                failures,
                category="resource_budget",
                message=(
                    f"candidate/matched peak multiplier {peak_ratio:.6f} "
                    f"exceeds cap {peak_cap:.6f}"
                ),
                cell=cell,
                arm="candidate-continuation",
                metric="peak_working_set_multiplier_candidate_over_matched",
            )
        resources = {
            "candidate_over_matched": {
                "wall_clock_multiplier": wall_ratio,
                "peak_working_set_multiplier": peak_ratio,
                "wall_clock_cap": wall_cap,
                "peak_working_set_cap": peak_cap,
                "wall_clock_pass": wall_pass,
                "peak_working_set_pass": peak_pass,
            },
            "fixed_large_minus_candidate": {
                field: float(fixed_resource[field]) - float(candidate_resource[field])
                for field in (
                    "wall_clock_seconds",
                    "peak_working_set_bytes",
                    "worker_parameter_bytes",
                    "checkpoint_write_bytes",
                    "inference_trace_count",
                )
            },
            "arm_resources": {
                arm_id: arm_summaries[arm_id]["resource"] for arm_id in ARM_IDS
            },
        }
    else:
        _record_failure(
            failures,
            category="resource_gate",
            message="candidate, matched, and fixed-large resources cannot be paired",
            cell=cell,
        )

    execution_clean = (
        cell_report.get("status") == "passed"
        and not cell_report.get("failures")
        and execution_row.get("status") == "executed_passed"
    )
    return {
        "cell": {"model_seed": cell[0], "course_seed": cell[1]},
        "status": "executed_passed" if execution_clean else "blocked",
        "execution_contract_digest": cell_report.get("execution_contract_digest"),
        "execution_report_path": execution_row.get("execution_report_path"),
        "capability": {
            "candidate_task_success_rate": candidate_rate,
            "fixed_large_task_success_rate": fixed_large_rate,
            "lesion_task_success_rate": lesion_rate,
            "candidate_holdout_floor_pass": candidate_floor_pass,
            "candidate_minus_lesion_task_success_rate": (
                None
                if not (_finite_number(candidate_rate) and _finite_number(lesion_rate))
                else float(candidate_rate) - float(lesion_rate)
            ),
            "lesion_breaks_gain": lesion_breaks_gain,
            "candidate_minus_frozen_parent_task_success_rate": None,
            "candidate_minus_matched_fixed_capacity_task_success_rate": None,
        },
        "resources": resources,
    }


def _aggregate_reports(
    *,
    manifest: Mapping[str, Any],
    execution: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    registries = {
        "parent": _index_registry(manifest, "parent_registry", failures),
        "worker": _index_registry(manifest, "worker_registry", failures),
        "fixed_large": _index_registry(manifest, "fixed_large_registry", failures),
    }
    if any(len(registry) != len(MODEL_SEEDS) for registry in registries.values()):
        _record_failure(
            failures,
            category="lineage",
            message="one or more content-addressed registries are incomplete",
        )

    ledger = execution.get("cell_ledger")
    if not isinstance(ledger, list):
        _record_failure(
            failures,
            category="input_contract",
            message="execution report has no cell_ledger list",
        )
        return {"cells": [], "metrics": {}, "gates": {}}
    row_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in ledger:
        if not isinstance(row, Mapping) or not isinstance(row.get("cell"), Mapping):
            _record_failure(
                failures,
                category="input_contract",
                message="ledger has malformed cell row",
            )
            continue
        key = (int(row["cell"]["model_seed"]), int(row["cell"]["course_seed"]))
        if key in row_by_key:
            _record_failure(
                failures,
                category="input_contract",
                message=f"ledger duplicates cell {key}",
            )
        row_by_key[key] = row
    if tuple(row_by_key) != EXPECTED_CELL_ORDER:
        _record_failure(
            failures,
            category="input_contract",
            message=f"ledger order/identity differs from {EXPECTED_CELL_ORDER!r}",
        )

    cell_results: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {
        "candidate_task_success_rate": [],
        "candidate_minus_lesion_task_success_rate": [],
        "candidate_over_matched_wall_clock_multiplier": [],
        "candidate_over_matched_peak_working_set_multiplier": [],
        "candidate_wall_clock_seconds": [],
        "matched_wall_clock_seconds": [],
        "candidate_peak_working_set_bytes": [],
        "matched_peak_working_set_bytes": [],
        "fixed_large_minus_candidate_wall_clock_seconds": [],
        "fixed_large_minus_candidate_peak_working_set_bytes": [],
        "fixed_large_minus_candidate_worker_parameter_bytes": [],
        "fixed_large_minus_candidate_checkpoint_write_bytes": [],
        "fixed_large_minus_candidate_inference_trace_count": [],
    }
    for key in EXPECTED_CELL_ORDER:
        row = row_by_key.get(key)
        if row is None:
            _record_failure(
                failures,
                category="input_contract",
                message=f"missing ledger cell {key}",
            )
            continue
        report_path_value = row.get("execution_report_path")
        if not isinstance(report_path_value, str):
            _record_failure(
                failures,
                category="input_contract",
                message=f"cell {key} has no report path",
            )
            continue
        cell_report_path = _resolve_repo_path(report_path_value)
        cell_report = _load_object(cell_report_path)
        if content_digest(cell_report) != row.get("execution_report_digest"):
            _record_failure(
                failures,
                category="lineage",
                message="cell report content digest differs from ledger digest",
                cell=key,
                metric="execution_report_digest",
            )
        cell_results.append(
            _validate_cell(
                cell_report,
                execution_row=row,
                manifest=manifest,
                registries=registries,
                failures=failures,
            )
        )
        result = cell_results[-1]
        capability = result.get("capability", {})
        resources = result.get("resources", {})
        for metric, value_key in (
            ("candidate_task_success_rate", "candidate_task_success_rate"),
            (
                "candidate_minus_lesion_task_success_rate",
                "candidate_minus_lesion_task_success_rate",
            ),
        ):
            value = capability.get(value_key)
            if _finite_number(value):
                metric_values[metric].append(float(value))
        candidate_resource = resources.get("arm_resources", {}).get(
            "candidate-continuation"
        )
        matched_resource = resources.get("arm_resources", {}).get(
            "matched-fixed-capacity"
        )
        fixed_delta = resources.get("fixed_large_minus_candidate", {})
        if isinstance(candidate_resource, Mapping) and isinstance(
            matched_resource, Mapping
        ):
            for metric, resource, field in (
                (
                    "candidate_wall_clock_seconds",
                    candidate_resource,
                    "wall_clock_seconds",
                ),
                (
                    "matched_wall_clock_seconds",
                    matched_resource,
                    "wall_clock_seconds",
                ),
                (
                    "candidate_peak_working_set_bytes",
                    candidate_resource,
                    "peak_working_set_bytes",
                ),
                (
                    "matched_peak_working_set_bytes",
                    matched_resource,
                    "peak_working_set_bytes",
                ),
            ):
                metric_values[metric].append(float(resource[field]))
        ratios = resources.get("candidate_over_matched", {})
        if isinstance(ratios, Mapping):
            for metric, field in (
                (
                    "candidate_over_matched_wall_clock_multiplier",
                    "wall_clock_multiplier",
                ),
                (
                    "candidate_over_matched_peak_working_set_multiplier",
                    "peak_working_set_multiplier",
                ),
            ):
                value = ratios.get(field)
                if _finite_number(value):
                    metric_values[metric].append(float(cast(float, value)))
        if isinstance(fixed_delta, Mapping):
            for metric, field in (
                (
                    "fixed_large_minus_candidate_wall_clock_seconds",
                    "wall_clock_seconds",
                ),
                (
                    "fixed_large_minus_candidate_peak_working_set_bytes",
                    "peak_working_set_bytes",
                ),
                (
                    "fixed_large_minus_candidate_worker_parameter_bytes",
                    "worker_parameter_bytes",
                ),
                (
                    "fixed_large_minus_candidate_checkpoint_write_bytes",
                    "checkpoint_write_bytes",
                ),
                (
                    "fixed_large_minus_candidate_inference_trace_count",
                    "inference_trace_count",
                ),
            ):
                value = fixed_delta.get(field)
                if _finite_number(value):
                    metric_values[metric].append(float(cast(float, value)))

    for metric in (
        "candidate_minus_frozen_parent_task_success_rate",
        "candidate_minus_matched_fixed_capacity_task_success_rate",
    ):
        for key in EXPECTED_CELL_ORDER:
            _record_failure(
                failures,
                category="capability_gate",
                message=(
                    "paired capability delta is unavailable because the current "
                    "control is detached; no zero/imputed value is allowed"
                ),
                cell=key,
                metric=metric,
            )

    metrics = {
        metric: _mean_summary(values)
        for metric, values in metric_values.items()
        if values
    }
    return {
        "cells": cell_results,
        "metrics": metrics,
        "gates": {
            "all_cells_executed_passed": len(cell_results) == len(EXPECTED_CELL_ORDER)
            and all(result["status"] == "executed_passed" for result in cell_results),
            "candidate_holdout_floor": all(
                result.get("capability", {}).get("candidate_holdout_floor_pass") is True
                for result in cell_results
            ),
            "k3_lesion_breaks_gain": all(
                result.get("capability", {}).get("lesion_breaks_gain") is True
                for result in cell_results
            ),
            "candidate_over_matched_wall_budget": all(
                result.get("resources", {})
                .get("candidate_over_matched", {})
                .get("wall_clock_pass")
                is True
                for result in cell_results
            ),
            "candidate_over_matched_peak_budget": all(
                result.get("resources", {})
                .get("candidate_over_matched", {})
                .get("peak_working_set_pass")
                is True
                for result in cell_results
            ),
            "paired_frozen_parent_delta_available": False,
            "paired_matched_fixed_capacity_delta_available": False,
            "default_runtime_attached": False,
            "provider_attached": False,
            "mcp_attached": False,
            "client_attached": False,
            "cuda_used": False,
        },
    }


def run_aggregate(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    execution_report_path: Path = DEFAULT_EXECUTION_REPORT,
    report_path: Path = DEFAULT_REPORT,
    parent_dir: Path = DEFAULT_PARENT_DIR,
    worker_dir: Path = DEFAULT_WORKER_DIR,
    fixed_large_dir: Path = DEFAULT_FIXED_LARGE_DIR,
) -> dict[str, Any]:
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": "blocked_aggregate",
        "manifest_path": str(manifest_path),
        "execution_report_path": str(execution_report_path),
        "parent_dir": str(parent_dir),
        "worker_dir": str(worker_dir),
        "fixed_large_dir": str(fixed_large_dir),
        "can_start_r6_formal": False,
        "can_promote": False,
        "default_runtime_attached": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "cuda_used": False,
        "training_performed": False,
    }
    try:
        manifest = _load_object(manifest_path)
        execution = _load_object(execution_report_path)
        report["manifest_digest"] = manifest.get("manifest_digest")
        if manifest.get("manifest_digest") != _manifest_digest(manifest):
            _record_failure(
                failures,
                category="lineage",
                message="input manifest digest is invalid",
            )
        if execution.get("manifest_digest") != manifest.get("manifest_digest"):
            _record_failure(
                failures,
                category="lineage",
                message="execution report manifest digest differs from input manifest",
            )
        input_report_value = execution.get("input_preflight_report")
        if not isinstance(input_report_value, str):
            _record_failure(
                failures,
                category="input_contract",
                message="execution report has no input preflight report",
            )
        else:
            input_report = _load_object(_resolve_repo_path(input_report_value))
            if input_report.get("status") != "passed":
                _record_failure(
                    failures,
                    category="input_contract",
                    message="input manifest preflight is not passed",
                )
            if input_report.get("manifest_digest") != manifest.get("manifest_digest"):
                _record_failure(
                    failures,
                    category="lineage",
                    message="input preflight manifest digest differs",
                )
            formal_preflight_path = execution_report_path.with_name(
                "taiji_m4v2_r6_formal_preflight_20260909.json"
            )
            formal_preflight = _load_object(formal_preflight_path)
            if formal_preflight.get("status") != "input_ready":
                _record_failure(
                    failures,
                    category="input_contract",
                    message="formal runner preflight is not input_ready",
                )
            if formal_preflight.get("manifest_digest") != manifest.get("manifest_digest"):
                _record_failure(
                    failures,
                    category="lineage",
                    message="formal runner preflight manifest digest differs",
                )
            if content_digest(formal_preflight) != execution.get(
                "input_preflight_report_digest"
            ):
                _record_failure(
                    failures,
                    category="lineage",
                    message="formal runner preflight digest differs from execution ledger",
                )
        if execution.get("failures"):
            _record_failure(
                failures,
                category="execution",
                message="execution ledger contains failures",
            )
        report.update(
            _aggregate_reports(
                manifest=manifest,
                execution=execution,
                failures=failures,
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _record_failure(
            failures,
            category="aggregate_input",
            message=f"aggregate input could not be verified: {type(exc).__name__}: {exc}",
        )
        report.update({"cells": [], "metrics": {}, "gates": {}})
    report["blocking_failures"] = failures
    report["failure_counts"] = {
        category: sum(1 for failure in failures if failure["category"] == category)
        for category in sorted({failure["category"] for failure in failures})
    }
    report["status"] = "passed" if not failures else "blocked_aggregate"
    report["elapsed_seconds"] = time.perf_counter() - started
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execution-report", type=Path, default=DEFAULT_EXECUTION_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--worker-dir", type=Path, default=DEFAULT_WORKER_DIR)
    parser.add_argument("--fixed-large-dir", type=Path, default=DEFAULT_FIXED_LARGE_DIR)
    args = parser.parse_args(argv)

    def project_path(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    report_path = project_path(args.report)
    report = run_aggregate(
        manifest_path=project_path(args.manifest),
        execution_report_path=project_path(args.execution_report),
        report_path=report_path,
        parent_dir=project_path(args.parent_dir),
        worker_dir=project_path(args.worker_dir),
        fixed_large_dir=project_path(args.fixed_large_dir),
    )
    print(
        json.dumps(
            {
                "report": str(report_path.resolve().relative_to(PROJECT_ROOT.resolve())).replace(
                    "\\", "/"
                ),
                "status": report["status"],
                "cell_count": len(report.get("cells", [])),
                "failure_counts": report["failure_counts"],
                "can_start_r6_formal": report["can_start_r6_formal"],
                "can_promote": report["can_promote"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
