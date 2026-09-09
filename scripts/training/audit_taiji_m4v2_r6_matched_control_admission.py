"""Fail-closed admission audit for the M4.V2.R6 matched-control revision.

This audit consumes only the content-addressed revision manifest and its
aggregate report.  A pass authorizes starting the preregistered formal runner;
it never attaches a runtime owner, mutates a checkpoint, or promotes a model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r6_formal_input_manifest_preflight import (  # noqa: E402
    _manifest_digest,
    _resolve_repo_path,
)
from taiji import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-r6-matched-control-admission-v1"
REVISION_FORMAT = "taiji-m4v2-r6-matched-control-v2"
AGGREGATE_FORMAT = "taiji-m4v2-r6-matched-control-aggregate-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_r6_matched_control_v2_20260910.json"
)
DEFAULT_AGGREGATE = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_r6_matched_control_aggregate_20260910.json"
)
EXPECTED_ARMS = (
    "frozen-parent",
    "matched-fixed-capacity",
    "candidate-continuation",
    "fixed-large",
    "lesion",
)
EXPECTED_THRESHOLDS = {
    "candidate_holdout_floor": 0.75,
    "paired_capability_delta_floor": 0.25,
    "peak_working_set_multiplier_cap": 1.25,
    "wall_clock_multiplier_cap": 1.5,
}
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON report must be an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _record_failure(
    failures: list[dict[str, Any]],
    *,
    category: str,
    message: str,
    cell: tuple[int, int] | None = None,
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
            "metric": metric,
            "message": message,
        }
    )


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and HEX_DIGEST.fullmatch(value) is not None


def _index_registry(
    manifest: Mapping[str, Any],
    name: str,
    failures: list[dict[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    raw = manifest.get(name)
    if not isinstance(raw, list):
        _record_failure(
            failures,
            category="manifest",
            message=f"manifest {name} is not a list",
        )
        return {}
    indexed: dict[int, Mapping[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            _record_failure(
                failures,
                category="manifest",
                message=f"manifest {name} contains a non-object entry",
            )
            continue
        try:
            model_seed = int(entry["model_seed"])
        except (KeyError, TypeError, ValueError):
            _record_failure(
                failures,
                category="manifest",
                message=f"manifest {name} entry has no model_seed",
            )
            continue
        if model_seed in indexed:
            _record_failure(
                failures,
                category="manifest",
                message=f"manifest {name} duplicates model_seed={model_seed}",
            )
            continue
        indexed[model_seed] = entry
    return indexed


def _check_path(
    value: Any,
    *,
    failures: list[dict[str, Any]],
    category: str,
    message: str,
) -> None:
    try:
        path = _resolve_repo_path(value)
    except (TypeError, ValueError):
        _record_failure(failures, category=category, message=message)
        return
    if not path.is_file():
        _record_failure(
            failures,
            category=category,
            message=f"{message}; file is missing: {value}",
        )


def _check_manifest(
    manifest: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> tuple[list[int], list[int], Mapping[str, Any] | None]:
    model_seeds = manifest.get("model_seeds")
    course_seeds = manifest.get("course_seeds")
    models = (
        [int(value) for value in model_seeds]
        if isinstance(model_seeds, list) and all(isinstance(value, int) for value in model_seeds)
        else []
    )
    courses = (
        [int(value) for value in course_seeds]
        if isinstance(course_seeds, list) and all(isinstance(value, int) for value in course_seeds)
        else []
    )
    if models != [17, 23, 31] or courses != [0, 1, 2]:
        _record_failure(
            failures,
            category="manifest",
            message="revision manifest model/course seed registry differs from 3x3 preregistration",
        )
    if manifest.get("manifest_digest") != _manifest_digest(manifest):
        _record_failure(
            failures,
            category="lineage",
            message="revision manifest content digest is invalid",
        )
    revision = manifest.get("control_revision")
    if not isinstance(revision, Mapping) or revision.get("format") != REVISION_FORMAT:
        _record_failure(
            failures,
            category="manifest",
            message="manifest is not the matched-control revision",
        )
        revision = None
    else:
        thresholds = revision.get("thresholds_unchanged")
        if thresholds != EXPECTED_THRESHOLDS:
            _record_failure(
                failures,
                category="threshold",
                message="revision threshold block differs from the frozen thresholds",
            )
    resource_contract = manifest.get("resource_contract")
    expected_resource_contract = {
        "candidate_parameter_bytes_frozen_before_run": True,
        "checkpoint_write_bytes_frozen_before_run": True,
        "cuda_required": False,
        "device": "cpu",
        "frozen_parent_explicit_k_baseline": True,
        "inference_trace_count_frozen_before_run": True,
        "matched_capacity_worker_attached": True,
        "peak_working_set_multiplier_cap": 1.25,
        "training_update_steps_frozen_before_run": True,
        "wall_clock_multiplier_cap": 1.5,
    }
    if resource_contract != expected_resource_contract:
        _record_failure(
            failures,
            category="resource",
            message="manifest resource contract is not the frozen CPU revision contract",
        )
    side_effect_contract = manifest.get("side_effect_contract")
    expected_side_effect_contract = {
        "candidate_namespace_isolated": True,
        "client_attached": False,
        "cuda_used": False,
        "default_runtime_attached": False,
        "mcp_attached": False,
        "network_used": False,
        "parent_namespace_read_only": True,
        "provider_attached": False,
        "read_only_workbench_routes": True,
        "rollback_required": True,
    }
    if side_effect_contract != expected_side_effect_contract:
        _record_failure(
            failures,
            category="side_effect",
            message="manifest side-effect contract is not isolated",
        )
    return models, courses, revision


def _check_registries(
    manifest: Mapping[str, Any],
    *,
    models: list[int],
    failures: list[dict[str, Any]],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    registries = {
        name: _index_registry(manifest, name, failures)
        for name in ("parent_registry", "worker_registry", "fixed_large_registry")
    }
    for name, registry in registries.items():
        if set(registry) != set(models):
            _record_failure(
                failures,
                category="lineage",
                message=f"{name} does not cover exactly the registered model seeds",
            )
    for model_seed in models:
        parent = registries["parent_registry"].get(model_seed, {})
        worker = registries["worker_registry"].get(model_seed, {})
        fixed_large = registries["fixed_large_registry"].get(model_seed, {})
        for entry, fields in (
            (
                parent,
                (
                    "checkpoint_digest",
                    "fresh_restore_digest",
                    "owner_graph_digest",
                    "resource_manifest_digest",
                    "source_manifest_digest",
                ),
            ),
            (
                worker,
                (
                    "bundle_digest",
                    "owner_graph_digest",
                    "parent_checkpoint_digest",
                    "resource_manifest_digest",
                    "source_manifest_digest",
                ),
            ),
            (
                fixed_large,
                (
                    "artifact_digest",
                    "ensemble_checkpoint_digest",
                    "k3_checkpoint_digest",
                    "owner_graph_digest",
                    "parent_checkpoint_digest",
                    "resource_manifest_digest",
                    "source_manifest_digest",
                ),
            ),
        ):
            for field in fields:
                if not _valid_digest(entry.get(field)):
                    _record_failure(
                        failures,
                        category="lineage",
                        message=f"model {model_seed} registry field {field} is not a digest",
                    )
        parent_digest = parent.get("checkpoint_digest")
        if worker.get("parent_checkpoint_digest") != parent_digest:
            _record_failure(
                failures,
                category="lineage",
                message=f"model {model_seed} worker is not rooted at its parent digest",
            )
        if fixed_large.get("parent_checkpoint_digest") != parent_digest:
            _record_failure(
                failures,
                category="lineage",
                message=f"model {model_seed} fixed-large is not rooted at its parent digest",
            )
        _check_path(
            parent.get("checkpoint_path"),
            failures=failures,
            category="checkpoint",
            message=f"model {model_seed} parent checkpoint path is invalid",
        )
        workers = worker.get("workers")
        if not isinstance(workers, Mapping) or set(workers) != {
            "k1.semantic",
            "k2.transition",
            "k3.outcome_projection",
        }:
            _record_failure(
                failures,
                category="checkpoint",
                message=f"model {model_seed} worker registry is incomplete",
            )
        else:
            for worker_id, worker_entry in workers.items():
                if not isinstance(worker_entry, Mapping):
                    _record_failure(
                        failures,
                        category="checkpoint",
                        message=f"model {model_seed} {worker_id} worker entry is malformed",
                    )
                    continue
                if not all(
                    _valid_digest(worker_entry.get(field))
                    for field in ("artifact_digest", "worker_checkpoint_digest")
                ):
                    _record_failure(
                        failures,
                        category="lineage",
                        message=f"model {model_seed} {worker_id} worker digest is invalid",
                    )
                _check_path(
                    worker_entry.get("path"),
                    failures=failures,
                    category="checkpoint",
                    message=f"model {model_seed} {worker_id} checkpoint path is invalid",
                )
        _check_path(
            fixed_large.get("artifact_path"),
            failures=failures,
            category="checkpoint",
            message=f"model {model_seed} fixed-large artifact path is invalid",
        )
    return registries


def _check_aggregate(
    aggregate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    models: list[int],
    courses: list[int],
    registries: dict[str, dict[int, Mapping[str, Any]]],
    revision: Mapping[str, Any] | None,
    failures: list[dict[str, Any]],
) -> dict[str, bool]:
    if aggregate.get("report_format") != AGGREGATE_FORMAT:
        _record_failure(
            failures,
            category="aggregate",
            message="aggregate report format is not the matched-control revision format",
        )
    if aggregate.get("status") != "passed" or aggregate.get("blocking_failures") != []:
        _record_failure(
            failures,
            category="aggregate",
            message="aggregate report is not a clean passed report",
        )
    if aggregate.get("manifest_digest") != manifest.get("manifest_digest"):
        _record_failure(
            failures,
            category="lineage",
            message="aggregate manifest digest differs from the revision manifest",
        )
    if aggregate.get("control_revision") != revision:
        _record_failure(
            failures,
            category="lineage",
            message="aggregate control revision differs from the revision manifest",
        )
    if aggregate.get("can_start_r6_formal") is not False or aggregate.get("can_promote") is not False:
        _record_failure(
            failures,
            category="boundary",
            message="aggregate report is already claiming formal start or promotion",
        )

    aggregate_gates = aggregate.get("gates")
    required_gate_names = (
        "all_cells_executed_passed",
        "candidate_holdout_floor",
        "k3_lesion_breaks_gain",
        "candidate_over_matched_wall_budget",
        "candidate_over_matched_peak_budget",
        "paired_frozen_parent_delta_available",
        "paired_frozen_parent_delta_floor",
        "paired_matched_fixed_capacity_delta_available",
        "paired_matched_fixed_capacity_delta_floor",
    )
    checks = {
        name: isinstance(aggregate_gates, Mapping) and aggregate_gates.get(name) is True
        for name in required_gate_names
    }
    for name, passed in checks.items():
        if not passed:
            _record_failure(
                failures,
                category="aggregate_gate",
                message=f"aggregate gate {name} is not true",
                metric=name,
            )
    if not (
        aggregate_gates.get("default_runtime_attached") is False
        and aggregate_gates.get("provider_attached") is False
        and aggregate_gates.get("mcp_attached") is False
        and aggregate_gates.get("client_attached") is False
        and aggregate_gates.get("cuda_used") is False
    ) if isinstance(aggregate_gates, Mapping) else True:
        _record_failure(
            failures,
            category="side_effect",
            message="aggregate external/runtime boundary is not closed",
        )

    expected_order = [
        {"model_seed": model_seed, "course_seed": course_seed}
        for model_seed in models
        for course_seed in courses
    ]
    cells = aggregate.get("cells")
    if not isinstance(cells, list) or [cell.get("cell") for cell in cells if isinstance(cell, Mapping)] != expected_order:
        _record_failure(
            failures,
            category="aggregate",
            message="aggregate cell order or identity differs from the 3x3 preregistration",
        )
        cells = []
    delta_floor = (
        _finite_float(revision.get("thresholds_unchanged", {}).get("paired_capability_delta_floor"))
        if isinstance(revision, Mapping)
        and isinstance(revision.get("thresholds_unchanged"), Mapping)
        else None
    )
    for raw_cell in cells:
        if not isinstance(raw_cell, Mapping):
            continue
        identity = raw_cell.get("cell")
        if not isinstance(identity, Mapping):
            continue
        cell = (int(identity["model_seed"]), int(identity["course_seed"]))
        if raw_cell.get("status") != "executed_passed":
            _record_failure(
                failures,
                category="aggregate",
                message="aggregate cell is not executed_passed",
                cell=cell,
            )
        contract_digest = raw_cell.get("execution_contract_digest")
        if not _valid_digest(contract_digest):
            _record_failure(
                failures,
                category="lineage",
                message="aggregate cell execution contract digest is invalid",
                cell=cell,
            )
        try:
            _resolve_repo_path(raw_cell.get("execution_report_path"))
        except (TypeError, ValueError):
            _record_failure(
                failures,
                category="lineage",
                message="aggregate cell execution report path is invalid",
                cell=cell,
            )
        capability = raw_cell.get("capability")
        if not isinstance(capability, Mapping):
            _record_failure(
                failures,
                category="capability",
                message="aggregate cell has no capability summary",
                cell=cell,
            )
        else:
            candidate_rate = _finite_float(capability.get("candidate_task_success_rate"))
            lesion_rate = _finite_float(capability.get("lesion_task_success_rate"))
            frozen_delta = _finite_float(
                capability.get("candidate_minus_frozen_parent_task_success_rate")
            )
            matched_delta = _finite_float(
                capability.get("candidate_minus_matched_fixed_capacity_task_success_rate")
            )
            if candidate_rate is None or candidate_rate < EXPECTED_THRESHOLDS["candidate_holdout_floor"]:
                _record_failure(
                    failures,
                    category="capability",
                    message="aggregate candidate rate is below the frozen floor",
                    cell=cell,
                )
            if lesion_rate is None or candidate_rate is None or candidate_rate <= lesion_rate:
                _record_failure(
                    failures,
                    category="causal",
                    message="aggregate candidate rate does not exceed lesion rate",
                    cell=cell,
                )
            if delta_floor is None or frozen_delta is None or frozen_delta < delta_floor:
                _record_failure(
                    failures,
                    category="capability",
                    message="aggregate frozen-parent paired delta is below floor",
                    cell=cell,
                )
            if delta_floor is None or matched_delta is None or matched_delta < delta_floor:
                _record_failure(
                    failures,
                    category="capability",
                    message="aggregate matched-capacity paired delta is below floor",
                    cell=cell,
                )
        resources = raw_cell.get("resources")
        arm_resources = resources.get("arm_resources") if isinstance(resources, Mapping) else None
        if not isinstance(arm_resources, Mapping) or set(arm_resources) != set(EXPECTED_ARMS):
            _record_failure(
                failures,
                category="resource",
                message="aggregate cell arm resource set is incomplete",
                cell=cell,
            )
            continue
        expected_parent = registries["parent_registry"].get(cell[0], {}).get(
            "resource_manifest_digest"
        )
        expected_worker = registries["worker_registry"].get(cell[0], {}).get(
            "resource_manifest_digest"
        )
        expected_fixed = registries["fixed_large_registry"].get(cell[0], {}).get(
            "resource_manifest_digest"
        )
        for arm_id, expected_digest in (
            ("frozen-parent", expected_parent),
            ("matched-fixed-capacity", expected_worker),
            ("candidate-continuation", expected_worker),
            ("lesion", expected_worker),
            ("fixed-large", expected_fixed),
        ):
            resource = arm_resources[arm_id]
            if not isinstance(resource, Mapping) or resource.get("resource_manifest_digest") != expected_digest:
                _record_failure(
                    failures,
                    category="lineage",
                    message=f"aggregate {arm_id} resource owner does not match manifest registry",
                    cell=cell,
                    metric="resource_manifest_digest",
                )
        candidate_resource = arm_resources["candidate-continuation"]
        matched_resource = arm_resources["matched-fixed-capacity"]
        if isinstance(candidate_resource, Mapping) and isinstance(matched_resource, Mapping):
            for field in (
                "candidate_parameter_bytes",
                "worker_parameter_count",
                "worker_parameter_bytes",
                "inference_trace_count",
                "training_update_steps",
            ):
                if candidate_resource.get(field) != matched_resource.get(field):
                    _record_failure(
                        failures,
                        category="resource",
                        message=f"aggregate candidate/matched {field} is not equal",
                        cell=cell,
                        metric=field,
                    )
    metrics = aggregate.get("metrics")
    for metric in (
        "candidate_minus_frozen_parent_task_success_rate",
        "candidate_minus_matched_fixed_capacity_task_success_rate",
    ):
        summary = metrics.get(metric) if isinstance(metrics, Mapping) else None
        if not isinstance(summary, Mapping) or summary.get("n") != len(models) * len(courses):
            _record_failure(
                failures,
                category="aggregate",
                message=f"aggregate metric {metric} does not cover all 9 cells",
                metric=metric,
            )
        lower_bound = _finite_float(
            summary.get("one_sided_95_student_t_lower_bound")
            if isinstance(summary, Mapping)
            else None
        )
        if delta_floor is None or lower_bound is None or lower_bound < delta_floor:
            _record_failure(
                failures,
                category="aggregate_gate",
                message=f"aggregate metric {metric} lower bound is below floor",
                metric=metric,
            )
    return checks


def audit(
    *,
    manifest: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    manifest_path: Path,
    aggregate_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    models, courses, revision = _check_manifest(manifest, failures)
    registries = _check_registries(manifest, models=models, failures=failures)
    checks = _check_aggregate(
        aggregate,
        manifest,
        models=models,
        courses=courses,
        registries=registries,
        revision=revision,
        failures=failures,
    )
    passed = not failures
    return {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": "passed" if passed else "blocked_admission",
        "manifest_path": str(manifest_path),
        "aggregate_path": str(aggregate_path),
        "manifest_digest": manifest.get("manifest_digest"),
        "aggregate_digest": content_digest(aggregate),
        "control_revision": revision,
        "sources": {
            "manifest": {
                "path": str(manifest_path),
                "content_digest": content_digest(manifest),
            },
            "aggregate": {
                "path": str(aggregate_path),
                "content_digest": content_digest(aggregate),
            },
        },
        "checks": checks,
        "blocking_failures": failures,
        "failure_counts": {
            category: sum(1 for failure in failures if failure["category"] == category)
            for category in sorted({failure["category"] for failure in failures})
        },
        "can_start_r6_formal": passed,
        "can_promote": False,
        "default_runtime_attached": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "cuda_used": False,
        "training_performed": False,
        "admission": "ready_to_start_formal" if passed else "blocked",
        "allowed_next_action": (
            "Run the preregistered R6 formal runner with the same manifest; do not attach "
            "default runtime/provider/MCP/client/CUDA and do not promote before formal results."
            if passed
            else "Resolve every blocking admission failure; do not start formal or promote."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    def project_path(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    manifest_path = project_path(args.manifest)
    aggregate_path = project_path(args.aggregate)
    report_path = project_path(args.report)
    try:
        report = audit(
            manifest=_load_object(manifest_path),
            aggregate=_load_object(aggregate_path),
            manifest_path=manifest_path,
            aggregate_path=aggregate_path,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "report_format": REPORT_FORMAT,
            "version": VERSION,
            "status": "blocked_admission",
            "manifest_path": str(manifest_path),
            "aggregate_path": str(aggregate_path),
            "blocking_failures": [
                {
                    "category": "admission_input",
                    "cell": None,
                    "metric": None,
                    "message": f"admission input could not be verified: {type(exc).__name__}: {exc}",
                }
            ],
            "failure_counts": {"admission_input": 1},
            "can_start_r6_formal": False,
            "can_promote": False,
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path.resolve().relative_to(PROJECT_ROOT.resolve())).replace(
                    "\\", "/"
                ),
                "status": report["status"],
                "can_start_r6_formal": report["can_start_r6_formal"],
                "can_promote": report["can_promote"],
                "failure_counts": report["failure_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
