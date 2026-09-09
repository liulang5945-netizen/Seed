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
    MEASUREMENT_SEMANTICS_FORMAT,
    _resource_gate_valid,
)
from taiji import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-r6-formal-aggregate-v1"
MATCHED_CONTROL_REPORT_FORMAT = "taiji-m4v2-r6-matched-control-aggregate-v1"
MATCHED_CONTROL_REVISION_FORMAT = "taiji-m4v2-r6-matched-control-v2"
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


def _finite_float(value: Any) -> float | None:
    return float(value) if _finite_number(value) else None


def _paired_delta_floor(control_revision: Any) -> float | None:
    if not isinstance(control_revision, Mapping):
        return None
    thresholds = control_revision.get("thresholds_unchanged")
    if not isinstance(thresholds, Mapping):
        return None
    return _finite_float(thresholds.get("paired_capability_delta_floor"))


def _learning_formal_ready(report: Mapping[str, Any]) -> bool:
    """Return whether a report is eligible for a learning, not wiring, gate."""

    return bool(
        report.get("run_kind") == "learning-formal"
        and report.get("training_performed") is True
        and report.get("candidate_training_performed") is True
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
    matched_control_revision: bool = False,
) -> str:
    if arm == "frozen-parent" or (
        arm == "matched-fixed-capacity" and not matched_control_revision
    ):
        return str(parent["resource_manifest_digest"])
    if arm in {"candidate-continuation", "lesion"} or (
        arm == "matched-fixed-capacity" and matched_control_revision
    ):
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
    worker_bound: bool = False,
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
    if worker_bound and arm.get("worker_bundle_digest") != expected_worker_digest:
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


def _phase_row(arm: Mapping[str, Any], phase: str) -> Mapping[str, Any] | None:
    rows = arm.get("phase_rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get("phase") == phase:
            return row
    return None


def _validate_matched_control_semantics(
    arms: Mapping[str, Any],
    *,
    cell: tuple[int, int],
    expected_parent_digest: str,
    failures: list[dict[str, Any]],
) -> None:
    """Validate the v2 controls instead of treating matched as a detached null arm."""

    def failure(arm: str, metric: str, message: str) -> None:
        _record_failure(
            failures,
            category="causal_gate",
            message=message,
            cell=cell,
            arm=arm,
            metric=metric,
        )

    frozen = arms["frozen-parent"]
    frozen_causal = frozen.get("causal")
    if frozen_causal != {
        "k_feedback_consumed": False,
        "k_task_admission": False,
        "status": "frozen_parent_zero_baseline",
    }:
        failure("frozen-parent", "causal", "frozen parent is not the explicit K zero baseline")
    frozen_resource = frozen.get("resource")
    if not isinstance(frozen_resource, Mapping) or any(
        frozen_resource.get(field) != 0
        for field in (
            "candidate_parameter_bytes",
            "checkpoint_write_bytes",
            "training_update_steps",
            "worker_parameter_bytes",
            "worker_parameter_count",
        )
    ):
        failure("frozen-parent", "resource", "frozen parent has non-zero K worker/update resource")
    frozen_k = _phase_row(frozen, "K")
    if not isinstance(frozen_k, Mapping) or frozen_k.get("status") != "rejected":
        failure("frozen-parent", "K", "frozen parent K phase was not explicitly rejected")
    frozen_checkpoint = frozen.get("checkpoint_ledger")
    if not isinstance(frozen_checkpoint, Mapping) or not (
        frozen_checkpoint.get("fresh_restore_digest") == expected_parent_digest
        and frozen_checkpoint.get("rollback_digest") == expected_parent_digest
        and frozen_checkpoint.get("rollback_matches_parent") is True
    ):
        failure("frozen-parent", "checkpoint_restore", "frozen parent did not restore and roll back to parent")

    matched = arms["matched-fixed-capacity"]
    matched_causal = matched.get("causal")
    matched_expected = {
        "feedback_admitted": False,
        "k3_projection_accepted": True,
        "lesion_k3": False,
        "real_workbench_success": True,
        "status": "matched_capacity_no_feedback",
    }
    if not isinstance(matched_causal, Mapping) or any(
        matched_causal.get(key) != value for key, value in matched_expected.items()
    ):
        failure("matched-fixed-capacity", "causal", "matched arm is not the no-feedback capacity control")
    matched_k = _phase_row(matched, "K")
    if not isinstance(matched_k, Mapping) or matched_k.get("status") != "executed_no_feedback":
        failure("matched-fixed-capacity", "K", "matched arm did not stop after K execution without feedback")
    matched_checkpoint = matched.get("checkpoint_ledger")
    if not isinstance(matched_checkpoint, Mapping) or not (
        matched_checkpoint.get("exchange_digest") is None
        and matched_checkpoint.get("rollback_record_explicit") is False
        and isinstance(matched_checkpoint.get("adapter_checkpoint"), Mapping)
        and matched_checkpoint["adapter_checkpoint"].get("exchange_checkpoint_digest")
        == matched_checkpoint["adapter_checkpoint"].get("rollback_checkpoint_digest")
    ):
        failure("matched-fixed-capacity", "checkpoint_restore", "matched arm created an exchange or lacked no-feedback rollback semantics")

    candidate = arms["candidate-continuation"]
    candidate_causal = candidate.get("causal")
    if not isinstance(candidate_causal, Mapping) or not (
        candidate_causal.get("feedback_admitted") is True
        and candidate_causal.get("k3_projection_accepted") is True
        and candidate_causal.get("status") == "candidate_path_verified"
    ):
        failure("candidate-continuation", "causal", "candidate arm did not admit the K feedback path")
    candidate_k = _phase_row(candidate, "K")
    if not isinstance(candidate_k, Mapping) or candidate_k.get("status") != "executed":
        failure("candidate-continuation", "K", "candidate K phase was not executed")

    lesion = arms["lesion"]
    lesion_causal = lesion.get("causal")
    if not isinstance(lesion_causal, Mapping) or not (
        lesion_causal.get("feedback_admitted") is False
        and lesion_causal.get("lesion_k3") is True
        and lesion_causal.get("k3_projection_accepted") is False
        and lesion_causal.get("status") == "lesion_verified"
    ):
        failure("lesion", "causal", "lesion arm did not disable K3 outcome feedback")
    lesion_k = _phase_row(lesion, "K")
    if not isinstance(lesion_k, Mapping) or lesion_k.get("status") != "lesion_rejected":
        failure("lesion", "K", "lesion K phase was not rejected")

    fixed_large = arms["fixed-large"]
    fixed_causal = fixed_large.get("causal")
    if not isinstance(fixed_causal, Mapping) or not (
        fixed_causal.get("k_task_equivalent") is True
        and fixed_causal.get("branch_lesion_observable") is True
        and fixed_causal.get("status") == "fixed_large_control_verified"
    ):
        failure("fixed-large", "causal", "fixed-large arm is not the K-task-equivalent control")


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
    control_revision = manifest.get("control_revision")
    matched_control_revision = (
        isinstance(control_revision, Mapping)
        and control_revision.get("format") == MATCHED_CONTROL_REVISION_FORMAT
    )
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
    if matched_control_revision:
        if cell_report.get("measurement_semantics") != MEASUREMENT_SEMANTICS_FORMAT:
            _record_failure(
                failures,
                category="input_contract",
                message=(
                    "cell does not declare the task-success measurement semantics; "
                    "legacy feedback-as-success reports are not admissible"
                ),
                cell=cell,
                metric="measurement_semantics",
            )
        if not _learning_formal_ready(cell_report):
            _record_failure(
                failures,
                category="learning_gate",
                message=(
                    "cell is a wiring canary or has no candidate parameter updates; "
                    "it cannot satisfy a learning formal gate"
                ),
                cell=cell,
                metric="training_performed",
            )
        if cell_report.get("manifest_digest") != manifest.get("manifest_digest"):
            _record_failure(
                failures,
                category="lineage",
                message="revised cell manifest digest differs from the input manifest",
                cell=cell,
                metric="manifest_digest",
            )
        if cell_report.get("control_revision") != control_revision:
            _record_failure(
                failures,
                category="input_contract",
                message="revised cell control revision differs from the input manifest",
                cell=cell,
                metric="control_revision",
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
                arm_id,
                parent=parent,
                worker=worker,
                fixed_large=fixed_large,
                matched_control_revision=matched_control_revision,
            ),
            expected_device=expected_device,
            worker_bound=(
                arm_id in {"candidate-continuation", "lesion"}
                or (matched_control_revision and arm_id == "matched-fixed-capacity")
            ),
            failures=failures,
        )

    candidate_rate = arm_summaries["candidate-continuation"]["task_success_rate"]
    fixed_large_rate = arm_summaries["fixed-large"]["task_success_rate"]
    lesion_rate = arm_summaries["lesion"]["task_success_rate"]
    frozen_rate = arm_summaries["frozen-parent"]["task_success_rate"]
    matched_rate = arm_summaries["matched-fixed-capacity"]["task_success_rate"]
    if matched_control_revision:
        _validate_matched_control_semantics(
            raw_arms,
            cell=cell,
            expected_parent_digest=expected_parent_digest,
            failures=failures,
        )
    rate_pairs = [
        ("candidate-continuation", candidate_rate),
        ("fixed-large", fixed_large_rate),
        ("lesion", lesion_rate),
    ]
    if matched_control_revision:
        # The frozen parent never attempts K.  Its missing value is coverage
        # information, not a zero-success behavioral observation.
        rate_pairs.append(("matched-fixed-capacity", matched_rate))
    for arm_id, rate in rate_pairs:
        if not _finite_number(rate):
            _record_failure(
                failures,
                category="causal",
                message="observed K arm has no finite task_success_rate",
                cell=cell,
                arm=arm_id,
                metric="task_success_rate",
            )
    candidate_task_floor_pass = _finite_number(candidate_rate) and float(candidate_rate) >= 0.75
    if not candidate_task_floor_pass:
        _record_failure(
            failures,
            category="capability_gate",
            message="candidate K task success is below the preregistered 0.75 floor",
            cell=cell,
            arm="candidate-continuation",
            metric="task_success_rate",
        )
    candidate_causal = raw_arms["candidate-continuation"].get("causal")
    lesion_causal = raw_arms["lesion"].get("causal")
    feedback_path_lesion_pass = (
        isinstance(candidate_causal, Mapping)
        and isinstance(lesion_causal, Mapping)
        and candidate_causal.get("feedback_admitted") is True
        and candidate_causal.get("k3_projection_accepted") is True
        and lesion_causal.get("feedback_admitted") is False
        and lesion_causal.get("k3_projection_accepted") is False
    )
    if not feedback_path_lesion_pass:
        _record_failure(
            failures,
            category="causal",
            message="K3 lesion does not isolate the feedback path",
            cell=cell,
            metric="feedback_path_lesion_pass",
        )
    paired_frozen_delta = None
    paired_matched_delta = (
        None
        if not (_finite_number(candidate_rate) and _finite_number(matched_rate))
        else float(candidate_rate) - float(matched_rate)
    )
    if matched_control_revision:
        delta_floor = _paired_delta_floor(control_revision)
        if delta_floor is None:
            _record_failure(
                failures,
                category="input_contract",
                message="revised control has no finite paired capability delta floor",
                cell=cell,
                metric="paired_capability_delta_floor",
            )
        else:
            for arm_id, delta in (("matched-fixed-capacity", paired_matched_delta),):
                delta_value = _finite_float(delta)
                if delta_value is None or delta_value < delta_floor:
                    _record_failure(
                        failures,
                        category="capability_gate",
                        message=(
                            f"candidate/{arm_id} capability delta {delta!r} is below "
                            f"floor {delta_floor:.6f}"
                        ),
                        cell=cell,
                        arm="candidate-continuation",
                        metric=f"candidate_minus_{arm_id}_task_success_rate",
                    )

    candidate_resource = arm_summaries["candidate-continuation"]["resource"]
    matched_resource = arm_summaries["matched-fixed-capacity"]["resource"]
    fixed_resource = arm_summaries["fixed-large"]["resource"]
    resources: dict[str, Any] = {}
    if all(
        isinstance(item, Mapping)
        for item in (candidate_resource, matched_resource, fixed_resource)
    ):
        if matched_control_revision:
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
                        category="resource_gate",
                        message=(
                            f"candidate/matched {field} differs: "
                            f"{candidate_resource.get(field)!r} != "
                            f"{matched_resource.get(field)!r}"
                        ),
                        cell=cell,
                        arm="matched-fixed-capacity",
                        metric=field,
                    )
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
            "candidate_task_success_floor_pass": candidate_task_floor_pass,
            "candidate_holdout_floor_pass": None,
            "candidate_minus_lesion_task_success_rate": (
                None
                if not (_finite_number(candidate_rate) and _finite_number(lesion_rate))
                else float(candidate_rate) - float(lesion_rate)
            ),
            "feedback_path_lesion_pass": feedback_path_lesion_pass,
            "lesion_breaks_gain": None,
            "candidate_minus_frozen_parent_task_success_rate": paired_frozen_delta,
            "candidate_minus_matched_fixed_capacity_task_success_rate": paired_matched_delta,
            "frozen_parent_task_not_attempted": frozen_rate is None,
        },
        "resources": resources,
    }


def _aggregate_reports(
    *,
    manifest: Mapping[str, Any],
    execution: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    control_revision = manifest.get("control_revision")
    matched_control_revision = (
        isinstance(control_revision, Mapping)
        and control_revision.get("format") == MATCHED_CONTROL_REVISION_FORMAT
    )
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
        "candidate_minus_frozen_parent_task_success_rate": [],
        "candidate_minus_matched_fixed_capacity_task_success_rate": [],
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
            (
                "candidate_minus_frozen_parent_task_success_rate",
                "candidate_minus_frozen_parent_task_success_rate",
            ),
            (
                "candidate_minus_matched_fixed_capacity_task_success_rate",
                "candidate_minus_matched_fixed_capacity_task_success_rate",
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

    aggregate_delta_floor = _paired_delta_floor(control_revision)
    metrics = {
        metric: _mean_summary(values)
        for metric, values in metric_values.items()
        if values
    }
    if matched_control_revision:
        if aggregate_delta_floor is not None:
            for metric in ("candidate_minus_matched_fixed_capacity_task_success_rate",):
                summary = metrics.get(metric)
                lower_bound = (
                    summary.get("one_sided_95_student_t_lower_bound")
                    if isinstance(summary, Mapping)
                    else None
                )
                lower_bound_value = _finite_float(lower_bound)
                if (
                    lower_bound_value is None
                    or lower_bound_value < aggregate_delta_floor
                ):
                    _record_failure(
                        failures,
                        category="aggregate_gate",
                        message=(
                            f"{metric} one-sided 95% lower bound {lower_bound!r} is below "
                            f"floor {aggregate_delta_floor:.6f}"
                        ),
                        metric=metric,
                    )
    else:
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
    return {
        "cells": cell_results,
        "metrics": metrics,
        "gates": {
            "all_cells_executed_passed": len(cell_results) == len(EXPECTED_CELL_ORDER)
            and all(result["status"] == "executed_passed" for result in cell_results),
            "candidate_task_success_floor": all(
                result.get("capability", {}).get("candidate_task_success_floor_pass") is True
                for result in cell_results
            ),
            "candidate_holdout_floor": False,
            "k3_lesion_isolates_feedback_path": all(
                result.get("capability", {}).get("feedback_path_lesion_pass") is True
                for result in cell_results
            ),
            "k3_lesion_breaks_gain": False,
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
            "paired_frozen_parent_delta_available": (
                False
            ),
            "paired_matched_fixed_capacity_delta_available": (
                matched_control_revision
                and len(
                    metric_values["candidate_minus_matched_fixed_capacity_task_success_rate"]
                )
                == len(EXPECTED_CELL_ORDER)
            ),
            "paired_frozen_parent_delta_floor": (
                None
            ),
            "paired_matched_fixed_capacity_delta_floor": (
                matched_control_revision
                and aggregate_delta_floor is not None
                and all(
                    value >= aggregate_delta_floor
                    for value in metric_values[
                        "candidate_minus_matched_fixed_capacity_task_success_rate"
                    ]
                )
            ),
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
        control_revision = manifest.get("control_revision")
        matched_control_revision = (
            isinstance(control_revision, Mapping)
            and control_revision.get("format") == MATCHED_CONTROL_REVISION_FORMAT
        )
        if matched_control_revision:
            report["report_format"] = MATCHED_CONTROL_REPORT_FORMAT
            report["control_revision"] = control_revision
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
        if matched_control_revision and execution.get("control_revision") != control_revision:
            _record_failure(
                failures,
                category="input_contract",
                message="execution report control revision differs from input manifest",
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
            formal_preflight_value = execution.get("formal_preflight_report")
            formal_preflight_path = (
                _resolve_repo_path(formal_preflight_value)
                if isinstance(formal_preflight_value, str)
                else execution_report_path.with_name(
                    "taiji_m4v2_r6_formal_preflight_20260909.json"
                )
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
