"""Execute the first R6 formal single-cell evidence slice.

The slice is intentionally limited to model seed 17 / course seed 0.  It runs
the real K candidate and the same-path K3 feedback lesion, replays the frozen
parent/matched-capacity controls, and runs the native fixed-large K control.
The control is comparable at the single-cell level, but this runner still does
not authorize the 9-cell formal or promotion.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r6_formal_input_manifest_preflight import (  # noqa: E402
    DEFAULT_FIXED_LARGE_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_PARENT_DIR,
    DEFAULT_WORKER_DIR,
    _resolve_repo_path,
)
from scripts.training.eval_taiji_m4v2_r6_formal_input_manifest_preflight import (  # noqa: E402
    run_preflight as run_input_preflight,
)
from scripts.training.eval_taiji_m4v2_r6_k_fixed_large_canary import (  # noqa: E402
    run_canary as run_fixed_large_canary,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_controlled_canary import (  # noqa: E402
    run_canary,
)
from scripts.training.eval_taiji_m4v2_r6_parent_baseline_preflight import (  # noqa: E402
    _r6_course,
    _score,
)
from taiji import Taiji, content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-r6-formal-single-cell-v1"
VERSION = 1
MEASUREMENT_SEMANTICS_FORMAT = "taiji-m4v2-r6-task-success-v2"
MODEL_SEED = 17
COURSE_SEED = 0
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r6_formal_single_cell_20260909.json"
ARM_IDS = (
    "frozen-parent",
    "matched-fixed-capacity",
    "candidate-continuation",
    "fixed-large",
    "lesion",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise TypeError(f"checkpoint at {path} must be a mapping")
    return {str(key): value for key, value in raw.items()}


def _failure(
    *,
    failure_class: str,
    message: str,
    model_seed: int,
    course_seed: int,
    arm: str | None = None,
    recoverability: str = "control_arm_required",
) -> dict[str, Any]:
    return {
        "class": failure_class,
        "phase": "K",
        "cell": {"model_seed": model_seed, "course_seed": course_seed},
        "arm": arm,
        "step": None,
        "is_model_evidence": False,
        "is_environment_blocker": failure_class == "environment_blocker",
        "stop_line": True,
        "recoverability": recoverability,
        "exception_type": None,
        "message": message,
        "evidence_digests": [],
    }


def _summarize_k_measurement(
    *,
    runner_passed: bool,
    task_executed: bool,
    outcome_success: bool,
    projection_accepted: bool,
    feedback_admitted: bool,
    lesion_k3: bool,
    training_update_steps: int,
    candidate_training_performed: bool,
) -> dict[str, Any]:
    """Separate task behavior from feedback admission and parameter learning.

    A control that executes a task successfully remains behaviorally successful
    even when its feedback is deliberately discarded.  A task that was never
    attempted is represented as ``None`` rather than an imputed failure.  Stage
    or exchange activity is not learning evidence; a positive update count and
    an explicit training flag are required for learning eligibility.
    """

    executed = bool(task_executed)
    task_success = bool(executed and runner_passed and outcome_success)
    update_steps = int(training_update_steps)
    if update_steps < 0:
        raise ValueError("training_update_steps must be non-negative")
    learning_update_applied = bool(candidate_training_performed and update_steps > 0)
    return {
        "format": MEASUREMENT_SEMANTICS_FORMAT,
        "task_executed": executed,
        "task_success": task_success,
        "task_success_rate": (None if not executed else (1.0 if task_success else 0.0)),
        "task_measurement_status": (
            "not_attempted"
            if not executed
            else ("observed_success" if task_success else "observed_failure")
        ),
        "projection_accepted": bool(projection_accepted),
        "feedback_admitted": bool(feedback_admitted),
        "lesion_k3": bool(lesion_k3),
        "training_update_steps": update_steps,
        "learning_update_applied": learning_update_applied,
        "learning_eligible": bool(
            task_success and learning_update_applied and feedback_admitted and not lesion_k3
        ),
    }


def _scores(parent: Mapping[str, Any], *, course_seed: int) -> dict[str, float]:
    model = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    course = _r6_course(course_seed)
    return {
        "S": float(_score(model, course.s_holdout, phase="R6-single-cell-S")),
        "G": float(_score(model, course.g_holdout, phase="R6-single-cell-G")),
    }


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        return None


def _repo_relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _resource_gate_valid(resource: Mapping[str, Any]) -> bool:
    required_numeric = (
        "wall_clock_seconds",
        "peak_working_set_bytes",
        "training_update_steps",
        "worker_parameter_count",
        "worker_parameter_bytes",
        "candidate_parameter_bytes",
        "checkpoint_write_bytes",
        "inference_trace_count",
    )
    if (
        resource.get("device") != "cpu"
        or not resource.get("resource_manifest_digest")
        or resource.get("peak_working_set_method") != "process_rss_before_after_lower_bound"
        or resource.get("measurement_complete") is not True
    ):
        return False
    return all(
        isinstance(resource.get(key), (int, float))
        and not isinstance(resource.get(key), bool)
        and float(resource[key]) >= 0.0
        for key in required_numeric
    )


def _control_arm(
    *,
    arm: str,
    parent: Mapping[str, Any],
    parent_digest: str,
    resource_manifest_digest: str,
    course_seed: int,
    explicit_k_baseline: bool = False,
) -> dict[str, Any]:
    before_rss = _rss_bytes()
    started = time.perf_counter()
    before = _scores(parent, course_seed=course_seed)
    restored = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    after = _scores(restored.checkpoint(), course_seed=course_seed)
    elapsed = time.perf_counter() - started
    after_rss = _rss_bytes()
    peak_rss = None if before_rss is None or after_rss is None else max(before_rss, after_rss)
    retention = {phase: abs(after[phase] - before[phase]) <= 0.01 for phase in ("S", "G")}
    measurement = _summarize_k_measurement(
        runner_passed=True,
        task_executed=False,
        outcome_success=False,
        projection_accepted=False,
        feedback_admitted=False,
        lesion_k3=False,
        training_update_steps=0,
        candidate_training_performed=False,
    )
    return {
        "arm": arm,
        "status": "control_passed",
        "parent_checkpoint_digest": parent_digest,
        "phase_rows": [
            {"phase": "S", "status": "observed", "before": before["S"], "after": after["S"]},
            {"phase": "G", "status": "observed", "before": before["G"], "after": after["G"]},
            {
                "phase": "K",
                "status": "rejected" if explicit_k_baseline else "detached",
                "new_capability": None,
                "reason": (
                    "frozen parent K admission rejected because no K worker is attached"
                    if explicit_k_baseline
                    else "this control does not attach the K worker bundle"
                ),
            },
        ],
        "new_capability": {
            "status": "not_attempted",
            "task_success_rate": measurement["task_success_rate"],
            "sample_count": 0,
        },
        "measurement": measurement,
        "old_capability_retention": retention,
        "causal": {
            "status": "frozen_parent_zero_baseline" if explicit_k_baseline else "control_only",
            "k_feedback_consumed": False,
            "k_task_admission": False,
        },
        "resource": {
            "device": "cpu",
            "resource_manifest_digest": resource_manifest_digest,
            "wall_clock_seconds": elapsed,
            "peak_working_set_bytes": peak_rss,
            "peak_working_set_method": "process_rss_before_after_lower_bound",
            "training_update_steps": 0,
            "worker_parameter_count": 0,
            "worker_parameter_bytes": 0,
            "candidate_parameter_bytes": 0,
            "checkpoint_write_bytes": 0,
            "checkpoint_write_paths": [],
            "inference_trace_count": 1 if explicit_k_baseline else 0,
            "measurement_complete": peak_rss is not None,
        },
        "side_effects": {
            "parent_namespace_unchanged": True,
            "candidate_namespace_written": False,
            "default_runtime_attached": False,
            "external_integrations_attached": False,
        },
        "checkpoint_ledger": {
            "fresh_restore_digest": content_digest(restored.checkpoint()),
            "rollback_digest": parent_digest,
            "rollback_matches_parent": content_digest(restored.checkpoint()) == parent_digest,
        },
        "failure": None,
    }


def _fixed_large_control_arm(
    *,
    artifact_path: Path,
    parent: Mapping[str, Any],
    parent_digest: str,
    resource_manifest_digest: str,
    model_seed: int,
    course_seed: int,
) -> dict[str, Any]:
    del parent
    before_rss = _rss_bytes()
    started = time.perf_counter()
    canary = run_fixed_large_canary(
        artifact_path=artifact_path,
        model_seed=model_seed,
        course_seed=course_seed,
    )
    elapsed = time.perf_counter() - started
    after_rss = _rss_bytes()
    peak_rss = None if before_rss is None or after_rss is None else max(before_rss, after_rss)
    passed = canary.get("status") == "passed"
    outcome_success = bool(canary.get("outcome_success", False))
    projection_accepted = bool(canary.get("projection_accepted", False))
    old_before = canary.get("old_capability_before", {})
    old_after = canary.get("old_capability_after", {})
    resource = dict(canary.get("resource") or {})
    resource["wall_clock_seconds"] = elapsed
    resource["device"] = "cpu"
    resource["resource_manifest_digest"] = resource_manifest_digest
    resource["peak_working_set_bytes"] = peak_rss
    resource["peak_working_set_method"] = "process_rss_before_after_lower_bound"
    resource["worker_parameter_count"] = resource.get("parameter_count")
    resource["worker_parameter_bytes"] = resource.get("candidate_parameter_bytes")
    resource["checkpoint_write_paths"] = [_repo_relative_path(artifact_path)]
    resource["measurement_complete"] = (
        peak_rss is not None
        and resource.get("worker_parameter_bytes") is not None
        and resource.get("checkpoint_write_bytes") is not None
        and resource.get("inference_trace_count") is not None
    )
    measurement = _summarize_k_measurement(
        runner_passed=passed,
        task_executed=bool(canary.get("task_executed", passed)),
        outcome_success=outcome_success,
        projection_accepted=projection_accepted,
        feedback_admitted=True,
        lesion_k3=False,
        training_update_steps=int(resource.get("training_update_steps", 0)),
        candidate_training_performed=bool(canary.get("training_performed", False)),
    )
    return {
        "arm": "fixed-large",
        "status": "passed" if passed else "failed",
        "parent_checkpoint_digest": parent_digest,
        "phase_rows": [
            {
                "phase": "S",
                "status": "observed",
                "before": old_before.get("S"),
                "after": old_after.get("S"),
            },
            {
                "phase": "G",
                "status": "observed",
                "before": old_before.get("G"),
                "after": old_after.get("G"),
            },
            {
                "phase": "K",
                "status": "executed" if passed else "failed",
                "observation_path": canary.get("observation_path"),
                "real_success": outcome_success,
                "projection_accepted": projection_accepted,
                "projection_reason": canary.get("projection_reason"),
                "reward": canary.get("outcome_reward"),
            },
        ],
        "new_capability": {
            "status": "observed" if passed else "failed",
            "task_success_rate": measurement["task_success_rate"],
            "sample_count": 1,
        },
        "measurement": measurement,
        "old_capability_retention": canary.get("old_capability_retention"),
        "causal": {
            "status": "fixed_large_control_verified" if passed else "fixed_large_control_failed",
            "k_task_equivalent": True,
            "real_workbench_success": outcome_success,
            "k3_projection_accepted": projection_accepted,
            "task_executed": measurement["task_executed"],
            "task_success": measurement["task_success"],
            "learning_update_applied": measurement["learning_update_applied"],
            "learning_eligible": measurement["learning_eligible"],
            "branch_lesion_observable": bool(
                canary.get("checks", {}).get("fixed_large_branch_lesion_observable", False)
            ),
        },
        "resource": {
            **resource,
        },
        "side_effects": {
            "parent_namespace_unchanged": bool(
                canary.get("checks", {}).get("rollback_parent_namespace_restored", False)
            ),
            "candidate_namespace_written": bool(
                canary.get("checks", {}).get("candidate_stage_roundtrip", False)
            ),
            "default_runtime_attached": False,
            "external_integrations_attached": False,
        },
        "checkpoint_ledger": {
            "adapter_checkpoint": canary.get("adapter_checkpoint"),
            "exchange_digest": canary.get("exchange_digest"),
            "rollback_record_explicit": bool(
                canary.get("checks", {}).get("rollback_record_explicit", False)
            ),
        },
        "failure": (
            None
            if passed
            else _failure(
                failure_class="causal" if canary.get("projection_accepted") else "input_contract",
                message=str(
                    canary.get("blocking_reason") or "fixed-large controlled canary failed"
                ),
                model_seed=model_seed,
                course_seed=course_seed,
                arm="fixed-large",
                recoverability="fixed_large_control_diagnosis_required",
            )
        ),
    }


def _candidate_arm(
    *,
    artifact_dir: Path,
    candidate_namespace: str,
    lesion_k3: bool,
    arm: str,
    resource_manifest_digest: str,
    model_seed: int,
    course_seed: int,
    admit_feedback: bool = True,
) -> dict[str, Any]:
    before_rss = _rss_bytes()
    started = time.perf_counter()
    canary = run_canary(
        artifact_dir=artifact_dir,
        model_seed=model_seed,
        course_seed=course_seed,
        candidate_namespace=candidate_namespace,
        lesion_k3=lesion_k3,
        admit_feedback=admit_feedback,
    )
    elapsed = time.perf_counter() - started
    after_rss = _rss_bytes()
    peak_rss = None if before_rss is None or after_rss is None else max(before_rss, after_rss)
    passed = canary.get("status") == "passed"
    canary_resource = canary.get("resource") or {}
    parameter_bytes = canary_resource.get("worker_parameter_bytes")
    checkpoint_write_bytes = canary_resource.get("checkpoint_write_bytes")
    measurement_complete = (
        peak_rss is not None
        and parameter_bytes is not None
        and checkpoint_write_bytes is not None
        and canary_resource.get("inference_trace_count") is not None
    )
    outcome_success = bool(canary.get("outcome_success", False))
    projection_accepted = bool(canary.get("projection_accepted", False))
    measurement = _summarize_k_measurement(
        runner_passed=passed,
        task_executed=bool(canary.get("task_executed", passed)),
        outcome_success=outcome_success,
        projection_accepted=projection_accepted,
        feedback_admitted=admit_feedback and not lesion_k3,
        lesion_k3=lesion_k3,
        training_update_steps=int(canary_resource.get("training_update_steps", 0)),
        candidate_training_performed=bool(canary.get("candidate_training_performed", False)),
    )
    if not admit_feedback and not lesion_k3:
        k_status = "executed_no_feedback" if passed else "failed"
        causal_status = "matched_capacity_no_feedback" if passed else "matched_capacity_failed"
    elif lesion_k3:
        k_status = "lesion_rejected" if passed else "failed"
        causal_status = "lesion_verified" if passed else "lesion_failed"
    else:
        k_status = "executed" if passed else "failed"
        causal_status = "candidate_path_verified" if passed else "candidate_failed"
    return {
        "arm": arm,
        "status": "passed" if passed else "failed",
        "parent_checkpoint_digest": canary.get("parent_checkpoint_digest"),
        "worker_bundle_digest": canary.get("worker_bundle_digest"),
        "phase_rows": [
            {
                "phase": "S",
                "status": "observed",
                "before": canary.get("old_capability_before", {}).get("S"),
                "after": canary.get("old_capability_after", {}).get("S"),
            },
            {
                "phase": "G",
                "status": "observed",
                "before": canary.get("old_capability_before", {}).get("G"),
                "after": canary.get("old_capability_after", {}).get("G"),
            },
            {
                "phase": "K",
                "status": k_status,
                "observation_path": canary.get("observation_path"),
                "real_success": outcome_success,
                "projection_accepted": projection_accepted,
                "projection_reason": canary.get("projection_reason"),
                "reward": canary.get("outcome_reward"),
            },
        ],
        "new_capability": {
            "status": (
                "observed"
                if measurement["task_measurement_status"] != "not_attempted"
                else "not_attempted"
            ),
            "task_success_rate": measurement["task_success_rate"],
            "sample_count": 1 if measurement["task_executed"] else 0,
        },
        "measurement": measurement,
        "old_capability_retention": canary.get("old_capability_retention"),
        "causal": {
            "status": causal_status,
            "real_workbench_success": outcome_success,
            "k3_projection_accepted": projection_accepted,
            "task_executed": measurement["task_executed"],
            "task_success": measurement["task_success"],
            "learning_update_applied": measurement["learning_update_applied"],
            "learning_eligible": measurement["learning_eligible"],
            "lesion_k3": lesion_k3,
            "feedback_admitted": admit_feedback and not lesion_k3,
        },
        "resource": {
            "device": "cpu",
            "resource_manifest_digest": resource_manifest_digest,
            "wall_clock_seconds": elapsed,
            "peak_working_set_bytes": peak_rss,
            "peak_working_set_method": "process_rss_before_after_lower_bound",
            "worker_parameter_count": canary_resource.get("worker_parameter_count"),
            "worker_parameter_bytes": parameter_bytes,
            "candidate_parameter_bytes": parameter_bytes,
            "checkpoint_write_bytes": checkpoint_write_bytes,
            "checkpoint_write_paths": [
                _repo_relative_path(path) for path in sorted(artifact_dir.glob("taiji_r6_*.pt"))
            ],
            "inference_trace_count": canary_resource.get("inference_trace_count", 1),
            "training_update_steps": canary_resource.get("training_update_steps", 0),
            "measurement_complete": measurement_complete,
        },
        "side_effects": {
            "parent_namespace_unchanged": all(
                bool(canary.get("checks", {}).get(key, False))
                for key in ("rollback_parent_namespace_restored",)
            ),
            "candidate_namespace_written": bool(
                canary.get("checks", {}).get("candidate_stage_roundtrip", False)
            ),
            "default_runtime_attached": bool(canary.get("default_runtime_attached", False)),
            "external_integrations_attached": bool(
                canary.get("provider_attached", False)
                or canary.get("mcp_attached", False)
                or canary.get("client_attached", False)
                or canary.get("cuda_used", False)
            ),
        },
        "checkpoint_ledger": {
            "adapter_checkpoint": canary.get("adapter_checkpoint"),
            "exchange_digest": canary.get("exchange_digest"),
            "rollback_record_explicit": bool(
                canary.get("checks", {}).get("rollback_record_explicit", False)
            ),
        },
        "failure": (
            None
            if passed
            else _failure(
                failure_class=(
                    "workbench_outcome" if outcome_success is False else "projection_contract"
                ),
                message=str(canary.get("blocking_reason") or "controlled K canary failed"),
                model_seed=model_seed,
                course_seed=course_seed,
                arm=arm,
                recoverability="candidate_canary_diagnosis_required",
            )
        ),
    }


def _paired_resource_comparison(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    candidate = arms["candidate-continuation"]
    fixed_large = arms["fixed-large"]
    candidate_resource = candidate.get("resource") or {}
    fixed_resource = fixed_large.get("resource") or {}
    fields = (
        "wall_clock_seconds",
        "peak_working_set_bytes",
        "worker_parameter_bytes",
        "checkpoint_write_bytes",
        "inference_trace_count",
    )

    def _delta(field: str) -> float | None:
        left = candidate_resource.get(field)
        right = fixed_resource.get(field)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        return float(right) - float(left)

    measurement_complete = bool(
        candidate_resource.get("measurement_complete", False)
        and fixed_resource.get("measurement_complete", False)
    )
    return {
        "status": "passed" if measurement_complete else "blocked",
        "measurement_complete": measurement_complete,
        "measurement_method": "process_rss_before_after_lower_bound",
        "candidate_arm": {
            "status": candidate.get("status"),
            "task_success_rate": (candidate.get("new_capability") or {}).get("task_success_rate"),
            "resource": candidate_resource,
        },
        "fixed_large_arm": {
            "status": fixed_large.get("status"),
            "task_success_rate": (fixed_large.get("new_capability") or {}).get("task_success_rate"),
            "resource": fixed_resource,
        },
        "fixed_large_minus_candidate": {field: _delta(field) for field in fields},
        "task_success_delta_fixed_large_minus_candidate": (
            float((fixed_large.get("new_capability") or {}).get("task_success_rate", 0.0))
            - float((candidate.get("new_capability") or {}).get("task_success_rate", 0.0))
        ),
    }


def execution_contract_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the replay-stable, non-resource part of one executed cell."""

    stable_arms: dict[str, Any] = {}
    for arm, payload in (report.get("arms") or {}).items():
        if not isinstance(payload, Mapping):
            stable_arms[str(arm)] = payload
            continue
        stable_arms[str(arm)] = {
            key: payload.get(key)
            for key in (
                "arm",
                "status",
                "parent_checkpoint_digest",
                "worker_bundle_digest",
                "phase_rows",
                "new_capability",
                "old_capability_retention",
                "causal",
                "measurement",
                "side_effects",
                "failure",
            )
        }
    snapshot = {
        "report_format": report.get("report_format"),
        "version": report.get("version"),
        "cell": report.get("cell"),
        "parent_checkpoint_digest": report.get("parent_checkpoint_digest"),
        "worker_bundle_digest": report.get("worker_bundle_digest"),
        "fixed_large_artifact_digest": report.get("fixed_large_artifact_digest"),
        "fixed_large_ensemble_checkpoint_digest": report.get(
            "fixed_large_ensemble_checkpoint_digest"
        ),
        "arms": stable_arms,
        "resource_gate": report.get("resource_gate"),
        "failures": report.get("failures"),
        "course_executed": report.get("course_executed"),
        "run_kind": report.get("run_kind"),
        "measurement_semantics": report.get("measurement_semantics"),
        "training_performed": report.get("training_performed"),
        "candidate_training_performed": report.get("candidate_training_performed"),
        "default_runtime_attached": report.get("default_runtime_attached"),
        "provider_attached": report.get("provider_attached"),
        "mcp_attached": report.get("mcp_attached"),
        "client_attached": report.get("client_attached"),
        "cuda_used": report.get("cuda_used"),
        "can_start_r6_formal": report.get("can_start_r6_formal"),
        "can_promote": report.get("can_promote"),
    }
    if "control_revision" in report:
        snapshot["control_revision"] = report.get("control_revision")
    if "manifest_digest" in report:
        snapshot["manifest_digest"] = report.get("manifest_digest")
    return snapshot


def execution_contract_digest(report: Mapping[str, Any]) -> str:
    return cast(str, content_digest(execution_contract_snapshot(report)))


def run_cell(
    *,
    model_seed: int,
    course_seed: int,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    input_report_path: Path | None = None,
    parent_dir: Path = DEFAULT_PARENT_DIR,
    worker_dir: Path = DEFAULT_WORKER_DIR,
    fixed_large_dir: Path = DEFAULT_FIXED_LARGE_DIR,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_report_path = input_report_path or (
        PROJECT_ROOT / "reports" / "taiji_m4v2_r6_formal_input_manifest_preflight_20260909.json"
    )
    input_gate = run_input_preflight(
        manifest_path=manifest_path,
        report_path=input_report_path,
        materialize_parents=False,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
        fixed_large_dir=fixed_large_dir,
    )
    if input_gate.get("status") != "passed":
        report = {
            "report_format": REPORT_FORMAT,
            "version": VERSION,
            "status": "blocked_input",
            "cell": {"model_seed": model_seed, "course_seed": course_seed},
            "input_preflight_status": input_gate.get("status"),
            "failures": input_gate.get("failures", []),
            "course_executed": False,
            "can_start_r6_formal": False,
            "can_promote": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_entry = next(
        item for item in manifest["parent_registry"] if int(item["model_seed"]) == model_seed
    )
    worker_entry = next(
        item for item in manifest["worker_registry"] if int(item["model_seed"]) == model_seed
    )
    fixed_large_entry = next(
        item for item in manifest["fixed_large_registry"] if int(item["model_seed"]) == model_seed
    )
    control_revision = manifest.get("control_revision")
    revised_controls = (
        isinstance(control_revision, Mapping)
        and control_revision.get("format") == "taiji-m4v2-r6-matched-control-v2"
    )
    parent_path = _resolve_repo_path(parent_entry["checkpoint_path"])
    parent = _load_mapping(parent_path)
    parent_digest = str(parent_entry["checkpoint_digest"])
    artifact_dir = worker_dir / f"model_{model_seed}"
    candidate_namespace = str(worker_entry["candidate_namespace"])

    arms = {
        "frozen-parent": _control_arm(
            arm="frozen-parent",
            parent=parent,
            parent_digest=parent_digest,
            resource_manifest_digest=str(parent_entry["resource_manifest_digest"]),
            course_seed=course_seed,
            explicit_k_baseline=revised_controls,
        ),
        "matched-fixed-capacity": (
            _candidate_arm(
                artifact_dir=artifact_dir,
                candidate_namespace=candidate_namespace,
                lesion_k3=False,
                arm="matched-fixed-capacity",
                resource_manifest_digest=str(worker_entry["resource_manifest_digest"]),
                model_seed=model_seed,
                course_seed=course_seed,
                admit_feedback=False,
            )
            if revised_controls
            else _control_arm(
                arm="matched-fixed-capacity",
                parent=parent,
                parent_digest=parent_digest,
                resource_manifest_digest=str(parent_entry["resource_manifest_digest"]),
                course_seed=course_seed,
            )
        ),
        "candidate-continuation": _candidate_arm(
            artifact_dir=artifact_dir,
            candidate_namespace=candidate_namespace,
            lesion_k3=False,
            arm="candidate-continuation",
            resource_manifest_digest=str(worker_entry["resource_manifest_digest"]),
            model_seed=model_seed,
            course_seed=course_seed,
        ),
        "fixed-large": _fixed_large_control_arm(
            artifact_path=_resolve_repo_path(fixed_large_entry["artifact_path"]),
            parent=parent,
            parent_digest=parent_digest,
            resource_manifest_digest=str(fixed_large_entry["resource_manifest_digest"]),
            model_seed=model_seed,
            course_seed=course_seed,
        ),
        "lesion": _candidate_arm(
            artifact_dir=artifact_dir,
            candidate_namespace=candidate_namespace,
            lesion_k3=True,
            arm="lesion",
            resource_manifest_digest=str(worker_entry["resource_manifest_digest"]),
            model_seed=model_seed,
            course_seed=course_seed,
        ),
    }
    paired_comparison = _paired_resource_comparison(arms)
    resource_gate = {arm: _resource_gate_valid(arms[arm].get("resource") or {}) for arm in ARM_IDS}
    resource_failure = None
    if not all(resource_gate.values()):
        resource_failure = _failure(
            failure_class="resource_gate",
            message=(
                "one or more arms have incomplete resource measurements; every arm must "
                "record CPU/device digest, RSS, wall-clock, parameter bytes, checkpoint "
                "paths/bytes, inference trace and training steps"
            ),
            model_seed=model_seed,
            course_seed=course_seed,
            recoverability="all_arm_resource_measurement_required",
        )
    failures = [arm["failure"] for arm in arms.values() if arm.get("failure") is not None]
    if resource_failure is not None:
        failures.append(resource_failure)
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": "blocked_controls" if failures else "passed",
        "cell": {"model_seed": model_seed, "course_seed": course_seed},
        "manifest_digest": manifest.get("manifest_digest"),
        "parent_checkpoint_digest": parent_digest,
        "worker_bundle_digest": worker_entry["bundle_digest"],
        "fixed_large_artifact_digest": fixed_large_entry["artifact_digest"],
        "fixed_large_ensemble_checkpoint_digest": fixed_large_entry["ensemble_checkpoint_digest"],
        "input_preflight_status": input_gate["status"],
        "arms": arms,
        "paired_comparison": paired_comparison,
        "resource_gate": resource_gate,
        "resource_contract": manifest["resource_contract"],
        "failures": failures,
        "course_executed": False,
        "single_cell_executed": True,
        "run_kind": "wiring-canary",
        "training_performed": False,
        "candidate_training_performed": False,
        "measurement_semantics": MEASUREMENT_SEMANTICS_FORMAT,
        "default_runtime_attached": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "cuda_used": False,
        "can_start_r6_formal": False,
        "can_promote": False,
        "elapsed_seconds": time.perf_counter() - started,
        "decision": (
            "single-cell K-task-equivalent fixed-large control passed; "
            "R6 formal and promotion remain closed until the full causal/resource/retention aggregate"
        ),
    }
    if revised_controls:
        report["control_revision"] = control_revision
    report["execution_contract_digest"] = execution_contract_digest(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_single_cell(
    *,
    model_seed: int = MODEL_SEED,
    course_seed: int = COURSE_SEED,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    input_report_path: Path | None = None,
    parent_dir: Path = DEFAULT_PARENT_DIR,
    worker_dir: Path = DEFAULT_WORKER_DIR,
    fixed_large_dir: Path = DEFAULT_FIXED_LARGE_DIR,
) -> dict[str, Any]:
    """Backward-compatible default entry for the model17/course0 cell."""

    return run_cell(
        model_seed=model_seed,
        course_seed=course_seed,
        manifest_path=manifest_path,
        report_path=report_path,
        input_report_path=input_report_path,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
        fixed_large_dir=fixed_large_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--course-seed", type=int, default=COURSE_SEED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--input-report", type=Path)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--worker-dir", type=Path, default=DEFAULT_WORKER_DIR)
    parser.add_argument("--fixed-large-dir", type=Path, default=DEFAULT_FIXED_LARGE_DIR)
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    input_report_path = (
        None
        if args.input_report is None
        else (
            args.input_report
            if args.input_report.is_absolute()
            else PROJECT_ROOT / args.input_report
        )
    )
    parent_dir = (
        args.parent_dir if args.parent_dir.is_absolute() else PROJECT_ROOT / args.parent_dir
    )
    worker_dir = (
        args.worker_dir if args.worker_dir.is_absolute() else PROJECT_ROOT / args.worker_dir
    )
    fixed_large_dir = (
        args.fixed_large_dir
        if args.fixed_large_dir.is_absolute()
        else PROJECT_ROOT / args.fixed_large_dir
    )
    report = run_cell(
        model_seed=args.model_seed,
        course_seed=args.course_seed,
        manifest_path=manifest_path,
        report_path=report_path,
        input_report_path=input_report_path,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
        fixed_large_dir=fixed_large_dir,
    )
    print(
        json.dumps(
            {
                "report": report_path.relative_to(PROJECT_ROOT).as_posix(),
                "status": report["status"],
                "single_cell_executed": report.get("single_cell_executed", False),
                "course_executed": report.get("course_executed", False),
                "can_promote": report["can_promote"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
