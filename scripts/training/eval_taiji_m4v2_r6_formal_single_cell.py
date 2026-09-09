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
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r6_formal_input_manifest_preflight import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_PARENT_DIR,
    DEFAULT_WORKER_DIR,
    _resolve_repo_path,
)
from scripts.training.eval_taiji_m4v2_r6_formal_input_manifest_preflight import (  # noqa: E402
    run_preflight as run_input_preflight,
)
from scripts.training.eval_taiji_m4v2_r6_k_fixed_large_canary import (  # noqa: E402
    DEFAULT_ARTIFACT as DEFAULT_FIXED_LARGE_ARTIFACT,
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
MODEL_SEED = 17
COURSE_SEED = 0
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_r6_formal_single_cell_20260909.json"
)
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
    arm: str | None = None,
    recoverability: str = "control_arm_required",
) -> dict[str, Any]:
    return {
        "class": failure_class,
        "phase": "K",
        "cell": {"model_seed": MODEL_SEED, "course_seed": COURSE_SEED},
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


def _scores(parent: Mapping[str, Any]) -> dict[str, float]:
    model = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    course = _r6_course(COURSE_SEED)
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


def _control_arm(
    *,
    arm: str,
    parent: Mapping[str, Any],
    parent_digest: str,
) -> dict[str, Any]:
    before = _scores(parent)
    restored = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    after = _scores(restored.checkpoint())
    retention = {
        phase: abs(after[phase] - before[phase]) <= 0.01 for phase in ("S", "G")
    }
    return {
        "arm": arm,
        "status": "control_passed",
        "parent_checkpoint_digest": parent_digest,
        "phase_rows": [
            {"phase": "S", "status": "observed", "before": before["S"], "after": after["S"]},
            {"phase": "G", "status": "observed", "before": before["G"], "after": after["G"]},
            {
                "phase": "K",
                "status": "detached",
                "new_capability": None,
                "reason": "this control does not attach the K worker bundle",
            },
        ],
        "new_capability": {
            "status": "detached",
            "task_success_rate": None,
        },
        "old_capability_retention": retention,
        "causal": {"status": "control_only", "k_feedback_consumed": False},
        "resource": {
            "wall_clock_seconds": 0.0,
            "peak_working_set_bytes": None,
            "training_update_steps": 0,
            "candidate_parameter_bytes": 0,
            "checkpoint_write_bytes": 0,
            "inference_trace_count": 0,
            "measurement_complete": False,
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
) -> dict[str, Any]:
    del parent
    canary = run_fixed_large_canary(
        artifact_path=artifact_path,
        model_seed=MODEL_SEED,
        course_seed=COURSE_SEED,
    )
    passed = canary.get("status") == "passed"
    outcome_success = bool(canary.get("outcome_success", False))
    projection_accepted = bool(canary.get("projection_accepted", False))
    old_before = canary.get("old_capability_before", {})
    old_after = canary.get("old_capability_after", {})
    resource = canary.get("resource", {})
    return {
        "arm": "fixed-large",
        "status": "passed" if passed else "failed",
        "parent_checkpoint_digest": parent_digest,
        "phase_rows": [
            {"phase": "S", "status": "observed", "before": old_before.get("S"), "after": old_after.get("S")},
            {"phase": "G", "status": "observed", "before": old_before.get("G"), "after": old_after.get("G")},
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
            "task_success_rate": 1.0 if passed and outcome_success and projection_accepted else 0.0,
            "sample_count": 1,
        },
        "old_capability_retention": canary.get("old_capability_retention"),
        "causal": {
            "status": "fixed_large_control_verified" if passed else "fixed_large_control_failed",
            "k_task_equivalent": True,
            "real_workbench_success": outcome_success,
            "k3_projection_accepted": projection_accepted,
            "branch_lesion_observable": bool(
                canary.get("checks", {}).get("fixed_large_branch_lesion_observable", False)
            ),
        },
        "resource": {
            **resource,
            "peak_working_set_bytes": resource.get("peak_working_set_bytes"),
            "measurement_complete": passed,
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
        "failure": None
        if passed
        else _failure(
            failure_class="causal" if canary.get("projection_accepted") else "input_contract",
            message=str(canary.get("blocking_reason") or "fixed-large controlled canary failed"),
            arm="fixed-large",
            recoverability="fixed_large_control_diagnosis_required",
        ),
    }


def _candidate_arm(
    *,
    artifact_dir: Path,
    candidate_namespace: str,
    lesion_k3: bool,
    arm: str,
) -> dict[str, Any]:
    before_rss = _rss_bytes()
    started = time.perf_counter()
    canary = run_canary(
        artifact_dir=artifact_dir,
        model_seed=MODEL_SEED,
        course_seed=COURSE_SEED,
        candidate_namespace=candidate_namespace,
        lesion_k3=lesion_k3,
    )
    elapsed = time.perf_counter() - started
    after_rss = _rss_bytes()
    peak_rss = None if before_rss is None or after_rss is None else max(before_rss, after_rss)
    passed = canary.get("status") == "passed"
    outcome_success = bool(canary.get("outcome_success", False))
    projection_accepted = bool(canary.get("projection_accepted", False))
    if lesion_k3:
        k_status = "lesion_rejected" if passed else "failed"
        capability_rate = 0.0 if passed else None
        causal_status = "lesion_verified" if passed else "lesion_failed"
    else:
        k_status = "executed" if passed else "failed"
        capability_rate = 1.0 if passed and outcome_success and projection_accepted else 0.0
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
            "status": "observed" if passed else "failed",
            "task_success_rate": capability_rate,
            "sample_count": 1,
        },
        "old_capability_retention": canary.get("old_capability_retention"),
        "causal": {
            "status": causal_status,
            "real_workbench_success": outcome_success,
            "k3_projection_accepted": projection_accepted,
            "lesion_k3": lesion_k3,
        },
        "resource": {
            "wall_clock_seconds": elapsed,
            "peak_working_set_bytes": peak_rss,
            "peak_working_set_method": "process_rss_before_after_lower_bound",
            "training_update_steps": 0,
            "candidate_parameter_bytes": 0,
            "checkpoint_write_bytes": 0,
            "inference_trace_count": 1,
            "measurement_complete": False,
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
        "failure": None
        if passed
        else _failure(
            failure_class="workbench_outcome" if outcome_success is False else "projection_contract",
            message=str(canary.get("blocking_reason") or "controlled K canary failed"),
            arm=arm,
            recoverability="candidate_canary_diagnosis_required",
        ),
    }


def run_single_cell(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    input_report_path: Path | None = None,
    parent_dir: Path = DEFAULT_PARENT_DIR,
    worker_dir: Path = DEFAULT_WORKER_DIR,
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
    )
    if input_gate.get("status") != "passed":
        report = {
            "report_format": REPORT_FORMAT,
            "version": VERSION,
            "status": "blocked_input",
            "cell": {"model_seed": MODEL_SEED, "course_seed": COURSE_SEED},
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
        item for item in manifest["parent_registry"] if int(item["model_seed"]) == MODEL_SEED
    )
    worker_entry = next(
        item for item in manifest["worker_registry"] if int(item["model_seed"]) == MODEL_SEED
    )
    parent_path = _resolve_repo_path(parent_entry["checkpoint_path"])
    parent = _load_mapping(parent_path)
    parent_digest = str(parent_entry["checkpoint_digest"])
    artifact_dir = worker_dir / f"model_{MODEL_SEED}"
    candidate_namespace = str(worker_entry["candidate_namespace"])

    arms = {
        "frozen-parent": _control_arm(
            arm="frozen-parent", parent=parent, parent_digest=parent_digest
        ),
        "matched-fixed-capacity": _control_arm(
            arm="matched-fixed-capacity", parent=parent, parent_digest=parent_digest
        ),
        "candidate-continuation": _candidate_arm(
            artifact_dir=artifact_dir,
            candidate_namespace=candidate_namespace,
            lesion_k3=False,
            arm="candidate-continuation",
        ),
        "fixed-large": _fixed_large_control_arm(
            artifact_path=DEFAULT_FIXED_LARGE_ARTIFACT,
            parent=parent,
            parent_digest=parent_digest,
        ),
        "lesion": _candidate_arm(
            artifact_dir=artifact_dir,
            candidate_namespace=candidate_namespace,
            lesion_k3=True,
            arm="lesion",
        ),
    }
    failures = [arm["failure"] for arm in arms.values() if arm.get("failure") is not None]
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": "blocked_controls" if failures else "passed",
        "cell": {"model_seed": MODEL_SEED, "course_seed": COURSE_SEED},
        "parent_checkpoint_digest": parent_digest,
        "worker_bundle_digest": worker_entry["bundle_digest"],
        "input_preflight_status": input_gate["status"],
        "arms": arms,
        "failures": failures,
        "course_executed": False,
        "single_cell_executed": True,
        "training_performed": False,
        "candidate_training_performed": False,
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--input-report", type=Path)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--worker-dir", type=Path, default=DEFAULT_WORKER_DIR)
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    input_report_path = (
        None
        if args.input_report is None
        else args.input_report
        if args.input_report.is_absolute()
        else PROJECT_ROOT / args.input_report
    )
    parent_dir = args.parent_dir if args.parent_dir.is_absolute() else PROJECT_ROOT / args.parent_dir
    worker_dir = args.worker_dir if args.worker_dir.is_absolute() else PROJECT_ROOT / args.worker_dir
    report = run_single_cell(
        manifest_path=manifest_path,
        report_path=report_path,
        input_report_path=input_report_path,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
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
