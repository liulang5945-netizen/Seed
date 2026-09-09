"""Preflight the M4.V2.R6 formal runner input and create its cell ledger.

This entry layer intentionally stops after the explicit manifest, parent, and
worker Gate.  It creates a 9-cell x 5-arm ledger with ``not_started`` rows; it
does not execute the S->G->K course, train a candidate, attach the default
runtime, or promote anything.  The execution layer can consume only this
content-addressed manifest after the preflight result is committed.
"""

from __future__ import annotations

import argparse
import copy
import json
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
    MANIFEST_FORMAT,
    MODEL_SEEDS,
    PHASE_ORDER,
    _relative_path,
)
from scripts.training.eval_taiji_m4v2_r6_formal_input_manifest_preflight import (  # noqa: E402
    DEFAULT_REPORT as DEFAULT_INPUT_REPORT,
)
from scripts.training.eval_taiji_m4v2_r6_formal_input_manifest_preflight import (  # noqa: E402
    run_preflight as run_input_preflight,
)
from scripts.training.eval_taiji_m4v2_r6_formal_single_cell import (  # noqa: E402
    run_cell,
)
from taiji import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-r6-formal-runner-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r6_formal_preflight_20260909.json"
DEFAULT_EXECUTION_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_r6_formal_execution_20260909.json"
)
DEFAULT_CELL_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "taiji_m4v2_r6_formal_cell_model_17_course_0_20260909.json"
)
EXECUTION_MODEL_SEED = 17
EXECUTION_COURSE_SEED = 0
EXECUTION_ORDER = tuple(
    (int(model_seed), int(course_seed))
    for model_seed in MODEL_SEEDS
    for course_seed in COURSE_SEEDS
)
MATCHED_CONTROL_REVISION_FORMAT = "taiji-m4v2-r6-matched-control-v2"


def _failure_record(
    *,
    failure_class: str,
    message: str,
    phase: str = "input",
    cell: Mapping[str, int] | None = None,
    arm: str | None = None,
    step: str | None = None,
    recoverability: str = "input_fix_required",
    evidence_digests: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "class": failure_class,
        "phase": phase,
        "cell": None if cell is None else dict(cell),
        "arm": arm,
        "step": step,
        "is_model_evidence": False,
        "is_environment_blocker": failure_class == "environment_blocker",
        "stop_line": True,
        "recoverability": recoverability,
        "exception_type": None,
        "message": message,
        "evidence_digests": list(evidence_digests),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("formal runner input manifest must be a JSON object")
    return {str(key): value for key, value in raw.items()}


def _requires_formal_admission(manifest: Mapping[str, Any]) -> bool:
    revision = manifest.get("control_revision")
    return (
        isinstance(revision, Mapping)
        and revision.get("format") == MATCHED_CONTROL_REVISION_FORMAT
    )


def _load_formal_admission(
    path: Path,
    *,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("formal admission report must be a JSON object")
    report = {str(key): value for key, value in raw.items()}
    if report.get("status") != "passed":
        raise ValueError("formal admission report is not passed")
    if report.get("can_start_r6_formal") is not True:
        raise ValueError("formal admission report does not authorize formal start")
    if report.get("can_promote") is not False:
        raise ValueError("formal admission report must keep promotion closed")
    if report.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("formal admission manifest digest differs from input manifest")
    if report.get("control_revision") != manifest.get("control_revision"):
        raise ValueError("formal admission control revision differs from input manifest")
    for key in (
        "default_runtime_attached",
        "provider_attached",
        "mcp_attached",
        "client_attached",
        "cuda_used",
        "training_performed",
    ):
        if report.get(key) is not False:
            raise ValueError(f"formal admission side-effect boundary is open: {key}")
    return report, content_digest(report)


def _cell_ledger(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    parents = {
        int(entry["model_seed"]): entry
        for entry in manifest["parent_registry"]
        if isinstance(entry, Mapping)
    }
    workers = {
        int(entry["model_seed"]): entry
        for entry in manifest["worker_registry"]
        if isinstance(entry, Mapping)
    }
    fixed_large = {
        int(entry["model_seed"]): entry
        for entry in manifest["fixed_large_registry"]
        if isinstance(entry, Mapping)
    }
    cells: list[dict[str, Any]] = []
    for model_seed in MODEL_SEEDS:
        parent = parents[model_seed]
        worker = workers[model_seed]
        fixed_large_entry = fixed_large[model_seed]
        for course_seed in COURSE_SEEDS:
            cell = {"model_seed": int(model_seed), "course_seed": int(course_seed)}
            cells.append(
                {
                    "cell": cell,
                    "status": "not_started",
                    "parent_checkpoint_digest": str(parent["checkpoint_digest"]),
                    "worker_bundle_digest": str(worker["bundle_digest"]),
                    "fixed_large_artifact_path": str(fixed_large_entry["artifact_path"]),
                    "fixed_large_artifact_digest": str(fixed_large_entry["artifact_digest"]),
                    "fixed_large_ensemble_checkpoint_digest": str(
                        fixed_large_entry["ensemble_checkpoint_digest"]
                    ),
                    "phase_order": list(PHASE_ORDER),
                    "arms": [
                        {
                            "arm": arm,
                            "status": "not_started",
                            "phase_rows": [],
                            "new_capability": None,
                            "old_capability_retention": None,
                            "causal": None,
                            "resource": None,
                            "side_effects": None,
                            "checkpoint_ledger": None,
                            "failure": None,
                        }
                        for arm in ARM_IDS
                    ],
                    "failure": None,
                }
            )
    return cells


def _apply_cell_execution(
    cell_ledger: list[dict[str, Any]],
    execution_report: Mapping[str, Any],
    *,
    execution_report_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Write exactly one executor result into the matching not_started ledger row."""

    target = execution_report.get("cell")
    if not isinstance(target, Mapping):
        return cell_ledger, [
            _failure_record(
                failure_class="input_contract",
                message="cell executor report has no structured cell identity",
            )
        ]
    target_key = (int(target.get("model_seed", -1)), int(target.get("course_seed", -1)))
    matches = [
        row
        for row in cell_ledger
        if (
            int(row["cell"]["model_seed"]),
            int(row["cell"]["course_seed"]),
        )
        == target_key
    ]
    if len(matches) != 1:
        return cell_ledger, [
            _failure_record(
                failure_class="input_contract",
                message=f"executor cell does not map to exactly one ledger row: {target_key}",
                cell={"model_seed": target_key[0], "course_seed": target_key[1]},
            )
        ]
    row = matches[0]
    if row.get("status") != "not_started":
        return cell_ledger, [
            _failure_record(
                failure_class="input_contract",
                message="executor refused to overwrite a non-not_started ledger row",
                cell={"model_seed": target_key[0], "course_seed": target_key[1]},
                recoverability="ledger_row_must_be_not_started",
            )
        ]
    execution_arms = execution_report.get("arms")
    if not isinstance(execution_arms, Mapping):
        return cell_ledger, [
            _failure_record(
                failure_class="input_contract",
                message="cell executor report has no arm ledger",
                cell={"model_seed": target_key[0], "course_seed": target_key[1]},
            )
        ]
    failures: list[dict[str, Any]] = []
    for arm_row in row["arms"]:
        arm = str(arm_row["arm"])
        execution_arm = execution_arms.get(arm)
        if not isinstance(execution_arm, Mapping):
            failures.append(
                _failure_record(
                    failure_class="input_contract",
                    message=f"cell executor omitted arm: {arm}",
                    cell={"model_seed": target_key[0], "course_seed": target_key[1]},
                    arm=arm,
                )
            )
            continue
        for key in (
            "status",
            "phase_rows",
            "new_capability",
            "old_capability_retention",
            "causal",
            "resource",
            "side_effects",
            "checkpoint_ledger",
            "failure",
        ):
            arm_row[key] = copy.deepcopy(cast(Any, execution_arm.get(key)))
    row["status"] = (
        "executed_passed"
        if execution_report.get("status") == "passed" and not failures
        else "executed_failed"
    )
    row["single_cell_executed"] = True
    row["execution_report_path"] = _relative_path(execution_report_path)
    row["execution_report_digest"] = content_digest(execution_report)
    row["execution_contract_digest"] = execution_report.get("execution_contract_digest")
    row["resource_gate"] = copy.deepcopy(execution_report.get("resource_gate"))
    row["failure"] = (
        copy.deepcopy(execution_report.get("failures", [])[0])
        if execution_report.get("failures")
        else (failures[0] if failures else None)
    )
    failures.extend(
        copy.deepcopy(cast(Any, item))
        for item in execution_report.get("failures", [])
        if isinstance(item, Mapping)
    )
    return cell_ledger, failures


def run_preflight(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    input_report_path: Path = DEFAULT_INPUT_REPORT,
    admission_report_path: Path | None = None,
    parent_dir: Path = DEFAULT_PARENT_DIR,
    worker_dir: Path = DEFAULT_WORKER_DIR,
    fixed_large_dir: Path = DEFAULT_FIXED_LARGE_DIR,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    report_path = report_path.resolve()
    input_report_path = input_report_path.resolve()
    admission_report_path = (
        None if admission_report_path is None else admission_report_path.resolve()
    )
    parent_dir = parent_dir.resolve()
    worker_dir = worker_dir.resolve()
    fixed_large_dir = fixed_large_dir.resolve()
    input_report = run_input_preflight(
        manifest_path=manifest_path,
        report_path=input_report_path,
        materialize_parents=False,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
        fixed_large_dir=fixed_large_dir,
    )
    failures = [copy.deepcopy(item) for item in input_report.get("failures", [])]
    manifest: dict[str, Any] | None = None
    if input_report.get("formal_input_ready"):
        try:
            manifest = _load_manifest(manifest_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(
                _failure_record(
                    failure_class="input_contract",
                    message=f"formal runner cannot reload the validated manifest: {exc}",
                )
            )
    formal_admission_required = manifest is not None and _requires_formal_admission(manifest)
    formal_admission_passed = not formal_admission_required
    formal_admission_digest = ""
    if manifest is not None and formal_admission_required:
        if admission_report_path is None:
            failures.append(
                _failure_record(
                    failure_class="input_contract",
                    message="matched-control revision requires a passed formal admission report",
                    recoverability="formal_admission_report_required",
                )
            )
        else:
            try:
                _, formal_admission_digest = _load_formal_admission(
                    admission_report_path,
                    manifest=manifest,
                )
                formal_admission_passed = True
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                failures.append(
                    _failure_record(
                        failure_class="input_contract",
                        message=f"formal admission report is not valid: {exc}",
                        recoverability="formal_admission_report_required",
                    )
                )
    checks = {
        "input_manifest_preflight_passed": input_report.get("status") == "passed",
        "manifest_format_stable": bool(
            manifest is not None and manifest.get("format") == MANIFEST_FORMAT
        ),
        "parent_worker_cell_ledger_materialized": False,
        "course_not_started": True,
        "candidate_training_not_started": True,
        "formal_admission_passed": formal_admission_passed,
        "default_runtime_detached": True,
        "external_integrations_detached": True,
        "cuda_unused": True,
    }
    cell_ledger: list[dict[str, Any]] = []
    if manifest is not None and not failures:
        try:
            cell_ledger = _cell_ledger(manifest)
            checks["parent_worker_cell_ledger_materialized"] = len(cell_ledger) == 9 and all(
                len(cell["arms"]) == len(ARM_IDS) for cell in cell_ledger
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                _failure_record(
                    failure_class="input_contract",
                    message=f"formal runner cannot materialize the cell ledger: {exc}",
                )
            )
    status = "input_ready" if not failures and all(checks.values()) else "blocked_input"
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": status,
        "manifest_path": _relative_path(manifest_path),
        "manifest_digest": (
            str(manifest.get("manifest_digest", "")) if manifest is not None else ""
        ),
        "input_preflight_report": _relative_path(input_report_path),
        "input_preflight_report_digest": content_digest(input_report),
        "formal_admission_report": (
            None
            if admission_report_path is None
            else _relative_path(admission_report_path)
        ),
        "formal_admission_report_digest": formal_admission_digest,
        "formal_admission_required": formal_admission_required,
        "formal_admission_passed": formal_admission_passed,
        "parent_dir": _relative_path(parent_dir),
        "worker_dir": _relative_path(worker_dir),
        "fixed_large_dir": _relative_path(fixed_large_dir),
        "matrix": {
            "model_seeds": list(MODEL_SEEDS),
            "course_seeds": list(COURSE_SEEDS),
            "phase_order": list(PHASE_ORDER),
            "arms": list(ARM_IDS),
        },
        "checks": checks,
        "failures": failures,
        "cell_ledger": cell_ledger,
        "formal_input_ready": input_report.get("formal_input_ready") is True and not failures,
        "course_executed": False,
        "training_performed": False,
        "candidate_training_performed": False,
        "candidate_promoted": False,
        "default_runtime_attached": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "cuda_used": False,
        "can_start_r6_formal": formal_admission_required and formal_admission_passed,
        "can_promote": False,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "Run the preregistered S->G->K formal execution under the passed admission "
            "report; keep promotion and all external owners closed."
            if formal_admission_required and formal_admission_passed
            else "Provide a passed formal admission report before starting the revised formal runner."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_execution(
    *,
    model_seed: int = EXECUTION_MODEL_SEED,
    course_seed: int = EXECUTION_COURSE_SEED,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_EXECUTION_REPORT,
    prior_execution_report: Path | None = None,
    input_report_path: Path = DEFAULT_INPUT_REPORT,
    preflight_report_path: Path | None = None,
    admission_report_path: Path | None = None,
    cell_report_path: Path = DEFAULT_CELL_REPORT,
    parent_dir: Path = DEFAULT_PARENT_DIR,
    worker_dir: Path = DEFAULT_WORKER_DIR,
    fixed_large_dir: Path = DEFAULT_FIXED_LARGE_DIR,
) -> dict[str, Any]:
    """Execute exactly one pre-registered cell and write it into the ledger."""

    started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    report_path = report_path.resolve()
    prior_execution_report = (
        report_path if prior_execution_report is None else prior_execution_report.resolve()
    )
    input_report_path = input_report_path.resolve()
    preflight_report_path = (
        report_path.with_name("taiji_m4v2_r6_formal_preflight_20260909.json")
        if preflight_report_path is None
        else preflight_report_path.resolve()
    )
    admission_report_path = (
        None if admission_report_path is None else admission_report_path.resolve()
    )
    cell_report_path = cell_report_path.resolve()
    parent_dir = parent_dir.resolve()
    worker_dir = worker_dir.resolve()
    fixed_large_dir = fixed_large_dir.resolve()
    preflight = run_preflight(
        manifest_path=manifest_path,
        report_path=preflight_report_path,
        input_report_path=input_report_path,
        admission_report_path=admission_report_path,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
        fixed_large_dir=fixed_large_dir,
    )
    report: dict[str, Any] = {
        "report_format": "taiji-m4v2-r6-formal-execution-v1",
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": "blocked_input",
        "manifest_path": _relative_path(manifest_path),
        "input_preflight_report": _relative_path(input_report_path),
        "formal_preflight_report": _relative_path(preflight_report_path),
        "formal_admission_report": (
            None
            if admission_report_path is None
            else _relative_path(admission_report_path)
        ),
        "parent_dir": _relative_path(parent_dir),
        "worker_dir": _relative_path(worker_dir),
        "fixed_large_dir": _relative_path(fixed_large_dir),
        "prior_execution_report": _relative_path(prior_execution_report),
        "target_cell": {"model_seed": model_seed, "course_seed": course_seed},
        "formal_input_ready": False,
        "cell_ledger": [],
        "failures": [],
        "course_executed": False,
        "single_cell_executed": False,
        "training_performed": False,
        "candidate_training_performed": False,
        "candidate_promoted": False,
        "default_runtime_attached": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "cuda_used": False,
        "can_start_r6_formal": False,
        "can_promote": False,
    }
    if preflight.get("status") != "input_ready":
        report["failures"] = copy.deepcopy(preflight.get("failures", []))
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    report["formal_admission_report_digest"] = preflight.get(
        "formal_admission_report_digest", ""
    )
    report["formal_admission_required"] = preflight.get(
        "formal_admission_required", False
    )
    report["formal_admission_passed"] = preflight.get(
        "formal_admission_passed", False
    )
    report["can_start_r6_formal"] = preflight.get("can_start_r6_formal", False)
    if (model_seed, course_seed) not in EXECUTION_ORDER:
        report["failures"] = [
            _failure_record(
                failure_class="input_contract",
                message=(
                    "execution target is outside the fixed manifest cell order "
                    "model17/course0 → model17/course1 → model17/course2 → "
                    "model23/course0 → model23/course1 → model23/course2 → "
                    "model31/course0 → model31/course1 → model31/course2"
                ),
                cell={"model_seed": model_seed, "course_seed": course_seed},
                recoverability="single_cell_execution_order",
            )
        ]
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    manifest = _load_manifest(manifest_path)
    ledger = _cell_ledger(manifest)
    if prior_execution_report.is_file():
        try:
            prior_raw = json.loads(prior_execution_report.read_text(encoding="utf-8"))
            if not isinstance(prior_raw, Mapping):
                raise ValueError("prior execution report must be an object")
            if str(prior_raw.get("manifest_digest", "")) != str(
                manifest.get("manifest_digest", "")
            ):
                raise ValueError("prior execution report manifest digest does not match")
            prior_ledger = prior_raw.get("cell_ledger")
            if not isinstance(prior_ledger, list) or len(prior_ledger) != len(ledger):
                raise ValueError("prior execution report has an incomplete cell ledger")
            ledger = copy.deepcopy(prior_ledger)
            report["prior_execution_report_digest"] = content_digest(prior_raw)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            report["failures"] = [
                _failure_record(
                    failure_class="input_contract",
                    message=f"cannot load prior execution ledger: {exc}",
                    recoverability="prior_execution_ledger_required",
                )
            ]
            report["elapsed_seconds"] = time.perf_counter() - started
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return report
    target_rows = [
        row
        for row in ledger
        if row["cell"] == {"model_seed": model_seed, "course_seed": course_seed}
    ]
    target_index = EXECUTION_ORDER.index((model_seed, course_seed))
    row_by_key = {
        (int(row["cell"]["model_seed"]), int(row["cell"]["course_seed"])): row
        for row in ledger
    }
    missing_predecessors = [
        key
        for key in EXECUTION_ORDER[:target_index]
        if row_by_key.get(key, {}).get("status") != "executed_passed"
    ]
    if missing_predecessors:
        report["failures"] = [
            _failure_record(
                failure_class="input_contract",
                message=f"execution predecessors are not passed: {missing_predecessors}",
                cell={"model_seed": model_seed, "course_seed": course_seed},
                recoverability="execution_order_predecessor_required",
            )
        ]
        report["cell_ledger"] = ledger
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    if len(target_rows) != 1 or target_rows[0].get("status") != "not_started":
        report["failures"] = [
            _failure_record(
                failure_class="input_contract",
                message="target cell is missing or is no longer not_started",
                cell={"model_seed": model_seed, "course_seed": course_seed},
                recoverability="ledger_row_must_be_not_started",
            )
        ]
        report["cell_ledger"] = ledger
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    cell_report = run_cell(
        model_seed=model_seed,
        course_seed=course_seed,
        manifest_path=manifest_path,
        report_path=cell_report_path,
        input_report_path=input_report_path,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
        fixed_large_dir=fixed_large_dir,
    )
    ledger, ledger_failures = _apply_cell_execution(
        ledger,
        cell_report,
        execution_report_path=cell_report_path,
    )
    report.update(
        {
            "status": "single_cell_executed"
            if cell_report.get("status") == "passed" and not ledger_failures
            else "blocked_execution",
            "manifest_digest": manifest.get("manifest_digest"),
            "control_revision": manifest.get("control_revision"),
            "input_preflight_report_digest": content_digest(preflight),
            "formal_input_ready": True,
            "cell_ledger": ledger,
            "failures": ledger_failures,
            "single_cell_executed": True,
            "execution_report_path": _relative_path(cell_report_path),
            "execution_report_digest": content_digest(cell_report),
            "execution_contract_digest": cell_report.get("execution_contract_digest"),
            "next_gate": (
                "Expand the same cell executor only after reviewing this written row; "
                "the remaining eight rows stay not_started and promotion remains closed."
            ),
        }
    )
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
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--input-report", type=Path, default=DEFAULT_INPUT_REPORT)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--worker-dir", type=Path, default=DEFAULT_WORKER_DIR)
    parser.add_argument("--fixed-large-dir", type=Path, default=DEFAULT_FIXED_LARGE_DIR)
    parser.add_argument(
        "--execute-cell",
        action="store_true",
        help="execute only the preregistered model17/course0 cell and write its ledger row",
    )
    parser.add_argument("--model-seed", type=int, default=EXECUTION_MODEL_SEED)
    parser.add_argument("--course-seed", type=int, default=EXECUTION_COURSE_SEED)
    parser.add_argument("--execution-report", type=Path, default=DEFAULT_EXECUTION_REPORT)
    parser.add_argument("--cell-report", type=Path, default=DEFAULT_CELL_REPORT)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--admission-report", type=Path)
    parser.add_argument("--prior-execution-report", type=Path)
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    input_report_path = (
        args.input_report if args.input_report.is_absolute() else PROJECT_ROOT / args.input_report
    )
    parent_dir = args.parent_dir if args.parent_dir.is_absolute() else PROJECT_ROOT / args.parent_dir
    worker_dir = args.worker_dir if args.worker_dir.is_absolute() else PROJECT_ROOT / args.worker_dir
    fixed_large_dir = (
        args.fixed_large_dir
        if args.fixed_large_dir.is_absolute()
        else PROJECT_ROOT / args.fixed_large_dir
    )
    if args.execute_cell:
        execution_report_path = (
            args.execution_report
            if args.execution_report.is_absolute()
            else PROJECT_ROOT / args.execution_report
        )
        cell_report_path = (
            args.cell_report
            if args.cell_report.is_absolute()
            else PROJECT_ROOT / args.cell_report
        )
        preflight_report_path = (
            None
            if args.preflight_report is None
            else (
                args.preflight_report
                if args.preflight_report.is_absolute()
                else PROJECT_ROOT / args.preflight_report
            )
        )
        admission_report_path = (
            None
            if args.admission_report is None
            else (
                args.admission_report
                if args.admission_report.is_absolute()
                else PROJECT_ROOT / args.admission_report
            )
        )
        prior_execution_report_path = (
            None
            if args.prior_execution_report is None
            else (
                args.prior_execution_report
                if args.prior_execution_report.is_absolute()
                else PROJECT_ROOT / args.prior_execution_report
            )
        )
        report = run_execution(
            model_seed=args.model_seed,
            course_seed=args.course_seed,
            manifest_path=manifest_path,
            report_path=execution_report_path,
            prior_execution_report=prior_execution_report_path,
            input_report_path=input_report_path,
            preflight_report_path=preflight_report_path,
            admission_report_path=admission_report_path,
            cell_report_path=cell_report_path,
            parent_dir=parent_dir,
            worker_dir=worker_dir,
            fixed_large_dir=fixed_large_dir,
        )
    else:
        admission_report_path = (
            None
            if args.admission_report is None
            else (
                args.admission_report
                if args.admission_report.is_absolute()
                else PROJECT_ROOT / args.admission_report
            )
        )
        report = run_preflight(
            manifest_path=manifest_path,
            report_path=report_path,
            input_report_path=input_report_path,
            admission_report_path=admission_report_path,
            parent_dir=parent_dir,
            worker_dir=worker_dir,
            fixed_large_dir=fixed_large_dir,
        )
    display_report_path = execution_report_path if args.execute_cell else report_path
    print(
        json.dumps(
            {
                "report": _relative_path(display_report_path),
                "status": report["status"],
                "formal_input_ready": report["formal_input_ready"],
                "course_executed": report["course_executed"],
                "single_cell_executed": report.get("single_cell_executed", False),
                "can_start_r6_formal": report["can_start_r6_formal"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] in {"input_ready", "single_cell_executed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
