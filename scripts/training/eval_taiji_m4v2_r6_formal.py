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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r6_formal_input_manifest_preflight import (  # noqa: E402
    ARM_IDS,
    COURSE_SEEDS,
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
from taiji import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-r6-formal-runner-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r6_formal_preflight_20260909.json"


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
    cells: list[dict[str, Any]] = []
    for model_seed in MODEL_SEEDS:
        parent = parents[model_seed]
        worker = workers[model_seed]
        for course_seed in COURSE_SEEDS:
            cell = {"model_seed": int(model_seed), "course_seed": int(course_seed)}
            cells.append(
                {
                    "cell": cell,
                    "status": "not_started",
                    "parent_checkpoint_digest": str(parent["checkpoint_digest"]),
                    "worker_bundle_digest": str(worker["bundle_digest"]),
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


def run_preflight(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    input_report_path: Path = DEFAULT_INPUT_REPORT,
    parent_dir: Path = DEFAULT_PARENT_DIR,
    worker_dir: Path = DEFAULT_WORKER_DIR,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    report_path = report_path.resolve()
    input_report_path = input_report_path.resolve()
    parent_dir = parent_dir.resolve()
    worker_dir = worker_dir.resolve()
    input_report = run_input_preflight(
        manifest_path=manifest_path,
        report_path=input_report_path,
        materialize_parents=False,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
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
    checks = {
        "input_manifest_preflight_passed": input_report.get("status") == "passed",
        "manifest_format_stable": bool(
            manifest is not None and manifest.get("format") == MANIFEST_FORMAT
        ),
        "parent_worker_cell_ledger_materialized": False,
        "course_not_started": True,
        "candidate_training_not_started": True,
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
        "parent_dir": _relative_path(parent_dir),
        "worker_dir": _relative_path(worker_dir),
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
        "can_start_r6_formal": False,
        "can_promote": False,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "Implement the S->G->K execution layer over this frozen ledger; do not run a "
            "cell until the per-arm baseline, causal, resource, retention, and rollback "
            "measurements are recorded."
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
    parser.add_argument("--input-report", type=Path, default=DEFAULT_INPUT_REPORT)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--worker-dir", type=Path, default=DEFAULT_WORKER_DIR)
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    input_report_path = (
        args.input_report if args.input_report.is_absolute() else PROJECT_ROOT / args.input_report
    )
    parent_dir = args.parent_dir if args.parent_dir.is_absolute() else PROJECT_ROOT / args.parent_dir
    worker_dir = args.worker_dir if args.worker_dir.is_absolute() else PROJECT_ROOT / args.worker_dir
    report = run_preflight(
        manifest_path=manifest_path,
        report_path=report_path,
        input_report_path=input_report_path,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
    )
    print(
        json.dumps(
            {
                "report": _relative_path(report_path),
                "status": report["status"],
                "formal_input_ready": report["formal_input_ready"],
                "course_executed": report["course_executed"],
                "can_start_r6_formal": report["can_start_r6_formal"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "input_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
