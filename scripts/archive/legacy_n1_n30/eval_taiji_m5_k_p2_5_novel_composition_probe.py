"""Probe a genuinely unseen recovery-after-list language/toolchain composition.

P2.3 used a Python ``inspect-language`` candidate that the inherited parent
already solved.  P2.5 keeps the recovery order but changes the candidate tuple
to TypeScript plus an available toolchain, which is absent from the P1
training combination.  The probe builds and audits two validation-only
records, saves their content-addressed manifest, and scores the frozen parent
and an existing P2 wake-only reference without calling ``fit``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_p2_3_recovery_continuation import (  # noqa: E402
    _jsonable,
    _restore_tensors,
)
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    _build_workspace,
    _observation,
    _registry,
    _schema,
    _semantic_example,
    _transition_examples,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _chain_row,
    _fresh_learners,
    _load_arm,
    _save_arm,
    _summarize,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    WORKER_ROOT,
    _checkpoint_preflight,
    _context,
)
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    StructuredSemanticExample,
    StructuredSemanticTransitionExample,
    content_digest,
)

P2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
P2_4_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_4_retention_canary_20260910.json"
REPORT_FORMAT = "taiji-m5-k-p2-5-novel-composition-probe-v1"
MANIFEST_FORMAT = "taiji-m5-k-p2-5-novel-composition-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_5_novel_composition_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_5_novel_composition_probe_20260910.json"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _build_record(
    *,
    root: Path,
    index: int,
    candidate_path: str,
    project_id: str,
    task_seed: int,
    source_manifest_digest: str,
    parent_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=False)
    _build_workspace(root, task_seed=task_seed)
    registry = _registry(typescript_available=True)
    environment = WorkbenchEnvironment(root=root, programming_language_registry=registry)
    schema = _schema()
    anchor = _observation(
        environment,
        tag=f"p2-5:{index}:anchor",
        path="missing_00.txt",
        schema=schema,
        tick=0,
    )
    initial = _observation(
        environment,
        tag=f"p2-5:{index}:initial",
        path="missing_novel.txt",
        schema=schema,
        tick=1,
    )
    candidate = _observation(
        environment,
        tag=f"p2-5:{index}:candidate",
        path=candidate_path,
        schema=schema,
        tick=2,
    )
    initial = replace(
        initial,
        project_id=project_id,
        task_id=f"{project_id}:missing_novel.txt",
    )
    candidate = replace(
        candidate,
        project_id=project_id,
        task_id=f"{project_id}:{candidate_path}",
    )
    candidate_semantic = _semantic_example(
        candidate,
        split=f"p2-5-validation-{index:04d}",
        tick=2,
    )
    candidate_transition = _transition_examples(
        (anchor, initial, candidate),
        split=f"p2-5-validation-{index:04d}",
    )[1]
    record = {
        "record_id": f"p2-5-novel-validation-{index:04d}",
        "split": "validation",
        "index": index,
        "class_key": "R-novel-composition",
        "project_id": project_id,
        "template_family_id": content_digest(
            {
                "format": "taiji-m5-k-p2-5-novel-template-v1",
                "candidate_path": candidate_path,
                "toolchain_available": True,
            }
        ),
        "source_p1_manifest_digest": source_manifest_digest,
        "parent_checkpoint_digest": parent_digest,
        "novel_tuple": {
            "recovery_phase": "after_workspace_list",
            "language_id": candidate.language_id,
            "toolchain_available": candidate.toolchain_available,
            "selection_state": candidate.selection_state,
            "content_id": candidate_semantic.content.content_id,
        },
        "initial": {
            "observation": _jsonable(initial.to_payload()),
            "percept": _jsonable(initial.to_percept_event(tick=1).to_payload()),
            "next_step": "workspace.list",
            "fit_eligible": False,
        },
        "candidate": {
            "observation": _jsonable(candidate.to_payload()),
            "semantic_example": _jsonable(candidate_semantic.to_payload()),
            "transition_example": _jsonable(candidate_transition.to_payload()),
            "k1_input_digest": candidate_semantic.input_digest,
            "k2_input_digest": candidate_transition.input_digest,
            "fit_eligible": False,
        },
    }
    record["record_digest"] = content_digest(record)
    runtime_case = {
        "index": index,
        "class_key": "R-novel-composition",
        "state_profile": "recovered-typescript-available",
        "root": root,
        "registry": registry,
        "observation": candidate,
        "record": record,
    }
    return record, runtime_case


def _audit_records(
    records: list[Mapping[str, Any]],
    *,
    p1_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    p1_paths = {
        str(path)
        for item in p1_manifest.get("records", [])
        for path in item.get("variant_paths", [])
    }
    candidate_paths: list[str] = []
    for record in records:
        record_id = str(record.get("record_id"))
        without_digest = dict(record)
        digest = without_digest.pop("record_digest", None)
        if digest != content_digest(without_digest):
            errors.append({"record_id": record_id, "reason": "record_digest_mismatch"})
        initial = record.get("initial", {})
        candidate = record.get("candidate", {})
        initial_observation = initial.get("observation", {})
        candidate_observation = candidate.get("observation", {})
        candidate_percept = candidate.get("semantic_example", {}).get("percept", {})
        if initial_observation.get("read_success") is not False:
            errors.append({"record_id": record_id, "reason": "initial_not_missing"})
        if float(initial.get("percept", {}).get("confidence", 1.0)) >= 0.55:
            errors.append({"record_id": record_id, "reason": "initial_confidence_not_low"})
        if initial.get("next_step") != "workspace.list":
            errors.append({"record_id": record_id, "reason": "initial_next_step_mismatch"})
        if candidate_observation.get("read_success") is not True:
            errors.append({"record_id": record_id, "reason": "candidate_not_readable"})
        if candidate_observation.get("language_id") != "typescript":
            errors.append({"record_id": record_id, "reason": "candidate_language_not_typescript"})
        if candidate_observation.get("selection_state") != "resolved":
            errors.append({"record_id": record_id, "reason": "candidate_not_resolved"})
        if candidate_observation.get("toolchain_available") is not True:
            errors.append({"record_id": record_id, "reason": "candidate_toolchain_not_available"})
        if float(candidate_percept.get("confidence", 0.0)) < 0.55:
            errors.append({"record_id": record_id, "reason": "candidate_confidence_below_floor"})
        candidate_path = str(candidate_observation.get("path"))
        candidate_paths.append(candidate_path)
        if candidate_path in p1_paths:
            errors.append({"record_id": record_id, "reason": "candidate_path_seen_in_p1"})
        if record.get("novel_tuple", {}).get("content_id") != "content:inspect-language":
            errors.append({"record_id": record_id, "reason": "candidate_content_target_mismatch"})
    return {
        "record_count": len(records),
        "candidate_paths": sorted(candidate_paths),
        "candidate_paths_disjoint_from_p1": not set(candidate_paths).intersection(p1_paths),
        "novel_tuple_count": len({content_digest(record["novel_tuple"]) for record in records}),
        "errors": errors,
        "passed": bool(records) and not errors,
    }


def run_probe(
    *,
    p1_manifest_path: Path = P1_MANIFEST,
    p2_report_path: Path = P2_REPORT,
    p2_4_report_path: Path = P2_4_REPORT,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p2_5_probe_{uuid4().hex}"
    scratch = run_dir / "scratch"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "sealed_payload_read": False,
        "can_promote": False,
        "p1_manifest": str(p1_manifest_path),
        "p2_report": str(p2_report_path),
        "p2_4_report": str(p2_4_report_path),
        "manifest": str(manifest_path),
    }
    try:
        p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
        p2_report = json.loads(p2_report_path.read_text(encoding="utf-8"))
        p2_4_report = json.loads(p2_4_report_path.read_text(encoding="utf-8"))
        if p1_manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
            raise ValueError("P2.5 requires P1 v2 manifest")
        if p2_report.get("status") != "completed" or p2_report.get("sealed_payload_read"):
            raise ValueError("P2.5 requires a completed unsealed P2 report")
        if p2_4_report.get("status") != "completed" or p2_4_report.get("sealed_payload_read"):
            raise ValueError("P2.5 requires a completed unsealed P2.4 report")
        _artifacts, parent_digest, bundle, _projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_report["contract"]["parent_checkpoint_digest"] != parent_digest:
            raise ValueError("P2.5 parent checkpoint digest drifted")
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        specs = ((0, "typescript_06.ts"), (1, "typescript_07.ts"))
        records: list[dict[str, Any]] = []
        runtime_cases: list[dict[str, Any]] = []
        for index, candidate_path in specs:
            record, case = _build_record(
                root=scratch / f"validation-{index:04d}",
                index=index,
                candidate_path=candidate_path,
                project_id=f"p2-5-validation-project-{index}",
                task_seed=5100 + index,
                source_manifest_digest=str(p1_manifest["manifest_digest"]),
                parent_digest=parent_digest,
            )
            records.append(record)
            runtime_cases.append(case)
        audit = _audit_records(records, p1_manifest=p1_manifest)
        if not audit["passed"]:
            raise RuntimeError(f"P2.5 novel composition contract failed: {audit['errors'][:3]}")
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p1_manifest_digest": str(p1_manifest["manifest_digest"]),
            "parent_checkpoint_digest": parent_digest,
            "novel_tuple_contract": {
                "language_id": "typescript",
                "toolchain_available": True,
                "selection_state": "resolved",
                "content_id": "content:inspect-language",
                "candidate_paths_must_be_disjoint_from_p1": True,
            },
            "records": records,
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        parent_preflight = _checkpoint_preflight(
            output_dir=run_dir / "preflight-parent",
            semantic_parent=_artifacts["k1.semantic"]["checkpoint"],
            transition_parent=_artifacts["k2.transition"]["checkpoint"],
        )
        if not parent_preflight["passed"]:
            raise RuntimeError("P2.5 parent checkpoint preflight failed")
        semantic_parent, transition_parent = _fresh_learners(
            _artifacts["k1.semantic"]["checkpoint"],
            _artifacts["k2.transition"]["checkpoint"],
        )
        parent_rows: list[dict[str, Any]] = []
        for case, record in zip(runtime_cases, records, strict=True):
            semantic_example = StructuredSemanticExample.from_payload(
                _restore_tensors(record["candidate"]["semantic_example"])
            )
            transition_example = StructuredSemanticTransitionExample.from_payload(
                _restore_tensors(record["candidate"]["transition_example"])
            )
            parent_rows.append(
                _chain_row(
                    semantic=semantic_parent,
                    transition=transition_parent,
                    case=case,
                    semantic_example=semantic_example,
                    transition_example=transition_example,
                    label="p2-5-novel-parent",
                )
            )
        parent_scores = _summarize(parent_rows)
        reference_semantic, reference_transition = _load_arm(
            p2_report["arms"]["wake-only"]["checkpoint"]
        )
        reference_rows: list[dict[str, Any]] = []
        for case, record in zip(runtime_cases, records, strict=True):
            semantic_example = StructuredSemanticExample.from_payload(
                _restore_tensors(record["candidate"]["semantic_example"])
            )
            transition_example = StructuredSemanticTransitionExample.from_payload(
                _restore_tensors(record["candidate"]["transition_example"])
            )
            reference_rows.append(
                _chain_row(
                    semantic=reference_semantic,
                    transition=reference_transition,
                    case=case,
                    semantic_example=semantic_example,
                    transition_example=transition_example,
                    label="p2-5-novel-p2-reference",
                )
            )
        reference_scores = _summarize(reference_rows)
        parent_checkpoint = _save_arm(
            run_dir / "arms" / "parent-frozen",
            semantic_parent,
            transition_parent,
        )
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "contract": audit,
                "novel_tuple": manifest["novel_tuple_contract"],
                "checkpoint_preflight": {
                    "parent_before_probe": parent_preflight,
                    "parent_saved_after_probe": parent_checkpoint,
                },
                "arms": {
                    "parent-frozen": {"scores": parent_scores},
                    "p2-wake-only-reference": {"scores": reference_scores},
                },
                "interpretation": (
                    "validation-only novel composition probe; no fit, no sealed payload, "
                    "host recovery is not model credit"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "run_dir": str(run_dir),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    _write_json_atomic(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-manifest", type=Path, default=P1_MANIFEST)
    parser.add_argument("--p2-report", type=Path, default=P2_REPORT)
    parser.add_argument("--p2-4-report", type=Path, default=P2_4_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_probe(
        p1_manifest_path=args.p1_manifest,
        p2_report_path=args.p2_report,
        p2_4_report_path=args.p2_4_report,
        manifest_path=args.manifest,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "report": str(args.report),
                "status": result["status"],
                "training_performed": result["training_performed"],
                "can_promote": result["can_promote"],
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
