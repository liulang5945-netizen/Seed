"""Probe holdout generalization of the P2.6 learned K2-content transition.

P2.6 learned one concrete K2 content transition from six novel paths while
preserving the old rehearsal cohort.  This step does not fit again.  It loads
that saved arm, verifies its content-addressed checkpoint and independent
restore evidence, then scores four entirely new paths across two projects.
The P2.6 validation rows are retained only as a learned-arm sanity check.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_p2_3_recovery_continuation import (  # noqa: E402
    _restore_tensors,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _chain_row,
    _fresh_learners,
    _independent_restore,
    _load_mapping,
    _save_arm,
    _summarize,
)
from scripts.training.eval_taiji_m5_k_p2_6_novel_learning import (  # noqa: E402
    _build_record as _build_novel_record,
)
from scripts.training.eval_taiji_m5_k_p2_output_action_diagnostic import (  # noqa: E402
    _build_validation_cases,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    WORKER_ROOT,
    _checkpoint_preflight,
    _context,
    _rebuild_and_verify_manifest,
)
from taiji import (  # noqa: E402
    StructuredSemanticExample,
    StructuredSemanticTransitionExample,
    content_digest,
)

P2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
P2_4_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_4_retention_canary_20260910.json"
P2_6_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_6_novel_learning_20260910.json"
P2_6_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_6_novel_learning_manifest_v1.json"
)
REPORT_FORMAT = "taiji-m5-k-p2-7-generalization-probe-v1"
MANIFEST_FORMAT = "taiji-m5-k-p2-7-generalization-manifest-v1"
VERSION = 1
CONFIDENCE_FLOOR = 0.55
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_7_generalization_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_7_generalization_20260910.json"
HOLDOUT_SPECS = (
    (0, "typescript_28.ts", "p2-7-holdout-project-a"),
    (1, "typescript_29.ts", "p2-7-holdout-project-a"),
    (2, "typescript_30.ts", "p2-7-holdout-project-b"),
    (3, "typescript_31.ts", "p2-7-holdout-project-b"),
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _relabel_holdout_record(
    record: Mapping[str, Any],
    *,
    source_p2_6_manifest_digest: str,
    index: int,
    project_id: str,
) -> dict[str, Any]:
    result = dict(record)
    result["record_id"] = f"p2-7-holdout-validation-{index:04d}"
    result["split"] = "validation"
    result["project_id"] = project_id
    result["source_p2_6_manifest_digest"] = source_p2_6_manifest_digest
    result["holdout_role"] = "unseen-path-unseen-project"
    result["candidate"] = dict(result["candidate"])
    result["candidate"]["fit_eligible"] = False
    result["template_family_id"] = content_digest(
        {
            "format": "taiji-m5-k-p2-7-holdout-template-v1",
            "candidate_path": result["candidate"]["observation"]["path"],
            "project_id": project_id,
        }
    )
    result.pop("record_digest", None)
    result["record_digest"] = content_digest(result)
    return result


def _audit_records(
    records: Sequence[Mapping[str, Any]],
    *,
    p1_manifest: Mapping[str, Any],
    p2_6_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    prior_paths = {
        str(path)
        for item in p1_manifest.get("records", [])
        for path in item.get("variant_paths", [])
    }
    for source_record in [
        *p2_6_manifest.get("train_records", []),
        *p2_6_manifest.get("validation_records", []),
    ]:
        prior_paths.add(str(source_record["candidate"]["observation"]["path"]))
    candidate_paths: list[str] = []
    projects: set[str] = set()
    tuple_digests: set[str] = set()
    for record in records:
        record_id = str(record.get("record_id"))
        without_digest = dict(record)
        digest = without_digest.pop("record_digest", None)
        if digest != content_digest(without_digest):
            errors.append({"record_id": record_id, "reason": "record_digest_mismatch"})
        candidate = record.get("candidate", {})
        observation = candidate.get("observation", {})
        percept = candidate.get("semantic_example", {}).get("percept", {})
        candidate_path = str(observation.get("path"))
        candidate_paths.append(candidate_path)
        projects.add(str(record.get("project_id")))
        tuple_digests.add(content_digest(record.get("novel_tuple", {})))
        if record.get("split") != "validation":
            errors.append({"record_id": record_id, "reason": "holdout_not_validation"})
        if candidate.get("fit_eligible") is not False:
            errors.append({"record_id": record_id, "reason": "holdout_fit_eligible"})
        if observation.get("read_success") is not True:
            errors.append({"record_id": record_id, "reason": "candidate_not_readable"})
        if observation.get("language_id") != "typescript":
            errors.append({"record_id": record_id, "reason": "candidate_language_not_typescript"})
        if observation.get("selection_state") != "resolved":
            errors.append({"record_id": record_id, "reason": "candidate_not_resolved"})
        if observation.get("toolchain_available") is not True:
            errors.append({"record_id": record_id, "reason": "candidate_toolchain_not_available"})
        if float(percept.get("confidence", 0.0)) < CONFIDENCE_FLOOR:
            errors.append({"record_id": record_id, "reason": "candidate_confidence_below_floor"})
        if record.get("novel_tuple", {}).get("content_id") != "content:inspect-language":
            errors.append({"record_id": record_id, "reason": "candidate_content_target_mismatch"})
        if candidate_path in prior_paths:
            errors.append({"record_id": record_id, "reason": "candidate_path_seen_in_prior_data"})
    return {
        "record_count": len(records),
        "candidate_paths": sorted(candidate_paths),
        "candidate_paths_unique": len(candidate_paths) == len(set(candidate_paths)),
        "candidate_paths_disjoint_from_prior": not set(candidate_paths).intersection(prior_paths),
        "project_count": len(projects),
        "novel_tuple_count": len(tuple_digests),
        "errors": errors,
        "passed": (
            len(records) == len(HOLDOUT_SPECS)
            and len(projects) >= 2
            and len(tuple_digests) == 1
            and len(candidate_paths) == len(set(candidate_paths))
            and not errors
        ),
    }


def _score_arm(
    *,
    semantic: Any,
    transition: Any,
    p1_cases: Sequence[Mapping[str, Any]],
    holdout_cases: Sequence[Mapping[str, Any]],
    holdout_records: Sequence[Mapping[str, Any]],
    sanity_cases: Sequence[Mapping[str, Any]],
    sanity_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def score_rows(
        cases: Sequence[Mapping[str, Any]],
        records: Sequence[Mapping[str, Any]],
        label: str,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for case, record in zip(cases, records, strict=True):
            candidate = record["candidate"]
            rows.append(
                _chain_row(
                    semantic=semantic,
                    transition=transition,
                    case=case,
                    semantic_example=StructuredSemanticExample.from_payload(
                        _restore_tensors(candidate["semantic_example"])
                    ),
                    transition_example=StructuredSemanticTransitionExample.from_payload(
                        _restore_tensors(candidate["transition_example"])
                    ),
                    label=label,
                )
            )
        return _summarize(rows)

    p1_rows: list[dict[str, Any]] = []
    for case in p1_cases:
        experience = case["experience"]
        p1_rows.append(
            _chain_row(
                semantic=semantic,
                transition=transition,
                case=case,
                semantic_example=experience.semantic_example,
                transition_example=experience.transition_example,
                label="p1-validation",
            )
        )
    return {
        "parameter_count": {
            "k1": int(semantic.parameter_count),
            "k2": int(transition.parameter_count),
            "total": int(semantic.parameter_count + transition.parameter_count),
        },
        "p1_validation": _summarize(p1_rows),
        "p2_6_validation_sanity": score_rows(
            sanity_cases,
            sanity_records,
            "p2-6-learned-sanity",
        ),
        "p2_7_holdout_validation": score_rows(
            holdout_cases,
            holdout_records,
            "p2-7-holdout-validation",
        ),
    }


def _non_decreasing(actual: Mapping[str, Any], baseline: Mapping[str, Any], key: str) -> bool:
    return int(actual[key]) >= int(baseline[key])


def run_probe(
    *,
    p1_manifest_path: Path = P1_MANIFEST,
    p2_report_path: Path = P2_REPORT,
    p2_4_report_path: Path = P2_4_REPORT,
    p2_6_report_path: Path = P2_6_REPORT,
    p2_6_manifest_path: Path = P2_6_MANIFEST,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p2_7_generalization_{uuid4().hex}"
    scratch = run_dir / "scratch"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "model_seed": MODEL_SEED,
        "training_performed": False,
        "sealed_payload_read": False,
        "can_promote": False,
        "p1_manifest": str(p1_manifest_path),
        "p2_report": str(p2_report_path),
        "p2_4_report": str(p2_4_report_path),
        "p2_6_report": str(p2_6_report_path),
        "p2_6_manifest": str(p2_6_manifest_path),
        "manifest": str(manifest_path),
    }
    try:
        p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
        p2_report = json.loads(p2_report_path.read_text(encoding="utf-8"))
        p2_4_report = json.loads(p2_4_report_path.read_text(encoding="utf-8"))
        p2_6_report = json.loads(p2_6_report_path.read_text(encoding="utf-8"))
        p2_6_manifest = json.loads(p2_6_manifest_path.read_text(encoding="utf-8"))
        if p1_manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
            raise ValueError("P2.7 requires the P1 v2 manifest")
        for name, report in (
            ("P2", p2_report),
            ("P2.4", p2_4_report),
            ("P2.6", p2_6_report),
        ):
            if report.get("status") != "completed" or report.get("sealed_payload_read"):
                raise ValueError(f"P2.7 requires a completed, unsealed {name} report")
        if not all(bool(value) for value in p2_4_report.get("retention_gate", {}).values()):
            raise ValueError("P2.4 retention gate is not fully passed")
        if not all(bool(value) for value in p2_6_report.get("retention_gate", {}).values()):
            raise ValueError("P2.6 retention gate is not fully passed")
        if p2_6_report.get("manifest_digest") != p2_6_manifest.get("manifest_digest"):
            raise ValueError("P2.6 manifest digest mismatch")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_report["contract"]["parent_checkpoint_digest"] != parent_digest:
            raise ValueError("P2 parent checkpoint digest drifted")
        if p2_6_manifest.get("parent_checkpoint_digest") != parent_digest:
            raise ValueError("P2.6 parent checkpoint digest drifted")
        learned_checkpoint = p2_6_report["arms"]["interleaved-rehearsal-novel"]["checkpoint"]
        if not learned_checkpoint.get("passed"):
            raise ValueError("P2.6 learned checkpoint did not pass its own save/restore Gate")
        learned_dir = Path(str(learned_checkpoint["files"]["k1"]["path"])).parent
        if not all(Path(str(item["path"])).is_file() for item in learned_checkpoint["files"].values()):
            raise FileNotFoundError("P2.6 learned checkpoint files are not available")
        learned_restore = _independent_restore(learned_dir)
        if not learned_restore.get("independent_process_restore"):
            raise RuntimeError("P2.6 learned checkpoint independent restore failed")
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        (
            _train_experiences,
            _train_metadata,
            p1_validation,
            p1_validation_metadata,
            p1_verification,
        ) = _rebuild_and_verify_manifest(
            scratch=scratch / "p1",
            manifest=p1_manifest,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        if not p1_verification["passed"]:
            raise RuntimeError("P1 manifest reconstruction failed before P2.7 probe")
        p1_cases, p1_case_mismatches = _build_validation_cases(
            scratch=scratch / "p1-cases",
            validation=p1_validation,
            validation_metadata=p1_validation_metadata,
        )
        if p1_case_mismatches:
            raise RuntimeError(f"P1 validation case mismatch: {p1_case_mismatches[:3]}")
        holdout_records: list[dict[str, Any]] = []
        holdout_cases: list[dict[str, Any]] = []
        for index, candidate_path, project_id in HOLDOUT_SPECS:
            source_record, case = _build_novel_record(
                root=scratch / f"holdout-{index:04d}",
                index=index,
                candidate_path=candidate_path,
                project_id=project_id,
                task_seed=6300 + index,
                split="validation",
                source_manifest_digest=str(p1_manifest["manifest_digest"]),
                parent_digest=parent_digest,
                fit_eligible=False,
            )
            record = _relabel_holdout_record(
                source_record,
                source_p2_6_manifest_digest=str(p2_6_manifest["manifest_digest"]),
                index=index,
                project_id=project_id,
            )
            holdout_records.append(record)
            case["record"] = record
            holdout_cases.append(case)
        audit = _audit_records(
            holdout_records,
            p1_manifest=p1_manifest,
            p2_6_manifest=p2_6_manifest,
        )
        if not audit["passed"]:
            raise RuntimeError(f"P2.7 holdout contract failed: {audit['errors'][:3]}")
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p1_manifest_digest": str(p1_manifest["manifest_digest"]),
            "source_p2_6_manifest_digest": str(p2_6_manifest["manifest_digest"]),
            "parent_checkpoint_digest": parent_digest,
            "learned_checkpoint_digests": learned_checkpoint["checkpoint_digests"],
            "holdout_tuple_contract": {
                "recovery_phase": "after_workspace_list",
                "language_id": "typescript",
                "toolchain_available": True,
                "selection_state": "resolved",
                "content_id": "content:inspect-language",
                "candidate_paths_must_be_disjoint_from_prior": True,
            },
            "records": holdout_records,
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        parent_preflight = _checkpoint_preflight(
            output_dir=run_dir / "preflight-parent",
            semantic_parent=_artifacts["k1.semantic"]["checkpoint"],
            transition_parent=_artifacts["k2.transition"]["checkpoint"],
        )
        if not parent_preflight["passed"]:
            raise RuntimeError("P2.7 parent checkpoint preflight failed")
        parent_semantic, parent_transition = _fresh_learners(
            _artifacts["k1.semantic"]["checkpoint"],
            _artifacts["k2.transition"]["checkpoint"],
        )
        learned_semantic, learned_transition = _fresh_learners(
            _load_mapping(Path(str(learned_checkpoint["files"]["k1"]["path"]))),
            _load_mapping(Path(str(learned_checkpoint["files"]["k2"]["path"]))),
        )
        sanity_records = p2_6_manifest["validation_records"]
        sanity_cases: list[dict[str, Any]] = []
        for record in sanity_records:
            index = int(record["index"])
            _source, case = _build_novel_record(
                root=scratch / f"sanity-{index:04d}",
                index=index,
                candidate_path=str(record["candidate"]["observation"]["path"]),
                project_id=str(record["project_id"]),
                task_seed=6200 + index,
                split="validation",
                source_manifest_digest=str(p1_manifest["manifest_digest"]),
                parent_digest=parent_digest,
                fit_eligible=False,
            )
            case["record"] = record
            sanity_cases.append(case)
        arm_payload: dict[str, Any] = {}
        for arm_name, semantic, transition in (
            ("parent-frozen", parent_semantic, parent_transition),
            ("p2-6-interleaved-learned", learned_semantic, learned_transition),
        ):
            checkpoint = _save_arm(run_dir / "arms" / arm_name, semantic, transition)
            if not checkpoint["passed"]:
                raise RuntimeError(f"P2.7 save/restore failed for {arm_name}")
            arm_payload[arm_name] = {
                "checkpoint": checkpoint,
                "training_steps_total": {
                    "k1": int(semantic.training_steps),
                    "k2": int(transition.training_steps),
                },
                "scores": _score_arm(
                    semantic=semantic,
                    transition=transition,
                    p1_cases=p1_cases,
                    holdout_cases=holdout_cases,
                    holdout_records=holdout_records,
                    sanity_cases=sanity_cases,
                    sanity_records=sanity_records,
                ),
            }
        parent_scores = arm_payload["parent-frozen"]["scores"]
        learned_scores = arm_payload["p2-6-interleaved-learned"]["scores"]
        p2_4_baseline = p2_4_report["arms"]["parent-frozen"]["scores"]["p1_validation"]
        sanity = learned_scores["p2_6_validation_sanity"]
        holdout = learned_scores["p2_7_holdout_validation"]
        learned_p1 = learned_scores["p1_validation"]
        generalization_gate = {
            "learned_sanity_k2_content_2_of_2": sanity["k2_content_hit_count"] == 2,
            "holdout_k2_content_4_of_4": holdout["k2_content_hit_count"] == 4,
            "holdout_workbench_4_of_4": holdout["workbench_success_count"] == 4,
            "holdout_k1_goal_4_of_4": holdout["k1_goal_hit_count"] == 4,
            "holdout_k2_goal_4_of_4": holdout["k2_goal_hit_count"] == 4,
            "p2_4_parent_k1_goal_non_decreasing": _non_decreasing(
                learned_p1, p2_4_baseline, "k1_goal_hit_count"
            ),
            "p2_4_parent_k2_goal_non_decreasing": _non_decreasing(
                learned_p1, p2_4_baseline, "k2_goal_hit_count"
            ),
            "p2_4_parent_safe_abstention_non_decreasing": _non_decreasing(
                learned_p1, p2_4_baseline, "safe_abstention_count"
            ),
            "parameter_count_stable": learned_scores["parameter_count"]
            == parent_scores["parameter_count"],
        }
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "contract": audit,
                "manifest_reconstruction": p1_verification,
                "checkpoint_preflight": {
                    "p2_6_learned_source": learned_restore,
                    "parent_before_probe": parent_preflight,
                    "all_saved_arms_after_probe": True,
                },
                "arms": arm_payload,
                "generalization_gate": generalization_gate,
                "can_promote": False,
                "interpretation": (
                    "validation-only holdout generalization probe; no fit, no sealed payload, "
                    "host recovery is not model credit; no promotion claim"
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
    parser.add_argument("--p2-6-report", type=Path, default=P2_6_REPORT)
    parser.add_argument("--p2-6-manifest", type=Path, default=P2_6_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_probe(
        p1_manifest_path=args.p1_manifest,
        p2_report_path=args.p2_report,
        p2_4_report_path=args.p2_4_report,
        p2_6_report_path=args.p2_6_report,
        p2_6_manifest_path=args.p2_6_manifest,
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
                "generalization_gate": result.get("generalization_gate"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
