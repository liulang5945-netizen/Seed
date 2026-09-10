"""Learn the P2.5 K2-content gap with a retention-preserving canary.

P2.5 showed that the inherited parent can recognize and execute the unseen
TypeScript + available-toolchain composition, but its K2 content readout stays
``None`` after the recovery continuation.  This experiment adds only that
missing continuation target:

* six novel train candidates and two disjoint validation candidates;
* the fixed 50-example P2 rehearsal stream from P2.4;
* ``parent-frozen``, ``rehearsal-only`` and interleaved arms.

The experiment is deliberately bounded.  It does not grow parameters, lower
the confidence floor, treat host recovery as model credit, read sealed
payloads, or promote a checkpoint.  A successful result means only that this
specific K2 content transition was learned without violating the frozen
retention constraints.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
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
    _summarize,
)
from scripts.training.eval_taiji_m5_k_p2_4_retention_canary import (  # noqa: E402
    _fit_stream,
    _interleaved_stream,
)
from scripts.training.eval_taiji_m5_k_p2_output_action_diagnostic import (  # noqa: E402
    _build_validation_cases,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    PILOT_PER_CLASS,
    WORKER_ROOT,
    _checkpoint_preflight,
    _context,
    _rebuild_and_verify_manifest,
    _select_balanced_wake,
)
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    StructuredSemanticExample,
    StructuredSemanticTransitionExample,
    content_digest,
)

P2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
P2_4_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_4_retention_canary_20260910.json"
P2_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_5_novel_composition_probe_20260910.json"
P2_5_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_5_novel_composition_manifest_v1.json"
)
REPORT_FORMAT = "taiji-m5-k-p2-6-novel-learning-v1"
MANIFEST_FORMAT = "taiji-m5-k-p2-6-novel-learning-manifest-v1"
VERSION = 1
CONFIDENCE_FLOOR = 0.55
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_6_novel_learning_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_6_novel_learning_20260910.json"

TRAIN_SPECS = tuple((index, f"typescript_{20 + index:02d}.ts") for index in range(6))
VALIDATION_SPECS = ((0, "typescript_26.ts"), (1, "typescript_27.ts"))


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
    split: str,
    source_manifest_digest: str,
    parent_digest: str,
    fit_eligible: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=False)
    _build_workspace(root, task_seed=task_seed)
    candidate_file = root / candidate_path
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_file.write_text(
        "interface NovelAnswer { value: number; }\n"
        f"export const novelAnswer{index} = (input: NovelAnswer): number => "
        f"input.value + {task_seed};\n",
        encoding="utf-8",
    )
    registry = _registry(typescript_available=True)
    environment = WorkbenchEnvironment(root=root, programming_language_registry=registry)
    schema = _schema()
    anchor = _observation(
        environment,
        tag=f"p2-6:{split}:{index}:anchor",
        path="missing_00.txt",
        schema=schema,
        tick=0,
    )
    initial = _observation(
        environment,
        tag=f"p2-6:{split}:{index}:initial",
        path="missing_novel.txt",
        schema=schema,
        tick=1,
    )
    candidate = _observation(
        environment,
        tag=f"p2-6:{split}:{index}:candidate",
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
        split=f"p2-6-{split}-{index:04d}",
        tick=2,
    )
    candidate_transition = _transition_examples(
        (anchor, initial, candidate),
        split=f"p2-6-{split}-{index:04d}",
    )[1]
    record = {
        "record_id": f"p2-6-novel-{split}-{index:04d}",
        "split": split,
        "index": index,
        "class_key": "R-novel-composition",
        "project_id": project_id,
        "template_family_id": content_digest(
            {
                "format": "taiji-m5-k-p2-6-novel-template-v1",
                "split": split,
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
            "fit_eligible": fit_eligible,
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


def _candidate_examples(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[StructuredSemanticExample], list[StructuredSemanticTransitionExample]]:
    semantics: list[StructuredSemanticExample] = []
    transitions: list[StructuredSemanticTransitionExample] = []
    for record in records:
        candidate = record["candidate"]
        semantics.append(
            StructuredSemanticExample.from_payload(
                _restore_tensors(candidate["semantic_example"])
            )
        )
        transitions.append(
            StructuredSemanticTransitionExample.from_payload(
                _restore_tensors(candidate["transition_example"])
            )
        )
    return semantics, transitions


def _audit_records(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    *,
    p1_manifest: Mapping[str, Any],
    p2_5_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    prior_paths = {
        str(path)
        for item in p1_manifest.get("records", [])
        for path in item.get("variant_paths", [])
    }
    prior_paths.update(
        str(record.get("candidate", {}).get("observation", {}).get("path"))
        for record in p2_5_manifest.get("records", [])
    )
    all_records = [*train_records, *validation_records]
    candidate_paths: list[str] = []
    project_ids: dict[str, str] = {}
    for record in all_records:
        record_id = str(record.get("record_id"))
        without_digest = dict(record)
        digest = without_digest.pop("record_digest", None)
        if digest != content_digest(without_digest):
            errors.append({"record_id": record_id, "reason": "record_digest_mismatch"})
        split = str(record.get("split"))
        fit_eligible = bool(record.get("candidate", {}).get("fit_eligible"))
        if (split == "train") != fit_eligible:
            errors.append({"record_id": record_id, "reason": "fit_eligibility_split_mismatch"})
        initial = record.get("initial", {})
        candidate = record.get("candidate", {})
        initial_observation = initial.get("observation", {})
        candidate_observation = candidate.get("observation", {})
        candidate_percept = candidate.get("semantic_example", {}).get("percept", {})
        if initial_observation.get("read_success") is not False:
            errors.append({"record_id": record_id, "reason": "initial_not_missing"})
        if float(initial.get("percept", {}).get("confidence", 1.0)) >= CONFIDENCE_FLOOR:
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
        if float(candidate_percept.get("confidence", 0.0)) < CONFIDENCE_FLOOR:
            errors.append({"record_id": record_id, "reason": "candidate_confidence_below_floor"})
        if record.get("novel_tuple", {}).get("content_id") != "content:inspect-language":
            errors.append({"record_id": record_id, "reason": "candidate_content_target_mismatch"})
        candidate_path = str(candidate_observation.get("path"))
        candidate_paths.append(candidate_path)
        if candidate_path in prior_paths:
            errors.append({"record_id": record_id, "reason": "candidate_path_seen_in_prior_data"})
        project_id = str(record.get("project_id"))
        prior_project = project_ids.setdefault(project_id, split)
        if prior_project != split:
            errors.append({"record_id": record_id, "reason": "project_crosses_split"})
    train_paths = {
        str(record["candidate"]["observation"]["path"]) for record in train_records
    }
    validation_paths = {
        str(record["candidate"]["observation"]["path"]) for record in validation_records
    }
    tuple_digests = {content_digest(record["novel_tuple"]) for record in all_records}
    return {
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "candidate_paths": sorted(candidate_paths),
        "candidate_paths_unique": len(candidate_paths) == len(set(candidate_paths)),
        "candidate_paths_disjoint_from_prior": not set(candidate_paths).intersection(prior_paths),
        "train_validation_paths_disjoint": not train_paths.intersection(validation_paths),
        "novel_tuple_count": len(tuple_digests),
        "errors": errors,
        "passed": (
            len(train_records) == len(TRAIN_SPECS)
            and len(validation_records) == len(VALIDATION_SPECS)
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
    novel_cases: Sequence[Mapping[str, Any]],
    novel_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
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
    novel_rows: list[dict[str, Any]] = []
    for case, record in zip(novel_cases, novel_records, strict=True):
        candidate = record["candidate"]
        novel_rows.append(
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
                label="p2-6-novel-validation",
            )
        )
    return {
        "parameter_count": {
            "k1": int(semantic.parameter_count),
            "k2": int(transition.parameter_count),
            "total": int(semantic.parameter_count + transition.parameter_count),
        },
        "p1_validation": _summarize(p1_rows),
        "p2_6_novel_validation": _summarize(novel_rows),
    }


def _non_decreasing(
    actual: Mapping[str, Any],
    baseline: Mapping[str, Any],
    key: str,
) -> bool:
    return int(actual[key]) >= int(baseline[key])


def run_pilot(
    *,
    p1_manifest_path: Path = P1_MANIFEST,
    p2_report_path: Path = P2_REPORT,
    p2_4_report_path: Path = P2_4_REPORT,
    p2_5_report_path: Path = P2_5_REPORT,
    p2_5_manifest_path: Path = P2_5_MANIFEST,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p2_6_novel_learning_{uuid4().hex}"
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
        "p2_5_report": str(p2_5_report_path),
        "p2_5_manifest": str(p2_5_manifest_path),
        "manifest": str(manifest_path),
    }
    try:
        p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
        p2_report = json.loads(p2_report_path.read_text(encoding="utf-8"))
        p2_4_report = json.loads(p2_4_report_path.read_text(encoding="utf-8"))
        p2_5_report = json.loads(p2_5_report_path.read_text(encoding="utf-8"))
        p2_5_manifest = json.loads(p2_5_manifest_path.read_text(encoding="utf-8"))
        if p1_manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
            raise ValueError("P2.6 requires the P1 v2 manifest")
        for name, report in (
            ("P2", p2_report),
            ("P2.4", p2_4_report),
            ("P2.5", p2_5_report),
        ):
            if report.get("status") != "completed" or report.get("sealed_payload_read"):
                raise ValueError(f"P2.6 requires a completed, unsealed {name} report")
        if not all(bool(value) for value in p2_4_report.get("retention_gate", {}).values()):
            raise ValueError("P2.4 retention gate is not fully passed")
        if not p2_5_report.get("contract", {}).get("passed"):
            raise ValueError("P2.5 novel composition contract is not passed")
        if p2_5_report.get("manifest_digest") != p2_5_manifest.get("manifest_digest"):
            raise ValueError("P2.5 manifest digest mismatch")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_report["contract"]["parent_checkpoint_digest"] != parent_digest:
            raise ValueError("P2 parent checkpoint digest drifted")
        if p2_5_manifest.get("parent_checkpoint_digest") != parent_digest:
            raise ValueError("P2.5 parent checkpoint digest drifted")
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        train_records: list[dict[str, Any]] = []
        validation_records: list[dict[str, Any]] = []
        validation_cases: list[dict[str, Any]] = []
        for index, candidate_path in TRAIN_SPECS:
            record, _case = _build_record(
                root=scratch / f"train-{index:04d}",
                index=index,
                candidate_path=candidate_path,
                project_id=f"p2-6-train-project-{index}",
                task_seed=6100 + index,
                split="train",
                source_manifest_digest=str(p1_manifest["manifest_digest"]),
                parent_digest=parent_digest,
                fit_eligible=True,
            )
            train_records.append(record)
        for index, candidate_path in VALIDATION_SPECS:
            record, case = _build_record(
                root=scratch / f"validation-{index:04d}",
                index=index,
                candidate_path=candidate_path,
                project_id=f"p2-6-validation-project-{index}",
                task_seed=6200 + index,
                split="validation",
                source_manifest_digest=str(p1_manifest["manifest_digest"]),
                parent_digest=parent_digest,
                fit_eligible=False,
            )
            validation_records.append(record)
            validation_cases.append(case)
        audit = _audit_records(
            train_records,
            validation_records,
            p1_manifest=p1_manifest,
            p2_5_manifest=p2_5_manifest,
        )
        if not audit["passed"]:
            raise RuntimeError(f"P2.6 novel learning contract failed: {audit['errors'][:3]}")
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p1_manifest_digest": str(p1_manifest["manifest_digest"]),
            "source_p2_5_manifest_digest": str(p2_5_manifest["manifest_digest"]),
            "parent_checkpoint_digest": parent_digest,
            "novel_tuple_contract": {
                "recovery_phase": "after_workspace_list",
                "language_id": "typescript",
                "toolchain_available": True,
                "selection_state": "resolved",
                "content_id": "content:inspect-language",
                "candidate_paths_must_be_disjoint_from_prior": True,
            },
            "train_records": train_records,
            "validation_records": validation_records,
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        (
            train_experiences,
            train_metadata,
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
            raise RuntimeError("P1 manifest reconstruction failed before P2.6 fit")
        p1_cases, p1_case_mismatches = _build_validation_cases(
            scratch=scratch / "p1-cases",
            validation=p1_validation,
            validation_metadata=p1_validation_metadata,
        )
        if p1_case_mismatches:
            raise RuntimeError(f"P1 validation case mismatch: {p1_case_mismatches[:3]}")
        wake, wake_metadata = _select_balanced_wake(train_experiences, train_metadata)
        expected_wake = p2_report["contract"]["wake_experience_digests"]
        actual_wake = [str(item.experience_digest) for item in wake]
        if actual_wake != [str(item) for item in expected_wake]:
            raise RuntimeError("P2 rehearsal experience digest/order drifted")
        if any(
            sum(item["class_key"] == class_key for item in wake_metadata) != PILOT_PER_CLASS
            for class_key in ("A", "B", "C", "D", "R")
        ):
            raise RuntimeError("P2 rehearsal class balance drifted")
        semantic_parent_payload = copy.deepcopy(_artifacts["k1.semantic"]["checkpoint"])
        transition_parent_payload = copy.deepcopy(_artifacts["k2.transition"]["checkpoint"])
        parent_preflight = _checkpoint_preflight(
            output_dir=run_dir / "preflight-parent",
            semantic_parent=semantic_parent_payload,
            transition_parent=transition_parent_payload,
        )
        if not parent_preflight["passed"]:
            raise RuntimeError("P2.6 checkpoint save/independent-restore preflight failed")
        train_semantic, train_transition = _candidate_examples(train_records)
        parent_semantic, parent_transition = _fresh_learners(
            semantic_parent_payload,
            transition_parent_payload,
        )
        rehearsal_semantic, rehearsal_transition = _fresh_learners(
            semantic_parent_payload,
            transition_parent_payload,
        )
        interleaved_semantic, interleaved_transition = _fresh_learners(
            semantic_parent_payload,
            transition_parent_payload,
        )
        rehearsal_stream = [
            ("rehearsal", item.semantic_example, item.transition_example) for item in wake
        ]
        interleaved_stream = _interleaved_stream(
            wake,
            train_semantic,
            train_transition,
        )
        rehearsal_counts = _fit_stream(
            rehearsal_semantic,
            rehearsal_transition,
            rehearsal_stream,
        )
        interleaved_counts = _fit_stream(
            interleaved_semantic,
            interleaved_transition,
            interleaved_stream,
        )
        arm_data: dict[str, dict[str, Any]] = {
            "parent-frozen": {
                "semantic": parent_semantic,
                "transition": parent_transition,
                "fit_counts": {"rehearsal": 0, "novel": 0},
            },
            "rehearsal-only": {
                "semantic": rehearsal_semantic,
                "transition": rehearsal_transition,
                "fit_counts": {"rehearsal": rehearsal_counts["rehearsal"], "novel": 0},
            },
            "interleaved-rehearsal-novel": {
                "semantic": interleaved_semantic,
                "transition": interleaved_transition,
                "fit_counts": {
                    "rehearsal": interleaved_counts["rehearsal"],
                    "novel": interleaved_counts["continuation"],
                },
            },
        }
        arm_payload: dict[str, Any] = {}
        for arm_name, arm in arm_data.items():
            from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import _save_arm

            checkpoint = _save_arm(
                run_dir / "arms" / arm_name,
                arm["semantic"],
                arm["transition"],
            )
            if not checkpoint["passed"]:
                raise RuntimeError(f"P2.6 independent restore failed for {arm_name}")
            scores = _score_arm(
                semantic=arm["semantic"],
                transition=arm["transition"],
                p1_cases=p1_cases,
                novel_cases=validation_cases,
                novel_records=validation_records,
            )
            arm_payload[arm_name] = {
                "checkpoint": checkpoint,
                "fit_counts": arm["fit_counts"],
                "training_steps_total": {
                    "k1": int(arm["semantic"].training_steps),
                    "k2": int(arm["transition"].training_steps),
                },
                "new_training_steps": {
                    "k1": int(arm["semantic"].training_steps) - int(parent_semantic.training_steps),
                    "k2": int(arm["transition"].training_steps) - int(parent_transition.training_steps),
                },
                "scores": scores,
            }
        parent_scores = arm_payload["parent-frozen"]["scores"]
        interleaved_scores = arm_payload["interleaved-rehearsal-novel"]["scores"]
        p2_4_baseline = p2_4_report["arms"]["parent-frozen"]["scores"]["p1_validation"]
        novel_parent = parent_scores["p2_6_novel_validation"]
        novel_interleaved = interleaved_scores["p2_6_novel_validation"]
        p1_interleaved = interleaved_scores["p1_validation"]
        retention_gate = {
            "p2_4_parent_k1_goal_non_decreasing": _non_decreasing(
                p1_interleaved, p2_4_baseline, "k1_goal_hit_count"
            ),
            "p2_4_parent_k2_goal_non_decreasing": _non_decreasing(
                p1_interleaved, p2_4_baseline, "k2_goal_hit_count"
            ),
            "p2_4_parent_safe_abstention_non_decreasing": _non_decreasing(
                p1_interleaved, p2_4_baseline, "safe_abstention_count"
            ),
            "p2_4_parent_workbench_non_decreasing": _non_decreasing(
                p1_interleaved, p2_4_baseline, "workbench_success_count"
            ),
            "novel_k1_goal_non_decreasing": _non_decreasing(
                novel_interleaved, novel_parent, "k1_goal_hit_count"
            ),
            "novel_k1_content_non_decreasing": _non_decreasing(
                novel_interleaved, novel_parent, "k1_content_hit_count"
            ),
            "novel_k2_goal_non_decreasing": _non_decreasing(
                novel_interleaved, novel_parent, "k2_goal_hit_count"
            ),
            "novel_k2_content_target_reached": int(
                novel_interleaved["k2_content_hit_count"]
            )
            == len(validation_records),
            "novel_workbench_success_non_decreasing": _non_decreasing(
                novel_interleaved, novel_parent, "workbench_success_count"
            ),
            "parameter_count_stable": all(
                arm["scores"]["parameter_count"] == parent_scores["parameter_count"]
                for arm in arm_payload.values()
            ),
        }
        payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "manifest_reconstruction": p1_verification,
                "p2_rehearsal": {
                    "count": len(wake),
                    "class_counts": {
                        key: sum(item["class_key"] == key for item in wake_metadata)
                        for key in ("A", "B", "C", "D", "R")
                    },
                    "experience_digest_match": True,
                    "stream_digest": content_digest(
                        [
                            {
                                "kind": kind,
                                "semantic_input_digest": str(semantic.input_digest),
                                "transition_input_digest": str(transition.input_digest),
                            }
                            for kind, semantic, transition in interleaved_stream
                        ]
                    ),
                },
                "contract": audit,
                "training_contract": {
                    "rehearsal_count": len(wake),
                    "novel_train_count": len(train_records),
                    "novel_validation_count": len(validation_records),
                    "validation_used_for_fit": False,
                    "replay_added": False,
                    "parameter_growth": False,
                    "confidence_floor": CONFIDENCE_FLOOR,
                    "fit_record_ids": [str(record["record_id"]) for record in train_records],
                    "validation_record_ids": [
                        str(record["record_id"]) for record in validation_records
                    ],
                },
                "checkpoint_preflight": {
                    "parent_before_fit": parent_preflight,
                    "all_saved_arms_after_fit": True,
                },
                "arms": arm_payload,
                "retention_gate": retention_gate,
                "can_promote": False,
                "interpretation": (
                    "retention-preserving novel-learning canary only; added target is K2 content; "
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
    parser.add_argument("--p2-5-report", type=Path, default=P2_5_REPORT)
    parser.add_argument("--p2-5-manifest", type=Path, default=P2_5_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_pilot(
        p1_manifest_path=args.p1_manifest,
        p2_report_path=args.p2_report,
        p2_4_report_path=args.p2_4_report,
        p2_5_report_path=args.p2_5_report,
        p2_5_manifest_path=args.p2_5_manifest,
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
                "retention_gate": result.get("retention_gate"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
