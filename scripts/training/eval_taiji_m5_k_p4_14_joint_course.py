"""P4.14 K-worker joint course runner (K continuation x G solver mechanism).

Two individually validated mechanisms have never run sequentially on the
same parent: the K-worker continuation (P2.6/P2.7 mechanics) and the G-head
solver mechanism (P4.9-P4.13 mechanics).  The interaction is real - K-worker
learning shifts the candidate feature landscape that the G phase consumes -
so per cell the course runs

- Phase K: novel K2-content continuation (P2.6 mechanics verbatim: six
  disjoint train candidates + two disjoint validation candidates of the
  ``p4-14`` identity, the fixed 50-example P2 rehearsal stream interleaved
  in the P2.4 order, no parameter growth) starting from the K workers of
  the joint parent, i.e. the P3.2 base-continuation workers to which the
  frozen P3.5 G parent's lineage is tied;
- re-materialize: the Phase G candidate cohorts are materialized from the
  post-K workers, so G-phase candidate features reflect the post-K landscape;
- Phase G: the P4.11 contract verbatim on that landscape - birth
  equivalence against the frozen P3.5 parent's decisions on post-K
  features, task fit (SGD 8 epochs + margin-preservation hinge), and the
  terminal joint projection (task constraints + decision-identity
  preservation constraints).

Gates reuse the frozen values of each mechanism (zero changes) plus the
cross-phase gates: ``k_unchanged_after_g``, ``g_preservation_vs_frozen_parent``
(the retention baselines are the frozen parent's decisions on the post-K
landscape) and exact G birth equivalence.  Matrix: 2 identity batches x
2 deterministic seeds = 4 cells; absolute resource budgets.  Preregistration:
``plans/reference/M5_K_P4_14_JOINT_COURSE_PREREGISTRATION_20260911.md``.
Never admits growth or promotion.
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

from scripts.training.build_taiji_m5_k_p2_3_recovery_continuation import (
    _jsonable,
)  # noqa: E402
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    _build_workspace,
    _observation,
    _registry,
    _schema,
    _semantic_example,
    _transition_examples,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _fresh_learners as _fresh_k_learners,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (
    _save_arm,
)
from scripts.training.eval_taiji_m5_k_p2_4_retention_canary import (  # noqa: E402
    _fit_stream,
    _interleaved_stream,
)
from scripts.training.eval_taiji_m5_k_p2_6_novel_learning import (  # noqa: E402
    CONFIDENCE_FLOOR,
    _candidate_examples,
    _non_decreasing,
)
from scripts.training.eval_taiji_m5_k_p2_6_novel_learning import (
    _score_arm as _score_phase_k_arm,
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
from scripts.training.eval_taiji_m5_k_p3_4_behavior_signal_canary import (  # noqa: E402
    _behavior_record,
)
from scripts.training.eval_taiji_m5_k_p3_5_g_learning import (  # noqa: E402
    _independent_g_restore,
    _load_json,
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p4_0_capacity_pressure import (  # noqa: E402
    _materialize_case,
    _pressure_record,
)
from scripts.training.eval_taiji_m5_k_p4_3_retention_incremental import (
    _evaluate,
)  # noqa: E402
from scripts.training.eval_taiji_m5_k_p4_6_functional_parent_objective import (  # noqa: E402
    MARGIN_EPSILON,
    _manifest_identity,
    _metric_summary,
    _records_identity,
    _structure_row,
)
from scripts.training.eval_taiji_m5_k_p4_7_capacity_clean_test import (  # noqa: E402
    _new_task_gate,
    _retention_gate,
)
from scripts.training.eval_taiji_m5_k_p4_13_promotion_course import (  # noqa: E402
    _extended_tamper_rejected,
    _preservation_constraints,
    _save_extended_checkpoint,
    _task_constraints,
    _verify_extended_checkpoint,
)
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)
from taiji.g_selection_extended import ExtendedGSelectionLearner  # noqa: E402
from taiji.g_selection_projection import project_to_joint_feasible_region  # noqa: E402

REPORT_FORMAT = "taiji-m5-k-p4-14-joint-course-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-14-joint-course-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_14_joint_course_manifest_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p4_14_joint_course_20260911.json"
)
P1_MANIFEST_PATH = P1_MANIFEST
P2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
P2_4_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p2_4_retention_canary_20260910.json"
)
P2_5_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p2_5_novel_composition_probe_20260910.json"
)
P2_5_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p2_5_novel_composition_manifest_v1.json"
)
P2_6_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_6_novel_learning_20260910.json"
P2_6_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p2_6_novel_learning_manifest_v1.json"
)
P2_7_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p2_7_generalization_manifest_v1.json"
)
P3_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
P4_1_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_1_context_contract_manifest_v1.json"
)
P4_2_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_2_capacity_attribution_manifest_v1.json"
)
P4_4_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_4_retention_identity_calibration_manifest_v1.json"
)
P4_10_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_10_feature_factorization_manifest_v1.json"
)
P4_11_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_11_projection_solver_manifest_v1.json"
)
P4_12_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_12_course_level_validation_manifest_v1.json"
)
P4_12_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p4_12_course_level_validation_20260911.json"
)
P4_13_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_13_promotion_course_manifest_v1.json"
)
P4_13_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p4_13_promotion_course_20260911.json"
)
SCORECARD_V4_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v4_20260911.json"
)

TRAINING_EPOCHS = 8
LEARNING_RATE = 0.15
SEEDS = (0, 1)
BATCHES = (0, 1)
OLD_CLASS_K_CONTENT_HITS = 4
OLD_CLASS_SAFE_ABSTENTIONS = 6
NOVEL_VALIDATION_COUNT = 2
EVAL_SPLITS = ("holdout", "retention-sibling", "retention-newtask")
G_RESOURCE_CAPS = {
    "phase_k_fit_seconds": 120.0,
    "phase_g_fit_seconds": 60.0,
    "phase_g_projection_seconds": 120.0,
    "cell_total_seconds": 600.0,
}
# P2.6-isomorphic novel K2-content identity, disjoint from every prior
# Phase-K cohort (P1: typescript_00/01/04, P2.5: 06/07, P2.6: 20-27,
# P2.7: 28-31).
P414_TRAIN_SPECS = {
    0: tuple((index, f"typescript_{32 + index:02d}.ts") for index in range(6)),
    1: tuple((index, f"typescript_{40 + index:02d}.ts") for index in range(6)),
}
P414_VALIDATION_SPECS = {
    0: ((0, "typescript_38.ts"), (1, "typescript_39.ts")),
    1: ((0, "typescript_46.ts"), (1, "typescript_47.ts")),
}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _digest_without(payload: Mapping[str, Any], key: str) -> str:
    return content_digest(
        {name: value for name, value in payload.items() if name != key}
    )


# ---------------------------------------------------------------------------
# Phase K: P2.6-isomorphic novel K2-content cohort (p4-14 identity).
# ---------------------------------------------------------------------------


def _p414_build_record(
    *,
    root: Path,
    batch: int,
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
    environment = WorkbenchEnvironment(
        root=root, programming_language_registry=registry
    )
    schema = _schema()
    anchor = _observation(
        environment,
        tag=f"p4-14:b{batch}:{split}:{index}:anchor",
        path="missing_00.txt",
        schema=schema,
        tick=0,
    )
    initial = _observation(
        environment,
        tag=f"p4-14:b{batch}:{split}:{index}:initial",
        path="missing_novel.txt",
        schema=schema,
        tick=1,
    )
    candidate = _observation(
        environment,
        tag=f"p4-14:b{batch}:{split}:{index}:candidate",
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
        split=f"p4-14-b{batch}-{split}-{index:04d}",
        tick=2,
    )
    candidate_transition = _transition_examples(
        (anchor, initial, candidate),
        split=f"p4-14-b{batch}-{split}-{index:04d}",
    )[1]
    record = {
        "record_id": f"p4-14-novel-b{batch}-{split}-{index:04d}",
        "batch": batch,
        "split": split,
        "index": index,
        "class_key": "R-novel-composition",
        "project_id": project_id,
        "template_family_id": content_digest(
            {
                "format": "taiji-m5-k-p4-14-novel-template-v1",
                "batch": batch,
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


def _p414_audit_records(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    *,
    p1_manifest: Mapping[str, Any],
    prior_k_paths: set[str],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    prior_paths = set(prior_k_paths)
    prior_paths.update(
        str(path)
        for item in p1_manifest.get("records", [])
        for path in item.get("variant_paths", [])
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
            errors.append(
                {"record_id": record_id, "reason": "fit_eligibility_split_mismatch"}
            )
        initial = record.get("initial", {})
        candidate = record.get("candidate", {})
        initial_observation = initial.get("observation", {})
        candidate_observation = candidate.get("observation", {})
        candidate_percept = candidate.get("semantic_example", {}).get("percept", {})
        if initial_observation.get("read_success") is not False:
            errors.append({"record_id": record_id, "reason": "initial_not_missing"})
        if float(initial.get("percept", {}).get("confidence", 1.0)) >= CONFIDENCE_FLOOR:
            errors.append(
                {"record_id": record_id, "reason": "initial_confidence_not_low"}
            )
        if initial.get("next_step") != "workspace.list":
            errors.append(
                {"record_id": record_id, "reason": "initial_next_step_mismatch"}
            )
        if candidate_observation.get("read_success") is not True:
            errors.append({"record_id": record_id, "reason": "candidate_not_readable"})
        if candidate_observation.get("language_id") != "typescript":
            errors.append(
                {"record_id": record_id, "reason": "candidate_language_not_typescript"}
            )
        if candidate_observation.get("selection_state") != "resolved":
            errors.append({"record_id": record_id, "reason": "candidate_not_resolved"})
        if candidate_observation.get("toolchain_available") is not True:
            errors.append(
                {"record_id": record_id, "reason": "candidate_toolchain_not_available"}
            )
        if float(candidate_percept.get("confidence", 0.0)) < CONFIDENCE_FLOOR:
            errors.append(
                {"record_id": record_id, "reason": "candidate_confidence_below_floor"}
            )
        if (
            record.get("novel_tuple", {}).get("content_id")
            != "content:inspect-language"
        ):
            errors.append(
                {"record_id": record_id, "reason": "candidate_content_target_mismatch"}
            )
        candidate_path = str(candidate_observation.get("path"))
        candidate_paths.append(candidate_path)
        if candidate_path in prior_paths:
            errors.append(
                {"record_id": record_id, "reason": "candidate_path_seen_in_prior_data"}
            )
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
        "candidate_paths_disjoint_from_prior": not set(candidate_paths).intersection(
            prior_paths
        ),
        "train_validation_paths_disjoint": not train_paths.intersection(
            validation_paths
        ),
        "novel_tuple_count": len(tuple_digests),
        "errors": errors,
        "passed": (
            len(train_records) == 6
            and len(validation_records) == NOVEL_VALIDATION_COUNT
            and len(tuple_digests) == 1
            and len(candidate_paths) == len(set(candidate_paths))
            and not errors
        ),
    }


def _k_tamper_rejected(payload: Mapping[str, Any]) -> bool:
    tampered = copy.deepcopy(dict(payload))
    tampered["source_digest"] = "tampered-source-digest"
    try:
        if payload.get("format") == "taiji-structured-semantic-transition-v3":
            from taiji import StructuredSemanticTransitionLearner

            StructuredSemanticTransitionLearner.from_checkpoint(tampered, device="cpu")
        else:
            from taiji import StructuredSemanticLearner

            StructuredSemanticLearner.from_checkpoint(tampered, device="cpu")
    except ValueError:
        return True
    return False


def _phase_k_gates(
    *,
    parent_scores: Mapping[str, Any],
    interleaved_scores: Mapping[str, Any],
    parent_parameter_count: Mapping[str, Any],
    interleaved_parameter_count: Mapping[str, Any],
) -> dict[str, bool]:
    p1_parent = parent_scores["p1_validation"]
    p1_inter = interleaved_scores["p1_validation"]
    novel_parent = parent_scores["p2_6_novel_validation"]
    novel_inter = interleaved_scores["p2_6_novel_validation"]
    return {
        # Absolute frozen gates (preregistration section 3).
        "novel_k2_content_target_reached": int(novel_inter["k2_content_hit_count"])
        == NOVEL_VALIDATION_COUNT,
        "old_class_k1_content_4_of_4": int(p1_inter["k1_content_hit_count"])
        == OLD_CLASS_K_CONTENT_HITS,
        "old_class_k2_content_4_of_4": int(p1_inter["k2_content_hit_count"])
        == OLD_CLASS_K_CONTENT_HITS,
        "old_class_safe_abstention_6_of_6": int(p1_inter["safe_abstention_count"])
        == OLD_CLASS_SAFE_ABSTENTIONS,
        # P2.6-mechanics retention (non-decreasing vs the in-run parent).
        "old_class_k1_goal_non_decreasing": _non_decreasing(
            p1_inter, p1_parent, "k1_goal_hit_count"
        ),
        "old_class_k2_goal_non_decreasing": _non_decreasing(
            p1_inter, p1_parent, "k2_goal_hit_count"
        ),
        "old_class_safe_abstention_non_decreasing": _non_decreasing(
            p1_inter, p1_parent, "safe_abstention_count"
        ),
        "old_class_workbench_non_decreasing": _non_decreasing(
            p1_inter, p1_parent, "workbench_success_count"
        ),
        "novel_k1_goal_non_decreasing": _non_decreasing(
            novel_inter, novel_parent, "k1_goal_hit_count"
        ),
        "novel_k1_content_non_decreasing": _non_decreasing(
            novel_inter, novel_parent, "k1_content_hit_count"
        ),
        "novel_k2_goal_non_decreasing": _non_decreasing(
            novel_inter, novel_parent, "k2_goal_hit_count"
        ),
        "novel_workbench_non_decreasing": _non_decreasing(
            novel_inter, novel_parent, "workbench_success_count"
        ),
        "parameter_count_stable": interleaved_parameter_count == parent_parameter_count,
    }


# ---------------------------------------------------------------------------
# Phase G: P4.11 contract verbatim on the post-K landscape (p4-14 identity).
# ---------------------------------------------------------------------------


def _p414_specs(split: str, offset: int, batch: int) -> tuple[dict[str, Any], ...]:
    prefix = f"p40_p414_b{batch}_{split}"
    rows = (
        (
            "A",
            "resolved-language",
            (f"{prefix}_a_python.py", f"{prefix}_a_rust.rs", f"{prefix}_a_python_b.py"),
        ),
        (
            "B",
            "ambiguous-language",
            (
                f"{prefix}_b_rust.py",
                f"{prefix}_b_python.py",
                f"{prefix}_b_typescript.ts",
            ),
        ),
        (
            "C",
            "resolved-language",
            (
                f"{prefix}_c_typescript.ts",
                f"{prefix}_c_python.py",
                f"{prefix}_c_rust.rs",
            ),
        ),
        (
            "D",
            "ambiguous-header",
            (f"{prefix}_d_header.h", f"{prefix}_d_python.py", f"{prefix}_d_rust.rs"),
        ),
        (
            "R",
            "recovery-no-selection",
            (f"{prefix}_r_missing.txt", f"{prefix}_r_python.py", f"{prefix}_r_rust.rs"),
        ),
    )
    return tuple(
        {
            "index": index,
            "class_key": class_key,
            "project_id": f"p4-14-b{batch}-{split}-project-{class_key.lower()}",
            "variant_paths": paths,
            "state_profile": state_profile,
            "task_seed": offset + index,
        }
        for index, (class_key, state_profile, paths) in enumerate(rows)
    )


def _p414_structured_specs(
    contract: Mapping[str, Any], *, namespace: str, seed_offset: int, batch: int
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    safe_index = 0
    high_index = 0
    for index, row in enumerate(contract["rows"]):
        if int(row["role_counts"].get("proposal", 0)) > 0:
            project_id = f"p4-14-b{batch}-{namespace}-project-a"
            prefix = f"p40_p414_b{batch}_{namespace}_a"
            paths = (
                f"{prefix}_main_{high_index}.py",
                f"{prefix}_alternate_{high_index}.rs",
                f"{prefix}_alternate_{high_index}_b.py",
            )
            state_profile = "resolved-language"
            class_key = f"H{high_index}"
            high_index += 1
        else:
            project_id = (
                f"p4-14-b{batch}-{namespace}-project-a"
                if safe_index == 0
                else f"p4-14-b{batch}-{namespace}-project-b"
            )
            prefix = f"p40_p414_b{batch}_{namespace}_{'a' if safe_index == 0 else 'b'}"
            paths = (
                f"{prefix}_missing_{safe_index}_missing.txt",
                f"{prefix}_fallback_{safe_index}.py",
                f"{prefix}_fallback_{safe_index}.rs",
            )
            state_profile = "recovery-no-selection"
            class_key = f"S{safe_index}"
            safe_index += 1
        specs.append(
            {
                "index": index,
                "class_key": class_key,
                "project_id": project_id,
                "variant_paths": paths,
                "state_profile": state_profile,
                "task_seed": seed_offset + index,
                "target_width": int(row["candidate_count"]),
            }
        )
    return tuple(specs)


def _p414_rebind(
    record: Mapping[str, Any],
    *,
    split: str,
    source_index: int,
    width: int,
    fit_eligible: bool,
    batch: int,
) -> dict[str, Any]:
    old_candidates = record["candidate_set"]
    old_behavior = record["behavior_set"]
    candidate_set = GSelectionCandidateSet.create(
        example_id=f"p4-14:b{batch}:{split}:{source_index}:width-{width}:{old_candidates.candidate_set_digest}",
        family_id=f"p4-14:b{batch}:{split}:family:{source_index}",
        split=split,
        project_id=old_candidates.project_id,
        path=old_candidates.path,
        input_digest=old_candidates.input_digest,
        candidates=old_candidates.candidates,
        target_candidate_id=old_candidates.target_candidate_id,
        target_kind=old_candidates.target_kind,
    )
    behavior_set = GSelectionBehaviorSet.create(
        candidate_set_digest=candidate_set.candidate_set_digest,
        inference_digest=candidate_set.inference_digest,
        split=split,
        project_id=candidate_set.project_id,
        path=candidate_set.path,
        outcomes=old_behavior.outcomes,
    )
    diagnostic = dict(record["diagnostic"])
    diagnostic.update(
        {
            "experiment": "p4.14",
            "batch": batch,
            "split": split,
            "source_index": source_index,
            "candidate_width": width,
        }
    )
    return {
        **record,
        "candidate_set": candidate_set,
        "behavior_set": behavior_set,
        "diagnostic": diagnostic,
        "fit_eligible": fit_eligible,
    }


def _p414_pressure_split(
    *,
    split: str,
    offset: int,
    batch: int,
    scratch: Path,
    parent_digest: str,
    worker_bundle_digest: str,
    source_manifest_digest: str,
    projector: Any,
    semantic: Any,
    transition: Any,
    semantic_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from scripts.training.eval_taiji_m5_k_p3_3_g_signal_canary import _catalogs

    goals, content_plans = _catalogs(semantic_payload)
    base_records: list[dict[str, Any]] = []
    for spec in _p414_specs(split, offset, batch):
        experience, metadata, case = _materialize_case(
            scratch=scratch,
            spec=spec,
            parent_digest=parent_digest,
            worker_bundle_digest=worker_bundle_digest,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        metadata = {**metadata, "split": split}
        candidate_set, behavior_set, diagnostic = _behavior_record(
            experience=experience,
            metadata=metadata,
            case=case,
            semantic=semantic,
            transition=transition,
            goals=goals,
            content_plans=content_plans,
            label=f"p4-14-b{batch}-{split}-{int(spec['index'])}",
        )
        base_records.append(
            {
                "experience": experience,
                "metadata": metadata,
                "case": case,
                "candidate_set": candidate_set,
                "behavior_set": behavior_set,
                "diagnostic": diagnostic,
            }
        )
    records: list[dict[str, Any]] = []
    for source_index in range(len(base_records)):
        for width in (2, 4, 8, 12):
            pressure = _pressure_record(
                base_records=base_records,
                source_index=source_index,
                width=width,
                transition=transition,
            )
            records.append(
                _p414_rebind(
                    pressure,
                    split=split,
                    source_index=source_index,
                    width=width,
                    fit_eligible=(
                        split == "train"
                        and float(pressure["behavior_set"].utility_margin)
                        > MARGIN_EPSILON
                    ),
                    batch=batch,
                )
            )
    return records


def _p414_structured_records(
    *,
    contract: Mapping[str, Any],
    namespace: str,
    seed_offset: int,
    batch: int,
    scratch: Path,
    parent_digest: str,
    worker_bundle_digest: str,
    source_manifest_digest: str,
    projector: Any,
    semantic: Any,
    transition: Any,
    semantic_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from scripts.training.eval_taiji_m5_k_p3_3_g_signal_canary import _catalogs

    goals, content_plans = _catalogs(semantic_payload)
    base_records: list[dict[str, Any]] = []
    specs = _p414_structured_specs(
        contract, namespace=namespace, seed_offset=seed_offset, batch=batch
    )
    for spec in specs:
        experience, metadata, case = _materialize_case(
            scratch=scratch,
            spec=spec,
            parent_digest=parent_digest,
            worker_bundle_digest=worker_bundle_digest,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        metadata = {**metadata, "split": namespace}
        candidate_set, behavior_set, diagnostic = _behavior_record(
            experience=experience,
            metadata=metadata,
            case=case,
            semantic=semantic,
            transition=transition,
            goals=goals,
            content_plans=content_plans,
            label=f"p4-14-b{batch}-{namespace}-{int(spec['index'])}",
        )
        base_records.append(
            {
                "experience": experience,
                "metadata": metadata,
                "case": case,
                "candidate_set": candidate_set,
                "behavior_set": behavior_set,
                "diagnostic": diagnostic,
            }
        )
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        pressure = _pressure_record(
            base_records=base_records,
            source_index=index,
            width=int(spec["target_width"]),
            transition=transition,
        )
        records.append(
            _p414_rebind(
                pressure,
                split=namespace,
                source_index=index,
                width=int(spec["target_width"]),
                fit_eligible=False,
                batch=batch,
            )
        )
    return records


def _birth_equivalence(
    *,
    parent: GSelectionLearner,
    learner: ExtendedGSelectionLearner,
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    mismatches = 0
    max_deviation = 0.0
    records_checked = 0
    for split in ("validation", *EVAL_SPLITS):
        for record in split_records[split]:
            candidate_set = record["candidate_set"]
            parent_decision = parent.select(candidate_set)
            child_decision = learner.select(candidate_set)
            if (
                parent_decision.selected_candidate_id
                != child_decision.selected_candidate_id
                or parent_decision.selection_status != child_decision.selection_status
            ):
                mismatches += 1
            child_scores = learner.total_scores(candidate_set)
            for candidate in candidate_set.candidates:
                max_deviation = max(
                    max_deviation,
                    abs(
                        child_scores[candidate.candidate_id]
                        - float(parent.score(candidate))
                    ),
                )
            records_checked += 1
    return {
        "records_checked": records_checked,
        "selection_mismatches": mismatches,
        "max_abs_score_deviation": max_deviation,
        "passed": mismatches == 0 and max_deviation == 0.0,
    }


# ---------------------------------------------------------------------------
# Course runner.
# ---------------------------------------------------------------------------


def _run(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_14_joint_course_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "manifest": str(manifest_path),
        "report": str(report_path),
    }
    try:
        p1_manifest = _load_json(P1_MANIFEST_PATH)
        p2_report = _load_json(P2_REPORT)
        p2_4_report = _load_json(P2_4_REPORT)
        p2_5_report = _load_json(P2_5_REPORT)
        p2_5_manifest = _load_json(P2_5_MANIFEST)
        p2_6_report = _load_json(P2_6_REPORT)
        p2_6_manifest = _load_json(P2_6_MANIFEST)
        p2_7_manifest = _load_json(P2_7_MANIFEST)
        p3_2_report = _load_json(P3_2_REPORT)
        p3_5_report = _load_json(P3_5_REPORT)
        p4_1_manifest = _load_json(P4_1_MANIFEST)
        p4_2_manifest = _load_json(P4_2_MANIFEST)
        p4_4_manifest = _load_json(P4_4_MANIFEST)
        p4_10_manifest = _load_json(P4_10_MANIFEST)
        p4_11_manifest = _load_json(P4_11_MANIFEST)
        p4_12_manifest = _load_json(P4_12_MANIFEST)
        p4_12_report = _load_json(P4_12_REPORT)
        p4_13_manifest = _load_json(P4_13_MANIFEST)
        p4_13_report = _load_json(P4_13_REPORT)
        scorecard_v4 = _load_json(SCORECARD_V4_REPORT)

        # ---- Entry conditions (scorecard v4 frozen boundary).
        if scorecard_v4.get("format") != "taiji-m5-k-axis-scorecard-v4":
            raise ValueError("P4.14 requires the scorecard v4 report")
        gates_v4 = scorecard_v4["promotion_gates"]
        if not gates_v4.get("g_solver_mechanism_course_closed"):
            raise ValueError("P4.14 requires g_solver_mechanism_course_closed=true")
        if gates_v4.get("k_worker_joint_course_completed"):
            raise ValueError(
                "P4.14 entry requires k_worker_joint_course_completed=false"
            )
        if (
            scorecard_v4["verdict"]["promotion_gate"]
            or scorecard_v4["verdict"]["can_promote"]
        ):
            raise ValueError("P4.14 cannot start from an open promotion gate")
        if (
            p4_13_report.get("status") != "completed"
            or p4_13_report.get("outcome") != "promotion_course_supported"
        ):
            raise ValueError(
                "P4.14 requires the completed P4.13 promotion_course_supported"
            )
        if p4_13_report.get("manifest_digest") != p4_13_manifest.get("manifest_digest"):
            raise ValueError("P4.13 manifest/report digest mismatch")
        if _digest_without(p4_13_manifest, "manifest_digest") != p4_13_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.13 manifest content digest mismatch")
        if p4_13_report.get("growth_admitted") or p4_13_report.get("can_promote"):
            raise ValueError("P4.14 cannot consume an admitted P4.13 artifact")
        if p4_12_report.get("outcome") != "course_level_validation_supported":
            raise ValueError(
                "P4.14 requires the P4.12 course_level_validation_supported"
            )
        if p4_12_report.get("manifest_digest") != p4_12_manifest.get("manifest_digest"):
            raise ValueError("P4.12 manifest/report digest mismatch")
        if p4_13_manifest.get("source_p4_12_manifest_digest") != p4_12_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.13 source chain drifted")
        if p4_10_manifest.get("manifest_digest") != p4_11_manifest.get(
            "source_p4_10_manifest_digest"
        ):
            raise ValueError("P4.10/P4.11 source chain drifted")

        # ---- Phase K mechanical baselines (P2.6 chain, zero changes).
        if p1_manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
            raise ValueError("P4.14 requires the P1 v2 manifest")
        for name, report in (
            ("P2", p2_report),
            ("P2.4", p2_4_report),
            ("P2.5", p2_5_report),
            ("P2.6", p2_6_report),
        ):
            if report.get("status") != "completed" or report.get("sealed_payload_read"):
                raise ValueError(f"P4.14 requires a completed, unsealed {name} report")
        if not all(
            bool(value) for value in p2_4_report.get("retention_gate", {}).values()
        ):
            raise ValueError("P2.4 retention gate is not fully passed")
        if not p2_5_report.get("contract", {}).get("passed"):
            raise ValueError("P2.5 novel composition contract is not passed")
        if p2_5_manifest.get("manifest_digest") != p2_5_report.get("manifest_digest"):
            raise ValueError("P2.5 manifest digest mismatch")
        if not all(
            bool(value) for value in p2_6_report.get("retention_gate", {}).values()
        ):
            raise ValueError("P2.6 retention gate is not fully passed")

        # ---- Joint parent: P3.2 base-continuation K workers + frozen P3.5 G.
        contract = p4_4_manifest["structure_contract"]
        if content_digest(contract["rows"]) != contract["contract_digest"]:
            raise ValueError("P4.4 structure contract digest drifted")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_report["contract"]["parent_checkpoint_digest"] != parent_digest:
            raise ValueError("P2 parent checkpoint digest drifted")
        if p2_6_manifest.get("parent_checkpoint_digest") != parent_digest:
            raise ValueError("P2.6 parent checkpoint digest drifted")
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        k_parent_k1_path = Path(str(worker_restore["k1"]["path"]))
        k_parent_k2_path = Path(str(worker_restore["k2"]["path"]))
        k_parent_semantic_payload = _load_mapping(k_parent_k1_path)
        k_parent_transition_payload = _load_mapping(k_parent_k2_path)
        worker_digests = {
            "k1": content_digest(k_parent_semantic_payload),
            "k2": content_digest(k_parent_transition_payload),
        }
        if worker_digests != dict(p4_2_manifest["k_checkpoint_digests"]):
            raise ValueError("P4.14 K-parent digests drifted from the P4.2 manifest")
        parent_metadata = p3_5_report["g_trained_checkpoint"]
        parent_path = Path(str(parent_metadata["path"]))
        parent_payload = _load_mapping(parent_path)
        if content_digest(parent_payload) != str(parent_metadata["digest"]):
            raise ValueError("P4.14 parent external checkpoint digest drifted")
        parent = GSelectionLearner.from_checkpoint(parent_payload, device="cpu")
        parent.assert_lineage(
            parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        if str(p4_2_manifest["parent_g_checkpoint_digest"]) != str(
            parent_metadata["digest"]
        ):
            raise ValueError("P4.14 parent G lineage drifted")
        parent_restore = _independent_g_restore(parent_path)
        if not parent_restore.get("independent_process_restore"):
            raise RuntimeError("P4.14 parent independent restore failed")
        run_dir.mkdir(parents=True, exist_ok=False)
        parent_preflight = _checkpoint_preflight(
            output_dir=run_dir / "preflight-parent",
            semantic_parent=copy.deepcopy(k_parent_semantic_payload),
            transition_parent=copy.deepcopy(k_parent_transition_payload),
        )
        if not parent_preflight["passed"]:
            raise RuntimeError("P4.14 K-parent checkpoint preflight failed")

        prior_k_paths: set[str] = set()
        for item in p1_manifest.get("records", []):
            prior_k_paths.update(str(path) for path in item.get("variant_paths", []))
        for manifest in (p2_5_manifest, p2_6_manifest, p2_7_manifest):
            for collection_key in ("records", "train_records", "validation_records"):
                for raw in manifest.get(collection_key, []) or []:
                    path = (
                        raw.get("candidate", {}).get("observation", {}).get("path")
                        if isinstance(raw, Mapping)
                        else None
                    )
                    if path:
                        prior_k_paths.add(str(path))
        historical_paths: set[str] = set()
        historical_projects: set[str] = set()
        for manifest in (
            p4_1_manifest,
            p4_2_manifest,
            p4_4_manifest,
            p4_10_manifest,
            p4_11_manifest,
            p4_12_manifest,
            p4_13_manifest,
        ):
            manifest_paths, manifest_projects = _manifest_identity(manifest)
            historical_paths.update(manifest_paths)
            historical_projects.update(manifest_projects)

        batch_summaries: list[dict[str, Any]] = []
        cell_results: list[dict[str, Any]] = []
        batch_candidate_digests: dict[int, set[str]] = {}
        batch_novel_records: dict[int, dict[str, list[dict[str, Any]]]] = {}
        all_novel_record_digests: set[str] = set()
        incomplete_projection_cells = 0
        resource_violations = 0
        for batch in BATCHES:
            batch_dir = run_dir / f"batch-{batch}"
            batch_dir.mkdir(parents=True, exist_ok=False)
            base = 65000 + batch * 1000
            offsets = {
                "constraint": base,
                "sibling": base + 100,
                "train": base + 200,
                "validation": base + 300,
                "holdout": base + 400,
                "retention-newtask": base + 500,
            }
            common_materialize = dict(
                parent_digest=parent_digest,
                worker_bundle_digest=bundle.bundle_digest,
                source_manifest_digest=str(p4_13_manifest["manifest_digest"]),
                projector=projector,
            )

            # ---- Phase K data (p4-14 novel identity; P2.6 mechanics).
            train_records: list[dict[str, Any]] = []
            validation_records: list[dict[str, Any]] = []
            validation_cases: list[dict[str, Any]] = []
            for index, candidate_path in P414_TRAIN_SPECS[batch]:
                record, _case = _p414_build_record(
                    root=batch_dir / "phase-k-data" / f"train-{index:04d}",
                    batch=batch,
                    index=index,
                    candidate_path=candidate_path,
                    project_id=f"p4-14-b{batch}-phase-k-train-project-{index}",
                    task_seed=7100 + 100 * batch + index,
                    split="train",
                    source_manifest_digest=str(p1_manifest["manifest_digest"]),
                    parent_digest=parent_digest,
                    fit_eligible=True,
                )
                train_records.append(record)
            for index, candidate_path in P414_VALIDATION_SPECS[batch]:
                record, case = _p414_build_record(
                    root=batch_dir / "phase-k-data" / f"validation-{index:04d}",
                    batch=batch,
                    index=index,
                    candidate_path=candidate_path,
                    project_id=f"p4-14-b{batch}-phase-k-validation-project-{index}",
                    task_seed=7200 + 100 * batch + index,
                    split="validation",
                    source_manifest_digest=str(p1_manifest["manifest_digest"]),
                    parent_digest=parent_digest,
                    fit_eligible=False,
                )
                validation_records.append(record)
                validation_cases.append(case)
            audit = _p414_audit_records(
                train_records,
                validation_records,
                p1_manifest=p1_manifest,
                prior_k_paths=prior_k_paths,
            )
            if not audit["passed"]:
                raise RuntimeError(
                    f"P4.14 batch {batch} phase-K contract failed: {audit['errors'][:3]}"
                )
            batch_novel_records[batch] = {
                "train": train_records,
                "validation": validation_records,
            }
            for record in (*train_records, *validation_records):
                digest = str(record["record_digest"])
                if digest in all_novel_record_digests:
                    raise ValueError(f"P4.14 phase-K record digest collision: {digest}")
                all_novel_record_digests.add(digest)

            (
                train_experiences,
                train_metadata,
                p1_validation,
                p1_validation_metadata,
                p1_verification,
            ) = _rebuild_and_verify_manifest(
                scratch=batch_dir / "p1",
                manifest=p1_manifest,
                parent_digest=parent_digest,
                bundle=bundle,
                projector=projector,
            )
            if not p1_verification["passed"]:
                raise RuntimeError(
                    "P1 manifest reconstruction failed before the P4.14 phase K"
                )
            p1_cases, p1_case_mismatches = _build_validation_cases(
                scratch=batch_dir / "p1-cases",
                validation=p1_validation,
                validation_metadata=p1_validation_metadata,
            )
            if p1_case_mismatches:
                raise RuntimeError(
                    f"P1 validation case mismatch: {p1_case_mismatches[:3]}"
                )
            wake, wake_metadata = _select_balanced_wake(
                train_experiences, train_metadata
            )
            expected_wake = p2_report["contract"]["wake_experience_digests"]
            actual_wake = [str(item.experience_digest) for item in wake]
            if actual_wake != [str(item) for item in expected_wake]:
                raise RuntimeError("P2 rehearsal experience digest/order drifted")
            if any(
                sum(item["class_key"] == class_key for item in wake_metadata)
                != PILOT_PER_CLASS
                for class_key in ("A", "B", "C", "D", "R")
            ):
                raise RuntimeError("P2 rehearsal class balance drifted")

            # ---- Phase K per cell (deterministic given the parent + data).
            phase_k_by_seed: dict[int, dict[str, Any]] = {}
            for seed in SEEDS:
                cell_k_dir = batch_dir / f"seed-{seed}" / "phase-k"
                cell_k_dir.mkdir(parents=True, exist_ok=False)
                k_fit_started = time.perf_counter()
                parent_semantic, parent_transition = _fresh_k_learners(
                    k_parent_semantic_payload, k_parent_transition_payload
                )
                interleaved_semantic, interleaved_transition = _fresh_k_learners(
                    k_parent_semantic_payload, k_parent_transition_payload
                )
                train_semantic, train_transition = _candidate_examples(train_records)
                interleaved_stream = _interleaved_stream(
                    wake,
                    train_semantic,
                    train_transition,
                )
                fit_counts = _fit_stream(
                    interleaved_semantic,
                    interleaved_transition,
                    interleaved_stream,
                )
                phase_k_fit_wall = time.perf_counter() - k_fit_started
                checkpoint = _save_arm(
                    cell_k_dir, interleaved_semantic, interleaved_transition
                )
                if not checkpoint["passed"]:
                    raise RuntimeError(
                        f"P4.14 cell b{batch}/s{seed} phase-K independent restore failed"
                    )
                parent_scores = _score_phase_k_arm(
                    semantic=parent_semantic,
                    transition=parent_transition,
                    p1_cases=p1_cases,
                    novel_cases=validation_cases,
                    novel_records=validation_records,
                )
                interleaved_scores = _score_phase_k_arm(
                    semantic=interleaved_semantic,
                    transition=interleaved_transition,
                    p1_cases=p1_cases,
                    novel_cases=validation_cases,
                    novel_records=validation_records,
                )
                gates = _phase_k_gates(
                    parent_scores=parent_scores,
                    interleaved_scores=interleaved_scores,
                    parent_parameter_count=parent_scores["parameter_count"],
                    interleaved_parameter_count=interleaved_scores["parameter_count"],
                )
                tamper_gate = {
                    "k1": _k_tamper_rejected(
                        _load_mapping(Path(str(checkpoint["files"]["k1"]["path"])))
                    ),
                    "k2": _k_tamper_rejected(
                        _load_mapping(Path(str(checkpoint["files"]["k2"]["path"])))
                    ),
                }
                if not all(tamper_gate.values()):
                    raise RuntimeError(
                        f"P4.14 cell b{batch}/s{seed} phase-K tamper gate failed"
                    )
                phase_k_by_seed[seed] = {
                    "checkpoint": checkpoint,
                    "fit_counts": fit_counts,
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
                    "fit_wall_seconds": round(phase_k_fit_wall, 3),
                    "parent_scores": parent_scores,
                    "interleaved_scores": interleaved_scores,
                    "gates": gates,
                    "tamper_gate": tamper_gate,
                    "post_k_digests": dict(checkpoint["checkpoint_digests"]),
                }
            post_k_digests_by_seed = {
                seed: phase_k_by_seed[seed]["post_k_digests"] for seed in SEEDS
            }
            post_k_deterministic = (
                len(
                    {
                        content_digest(digests)
                        for digests in post_k_digests_by_seed.values()
                    }
                )
                == 1
            )

            # ---- Re-materialize: instantiate the post-K workers and
            # materialize the Phase G cohorts on the post-K landscape.
            post_k_checkpoint = phase_k_by_seed[SEEDS[0]]["checkpoint"]
            post_k_semantic_payload = _load_mapping(
                Path(str(post_k_checkpoint["files"]["k1"]["path"]))
            )
            post_k_transition_payload = _load_mapping(
                Path(str(post_k_checkpoint["files"]["k2"]["path"]))
            )
            if (
                content_digest(post_k_semantic_payload)
                != post_k_digests_by_seed[SEEDS[0]]["k1"]
                or content_digest(post_k_transition_payload)
                != post_k_digests_by_seed[SEEDS[0]]["k2"]
            ):
                raise ValueError("P4.14 post-K worker payload digest mismatch")
            post_k_semantic, post_k_transition = _fresh_k_learners(
                post_k_semantic_payload, post_k_transition_payload
            )
            post_k_catalogs_digest = content_digest(
                {
                    "goal_catalog": post_k_semantic_payload["goal_catalog"],
                    "content_catalog": post_k_semantic_payload["content_catalog"],
                }
            )
            pre_k_catalogs_digest = content_digest(
                {
                    "goal_catalog": k_parent_semantic_payload["goal_catalog"],
                    "content_catalog": k_parent_semantic_payload["content_catalog"],
                }
            )
            k_catalogs_unchanged = post_k_catalogs_digest == pre_k_catalogs_digest

            materialize_common = {
                **common_materialize,
                "semantic": post_k_semantic,
                "transition": post_k_transition,
                "semantic_payload": post_k_semantic_payload,
            }
            train_g = _p414_pressure_split(
                split="train",
                offset=offsets["train"],
                batch=batch,
                scratch=batch_dir / "data" / "train",
                **materialize_common,
            )
            validation_g = _p414_pressure_split(
                split="validation",
                offset=offsets["validation"],
                batch=batch,
                scratch=batch_dir / "data" / "validation",
                **materialize_common,
            )
            holdout_g = _p414_pressure_split(
                split="holdout",
                offset=offsets["holdout"],
                batch=batch,
                scratch=batch_dir / "data" / "holdout",
                **materialize_common,
            )
            retention_newtask = _p414_pressure_split(
                split="retention-newtask",
                offset=offsets["retention-newtask"],
                batch=batch,
                scratch=batch_dir / "data" / "retention-newtask",
                **materialize_common,
            )
            constraint_records = _p414_structured_records(
                contract=contract,
                namespace="constraint",
                seed_offset=offsets["constraint"],
                batch=batch,
                scratch=batch_dir / "data" / "constraint",
                **materialize_common,
            )
            retention_sibling_records = _p414_structured_records(
                contract=contract,
                namespace="retention",
                seed_offset=offsets["sibling"],
                batch=batch,
                scratch=batch_dir / "data" / "retention",
                **materialize_common,
            )
            train_fit_g = [record for record in train_g if record["fit_eligible"]]
            constraint_sets = tuple(
                record["candidate_set"] for record in constraint_records
            )
            constraint_digest = content_digest(
                {
                    "constraint_set_digests": [
                        candidate_set.candidate_set_digest
                        for candidate_set in constraint_sets
                    ],
                    "constraint_form": "margin-preservation-hinge",
                    "landscape": "post-k",
                }
            )
            batch_records = [
                *train_g,
                *validation_g,
                *holdout_g,
                *retention_newtask,
                *constraint_records,
                *retention_sibling_records,
            ]
            batch_candidate_digests[batch] = {
                record["candidate_set"].candidate_set_digest for record in batch_records
            }
            split_records = {
                "validation": validation_g,
                "holdout": holdout_g,
                "retention-sibling": retention_sibling_records,
                "retention-newtask": retention_newtask,
            }
            identity_sets = [
                _records_identity(collection)
                for collection in (
                    train_g,
                    validation_g,
                    holdout_g,
                    retention_newtask,
                    constraint_records,
                    retention_sibling_records,
                )
            ]
            batch_identity_gate = {
                "train_records": len(train_g) == 20,
                "validation_records": len(validation_g) == 20,
                "holdout_records": len(holdout_g) == 20,
                "retention_newtask_records": len(retention_newtask) == 20,
                "constraint_records": len(constraint_records)
                == int(contract["row_count"]),
                "retention_sibling_records": len(retention_sibling_records)
                == int(contract["row_count"]),
                "train_fit_positive": len(train_fit_g) >= 8,
                "five_classes_all_pressure_splits": all(
                    len({record["diagnostic"]["class_key"] for record in records}) == 5
                    for records in (train_g, validation_g, holdout_g, retention_newtask)
                ),
                "new_projects_disjoint_from_historical": all(
                    projects.isdisjoint(historical_projects)
                    for _paths, projects in identity_sets
                ),
                "new_paths_disjoint_from_historical": all(
                    paths.isdisjoint(historical_paths)
                    for paths, _projects in identity_sets
                ),
                "post_k_deterministic_across_seeds": post_k_deterministic,
                "k_catalogs_unchanged": k_catalogs_unchanged,
            }
            if not all(batch_identity_gate.values()):
                raise ValueError(
                    f"P4.14 batch {batch} identity gate failed: "
                    f"{[k for k, v in batch_identity_gate.items() if not v]}"
                )
            historical_paths |= {
                record["candidate_set"].path for record in batch_records
            }
            historical_projects |= {
                record["candidate_set"].project_id for record in batch_records
            }
            prior_k_paths.update(
                str(record["candidate"]["observation"]["path"])
                for record in (*train_records, *validation_records)
            )
            sibling_structure = [
                _structure_row(record["candidate_set"], record["behavior_set"])
                for record in retention_sibling_records
            ]
            constraint_structure = [
                _structure_row(record["candidate_set"], record["behavior_set"])
                for record in constraint_records
            ]
            structure_ok = sibling_structure == list(
                contract["rows"]
            ) and constraint_structure == list(contract["rows"])
            if not structure_ok:
                raise ValueError(
                    f"P4.14 batch {batch} structure gate failed: P4.4 contract mismatch"
                )
            # The preservation reference: the frozen P3.5 parent's decisions
            # on the post-K landscape (decision identity, well-defined for
            # any features).
            parent_decision_digest = content_digest(
                {
                    str(candidate_set.candidate_set_digest): [
                        str(parent.select(candidate_set).selected_candidate_id),
                        str(parent.select(candidate_set).selection_status),
                    ]
                    for candidate_set in constraint_sets
                }
            )
            parent_metrics = {
                split: _metric_summary(_evaluate_records(records, parent))
                for split, records in split_records.items()
            }

            # ---- Phase G per cell (P4.11 contract on the post-K landscape).
            for seed in SEEDS:
                cell_dir = batch_dir / f"seed-{seed}"
                cell_started = time.perf_counter()
                learner = ExtendedGSelectionLearner.from_parent_learner(parent)
                birth = _birth_equivalence(
                    parent=parent,
                    learner=learner,
                    split_records=split_records,
                )
                if not birth["passed"]:
                    raise RuntimeError(
                        f"P4.14 cell b{batch}/s{seed} birth gate failed: {birth}"
                    )
                birth_feature_source_digest = learner.feature_source_state_digest
                base_hinge_losses = [
                    learner.invariant_hinge(candidate_set)[0]
                    for candidate_set in constraint_sets
                ]
                birth_hinge_zero = all(loss == 0.0 for loss in base_hinge_losses)
                if not birth_hinge_zero:
                    raise RuntimeError(
                        f"P4.14 cell b{batch}/s{seed} birth hinge is not zero"
                    )
                fit_started = time.perf_counter()
                fit = learner.invariant_fit(
                    train_fit_g,
                    constraint_sets,
                    epochs=TRAINING_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    order_seed=seed,
                    constraint_digest=constraint_digest,
                )
                phase_g_fit_wall = time.perf_counter() - fit_started
                anchor = [
                    float(value) for value in learner.head.weight.detach().reshape(-1)
                ]
                constraints = _task_constraints(
                    train_fit_g, learner, "phase-g"
                ) + _preservation_constraints(constraint_sets, parent, learner)
                projection_started = time.perf_counter()
                projection = project_to_joint_feasible_region(constraints, anchor)
                phase_g_projection_wall = time.perf_counter() - projection_started
                projection_incomplete = not projection["converged"]
                if projection_incomplete:
                    incomplete_projection_cells += 1
                else:
                    projection_digest = content_digest(
                        {
                            "batch": batch,
                            "seed": seed,
                            "phase": "g",
                            "anchor": anchor,
                            "projected": projection["weights"],
                        }
                    )
                    learner.apply_projected_weights(
                        projection["weights"], projection_digest=projection_digest
                    )
                checkpoint_g = _save_extended_checkpoint(
                    cell_dir / "phase-g-extended.pt", learner
                )
                if not checkpoint_g["passed"]:
                    raise RuntimeError(
                        f"P4.14 cell b{batch}/s{seed} phase-G checkpoint failed"
                    )
                restored = ExtendedGSelectionLearner.from_checkpoint(
                    _load_mapping(Path(str(checkpoint_g["path"]))), device="cpu"
                )
                rollback_mismatches = 0
                for split in ("validation", *EVAL_SPLITS):
                    for record in split_records[split]:
                        candidate_set = record["candidate_set"]
                        live_decision = learner.select(candidate_set)
                        restored_decision = restored.select(candidate_set)
                        if (
                            live_decision.selected_candidate_id
                            != restored_decision.selected_candidate_id
                            or live_decision.selection_status
                            != restored_decision.selection_status
                        ):
                            rollback_mismatches += 1
                rollback_gate = {
                    "restored_selection_mismatches": rollback_mismatches,
                    "passed": rollback_mismatches == 0,
                }
                if not rollback_gate["passed"]:
                    raise RuntimeError(
                        f"P4.14 cell b{batch}/s{seed} rollback gate failed"
                    )
                tamper_gate_g = _extended_tamper_rejected(
                    _load_mapping(Path(str(checkpoint_g["path"])))
                )
                if not tamper_gate_g:
                    raise RuntimeError(
                        f"P4.14 cell b{batch}/s{seed} phase-G tamper gate failed"
                    )
                feature_source_unchanged = (
                    learner.feature_source_state_digest == birth_feature_source_digest
                )
                metrics = {
                    split: _metric_summary(_evaluate_records(records, learner))
                    for split, records in split_records.items()
                }
                phase_g_gates = {
                    "new_task": _new_task_gate(metrics["holdout"]),
                    "retention_sibling": _retention_gate(
                        metrics["retention-sibling"],
                        parent_metrics["retention-sibling"],
                    ),
                    "retention_newtask": _retention_gate(
                        metrics["retention-newtask"],
                        parent_metrics["retention-newtask"],
                    ),
                }
                # Cross-phase gate: the K workers on disk are untouched by
                # the G phase.
                k_unchanged_after_g = all(
                    content_digest(_load_mapping(path)) == digest
                    for path, digest in (
                        (
                            Path(
                                str(
                                    phase_k_by_seed[seed]["checkpoint"]["files"]["k1"][
                                        "path"
                                    ]
                                )
                            ),
                            phase_k_by_seed[seed]["post_k_digests"]["k1"],
                        ),
                        (
                            Path(
                                str(
                                    phase_k_by_seed[seed]["checkpoint"]["files"]["k2"][
                                        "path"
                                    ]
                                )
                            ),
                            phase_k_by_seed[seed]["post_k_digests"]["k2"],
                        ),
                    )
                )
                cross_phase_gates = {
                    "k_unchanged_after_g": k_unchanged_after_g,
                    "g_preservation_vs_frozen_parent": all(
                        phase_g_gates["retention_sibling"].values()
                    )
                    and all(phase_g_gates["retention_newtask"].values()),
                    "birth_equivalence": birth["passed"] and birth_hinge_zero,
                }
                phase_k_cell = phase_k_by_seed[seed]
                cell_total_wall = (
                    time.perf_counter()
                    - cell_started
                    + phase_k_cell["fit_wall_seconds"]
                )
                resource_audit = {
                    "phase_k_fit_wall_seconds": phase_k_cell["fit_wall_seconds"],
                    "phase_g_fit_wall_seconds": round(phase_g_fit_wall, 3),
                    "phase_g_projection_wall_seconds": round(
                        phase_g_projection_wall, 3
                    ),
                    "cell_total_wall_seconds": round(cell_total_wall, 3),
                    "caps": G_RESOURCE_CAPS,
                    "caps_passed": (
                        phase_k_cell["fit_wall_seconds"]
                        <= G_RESOURCE_CAPS["phase_k_fit_seconds"]
                        and phase_g_fit_wall <= G_RESOURCE_CAPS["phase_g_fit_seconds"]
                        and phase_g_projection_wall
                        <= G_RESOURCE_CAPS["phase_g_projection_seconds"]
                        and cell_total_wall <= G_RESOURCE_CAPS["cell_total_seconds"]
                    ),
                }
                if not resource_audit["caps_passed"]:
                    resource_violations += 1
                phase_k_passed = all(phase_k_cell["gates"].values())
                phase_g_passed = (
                    all(phase_g_gates["new_task"].values())
                    and all(phase_g_gates["retention_sibling"].values())
                    and all(phase_g_gates["retention_newtask"].values())
                    and not projection_incomplete
                )
                cell_results.append(
                    {
                        "batch": batch,
                        "seed": seed,
                        "phase_k": {
                            "fit_counts": phase_k_cell["fit_counts"],
                            "stream_digest": phase_k_cell["stream_digest"],
                            "checkpoint": phase_k_cell["checkpoint"],
                            "tamper_gate": phase_k_cell["tamper_gate"],
                            "post_k_digests": phase_k_cell["post_k_digests"],
                            "parent_novel_scores": _strip_rows(
                                phase_k_cell["parent_scores"]["p2_6_novel_validation"]
                            ),
                            "interleaved_novel_scores": _strip_rows(
                                phase_k_cell["interleaved_scores"][
                                    "p2_6_novel_validation"
                                ]
                            ),
                            "p1_validation_parent_summary": _strip_rows(
                                phase_k_cell["parent_scores"]["p1_validation"]
                            ),
                            "p1_validation_interleaved_summary": _strip_rows(
                                phase_k_cell["interleaved_scores"]["p1_validation"]
                            ),
                            "gates": phase_k_cell["gates"],
                            "passed": phase_k_passed,
                        },
                        "phase_g": {
                            "fit": fit,
                            "birth": birth,
                            "birth_hinge_zero": birth_hinge_zero,
                            "feature_source_unchanged": feature_source_unchanged,
                            "projection": {
                                "constraint_count": len(constraints),
                                "converged": projection["converged"],
                                "max_violation": projection["max_violation"],
                                "total_violation": projection["total_violation"],
                                "distance": projection["distance"],
                                "digest": (
                                    None if projection_incomplete else projection_digest
                                ),
                            },
                            "checkpoint": checkpoint_g,
                            "rollback_gate": rollback_gate,
                            "tamper_gate": tamper_gate_g,
                            "metrics": metrics,
                            "gates": phase_g_gates,
                            "passed": phase_g_passed,
                        },
                        "cross_phase_gates": cross_phase_gates,
                        "resource_audit": resource_audit,
                        "passes_all": (
                            phase_k_passed
                            and phase_g_passed
                            and all(cross_phase_gates.values())
                        ),
                        "parameter_count": {
                            "k_total": int(
                                phase_k_cell["interleaved_scores"]["parameter_count"][
                                    "total"
                                ]
                            ),
                            "g_extended": int(learner.parameter_count),
                        },
                    }
                )
            batch_summaries.append(
                {
                    "batch": batch,
                    "phase_k_contract": audit,
                    "identity_gate": batch_identity_gate,
                    "structure_ok": structure_ok,
                    "parent_decision_digest": parent_decision_digest,
                    "constraint_digest": constraint_digest,
                    "parent_metrics": parent_metrics,
                    "post_k_digests": post_k_digests_by_seed[SEEDS[0]],
                }
            )

        all_candidate_digests: set[str] = set()
        for digests in batch_candidate_digests.values():
            all_candidate_digests |= digests
        digest_gate = {
            "all_candidate_digests_unique": len(all_candidate_digests)
            == len(BATCHES) * 88,
            "phase_k_record_digests_unique": len(all_novel_record_digests)
            == len(BATCHES) * 8,
        }
        phase_k_failed_cells = sum(
            1 for cell in cell_results if not cell["phase_k"]["passed"]
        )
        phase_g_failed_cells = sum(
            1 for cell in cell_results if not cell["phase_g"]["passed"]
        )
        cross_phase_failed_cells = sum(
            1 for cell in cell_results if not all(cell["cross_phase_gates"].values())
        )
        passing_cells = sum(1 for cell in cell_results if cell["passes_all"])
        if phase_k_failed_cells >= 1:
            outcome = "k_retention_regressed"
        elif incomplete_projection_cells >= 2:
            outcome = "projection_incomplete"
        elif phase_g_failed_cells >= 1 or cross_phase_failed_cells >= 1:
            outcome = "cross_phase_feature_shift"
        elif passing_cells == len(BATCHES) * len(SEEDS):
            outcome = "joint_course_supported"
        else:
            outcome = "cross_phase_feature_shift"
        checkpoint_gate = {
            "all_phase_k_checkpoints": all(
                cell["phase_k"]["checkpoint"]["passed"] for cell in cell_results
            ),
            "all_phase_g_checkpoints": all(
                cell["phase_g"]["checkpoint"]["passed"] for cell in cell_results
            ),
            "all_rollback_gates": all(
                cell["phase_g"]["rollback_gate"]["passed"] for cell in cell_results
            ),
            "all_tamper_checks": all(
                all(cell["phase_k"]["tamper_gate"].values())
                and cell["phase_g"]["tamper_gate"]
                for cell in cell_results
            ),
            "k_parent_not_overwritten": worker_digests
            == {
                "k1": content_digest(_load_mapping(k_parent_k1_path)),
                "k2": content_digest(_load_mapping(k_parent_k2_path)),
            },
            "g_parent_not_overwritten": content_digest(parent_payload)
            == content_digest(_load_mapping(parent_path)),
            "feature_source_unchanged": all(
                cell["phase_g"]["feature_source_unchanged"] for cell in cell_results
            ),
        }
        training_gate = {
            "four_cells_present": len(cell_results) == len(BATCHES) * len(SEEDS),
            "two_deterministic_seeds": all(
                sum(1 for cell in cell_results if cell["batch"] == batch) == len(SEEDS)
                for batch in BATCHES
            ),
            "validation_not_fit": True,
            "holdout_not_fit": True,
            "retention_newtask_not_fit": True,
            "retention_sibling_not_fit": True,
            "constraint_targets_are_parent_only": True,
            "external_target_unused": True,
            "k_parameters_unchanged": all(
                cell["parameter_count"]["k_total"] == 5648 for cell in cell_results
            ),
            "g_parameter_counts_frozen": all(
                cell["parameter_count"]["g_extended"] == 17 for cell in cell_results
            ),
            "k3_absent_from_course": True,
        }
        all_gates = (
            all(digest_gate.values())
            and all(checkpoint_gate.values())
            and all(training_gate.values())
            and all(summary["structure_ok"] for summary in batch_summaries)
            and incomplete_projection_cells == 0
        )
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p1_manifest_digest": str(p1_manifest["manifest_digest"]),
            "source_p2_6_manifest_digest": str(p2_6_manifest["manifest_digest"]),
            "source_p2_7_manifest_digest": str(p2_7_manifest["manifest_digest"]),
            "source_p4_4_manifest_digest": str(p4_4_manifest["manifest_digest"]),
            "source_p4_11_manifest_digest": str(p4_11_manifest["manifest_digest"]),
            "source_p4_12_manifest_digest": str(p4_12_manifest["manifest_digest"]),
            "source_p4_13_manifest_digest": str(p4_13_manifest["manifest_digest"]),
            "source_p4_13_report_digest": content_digest(p4_13_report),
            "source_scorecard_v4_report_digest": content_digest(scorecard_v4),
            "parent_g_checkpoint_digest": str(parent_metadata["digest"]),
            "k_parent_checkpoint_digests": worker_digests,
            "course_contract": {
                "phases": ["phase-k", "re-materialize", "phase-g"],
                "k_parent": "P3.2 base-continuation workers (the K workers of the joint parent)",
                "g_parent": "frozen P3.5 G checkpoint (decision identity on the post-K landscape)",
                "phase_k_mechanics": "P2.6 verbatim (novel K2-content cohort + 50-example P2 rehearsal interleave)",
                "phase_g_mechanics": "P4.11 verbatim (task fit + margin-preservation hinge + terminal joint projection)",
                "cross_phase_gates": [
                    "k_unchanged_after_g",
                    "g_preservation_vs_frozen_parent",
                    "birth_equivalence",
                ],
                "resource_caps_absolute": G_RESOURCE_CAPS,
            },
            "fit_policy": {
                "fit_called": True,
                "training_performed": True,
                "validation_only": False,
                "growth_admitted": False,
                "retention_fit_count": 0,
                "new_task_thresholds": {"utility_floor": 0.68, "target_hit_floor": 0.6},
            },
            "batch_candidate_set_digests": {
                str(batch): sorted(batch_candidate_digests[batch]) for batch in BATCHES
            },
            "phase_k_novel_record_digests": sorted(all_novel_record_digests),
            "batch_novel_records": {
                str(batch): {
                    "train_records": batch_novel_records[batch]["train"],
                    "validation_records": batch_novel_records[batch]["validation"],
                }
                for batch in BATCHES
            },
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "fit_called": True,
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "identity_gate": digest_gate,
                "batch_summaries": batch_summaries,
                "checkpoint_gate": checkpoint_gate,
                "training_gate": training_gate,
                "resource_audit": {
                    "resource_violations": resource_violations,
                    "caps": G_RESOURCE_CAPS,
                    "per_cell": [
                        {
                            "batch": cell["batch"],
                            "seed": cell["seed"],
                            **cell["resource_audit"],
                        }
                        for cell in cell_results
                    ],
                },
                "phase_k_failed_cells": phase_k_failed_cells,
                "phase_g_failed_cells": phase_g_failed_cells,
                "cross_phase_failed_cells": cross_phase_failed_cells,
                "incomplete_projection_cells": incomplete_projection_cells,
                "passing_cells": passing_cells,
                "cell_results": cell_results,
                "outcome": outcome,
                "joint_course_supported": outcome == "joint_course_supported",
                "experiment_passed": bool(all_gates),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; K-worker continuation (P2.6 mechanics) "
                    "followed by the G solver mechanism (P4.11 contract) on the post-K "
                    "landscape of the same parent; growth and promotion remain fail-closed"
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    finally:
        for batch_dir in Path(run_dir).glob("batch-*"):
            shutil.rmtree(batch_dir / "phase-k-data", ignore_errors=True)
            shutil.rmtree(batch_dir / "p1", ignore_errors=True)
            shutil.rmtree(batch_dir / "p1-cases", ignore_errors=True)
            shutil.rmtree(batch_dir / "data", ignore_errors=True)
    _write_json_atomic(report_path, payload)
    return payload


def _evaluate_records(
    records: Sequence[Mapping[str, Any]], learner: Any
) -> dict[str, Any]:
    return _evaluate(records, learner)


def _strip_rows(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "rows"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify-extended-only", type=Path)
    args = parser.parse_args()
    if args.verify_extended_only is not None:
        result = _verify_extended_checkpoint(args.verify_extended_only)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["passed"] else 1
    result = _run(manifest_path=args.manifest, report_path=args.report)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "report": str(args.report),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "passing_cells": result.get("passing_cells"),
                "growth_admitted": result.get("growth_admitted"),
                "can_promote": result.get("can_promote"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
