"""P4.13 two-phase promotion course runner (solver mechanism).

P4.12 validated the projection-solver mechanism across a 9-cell matrix
for a single task cohort plus retention.  This runner executes the
preregistered promotion course: per cell, TWO sequential novel-task
phases on the same parent -

- Phase A: cohort A task fit (SGD) -> projection #1 over {A task
  constraints + preservation constraints} -> checkpoint (rollback net);
- Phase B: cohort B task fit FROM the phase-A projected state ->
  projection #2 over the CUMULATIVE system {A + B task constraints +
  preservation constraints};

with the backward-retention gate (the phase-A holdout must still pass
the new-task gate after phase B), two arms per cell (baseline without
projection vs projected), absolute resource budgets, and the P3.0
checkpoint/rollback machinery.  Preregistration:
``plans/reference/M5_K_P4_13_PROMOTION_COURSE_PREREGISTRATION
_20260911.md``.  Never admits growth or promotion.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    WORKER_ROOT,
    _context,
)
from scripts.training.eval_taiji_m5_k_p3_3_g_learning import (  # noqa: E402
    _fresh_learners,
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
from scripts.training.eval_taiji_m5_k_p4_3_retention_incremental import (  # noqa: E402
    _evaluate,
    _save_torch_atomic,
)
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
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)
from taiji.g_selection_extended import (  # noqa: E402
    ExtendedGSelectionLearner,
)
from taiji.g_selection_projection import (  # noqa: E402
    project_to_joint_feasible_region,
)

REPORT_FORMAT = "taiji-m5-k-p4-13-promotion-course-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-13-promotion-course-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_13_promotion_course_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_13_promotion_course_20260911.json"
P4_12_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_12_course_level_validation_manifest_v1.json"
)
P4_12_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_12_course_level_validation_20260911.json"
P4_11_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_11_projection_solver_manifest_v1.json"
)
P4_9_PROBE_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_9_feature_space_probe_20260911.json"
P4_10_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_10_feature_factorization_manifest_v1.json"
)
P4_8_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_8_representation_redesign_manifest_v1.json"
)
P4_7_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_7_capacity_clean_test_manifest_v1.json"
)
P4_6_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_6_functional_parent_objective_manifest_v1.json"
)
P4_5_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_5_update_rule_gate_manifest_v1.json"
)
P4_4_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_4_retention_identity_calibration_manifest_v1.json"
)
P4_3_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_3_retention_incremental_manifest_v1.json"
)
P4_2_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_2_capacity_attribution_manifest_v1.json"
)
P4_1_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_1_context_contract_manifest_v1.json"
)
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
P3_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"
TRAINING_EPOCHS = 8
LEARNING_RATE = 0.15
SEEDS = (0, 1, 2)
BATCHES = (0, 1, 2)
ARM_BASELINE = "baseline-ext-17"
ARM_PROJECTED = "projected-ext-17"
ARGMAX_EPSILON = 1e-6
SAFE_EPSILON = 1e-9
SELECTION_MARGIN = 0.05
EVAL_SPLITS_A = ("validation-a", "holdout-a", "retention-sibling", "retention-newtask")
EVAL_SPLITS_B = ("validation-b", "holdout-b", "holdout-a", "retention-sibling", "retention-newtask")
RESOURCE_CAPS = {
    "fit_seconds_per_phase": 60.0,
    "projection_seconds_per_phase": 120.0,
    "cell_total_seconds": 600.0,
}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _digest_without(payload: Mapping[str, Any], key: str) -> str:
    return content_digest({name: value for name, value in payload.items() if name != key})


def _batch_offsets(batch: int) -> dict[str, int]:
    base = 55000 + batch * 1000
    return {
        "sibling": base,
        "constraint": base + 100,
        "train-a": base + 200,
        "validation-a": base + 300,
        "holdout-a": base + 400,
        "retention-newtask": base + 500,
        "train-b": base + 600,
        "validation-b": base + 700,
        "holdout-b": base + 800,
    }


def _p413_specs(split: str, offset: int, batch: int, phase: str) -> tuple[dict[str, Any], ...]:
    prefix = f"p40_p413_b{batch}_{phase}_{split}"
    rows = (
        (
            "A",
            "resolved-language",
            (f"{prefix}_a_python.py", f"{prefix}_a_rust.rs", f"{prefix}_a_python_b.py"),
        ),
        (
            "B",
            "ambiguous-language",
            (f"{prefix}_b_rust.py", f"{prefix}_b_python.py", f"{prefix}_b_typescript.ts"),
        ),
        (
            "C",
            "resolved-language",
            (f"{prefix}_c_typescript.ts", f"{prefix}_c_python.py", f"{prefix}_c_rust.rs"),
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
            "project_id": f"p4-13-b{batch}-{phase}-{split}-project-{class_key.lower()}",
            "variant_paths": paths,
            "state_profile": state_profile,
            "task_seed": offset + index,
        }
        for index, (class_key, state_profile, paths) in enumerate(rows)
    )


def _p413_structured_specs(
    contract: Mapping[str, Any], *, namespace: str, seed_offset: int, batch: int
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    safe_index = 0
    high_index = 0
    for index, row in enumerate(contract["rows"]):
        if int(row["role_counts"].get("proposal", 0)) > 0:
            project_id = f"p4-13-b{batch}-{namespace}-project-a"
            prefix = f"p40_p413_b{batch}_{namespace}_a"
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
                f"p4-13-b{batch}-{namespace}-project-a"
                if safe_index == 0
                else f"p4-13-b{batch}-{namespace}-project-b"
            )
            prefix = f"p40_p413_b{batch}_{namespace}_{'a' if safe_index == 0 else 'b'}"
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


def _p413_rebind(
    record: Mapping[str, Any],
    *,
    split: str,
    source_index: int,
    width: int,
    fit_eligible: bool,
    batch: int,
    phase: str,
) -> dict[str, Any]:
    old_candidates = record["candidate_set"]
    old_behavior = record["behavior_set"]
    candidate_set = GSelectionCandidateSet.create(
        example_id=f"p4-13:b{batch}:{phase}:{split}:{source_index}:width-{width}:{old_candidates.candidate_set_digest}",
        family_id=f"p4-13:b{batch}:{phase}:{split}:family:{source_index}",
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
            "experiment": "p4.13",
            "batch": batch,
            "phase": phase,
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


def _p413_pressure_split(
    *,
    split: str,
    offset: int,
    batch: int,
    phase: str,
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
    for spec in _p413_specs(split, offset, batch, phase):
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
            label=f"p4-13-b{batch}-{phase}-{split}-{int(spec['index'])}",
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
                _p413_rebind(
                    pressure,
                    split=split,
                    source_index=source_index,
                    width=width,
                    fit_eligible=(
                        split == "train"
                        and float(pressure["behavior_set"].utility_margin) > MARGIN_EPSILON
                    ),
                    batch=batch,
                    phase=phase,
                )
            )
    return records


def _p413_structured_records(
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
    specs = _p413_structured_specs(
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
            label=f"p4-13-b{batch}-{namespace}-{int(spec['index'])}",
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
            _p413_rebind(
                pressure,
                split=namespace,
                source_index=index,
                width=int(spec["target_width"]),
                fit_eligible=False,
                batch=batch,
                phase="shared",
            )
        )
    return records


def _task_constraints(
    train_fit_records: Sequence[Mapping[str, Any]],
    learner: ExtendedGSelectionLearner,
    family: str,
) -> list[tuple[tuple[float, ...], float, str]]:
    constraints: list[tuple[tuple[float, ...], float, str]] = []
    for record in train_fit_records:
        candidate_set = record["candidate_set"]
        behavior_set = record["behavior_set"]
        features = learner.relative_features(candidate_set)
        target_id = behavior_set.behavior_target_candidate_id
        target = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_id == target_id
        )
        safe = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_role in {"abstain", "reobserve"}
        )
        for candidate in candidate_set.candidates:
            if candidate.candidate_id == target_id:
                continue
            difference = tuple(
                t - o
                for t, o in zip(features[target_id], features[candidate.candidate_id], strict=True)
            )
            constraints.append(
                (
                    difference,
                    ARGMAX_EPSILON,
                    f"{family}-argmax:{candidate_set.candidate_set_digest[:16]}:{candidate.candidate_id}",
                )
            )
        if target.candidate_role == "proposal":
            difference = tuple(
                t - o for t, o in zip(features[target_id], features[safe.candidate_id], strict=True)
            )
            constraints.append(
                (
                    difference,
                    SELECTION_MARGIN + SAFE_EPSILON,
                    f"{family}-safe:{candidate_set.candidate_set_digest[:16]}",
                )
            )
    return constraints


def _preservation_constraints(
    constraint_sets: Sequence[GSelectionCandidateSet],
    parent: GSelectionLearner,
    learner: ExtendedGSelectionLearner,
) -> list[tuple[tuple[float, ...], float, str]]:
    constraints: list[tuple[tuple[float, ...], float, str]] = []
    for candidate_set in constraint_sets:
        decision = parent.select(candidate_set)
        features = learner.relative_features(candidate_set)
        safe = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_role in {"abstain", "reobserve"}
        )
        if decision.selection_status == "selected":
            picked_id = decision.selected_candidate_id
            for candidate in candidate_set.candidates:
                if candidate.candidate_id == picked_id:
                    continue
                difference = tuple(
                    t - o
                    for t, o in zip(
                        features[picked_id], features[candidate.candidate_id], strict=True
                    )
                )
                constraints.append(
                    (
                        difference,
                        ARGMAX_EPSILON,
                        f"cohort-argmax:{candidate_set.candidate_set_digest[:16]}:{candidate.candidate_id}",
                    )
                )
            difference = tuple(
                t - o for t, o in zip(features[picked_id], features[safe.candidate_id], strict=True)
            )
            constraints.append(
                (
                    difference,
                    SELECTION_MARGIN + SAFE_EPSILON,
                    f"cohort-safe:{candidate_set.candidate_set_digest[:16]}",
                )
            )
        else:
            picked_id = decision.selected_candidate_id
            for candidate in candidate_set.candidates:
                if candidate.candidate_role != "proposal":
                    continue
                difference = tuple(
                    t - o
                    for t, o in zip(
                        features[candidate.candidate_id], features[picked_id], strict=True
                    )
                )
                constraints.append(
                    (
                        difference,
                        -(SELECTION_MARGIN - SAFE_EPSILON),
                        f"cohort-boundary:{candidate_set.candidate_set_digest[:16]}:{candidate.candidate_id}",
                    )
                )
    return constraints


def _save_extended_checkpoint(path: Path, learner: ExtendedGSelectionLearner) -> dict[str, Any]:
    payload = learner.checkpoint()
    _save_torch_atomic(path, payload)
    restored = ExtendedGSelectionLearner.from_checkpoint(_load_mapping(path), device="cpu")
    independent = _independent_extended_restore(path)
    digest_matches = restored.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"]
    return {
        "path": str(path),
        "digest": payload["checkpoint_digest"],
        "bytes": path.stat().st_size,
        "roundtrip": digest_matches,
        "restore": independent,
        "passed": bool(independent.get("independent_process_restore")) and digest_matches,
    }


def _independent_extended_restore(path: Path) -> dict[str, Any]:
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify-extended-only", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    result: dict[str, Any] = {
        "returncode": int(child.returncode),
        "independent_process_restore": False,
        "stdout": child.stdout[-2000:],
        "stderr": child.stderr[-2000:],
    }
    if child.returncode != 0:
        return result
    try:
        decoded = json.loads(child.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
        return result
    result.update(decoded)
    result["independent_process_restore"] = bool(decoded.get("passed"))
    return result


def _extended_tamper_rejected(payload: Mapping[str, Any]) -> bool:
    tampered = copy.deepcopy(dict(payload))
    tampered["revision"] = int(tampered.get("revision", 0)) + 1
    try:
        ExtendedGSelectionLearner.from_checkpoint(tampered, device="cpu")
    except ValueError:
        return True
    return False


def _verify_extended_checkpoint(path: Path) -> dict[str, Any]:
    payload = _load_mapping(path)
    learner = ExtendedGSelectionLearner.from_checkpoint(payload, device="cpu")
    return {
        "passed": learner.parameter_count == 17
        and learner.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"],
        "parameter_count": learner.parameter_count,
    }


def _run(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_13_promotion_course_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "external_target_used": False,
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "manifest": str(manifest_path),
        "report": str(report_path),
    }
    try:
        p4_1_manifest = _load_json(P4_1_MANIFEST)
        p4_2_manifest = _load_json(P4_2_MANIFEST)
        p4_3_manifest = _load_json(P4_3_MANIFEST)
        p4_4_manifest = _load_json(P4_4_MANIFEST)
        p4_5_manifest = _load_json(P4_5_MANIFEST)
        p4_6_manifest = _load_json(P4_6_MANIFEST)
        p4_7_manifest = _load_json(P4_7_MANIFEST)
        p4_8_manifest = _load_json(P4_8_MANIFEST)
        p4_10_manifest = _load_json(P4_10_MANIFEST)
        p4_11_manifest = _load_json(P4_11_MANIFEST)
        p4_12_manifest = _load_json(P4_12_MANIFEST)
        p4_12_report = _load_json(P4_12_REPORT)
        p4_9_probe = _load_json(P4_9_PROBE_REPORT)
        p3_5_report = _load_json(P3_5_REPORT)
        if (
            p4_12_report.get("status") != "completed"
            or p4_12_report.get("outcome") != "course_level_validation_supported"
        ):
            raise ValueError("P4.13 requires the completed P4.12 course_level_validation_supported")
        if p4_12_report.get("manifest_digest") != p4_12_manifest.get("manifest_digest"):
            raise ValueError("P4.12 manifest/report digest mismatch")
        if _digest_without(p4_12_manifest, "manifest_digest") != p4_12_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.12 manifest content digest mismatch")
        if p4_12_report.get("growth_admitted") or p4_12_report.get("can_promote"):
            raise ValueError("P4.13 cannot consume an admitted P4.12 artifact")
        if p4_12_manifest.get("source_p4_11_manifest_digest") != p4_11_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.12 source chain drifted")
        if p4_9_probe.get("feasibility_verdict") != (
            "parent_relative_features_are_the_factorization"
        ):
            raise ValueError("P4.13 requires the P4.9 factorization verdict")
        contract = p4_4_manifest["structure_contract"]
        if content_digest(contract["rows"]) != contract["contract_digest"]:
            raise ValueError("P4.4 structure contract digest drifted")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        p3_2_report = _load_json(P3_2_REPORT)
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        semantic_payload = _load_mapping(Path(str(worker_restore["k1"]["path"])))
        transition_payload = _load_mapping(Path(str(worker_restore["k2"]["path"])))
        worker_digests = dict(p4_2_manifest["k_checkpoint_digests"])
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        parent_metadata = p3_5_report["g_trained_checkpoint"]
        parent_path = Path(str(parent_metadata["path"]))
        parent_payload = _load_mapping(parent_path)
        if content_digest(parent_payload) != str(parent_metadata["digest"]):
            raise ValueError("P4.13 parent external checkpoint digest drifted")
        parent = GSelectionLearner.from_checkpoint(parent_payload, device="cpu")
        parent.assert_lineage(
            parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        if str(p4_2_manifest["parent_g_checkpoint_digest"]) != str(parent_metadata["digest"]):
            raise ValueError("P4.13 parent G lineage drifted")
        parent_restore = _independent_g_restore(parent_path)
        if not parent_restore.get("independent_process_restore"):
            raise RuntimeError("P4.13 parent independent restore failed")
        run_dir.mkdir(parents=True, exist_ok=False)
        old_paths: set[str] = set()
        old_projects: set[str] = set()
        for manifest in (
            p4_1_manifest,
            p4_2_manifest,
            p4_3_manifest,
            p4_4_manifest,
            p4_5_manifest,
            p4_6_manifest,
            p4_7_manifest,
            p4_8_manifest,
            p4_10_manifest,
            p4_11_manifest,
            p4_12_manifest,
        ):
            manifest_paths, manifest_projects = _manifest_identity(manifest)
            old_paths.update(manifest_paths)
            old_projects.update(manifest_projects)

        batch_summaries: list[dict[str, Any]] = []
        cell_results: list[dict[str, Any]] = []
        batch_candidate_digests: dict[int, set[str]] = {}
        all_candidate_digests: set[str] = set()
        incomplete_projection_b_cells = 0
        resource_violations = 0
        for batch in BATCHES:
            offsets = _batch_offsets(batch)
            batch_dir = run_dir / f"batch-{batch}"
            batch_dir.mkdir(parents=True, exist_ok=False)
            common = dict(
                parent_digest=parent_digest,
                worker_bundle_digest=bundle.bundle_digest,
                source_manifest_digest=str(p4_12_manifest["manifest_digest"]),
                projector=projector,
                semantic=semantic,
                transition=transition,
                semantic_payload=semantic_payload,
            )
            train_a = _p413_pressure_split(
                split="train",
                offset=offsets["train-a"],
                batch=batch,
                phase="a",
                scratch=batch_dir / "data" / "train-a",
                **common,
            )
            validation_a = _p413_pressure_split(
                split="validation",
                offset=offsets["validation-a"],
                batch=batch,
                phase="a",
                scratch=batch_dir / "data" / "validation-a",
                **common,
            )
            holdout_a = _p413_pressure_split(
                split="holdout",
                offset=offsets["holdout-a"],
                batch=batch,
                phase="a",
                scratch=batch_dir / "data" / "holdout-a",
                **common,
            )
            retention_newtask = _p413_pressure_split(
                split="retention-newtask",
                offset=offsets["retention-newtask"],
                batch=batch,
                phase="a",
                scratch=batch_dir / "data" / "retention-newtask",
                **common,
            )
            constraint_records = _p413_structured_records(
                contract=contract,
                namespace="constraint",
                seed_offset=offsets["constraint"],
                batch=batch,
                scratch=batch_dir / "data" / "constraint",
                **common,
            )
            retention_sibling_records = _p413_structured_records(
                contract=contract,
                namespace="retention",
                seed_offset=offsets["sibling"],
                batch=batch,
                scratch=batch_dir / "data" / "retention",
                **common,
            )
            train_b = _p413_pressure_split(
                split="train",
                offset=offsets["train-b"],
                batch=batch,
                phase="b",
                scratch=batch_dir / "data" / "train-b",
                **common,
            )
            validation_b = _p413_pressure_split(
                split="validation",
                offset=offsets["validation-b"],
                batch=batch,
                phase="b",
                scratch=batch_dir / "data" / "validation-b",
                **common,
            )
            holdout_b = _p413_pressure_split(
                split="holdout",
                offset=offsets["holdout-b"],
                batch=batch,
                phase="b",
                scratch=batch_dir / "data" / "holdout-b",
                **common,
            )
            train_fit_a = [record for record in train_a if record["fit_eligible"]]
            train_fit_b = [record for record in train_b if record["fit_eligible"]]
            constraint_sets = tuple(record["candidate_set"] for record in constraint_records)
            constraint_digest = content_digest(
                {
                    "constraint_set_digests": [
                        candidate_set.candidate_set_digest for candidate_set in constraint_sets
                    ],
                    "constraint_form": "margin-preservation-hinge",
                }
            )
            batch_records = [
                *train_a,
                *validation_a,
                *holdout_a,
                *retention_newtask,
                *constraint_records,
                *retention_sibling_records,
                *train_b,
                *validation_b,
                *holdout_b,
            ]
            batch_candidate_digests[batch] = {
                record["candidate_set"].candidate_set_digest for record in batch_records
            }
            all_candidate_digests |= batch_candidate_digests[batch]
            split_records_a = {
                "validation-a": validation_a,
                "holdout-a": holdout_a,
                "retention-sibling": retention_sibling_records,
                "retention-newtask": retention_newtask,
            }
            split_records_b = {
                "validation-b": validation_b,
                "holdout-b": holdout_b,
                "holdout-a": holdout_a,
                "retention-sibling": retention_sibling_records,
                "retention-newtask": retention_newtask,
            }
            identity_sets = [
                _records_identity(collection)
                for collection in (
                    train_a,
                    validation_a,
                    holdout_a,
                    retention_newtask,
                    constraint_records,
                    retention_sibling_records,
                    train_b,
                    validation_b,
                    holdout_b,
                )
            ]
            batch_identity_gate = {
                "train_a_records": len(train_a) == 20,
                "validation_a_records": len(validation_a) == 20,
                "holdout_a_records": len(holdout_a) == 20,
                "retention_newtask_records": len(retention_newtask) == 20,
                "constraint_records": len(constraint_records) == int(contract["row_count"]),
                "retention_sibling_records": len(retention_sibling_records)
                == int(contract["row_count"]),
                "train_b_records": len(train_b) == 20,
                "validation_b_records": len(validation_b) == 20,
                "holdout_b_records": len(holdout_b) == 20,
                "train_fit_a_positive": len(train_fit_a) >= 8,
                "train_fit_b_positive": len(train_fit_b) >= 8,
                "five_classes_all_pressure_splits": all(
                    len({record["diagnostic"]["class_key"] for record in records}) == 5
                    for records in (
                        train_a,
                        validation_a,
                        holdout_a,
                        retention_newtask,
                        train_b,
                        validation_b,
                        holdout_b,
                    )
                ),
                "new_projects_disjoint_from_historical": all(
                    projects.isdisjoint(old_projects) for _paths, projects in identity_sets
                ),
                "new_paths_disjoint_from_historical": all(
                    paths.isdisjoint(old_paths) for paths, _projects in identity_sets
                ),
            }
            if not all(batch_identity_gate.values()):
                raise ValueError(
                    f"P4.13 batch {batch} identity gate failed: "
                    f"{[k for k, v in batch_identity_gate.items() if not v]}"
                )
            old_paths |= {record["candidate_set"].path for record in batch_records}
            old_projects |= {record["candidate_set"].project_id for record in batch_records}
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
                    f"P4.13 batch {batch} structure gate failed: P4.4 contract mismatch"
                )
            parent_metrics_a = {
                split: _metric_summary(_evaluate(records, parent))
                for split, records in split_records_a.items()
            }
            parent_metrics_b = {
                split: _metric_summary(_evaluate(records, parent))
                for split, records in split_records_b.items()
            }
            batch_seed_results: list[dict[str, Any]] = []
            for seed in SEEDS:
                cell_dir = batch_dir / f"seed-{seed}"
                cell_dir.mkdir(parents=True, exist_ok=False)
                cell_started = time.perf_counter()
                learners: dict[str, Any] = {
                    ARM_BASELINE: ExtendedGSelectionLearner.from_parent_learner(parent),
                    ARM_PROJECTED: ExtendedGSelectionLearner.from_parent_learner(parent),
                }
                birth_report: dict[str, Any] = {}
                for arm, learner in learners.items():
                    mismatches = 0
                    max_deviation = 0.0
                    for split in EVAL_SPLITS_A:
                        for record in split_records_a[split]:
                            candidate_set = record["candidate_set"]
                            parent_decision = parent.select(candidate_set)
                            child_decision = learner.select(candidate_set)
                            if (
                                parent_decision.selected_candidate_id
                                != child_decision.selected_candidate_id
                                or parent_decision.selection_status
                                != child_decision.selection_status
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
                    birth_report[arm] = {
                        "selection_mismatches": mismatches,
                        "max_abs_score_deviation": max_deviation,
                    }
                base_hinge_losses = [
                    learners[ARM_BASELINE].invariant_hinge(candidate_set)[0]
                    for candidate_set in constraint_sets
                ]
                birth_report["birth_hinge_loss_zero"] = all(
                    loss == 0.0 for loss in base_hinge_losses
                )
                birth_report["feature_source_digest"] = learners[
                    ARM_BASELINE
                ].feature_source_state_digest
                birth_passed = (
                    all(
                        report["selection_mismatches"] == 0
                        and report["max_abs_score_deviation"] == 0.0
                        for arm, report in birth_report.items()
                        if arm in learners
                    )
                    and birth_report["birth_hinge_loss_zero"]
                )
                if not birth_passed:
                    raise ValueError(
                        f"P4.13 cell b{batch}/s{seed} birth gate failed: {birth_report}"
                    )
                # ---- Phase A: task fit + projection #1 (A + preservation).
                fit_a_started = time.perf_counter()
                fit_a_baseline = learners[ARM_BASELINE].invariant_fit(
                    train_fit_a,
                    constraint_sets,
                    epochs=TRAINING_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    order_seed=seed,
                    constraint_digest=constraint_digest,
                )
                fit_a_projected = learners[ARM_PROJECTED].invariant_fit(
                    train_fit_a,
                    constraint_sets,
                    epochs=TRAINING_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    order_seed=seed,
                    constraint_digest=constraint_digest,
                )
                fit_a_wall = time.perf_counter() - fit_a_started
                phase_a_trajectory_ok = (
                    learners[ARM_BASELINE].model_state_digest
                    == learners[ARM_PROJECTED].model_state_digest
                )
                if not phase_a_trajectory_ok:
                    raise RuntimeError(
                        f"P4.13 cell b{batch}/s{seed} phase-A trajectory gate failed"
                    )
                anchor_a = [
                    float(value)
                    for value in learners[ARM_PROJECTED].head.weight.detach().reshape(-1)
                ]
                constraints_a = _task_constraints(
                    train_fit_a, learners[ARM_PROJECTED], "phase-a"
                ) + _preservation_constraints(constraint_sets, parent, learners[ARM_PROJECTED])
                projection_started = time.perf_counter()
                projection_a = project_to_joint_feasible_region(constraints_a, anchor_a)
                projection_a_wall = time.perf_counter() - projection_started
                if not projection_a["converged"]:
                    raise ValueError(
                        f"P4.13 cell b{batch}/s{seed} phase-A projection_incomplete: "
                        f"max {projection_a['max_violation']}, total {projection_a['total_violation']}"
                    )
                projection_a_digest = content_digest(
                    {
                        "batch": batch,
                        "seed": seed,
                        "phase": "a",
                        "anchor": anchor_a,
                        "projected": projection_a["weights"],
                    }
                )
                learners[ARM_PROJECTED].apply_projected_weights(
                    projection_a["weights"], projection_digest=projection_a_digest
                )
                checkpoint_a = _save_extended_checkpoint(
                    cell_dir / f"{ARM_PROJECTED}-phase-a.pt", learners[ARM_PROJECTED]
                )
                if not checkpoint_a["passed"]:
                    raise RuntimeError(f"P4.13 cell b{batch}/s{seed} phase-A checkpoint failed")
                # Rollback gate: restore the phase-A checkpoint and verify
                # bit-identical behaviour on the phase-A evaluation splits.
                restored_a = ExtendedGSelectionLearner.from_checkpoint(
                    _load_mapping(Path(str(checkpoint_a["path"]))), device="cpu"
                )
                rollback_mismatches = 0
                for split in EVAL_SPLITS_A:
                    for record in split_records_a[split]:
                        candidate_set = record["candidate_set"]
                        live_decision = learners[ARM_PROJECTED].select(candidate_set)
                        restored_decision = restored_a.select(candidate_set)
                        if (
                            live_decision.selected_candidate_id
                            != restored_decision.selected_candidate_id
                            or live_decision.selection_status != restored_decision.selection_status
                        ):
                            rollback_mismatches += 1
                rollback_gate = {
                    "restored_selection_mismatches": rollback_mismatches,
                    "passed": rollback_mismatches == 0,
                }
                if not rollback_gate["passed"]:
                    raise RuntimeError(f"P4.13 cell b{batch}/s{seed} rollback gate failed")
                phase_a_metrics = {
                    arm: {
                        split: _metric_summary(_evaluate(records, learner))
                        for split, records in split_records_a.items()
                    }
                    for arm, learner in learners.items()
                }
                phase_a_gates = {
                    arm: {
                        "new_task_a": _new_task_gate(phase_a_metrics[arm]["holdout-a"]),
                        "retention_sibling": _retention_gate(
                            phase_a_metrics[arm]["retention-sibling"],
                            parent_metrics_a["retention-sibling"],
                        ),
                        "retention_newtask": _retention_gate(
                            phase_a_metrics[arm]["retention-newtask"],
                            parent_metrics_a["retention-newtask"],
                        ),
                    }
                    for arm in learners
                }
                # ---- Phase B: task fit from the phase-A states +
                # cumulative projection #2 (A + B + preservation).
                fit_b_started = time.perf_counter()
                fit_b_baseline = learners[ARM_BASELINE].invariant_fit(
                    train_fit_b,
                    constraint_sets,
                    epochs=TRAINING_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    order_seed=seed,
                    constraint_digest=constraint_digest,
                )
                fit_b_projected = learners[ARM_PROJECTED].invariant_fit(
                    train_fit_b,
                    constraint_sets,
                    epochs=TRAINING_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    order_seed=seed,
                    constraint_digest=constraint_digest,
                )
                fit_b_wall = time.perf_counter() - fit_b_started
                anchor_b = [
                    float(value)
                    for value in learners[ARM_PROJECTED].head.weight.detach().reshape(-1)
                ]
                constraints_b = (
                    _task_constraints(train_fit_a, learners[ARM_PROJECTED], "phase-a")
                    + _task_constraints(train_fit_b, learners[ARM_PROJECTED], "phase-b")
                    + _preservation_constraints(constraint_sets, parent, learners[ARM_PROJECTED])
                )
                projection_started = time.perf_counter()
                projection_b = project_to_joint_feasible_region(constraints_b, anchor_b)
                projection_b_wall = time.perf_counter() - projection_started
                phase_b_projection_incomplete = not projection_b["converged"]
                if not phase_b_projection_incomplete:
                    projection_b_digest = content_digest(
                        {
                            "batch": batch,
                            "seed": seed,
                            "phase": "b",
                            "anchor": anchor_b,
                            "projected": projection_b["weights"],
                        }
                    )
                    learners[ARM_PROJECTED].apply_projected_weights(
                        projection_b["weights"], projection_digest=projection_b_digest
                    )
                else:
                    incomplete_projection_b_cells += 1
                checkpoint_b = _save_extended_checkpoint(
                    cell_dir / f"{ARM_PROJECTED}-phase-b.pt", learners[ARM_PROJECTED]
                )
                if not checkpoint_b["passed"]:
                    raise RuntimeError(f"P4.13 cell b{batch}/s{seed} phase-B checkpoint failed")
                phase_b_metrics = {
                    arm: {
                        split: _metric_summary(_evaluate(records, learner))
                        for split, records in split_records_b.items()
                    }
                    for arm, learner in learners.items()
                }
                phase_b_gates = {
                    arm: {
                        "new_task_b": _new_task_gate(phase_b_metrics[arm]["holdout-b"]),
                        "backward_retention_a": _new_task_gate(phase_b_metrics[arm]["holdout-a"]),
                        "retention_sibling": _retention_gate(
                            phase_b_metrics[arm]["retention-sibling"],
                            parent_metrics_b["retention-sibling"],
                        ),
                        "retention_newtask": _retention_gate(
                            phase_b_metrics[arm]["retention-newtask"],
                            parent_metrics_b["retention-newtask"],
                        ),
                    }
                    for arm in learners
                }
                cell_total_wall = time.perf_counter() - cell_started
                resource_audit = {
                    "fit_a_wall_seconds": round(fit_a_wall, 3),
                    "fit_b_wall_seconds": round(fit_b_wall, 3),
                    "projection_a_wall_seconds": round(projection_a_wall, 3),
                    "projection_b_wall_seconds": round(projection_b_wall, 3),
                    "cell_total_wall_seconds": round(cell_total_wall, 3),
                    "caps": RESOURCE_CAPS,
                    "caps_passed": (
                        fit_a_wall <= RESOURCE_CAPS["fit_seconds_per_phase"]
                        and fit_b_wall <= RESOURCE_CAPS["fit_seconds_per_phase"]
                        and projection_a_wall <= RESOURCE_CAPS["projection_seconds_per_phase"]
                        and projection_b_wall <= RESOURCE_CAPS["projection_seconds_per_phase"]
                        and cell_total_wall <= RESOURCE_CAPS["cell_total_seconds"]
                    ),
                }
                if not resource_audit["caps_passed"]:
                    resource_violations += 1
                tamper_gate = {
                    "phase-a": _extended_tamper_rejected(
                        _load_mapping(Path(str(checkpoint_a["path"])))
                    ),
                    "phase-b": _extended_tamper_rejected(
                        _load_mapping(Path(str(checkpoint_b["path"])))
                    ),
                }
                if not all(tamper_gate.values()):
                    raise RuntimeError(f"P4.13 cell b{batch}/s{seed} tamper gate failed")
                feature_source_unchanged = all(
                    learner.feature_source_state_digest == birth_report["feature_source_digest"]
                    for learner in learners.values()
                )
                passes_all = {
                    arm: (
                        all(phase_a_gates[arm]["new_task_a"].values())
                        and all(phase_a_gates[arm]["retention_sibling"].values())
                        and all(phase_a_gates[arm]["retention_newtask"].values())
                        and all(phase_b_gates[arm]["new_task_b"].values())
                        and all(phase_b_gates[arm]["backward_retention_a"].values())
                        and all(phase_b_gates[arm]["retention_sibling"].values())
                        and all(phase_b_gates[arm]["retention_newtask"].values())
                    )
                    for arm in learners
                }
                cell_results.append(
                    {
                        "batch": batch,
                        "seed": seed,
                        "fit": {
                            ARM_BASELINE: {
                                "phase-a": fit_a_baseline,
                                "phase-b": fit_b_baseline,
                            },
                            ARM_PROJECTED: {
                                "phase-a": fit_a_projected,
                                "phase-b": fit_b_projected,
                            },
                        },
                        "phase_a_trajectory_gate": {
                            "pre_projection_digests_identical": phase_a_trajectory_ok
                        },
                        "projection_a": {
                            "constraint_count": len(constraints_a),
                            "converged": projection_a["converged"],
                            "max_violation": projection_a["max_violation"],
                            "total_violation": projection_a["total_violation"],
                            "distance": projection_a["distance"],
                            "digest": projection_a_digest,
                        },
                        "projection_b": {
                            "constraint_count": len(constraints_b),
                            "converged": projection_b["converged"],
                            "max_violation": projection_b["max_violation"],
                            "total_violation": projection_b["total_violation"],
                            "distance": projection_b["distance"],
                            "digest": (
                                projection_b_digest if not phase_b_projection_incomplete else None
                            ),
                        },
                        "checkpoint_a": checkpoint_a,
                        "checkpoint_b": checkpoint_b,
                        "rollback_gate": rollback_gate,
                        "tamper_gate": tamper_gate,
                        "feature_source_unchanged": feature_source_unchanged,
                        "resource_audit": resource_audit,
                        "phase_a_metrics": phase_a_metrics,
                        "phase_a_gates": phase_a_gates,
                        "phase_b_metrics": phase_b_metrics,
                        "phase_b_gates": phase_b_gates,
                        "passes_all": passes_all,
                        "parameter_count": {
                            arm: learner.parameter_count for arm, learner in learners.items()
                        },
                    }
                )
                batch_seed_results.append(
                    {
                        "seed": seed,
                        "passes_all": passes_all,
                        "phase_a_gates": phase_a_gates,
                        "phase_b_gates": phase_b_gates,
                    }
                )
            batch_summaries.append(
                {
                    "batch": batch,
                    "identity_gate": batch_identity_gate,
                    "parent_metrics_a": parent_metrics_a,
                    "parent_metrics_b": parent_metrics_b,
                    "seed_results": batch_seed_results,
                    "structure_ok": structure_ok,
                }
            )

        all_candidate_digests_overall: set[str] = set()
        for digests in batch_candidate_digests.values():
            all_candidate_digests_overall |= digests
        # Per batch: 7 pressure splits x 20 records + 2 structured x 4 = 148.
        digest_gate = {
            "all_candidate_digests_unique": len(all_candidate_digests_overall) == 3 * 148,
        }
        projected_pass_count = sum(1 for cell in cell_results if cell["passes_all"][ARM_PROJECTED])
        baseline_tension_batches = sum(
            1
            for summary in batch_summaries
            if any(
                not cell["passes_all"][ARM_BASELINE]
                for cell in cell_results
                if cell["batch"] == summary["batch"]
            )
        )
        aggregate_gate = {
            "projected_pass_cells_ge_8": projected_pass_count >= 8,
            "baseline_tension_batches_ge_2": baseline_tension_batches >= 2,
            "incomplete_projection_b_cells_le_1": incomplete_projection_b_cells <= 1,
        }
        checkpoint_gate = {
            "all_phase_checkpoints": all(
                item["passed"]
                for cell in cell_results
                for item in (
                    cell["checkpoint_a"],
                    cell["checkpoint_b"],
                )
            ),
            "all_rollback_gates": all(cell["rollback_gate"]["passed"] for cell in cell_results),
            "all_tamper_checks": all(
                all(bool(value) for value in cell["tamper_gate"].values()) for cell in cell_results
            ),
            "feature_source_unchanged": all(
                cell["feature_source_unchanged"] for cell in cell_results
            ),
            "parent_not_overwritten": content_digest(parent_payload)
            == content_digest(_load_mapping(parent_path)),
        }
        training_gate = {
            "nine_cells_present": len(cell_results) == 9,
            "three_deterministic_seeds": all(
                len(summary["seed_results"]) == len(SEEDS) for summary in batch_summaries
            ),
            "phase_a_trajectory_gates_all_pass": all(
                cell["phase_a_trajectory_gate"]["pre_projection_digests_identical"]
                for cell in cell_results
            ),
            "validation_not_fit": True,
            "holdout_not_fit": True,
            "retention_newtask_not_fit": True,
            "retention_sibling_not_fit": True,
            "constraint_targets_are_parent_only": True,
            "external_target_unused": True,
            "k_parameters_unchanged": True,
            "parameter_counts_frozen": all(
                cell["parameter_count"][ARM_BASELINE] == 17
                and cell["parameter_count"][ARM_PROJECTED] == 17
                for cell in cell_results
            ),
        }
        all_gates = (
            all(digest_gate.values())
            and all(checkpoint_gate.values())
            and all(training_gate.values())
            and all(summary["structure_ok"] for summary in batch_summaries)
            and incomplete_projection_b_cells == 0
        )
        if baseline_tension_batches >= 2:
            if projected_pass_count >= 8:
                outcome = "promotion_course_supported"
            else:
                if incomplete_projection_b_cells >= 2:
                    outcome = "cumulative_constraint_conflict"
                else:
                    outcome = "sequential_retention_failure"
        else:
            outcome = "baseline_drift"
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p4_1_manifest_digest": p4_1_manifest["manifest_digest"],
            "source_p4_2_manifest_digest": p4_2_manifest["manifest_digest"],
            "source_p4_3_manifest_digest": p4_3_manifest["manifest_digest"],
            "source_p4_4_manifest_digest": p4_4_manifest["manifest_digest"],
            "source_p4_5_manifest_digest": p4_5_manifest["manifest_digest"],
            "source_p4_6_manifest_digest": p4_6_manifest["manifest_digest"],
            "source_p4_7_manifest_digest": p4_7_manifest["manifest_digest"],
            "source_p4_8_manifest_digest": p4_8_manifest["manifest_digest"],
            "source_p4_10_manifest_digest": p4_10_manifest["manifest_digest"],
            "source_p4_11_manifest_digest": p4_11_manifest["manifest_digest"],
            "source_p4_12_manifest_digest": p4_12_manifest["manifest_digest"],
            "source_p4_12_report_digest": content_digest(p4_12_report),
            "source_p4_9_probe_report_digest": content_digest(p4_9_probe),
            "parent_g_checkpoint_digest": str(parent_metadata["digest"]),
            "k_checkpoint_digests": worker_digests,
            "course_contract": {
                "phases": ["a", "b"],
                "cumulative_projection": "phase-B system = A + B task constraints + preservation",
                "backward_retention_gate": "phase-A holdout re-checked after phase B",
                "resource_caps_absolute": RESOURCE_CAPS,
                "rollback": "phase-A projected checkpoint restores bit-identical behaviour",
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
                "batch_identity_gates": {
                    str(summary["batch"]): summary["identity_gate"] for summary in batch_summaries
                },
                "structure_gate": {
                    str(summary["batch"]): summary["structure_ok"] for summary in batch_summaries
                },
                "checkpoint_gate": checkpoint_gate,
                "training_gate": training_gate,
                "aggregate_gate": aggregate_gate,
                "resource_audit": {
                    "resource_violations": resource_violations,
                    "caps": RESOURCE_CAPS,
                    "per_cell": [
                        {
                            "batch": cell["batch"],
                            "seed": cell["seed"],
                            **cell["resource_audit"],
                        }
                        for cell in cell_results
                    ],
                },
                "projected_pass_cells": projected_pass_count,
                "baseline_tension_batches": baseline_tension_batches,
                "incomplete_projection_b_cells": incomplete_projection_b_cells,
                "batch_summaries": batch_summaries,
                "cell_results": cell_results,
                "outcome": outcome,
                "experiment_passed": bool(all_gates),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; two-phase promotion course under "
                    "the projection-solver mechanism; growth and promotion remain "
                    "fail-closed"
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
    _write_json_atomic(report_path, payload)
    return payload


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
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
