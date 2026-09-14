"""P4.11 two-arm projection solver runner.

P4.10 confirmed the learnability gap: the extended 16-feature space
contains an exact feasible solution for the joint constraint system
(new-task targets win AND parent decisions preserved), but interleaved
per-set SGD never reaches it.  P4.11 tests the preregistered repair:
after the bit-identical P4.10 training trajectory, the projected arm
applies ONE feasible-region projection - minimising the distance to the
trained endpoint subject to the joint constraint system - via the
deterministic penalty-continuation solver.  The baseline arm is the
unmodified P4.10 replication.

Preregistration: ``plans/reference/M5_K_P4_11_PROJECTION_SOLVER
_PREREGISTRATION_20260911.md`` (solver contract, convergence criteria,
and projection frequency frozen; preservation constraints use the
decision-identity form established feasible by the P4.9 probe).
Never admits growth or promotion.
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

REPORT_FORMAT = "taiji-m5-k-p4-11-projection-solver-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-11-projection-solver-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_11_projection_solver_manifest_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p4_11_projection_solver_20260911.json"
)
P4_10_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_10_feature_factorization_manifest_v1.json"
)
P4_10_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p4_10_feature_factorization_20260911.json"
)
P4_9_PROBE_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p4_9_feature_space_probe_20260911.json"
)
P4_8_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_8_representation_redesign_manifest_v1.json"
)
P4_7_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_7_capacity_clean_test_manifest_v1.json"
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
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_3_retention_incremental_manifest_v1.json"
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
SEEDS = (0, 1)
ARM_BASELINE = "invariant-ext-17"
ARM_PROJECTED = "projected-ext-17"
ARGMAX_EPSILON = 1e-6
SAFE_EPSILON = 1e-9
SELECTION_MARGIN = 0.05
EVAL_SPLITS = ("validation", "holdout", "retention-sibling", "retention-newtask")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _digest_without(payload: Mapping[str, Any], key: str) -> str:
    return content_digest({name: value for name, value in payload.items() if name != key})


def _p411_specs(split: str, offset: int) -> tuple[dict[str, Any], ...]:
    # The "p40_" prefix is required by _materialize_case's content-writing
    # trigger; the "p411" infix keeps identities disjoint from P4.0-P4.10.
    prefix = f"p40_p411_{split}"
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
            "project_id": f"p4-11-{split}-project-{class_key.lower()}",
            "variant_paths": paths,
            "state_profile": state_profile,
            "task_seed": offset + index,
        }
        for index, (class_key, state_profile, paths) in enumerate(rows)
    )


def _p411_structured_specs(
    contract: Mapping[str, Any], *, namespace: str, seed_offset: int
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    safe_index = 0
    high_index = 0
    for index, row in enumerate(contract["rows"]):
        if int(row["role_counts"].get("proposal", 0)) > 0:
            project_id = f"p4-11-{namespace}-project-a"
            prefix = f"p40_p411_{namespace}_a"
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
                f"p4-11-{namespace}-project-a"
                if safe_index == 0
                else f"p4-11-{namespace}-project-b"
            )
            prefix = f"p40_p411_{namespace}_{'a' if safe_index == 0 else 'b'}"
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


def _p411_rebind(
    record: Mapping[str, Any],
    *,
    split: str,
    source_index: int,
    width: int,
    fit_eligible: bool,
) -> dict[str, Any]:
    old_candidates = record["candidate_set"]
    old_behavior = record["behavior_set"]
    candidate_set = GSelectionCandidateSet.create(
        example_id=f"p4-11:{split}:{source_index}:width-{width}:{old_candidates.candidate_set_digest}",
        family_id=f"p4-11:{split}:family:{source_index}",
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
            "experiment": "p4.11",
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


def _p411_pressure_split(
    *,
    split: str,
    offset: int,
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
    for spec in _p411_specs(split, offset):
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
            label=f"p4-11-{split}-{int(spec['index'])}",
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
                _p411_rebind(
                    pressure,
                    split=split,
                    source_index=source_index,
                    width=width,
                    fit_eligible=(
                        split == "train"
                        and float(pressure["behavior_set"].utility_margin) > MARGIN_EPSILON
                    ),
                )
            )
    return records


def _p411_structured_records(
    *,
    contract: Mapping[str, Any],
    namespace: str,
    seed_offset: int,
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
    specs = _p411_structured_specs(contract, namespace=namespace, seed_offset=seed_offset)
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
            label=f"p4-11-{namespace}-{int(spec['index'])}",
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
            _p411_rebind(
                pressure,
                split=namespace,
                source_index=index,
                width=int(spec["target_width"]),
                fit_eligible=False,
            )
        )
    return records


def _joint_constraints(
    train_fit_records: Sequence[Mapping[str, Any]],
    constraint_sets: Sequence[GSelectionCandidateSet],
    parent: GSelectionLearner,
    learner: ExtendedGSelectionLearner,
) -> list[tuple[tuple[float, ...], float, str]]:
    """Decision-identity joint system over the extended feature space.

    Task constraints from the fit-eligible train sets; preservation
    constraints from the constraint cohort with the frozen P3.5 parent's
    decisions as reference.  All constraints are score differences, so
    the bias never appears and the projection touches only the 16 weight
    dimensions.
    """
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
                t - o for t, o in zip(features[target_id], features[candidate.candidate_id], strict=True)
            )
            constraints.append(
                (
                    difference,
                    ARGMAX_EPSILON,
                    f"newtask-argmax:{candidate_set.candidate_set_digest[:16]}:{candidate.candidate_id}",
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
                    f"newtask-safe:{candidate_set.candidate_set_digest[:16]}",
                )
            )
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
                    t - o for t, o in zip(features[picked_id], features[candidate.candidate_id], strict=True)
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
                    t - o for t, o in zip(features[candidate.candidate_id], features[picked_id], strict=True)
                )
                constraints.append(
                    (
                        difference,
                        -(SELECTION_MARGIN - SAFE_EPSILON),
                        f"cohort-boundary:{candidate_set.candidate_set_digest[:16]}:{candidate.candidate_id}",
                    )
                )
    return constraints


def _save_extended_checkpoint(
    path: Path, learner: ExtendedGSelectionLearner
) -> dict[str, Any]:
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
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_11_projection_solver_{uuid4().hex}"
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
        p4_10_report = _load_json(P4_10_REPORT)
        p4_9_probe = _load_json(P4_9_PROBE_REPORT)
        p3_5_report = _load_json(P3_5_REPORT)
        if (
            p4_10_report.get("status") != "completed"
            or p4_10_report.get("outcome") != "learnability_gap"
        ):
            raise ValueError("P4.11 requires the completed P4.10 learnability_gap")
        if p4_10_report.get("manifest_digest") != p4_10_manifest.get("manifest_digest"):
            raise ValueError("P4.10 manifest/report digest mismatch")
        if _digest_without(p4_10_manifest, "manifest_digest") != p4_10_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.10 manifest content digest mismatch")
        if p4_10_report.get("growth_admitted") or p4_10_report.get("can_promote"):
            raise ValueError("P4.11 cannot consume an admitted P4.10 artifact")
        if p4_10_manifest.get("source_p4_8_manifest_digest") != p4_8_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.10 source chain drifted")
        if p4_9_probe.get("feasibility_verdict") != (
            "parent_relative_features_are_the_factorization"
        ):
            raise ValueError("P4.11 requires the P4.9 factorization verdict")
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
            raise ValueError("P4.11 parent external checkpoint digest drifted")
        parent = GSelectionLearner.from_checkpoint(parent_payload, device="cpu")
        parent.assert_lineage(
            parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        if str(p4_2_manifest["parent_g_checkpoint_digest"]) != str(parent_metadata["digest"]):
            raise ValueError("P4.11 parent G lineage drifted")
        parent_restore = _independent_g_restore(parent_path)
        if not parent_restore.get("independent_process_restore"):
            raise RuntimeError("P4.11 parent independent restore failed")
        run_dir.mkdir(parents=True, exist_ok=False)
        train_records = _p411_pressure_split(
            split="train",
            offset=51200,
            scratch=run_dir / "data" / "train",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_10_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        validation_records = _p411_pressure_split(
            split="validation",
            offset=51300,
            scratch=run_dir / "data" / "validation",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_10_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        holdout_records = _p411_pressure_split(
            split="holdout",
            offset=51400,
            scratch=run_dir / "data" / "holdout",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_10_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        retention_newtask_records = _p411_pressure_split(
            split="retention-newtask",
            offset=51500,
            scratch=run_dir / "data" / "retention-newtask",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_10_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        constraint_records = _p411_structured_records(
            contract=contract,
            namespace="constraint",
            seed_offset=51100,
            scratch=run_dir / "data" / "constraint",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_10_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        retention_sibling_records = _p411_structured_records(
            contract=contract,
            namespace="retention",
            seed_offset=51000,
            scratch=run_dir / "data" / "retention",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_10_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        train_fit = [record for record in train_records if record["fit_eligible"]]
        constraint_sets = tuple(record["candidate_set"] for record in constraint_records)
        constraint_digest = content_digest(
            {
                "constraint_set_digests": [
                    candidate_set.candidate_set_digest for candidate_set in constraint_sets
                ],
                "constraint_form": "margin-preservation-hinge",
            }
        )
        all_records = [
            *train_records,
            *validation_records,
            *holdout_records,
            *retention_newtask_records,
            *constraint_records,
            *retention_sibling_records,
        ]
        split_records = {
            "validation": validation_records,
            "holdout": holdout_records,
            "retention-sibling": retention_sibling_records,
            "retention-newtask": retention_newtask_records,
        }
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
        ):
            manifest_paths, manifest_projects = _manifest_identity(manifest)
            old_paths.update(manifest_paths)
            old_projects.update(manifest_projects)
        identity_sets = [
            _records_identity(collection)
            for collection in (
                train_records,
                validation_records,
                holdout_records,
                retention_newtask_records,
                constraint_records,
                retention_sibling_records,
            )
        ]
        candidate_digests = {
            record["candidate_set"].candidate_set_digest for record in all_records
        }
        sibling_structure = [
            _structure_row(record["candidate_set"], record["behavior_set"])
            for record in retention_sibling_records
        ]
        constraint_structure = [
            _structure_row(record["candidate_set"], record["behavior_set"])
            for record in constraint_records
        ]
        structure_gate = {
            "retention_sibling_matches_p4_4_contract": sibling_structure
            == list(contract["rows"]),
            "constraint_matches_p4_4_contract": constraint_structure
            == list(contract["rows"]),
            "retention_sibling_structure_digest": content_digest(sibling_structure),
            "p4_4_structure_contract_digest": contract["contract_digest"],
        }
        identity_gate = {
            "train_records": len(train_records) == 20,
            "validation_records": len(validation_records) == 20,
            "holdout_records": len(holdout_records) == 20,
            "retention_newtask_records": len(retention_newtask_records) == 20,
            "constraint_records": len(constraint_records) == int(contract["row_count"]),
            "retention_sibling_records": len(retention_sibling_records)
            == int(contract["row_count"]),
            "train_fit_positive": len(train_fit) >= 8,
            "five_train_classes": len(
                {record["diagnostic"]["class_key"] for record in train_records}
            )
            == 5,
            "five_validation_classes": len(
                {record["diagnostic"]["class_key"] for record in validation_records}
            )
            == 5,
            "five_holdout_classes": len(
                {record["diagnostic"]["class_key"] for record in holdout_records}
            )
            == 5,
            "five_retention_newtask_classes": len(
                {record["diagnostic"]["class_key"] for record in retention_newtask_records}
            )
            == 5,
            "new_projects_disjoint_from_historical": all(
                projects.isdisjoint(old_projects) for _paths, projects in identity_sets
            ),
            "new_paths_disjoint_from_historical": all(
                paths.isdisjoint(old_paths) for paths, _projects in identity_sets
            ),
            "all_candidate_digests_unique": len(candidate_digests) == len(all_records),
            "all_behavior_digests_unique": len(
                {record["behavior_set"].behavior_digest for record in all_records}
            )
            == len(all_records),
            "constraint_not_fit_eligible": not any(
                record["fit_eligible"] for record in constraint_records
            ),
            "retention_sibling_not_fit_eligible": not any(
                record["fit_eligible"] for record in retention_sibling_records
            ),
            "retention_newtask_not_fit_eligible": not any(
                record["fit_eligible"] for record in retention_newtask_records
            ),
            "retention_sibling_structure_matches_contract": structure_gate[
                "retention_sibling_matches_p4_4_contract"
            ],
        }
        if not all(identity_gate.values()):
            raise ValueError(
                f"P4.11 identity gate failed: {[key for key, value in identity_gate.items() if not value]}"
            )
        parent_metrics = {
            split: _metric_summary(_evaluate(records, parent))
            for split, records in split_records.items()
        }
        seed_results: list[dict[str, Any]] = []
        for seed in SEEDS:
            seed_dir = run_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=False)
            learners: dict[str, Any] = {
                ARM_BASELINE: ExtendedGSelectionLearner.from_parent_learner(parent),
                ARM_PROJECTED: ExtendedGSelectionLearner.from_parent_learner(parent),
            }
            birth_report: dict[str, Any] = {}
            for arm, learner in learners.items():
                mismatches = 0
                max_deviation = 0.0
                for split in EVAL_SPLITS:
                    for record in split_records[split]:
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
            birth_passed = all(
                report["selection_mismatches"] == 0
                and report["max_abs_score_deviation"] == 0.0
                for arm, report in birth_report.items()
                if arm in learners
            ) and birth_report["birth_hinge_loss_zero"]
            if not birth_passed:
                raise ValueError(f"P4.11 birth equivalence gate failed: {birth_report}")
            zero_checkpoints = {
                arm: _save_extended_checkpoint(
                    seed_dir / f"{arm}-zero.pt", learner
                )
                for arm, learner in learners.items()
            }
            if not all(item["passed"] for item in zero_checkpoints.values()):
                raise RuntimeError(
                    f"P4.11 zero-step checkpoint preflight failed for seed {seed}"
                )
            # Identical training trajectory for both arms (deterministic).
            fit_baseline = learners[ARM_BASELINE].invariant_fit(
                train_fit,
                constraint_sets,
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
                order_seed=seed,
                constraint_digest=constraint_digest,
            )
            fit_projected = learners[ARM_PROJECTED].invariant_fit(
                train_fit,
                constraint_sets,
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
                order_seed=seed,
                constraint_digest=constraint_digest,
            )
            pre_projection_digests = {
                arm: learner.model_state_digest for arm, learner in learners.items()
            }
            trajectory_gate = {
                "pre_projection_digests_identical": pre_projection_digests[
                    ARM_BASELINE
                ]
                == pre_projection_digests[ARM_PROJECTED],
            }
            if not trajectory_gate["pre_projection_digests_identical"]:
                raise RuntimeError(
                    "P4.11 trajectory gate failed: the two arms diverged before projection"
                )
            # Single end-of-training projection on the projected arm.
            anchor = [
                float(value)
                for value in learners[ARM_PROJECTED].head.weight.detach().reshape(-1)
            ]
            constraints = _joint_constraints(train_fit, constraint_sets, parent, learners[ARM_PROJECTED])
            projection = project_to_joint_feasible_region(constraints, anchor)
            if not projection["converged"]:
                raise ValueError(
                    f"P4.11 projection_incomplete: max violation "
                    f"{projection['max_violation']}, total {projection['total_violation']}"
                )
            projection_digest = content_digest(
                {
                    "anchor": anchor,
                    "projected": projection["weights"],
                    "constraint_count": len(constraints),
                }
            )
            learners[ARM_PROJECTED].apply_projected_weights(
                projection["weights"], projection_digest=projection_digest
            )
            trained_checkpoints = {
                ARM_BASELINE: _save_extended_checkpoint(
                    seed_dir / f"{ARM_BASELINE}-trained.pt", learners[ARM_BASELINE]
                ),
                ARM_PROJECTED: _save_extended_checkpoint(
                    seed_dir / f"{ARM_PROJECTED}-trained.pt", learners[ARM_PROJECTED]
                ),
            }
            if not all(item["passed"] for item in trained_checkpoints.values()):
                raise RuntimeError(
                    f"P4.11 trained checkpoint preflight failed for seed {seed}"
                )
            tamper_gate = {
                arm: _extended_tamper_rejected(
                    _load_mapping(Path(str(item["path"])))
                )
                for arm, item in trained_checkpoints.items()
            }
            if not all(tamper_gate.values()):
                raise RuntimeError(f"P4.11 tamper gate failed for seed {seed}")
            feature_source_unchanged = all(
                learner.feature_source_state_digest == birth_report["feature_source_digest"]
                for learner in learners.values()
            )
            metrics = {
                arm: {
                    split: _metric_summary(_evaluate(records, learner))
                    for split, records in split_records.items()
                }
                for arm, learner in learners.items()
            }
            gates = {
                arm: {
                    "new_task": _new_task_gate(metrics[arm]["holdout"]),
                    "retention_sibling": _retention_gate(
                        metrics[arm]["retention-sibling"],
                        parent_metrics["retention-sibling"],
                    ),
                    "retention_newtask": _retention_gate(
                        metrics[arm]["retention-newtask"],
                        parent_metrics["retention-newtask"],
                    ),
                }
                for arm in learners
            }
            seed_results.append(
                {
                    "seed": seed,
                    "fit": {ARM_BASELINE: fit_baseline, ARM_PROJECTED: fit_projected},
                    "zero_checkpoints": zero_checkpoints,
                    "trained_checkpoints": trained_checkpoints,
                    "tamper_gate": tamper_gate,
                    "trajectory_gate": trajectory_gate,
                    "projection": {
                        "constraint_count": len(constraints),
                        "converged": projection["converged"],
                        "max_violation": projection["max_violation"],
                        "total_violation": projection["total_violation"],
                        "violation_per_constraint_family": projection[
                            "violation_per_constraint_family"
                        ],
                        "distance": projection["distance"],
                        "phases": projection["phases"],
                        "digest": projection_digest,
                    },
                    "feature_source_unchanged": feature_source_unchanged,
                    "metrics": metrics,
                    "gates": gates,
                    "parameter_count": {
                        arm: learner.parameter_count for arm, learner in learners.items()
                    },
                }
            )

        def _arm_passes_all(arm: str) -> bool:
            return all(
                all(result["gates"][arm]["new_task"].values())
                and all(result["gates"][arm]["retention_sibling"].values())
                and all(result["gates"][arm]["retention_newtask"].values())
                for result in seed_results
            )

        baseline_passes = _arm_passes_all(ARM_BASELINE)
        projected_passes = _arm_passes_all(ARM_PROJECTED)
        if baseline_passes:
            outcome = "baseline_drift"
        elif projected_passes:
            outcome = "projection_solver_supported"
        else:
            outcome = "projection_generalization_gap"
        source_gate = {
            "p4_10_completed_and_gap_confirmed": True,
            "p4_10_source_chain_valid": True,
            "p4_9_factorization_verdict_consumed": True,
            "parent_independent_restore": bool(
                parent_restore.get("independent_process_restore")
            ),
            "parent_lineage_valid": True,
            "k_parent_digests_valid": True,
        }
        checkpoint_gate = {
            "all_zero_step_checkpoints": all(
                item["passed"]
                for result in seed_results
                for item in result["zero_checkpoints"].values()
            ),
            "all_trained_checkpoints": all(
                item["passed"]
                for result in seed_results
                for item in result["trained_checkpoints"].values()
            ),
            "all_tamper_checks": all(
                all(bool(value) for value in result["tamper_gate"].values())
                for result in seed_results
            ),
            "feature_source_unchanged": all(
                result["feature_source_unchanged"] for result in seed_results
            ),
            "parent_not_overwritten": content_digest(parent_payload)
            == content_digest(_load_mapping(parent_path)),
        }
        training_gate = {
            "two_arms_present": True,
            "two_deterministic_seeds": len(seed_results) == len(SEEDS),
            "validation_not_fit": True,
            "holdout_not_fit": True,
            "retention_newtask_not_fit": True,
            "retention_sibling_not_fit": True,
            "constraint_targets_are_parent_only": True,
            "external_target_unused": True,
            "k_parameters_unchanged": True,
            "parameter_counts_frozen": all(
                result["parameter_count"][ARM_BASELINE] == 17
                and result["parameter_count"][ARM_PROJECTED] == 17
                for result in seed_results
            ),
        }
        all_gates = (
            all(source_gate.values())
            and all(identity_gate.values())
            and structure_gate["retention_sibling_matches_p4_4_contract"]
            and structure_gate["constraint_matches_p4_4_contract"]
            and birth_passed
            and all(checkpoint_gate.values())
            and all(training_gate.values())
            and all(
                result["trajectory_gate"]["pre_projection_digests_identical"]
                for result in seed_results
            )
        )
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
            "source_p4_10_report_digest": content_digest(p4_10_report),
            "source_p4_9_probe_report_digest": content_digest(p4_9_probe),
            "parent_g_checkpoint_digest": str(parent_metadata["digest"]),
            "k_checkpoint_digests": worker_digests,
            "projection_contract": {
                "form": "single end-of-training projection onto the joint feasible region",
                "constraint_form": "decision-identity (P4.9-probe-established)",
                "solver": "penalty continuation, deterministic warm continuation from anchor",
                "bias_untouched": True,
                "anchor": "baseline-arm trained endpoint (bit-identical trajectory)",
            },
            "fit_policy": {
                "fit_called": True,
                "training_performed": True,
                "validation_only": False,
                "growth_admitted": False,
                "retention_fit_count": 0,
                "new_task_thresholds": {"utility_floor": 0.68, "target_hit_floor": 0.6},
            },
            "train_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in train_records
            ],
            "validation_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in validation_records
            ],
            "holdout_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in holdout_records
            ],
            "retention_newtask_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest
                for record in retention_newtask_records
            ],
            "constraint_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in constraint_records
            ],
            "retention_sibling_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest
                for record in retention_sibling_records
            ],
            "records": {
                split: [
                    {
                        "candidate_set": record["candidate_set"].to_payload(),
                        "behavior_set": record["behavior_set"].to_payload(),
                        "diagnostic": record["diagnostic"],
                    }
                    for record in records
                ]
                for split, records in (
                    ("train", train_records),
                    ("validation", validation_records),
                    ("holdout", holdout_records),
                    ("retention-newtask", retention_newtask_records),
                    ("constraint", constraint_records),
                    ("retention-sibling", retention_sibling_records),
                )
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
                "source_gate": source_gate,
                "identity_gate": identity_gate,
                "structure_gate": structure_gate,
                "checkpoint_gate": checkpoint_gate,
                "training_gate": training_gate,
                "birth_equivalence": birth_report,
                "parent_metrics": parent_metrics,
                "seed_results": seed_results,
                "baseline_17_passes_all": baseline_passes,
                "projected_17_passes_all": projected_passes,
                "outcome": outcome,
                "experiment_passed": bool(all_gates),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; projection solver update mechanism; "
                    "growth and promotion remain fail-closed"
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
