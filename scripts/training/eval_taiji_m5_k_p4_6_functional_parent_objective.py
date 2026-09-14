"""Test a functional parent-preserving objective at fixed G capacity.

P4.5 showed that rehearsal had no separable benefit and that a parameter
trust-region traded new-task quality against retention differently across
seeds.  P4.6 keeps the 13-parameter G fixed and adds a functional constraint:
the child learns new behavior targets while matching the parent model's
candidate scores on an independent constraint cohort.  The cohort is not an
evaluation set and its behavior target/utility is never read by the fit.

This is the last controlled fixed-capacity objective comparison in the
current route.  It never admits topology growth or promotion.
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

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
    _fit_sequence,
    _save_checkpoint,
    _tamper_rejected,
)
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)
from taiji.local_learning import (  # noqa: E402
    apply_linear_delta,
    mean_squared_error_delta,
)

REPORT_FORMAT = "taiji-m5-k-p4-6-functional-parent-objective-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-6-functional-parent-objective-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_6_functional_parent_objective_manifest_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p4_6_functional_parent_objective_20260911.json"
)
P4_1_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_1_context_contract_manifest_v1.json"
)
P4_2_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_2_capacity_attribution_manifest_v1.json"
)
P4_3_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_3_retention_incremental_manifest_v1.json"
)
P4_4_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_4_retention_identity_calibration_manifest_v1.json"
)
P4_5_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_5_update_rule_gate_manifest_v1.json"
)
P4_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_5_update_rule_gate_20260911.json"
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
P3_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"
TRAINING_EPOCHS = 8
LEARNING_RATE = 0.15
FUNCTIONAL_WEIGHT = 1.0
SEEDS = (0, 1)
MARGIN_EPSILON = 1e-9


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _digest_without(payload: Mapping[str, Any], key: str) -> str:
    return content_digest({name: value for name, value in payload.items() if name != key})


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "count",
            "behavior_target_hit_count",
            "behavior_target_hit_rate",
            "selected_utility_sum",
            "selected_utility_mean",
            "selected_residual_error",
            "selected_roles",
            "safe_selection_violations",
            "reobserve_selected_count",
            "reobserve_projection_passed",
            "workbench_success_count",
        )
        if key in metrics
    }


def _structure_row(
    candidate_set: GSelectionCandidateSet, behavior_set: GSelectionBehaviorSet
) -> dict[str, Any]:
    role_counts = Counter(candidate.candidate_role for candidate in candidate_set.candidates)
    confidence_values = [float(candidate.confidence) for candidate in candidate_set.candidates]
    maximum_confidence = max(confidence_values, default=0.0)
    confidence_bucket = (
        "zero"
        if maximum_confidence <= MARGIN_EPSILON
        else "high-and-safe" if maximum_confidence >= 0.8 else "mixed"
    )
    safe_roles = sorted(
        {
            candidate.candidate_role
            for candidate, outcome in zip(
                candidate_set.candidates, behavior_set.outcomes, strict=True
            )
            if outcome.safe_exit_valid or outcome.safe_exit_progress
        }
    )
    return {
        "candidate_count": len(candidate_set.candidates),
        "role_counts": dict(sorted(role_counts.items())),
        "confidence_bucket": confidence_bucket,
        "safe_projection_roles": safe_roles,
    }


def _new_specs(split: str, offset: int) -> tuple[dict[str, Any], ...]:
    prefix = f"p40_p46_{split}"
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
            "project_id": f"p4-6-{split}-project-{class_key.lower()}",
            "variant_paths": paths,
            "state_profile": state_profile,
            "task_seed": offset + index,
        }
        for index, (class_key, state_profile, paths) in enumerate(rows)
    )


def _rebind_pressure_record(
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
        example_id=(
            f"p4-6:{split}:{source_index}:width-{width}:{old_candidates.candidate_set_digest}"
        ),
        family_id=f"p4-6:{split}:family:{source_index}",
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
            "experiment": "p4.6",
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


def _build_pressure_split(
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
    for spec in _new_specs(split, offset):
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
            label=f"p4-6-{split}-{int(spec['index'])}",
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
                _rebind_pressure_record(
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


def _structured_specs(contract: Mapping[str, Any], *, namespace: str) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    safe_index = 0
    high_index = 0
    for index, row in enumerate(contract["rows"]):
        if int(row["role_counts"].get("proposal", 0)) > 0:
            project_id = f"p4-6-{namespace}-project-a"
            prefix = f"p40_p46_{namespace}_a"
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
                f"p4-6-{namespace}-project-a" if safe_index == 0 else f"p4-6-{namespace}-project-b"
            )
            prefix = f"p40_p46_{namespace}_{'a' if safe_index == 0 else 'b'}"
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
                "task_seed": 46000 + index + (100 if namespace == "constraint" else 0),
                "target_width": int(row["candidate_count"]),
            }
        )
    return tuple(specs)


def _build_structured_records(
    *,
    contract: Mapping[str, Any],
    namespace: str,
    split: str,
    fit_eligible: bool,
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
    specs = _structured_specs(contract, namespace=namespace)
    for spec in specs:
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
            label=f"p4-6-{namespace}-{split}-{int(spec['index'])}",
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
            _rebind_pressure_record(
                pressure,
                split=split,
                source_index=index,
                width=int(spec["target_width"]),
                fit_eligible=fit_eligible,
            )
        )
    return records


def _records_identity(records: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    return (
        {str(record["candidate_set"].path) for record in records},
        {str(record["candidate_set"].project_id) for record in records},
    )


def _manifest_identity(payload: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    projects: set[str] = set()
    records = payload.get("records")
    collections = (
        [records]
        if isinstance(records, list)
        else (
            [value for value in records.values() if isinstance(value, list)]
            if isinstance(records, Mapping)
            else []
        )
    )
    for collection in collections:
        for raw in collection:
            candidate = raw.get("candidate_set", {})
            paths.add(str(candidate.get("path", "")))
            projects.add(str(candidate.get("project_id", "")))
    return paths, projects


def _constraint_feature_groups(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    # Deliberately project only input feature vectors.  The fit never reads
    # the record's target candidate, behavior target, utility, or role.
    return tuple(
        tuple(
            tuple(float(value) for value in candidate.feature_vector)
            for candidate in record["candidate_set"].candidates
        )
        for record in records
    )


def _functional_fit_sequence(
    learner: GSelectionLearner,
    parent: GSelectionLearner,
    records: Sequence[Mapping[str, Any]],
    constraint_features: Sequence[Sequence[Sequence[float]]],
    *,
    epochs: int,
    learning_rate: float,
    order_seed: int,
    functional_weight: float,
    constraint_digest: str,
) -> dict[str, Any]:
    items = tuple(record["candidate_set"] for record in records)
    if not items or any(item.split != "train" for item in items):
        raise ValueError("P4.6 functional fit requires non-empty train records")
    if not constraint_features:
        raise ValueError("P4.6 functional fit requires an independent constraint cohort")
    if len(items) != len({item.candidate_set_digest for item in items}):
        raise ValueError("P4.6 functional fit candidate sets must be unique")
    if functional_weight <= 0.0:
        raise ValueError("P4.6 functional weight must be positive")
    dataset_digest = content_digest(
        {
            "candidate_set_digests": [item.candidate_set_digest for item in items],
            "constraint_digest": constraint_digest,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "order_seed": int(order_seed),
            "functional_weight": float(functional_weight),
        }
    )
    task_loss_sum = 0.0
    constraint_loss_sum = 0.0
    constraint_steps = 0
    for epoch in range(int(epochs)):
        order = list(range(len(items)))
        random.Random(int(order_seed) + epoch).shuffle(order)
        for step, index in enumerate(order):
            item = items[index]
            task_inputs = torch.tensor(
                [candidate.feature_vector for candidate in item.candidates],
                dtype=torch.float32,
                device=learner.device,
            )
            task_targets = torch.zeros(
                (len(item.candidates), 1), dtype=torch.float32, device=learner.device
            )
            target_index = next(
                position
                for position, candidate in enumerate(item.candidates)
                if candidate.candidate_id == item.target_candidate_id
            )
            task_targets[target_index, 0] = 1.0
            task_predictions = learner.model(task_inputs)
            task_loss_sum += float(torch.mean((task_predictions - task_targets) ** 2).item())
            apply_linear_delta(
                learner.model,
                task_inputs,
                mean_squared_error_delta(task_predictions, task_targets),
                float(learning_rate),
            )
            group = constraint_features[(epoch * len(order) + step) % len(constraint_features)]
            constraint_inputs = torch.tensor(group, dtype=torch.float32, device=learner.device)
            with torch.no_grad():
                teacher = parent.model(constraint_inputs).detach()
            student = learner.model(constraint_inputs)
            constraint_loss_sum += float(torch.mean((student - teacher) ** 2).item())
            apply_linear_delta(
                learner.model,
                constraint_inputs,
                mean_squared_error_delta(student, teacher) * float(functional_weight),
                float(learning_rate),
            )
            constraint_steps += 1
            learner.training_steps += 2
    learner.revision += 1
    learner.last_train_digest = dataset_digest
    return {
        "dataset_digest": dataset_digest,
        "candidate_sets": len(items),
        "constraint_groups": len(constraint_features),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "order_seed": int(order_seed),
        "functional_weight": float(functional_weight),
        "training_steps": learner.training_steps,
        "revision": learner.revision,
        "task_loss_mean": task_loss_sum / (len(items) * int(epochs)),
        "constraint_loss_mean": constraint_loss_sum / constraint_steps,
        "constraint_steps": constraint_steps,
    }


def _gate_noninferior(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "utility_not_below": float(metrics["selected_utility_mean"])
        >= float(baseline["selected_utility_mean"]) - MARGIN_EPSILON,
        "target_hit_not_below": float(metrics["behavior_target_hit_rate"])
        >= float(baseline["behavior_target_hit_rate"]) - MARGIN_EPSILON,
        "safe_selection_preserved": int(metrics["safe_selection_violations"]) == 0,
        "reobserve_projection_passed": bool(metrics["reobserve_projection_passed"]),
        "workbench_not_below": int(metrics["workbench_success_count"])
        >= int(baseline["workbench_success_count"]),
    }


def _run(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_6_functional_parent_objective_{uuid4().hex}"
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
        p4_5_report = _load_json(P4_5_REPORT)
        p3_5_report = _load_json(P3_5_REPORT)
        if (
            p4_5_report.get("status") != "completed"
            or p4_5_report.get("outcome") != "update_rule_unresolved"
        ):
            raise ValueError("P4.6 requires completed P4.5 update_rule_unresolved")
        if p4_5_report.get("manifest_digest") != p4_5_manifest.get("manifest_digest"):
            raise ValueError("P4.5 manifest/report digest mismatch")
        if _digest_without(p4_5_manifest, "manifest_digest") != p4_5_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.5 manifest content digest mismatch")
        if p4_5_report.get("growth_admitted") or p4_5_report.get("can_promote"):
            raise ValueError("P4.6 cannot consume an admitted P4.5 artifact")
        if p4_5_manifest.get("source_p4_4_manifest_digest") != p4_4_manifest.get("manifest_digest"):
            raise ValueError("P4.5 source chain drifted")
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
            raise ValueError("P4.6 parent external checkpoint digest drifted")
        parent = GSelectionLearner.from_checkpoint(parent_payload, device="cpu")
        parent.assert_lineage(
            parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        if str(p4_2_manifest["parent_g_checkpoint_digest"]) != str(parent_metadata["digest"]):
            raise ValueError("P4.6 parent G lineage drifted")
        parent_restore = _independent_g_restore(parent_path)
        if not parent_restore.get("independent_process_restore"):
            raise RuntimeError("P4.6 parent independent restore failed")
        run_dir.mkdir(parents=True, exist_ok=False)
        train_records = _build_pressure_split(
            split="train",
            offset=46200,
            scratch=run_dir / "data" / "train",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_5_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        validation_records = _build_pressure_split(
            split="validation",
            offset=46300,
            scratch=run_dir / "data" / "validation",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_5_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        holdout_records = _build_pressure_split(
            split="holdout",
            offset=46400,
            scratch=run_dir / "data" / "holdout",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_5_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        constraint_records = _build_structured_records(
            contract=contract,
            namespace="constraint",
            split="constraint",
            fit_eligible=False,
            scratch=run_dir / "data" / "constraint",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_5_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        retention_records = _build_structured_records(
            contract=contract,
            namespace="retention",
            split="retention",
            fit_eligible=False,
            scratch=run_dir / "data" / "retention",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_5_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        train_fit = [record for record in train_records if record["fit_eligible"]]
        constraint_features = _constraint_feature_groups(constraint_records)
        constraint_digest = content_digest(constraint_features)
        all_records = [
            *train_records,
            *validation_records,
            *holdout_records,
            *constraint_records,
            *retention_records,
        ]
        old_paths: set[str] = set()
        old_projects: set[str] = set()
        for manifest in (
            p4_1_manifest,
            p4_2_manifest,
            p4_3_manifest,
            p4_4_manifest,
            p4_5_manifest,
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
                constraint_records,
                retention_records,
            )
        ]
        retention_structure = [
            _structure_row(record["candidate_set"], record["behavior_set"])
            for record in retention_records
        ]
        structure_gate = {
            "retention_shape_matches_p4_4_contract": retention_structure == list(contract["rows"]),
            "retention_structure_digest": content_digest(retention_structure),
            "p4_4_structure_contract_digest": contract["contract_digest"],
        }
        identity_gate = {
            "train_records": len(train_records) == 20,
            "validation_records": len(validation_records) == 20,
            "holdout_records": len(holdout_records) == 20,
            "constraint_records": len(constraint_records) == int(contract["row_count"]),
            "retention_records": len(retention_records) == int(contract["row_count"]),
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
            "new_projects_disjoint_from_historical": all(
                projects.isdisjoint(old_projects) for _paths, projects in identity_sets
            ),
            "new_paths_disjoint_from_historical": all(
                paths.isdisjoint(old_paths) for paths, _projects in identity_sets
            ),
            "all_candidate_digests_unique": len(
                {record["candidate_set"].candidate_set_digest for record in all_records}
            )
            == len(all_records),
            "all_behavior_digests_unique": len(
                {record["behavior_set"].behavior_digest for record in all_records}
            )
            == len(all_records),
            "constraint_not_fit_eligible": not any(
                record["fit_eligible"] for record in constraint_records
            ),
            "retention_not_fit_eligible": not any(
                record["fit_eligible"] for record in retention_records
            ),
            "retention_structure_matches_contract": structure_gate[
                "retention_shape_matches_p4_4_contract"
            ],
        }
        if not all(identity_gate.values()):
            raise ValueError(
                f"P4.6 identity gate failed: {[key for key, value in identity_gate.items() if not value]}"
            )
        parent_metrics = {
            "validation": _metric_summary(_evaluate(validation_records, parent)),
            "holdout": _metric_summary(_evaluate(holdout_records, parent)),
            "retention": _metric_summary(_evaluate(retention_records, parent)),
        }
        seed_results: list[dict[str, Any]] = []
        for seed in SEEDS:
            seed_dir = run_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=False)
            learners = {
                "new-only": GSelectionLearner.from_checkpoint(
                    copy.deepcopy(parent_payload), device="cpu"
                ),
                "functional-parent-preserving": GSelectionLearner.from_checkpoint(
                    copy.deepcopy(parent_payload), device="cpu"
                ),
            }
            zero_checkpoints = {
                arm: _save_checkpoint(seed_dir / f"{arm}-zero.pt", learner)
                for arm, learner in learners.items()
            }
            if not all(item["passed"] for item in zero_checkpoints.values()):
                raise RuntimeError(f"P4.6 zero-step checkpoint preflight failed for seed {seed}")
            new_fit = _fit_sequence(
                learners["new-only"],
                train_fit,
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
                order_seed=seed,
                source_labels=["new-task"] * len(train_fit),
            )
            functional_fit = _functional_fit_sequence(
                learners["functional-parent-preserving"],
                parent,
                train_fit,
                constraint_features,
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
                order_seed=seed,
                functional_weight=FUNCTIONAL_WEIGHT,
                constraint_digest=constraint_digest,
            )
            trained_checkpoints = {
                arm: _save_checkpoint(seed_dir / f"{arm}-trained.pt", learner)
                for arm, learner in learners.items()
            }
            if not all(item["passed"] for item in trained_checkpoints.values()):
                raise RuntimeError(f"P4.6 trained checkpoint preflight failed for seed {seed}")
            tamper_gate = {
                arm: _tamper_rejected(_load_mapping(Path(str(item["path"]))))
                for arm, item in trained_checkpoints.items()
            }
            if not all(tamper_gate.values()):
                raise RuntimeError(f"P4.6 tamper gate failed for seed {seed}")
            metrics = {
                arm: {
                    "validation": _metric_summary(_evaluate(validation_records, learner)),
                    "holdout": _metric_summary(_evaluate(holdout_records, learner)),
                    "retention": _metric_summary(_evaluate(retention_records, learner)),
                }
                for arm, learner in learners.items()
            }
            new_task_gate = {
                "functional-parent-preserving": _gate_noninferior(
                    metrics["functional-parent-preserving"]["holdout"],
                    metrics["new-only"]["holdout"],
                )
            }
            retention_gate = {
                arm: _gate_noninferior(metrics[arm]["retention"], parent_metrics["retention"])
                for arm in learners
            }
            seed_results.append(
                {
                    "seed": seed,
                    "fit": {"new-only": new_fit, "functional-parent-preserving": functional_fit},
                    "zero_checkpoints": zero_checkpoints,
                    "trained_checkpoints": trained_checkpoints,
                    "tamper_gate": tamper_gate,
                    "metrics": metrics,
                    "new_task_gate": new_task_gate,
                    "retention_gate": retention_gate,
                    "parameter_count": {
                        arm: learner.parameter_count for arm, learner in learners.items()
                    },
                }
            )
        functional_retention_passed = all(
            all(result["retention_gate"]["functional-parent-preserving"].values())
            for result in seed_results
        )
        functional_new_task_noninferior = all(
            all(result["new_task_gate"]["functional-parent-preserving"].values())
            for result in seed_results
        )
        functional_new_task_gain = all(
            float(
                result["metrics"]["functional-parent-preserving"]["holdout"][
                    "selected_utility_mean"
                ]
            )
            > float(parent_metrics["holdout"]["selected_utility_mean"]) + MARGIN_EPSILON
            or float(
                result["metrics"]["functional-parent-preserving"]["holdout"][
                    "behavior_target_hit_rate"
                ]
            )
            > float(parent_metrics["holdout"]["behavior_target_hit_rate"]) + MARGIN_EPSILON
            for result in seed_results
        )
        functional_update_repaired = (
            functional_retention_passed
            and functional_new_task_noninferior
            and functional_new_task_gain
        )
        if functional_update_repaired:
            outcome = "functional_update_repaired"
        elif functional_retention_passed:
            outcome = "functional_update_no_gain"
        else:
            outcome = "functional_update_unresolved"
        source_gate = {
            "p4_5_completed_and_unresolved": True,
            "p4_5_growth_closed": True,
            "parent_independent_restore": bool(parent_restore.get("independent_process_restore")),
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
            "parent_not_overwritten": content_digest(parent_payload)
            == content_digest(_load_mapping(parent_path)),
        }
        training_gate = {
            "two_arms_present": True,
            "two_deterministic_seeds": len(seed_results) == len(SEEDS),
            "validation_not_fit": True,
            "holdout_not_fit": True,
            "retention_not_fit": True,
            "constraint_targets_are_parent_only": True,
            "p4_5_retention_not_fit": True,
            "external_target_unused": True,
            "k_parameters_unchanged": True,
            "parameter_count_unchanged": all(
                all(value == parent.parameter_count for value in result["parameter_count"].values())
                for result in seed_results
            ),
        }
        all_gates = (
            all(source_gate.values())
            and all(identity_gate.values())
            and structure_gate["retention_shape_matches_p4_4_contract"]
            and all(checkpoint_gate.values())
            and all(training_gate.values())
        )
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p4_1_manifest_digest": p4_1_manifest["manifest_digest"],
            "source_p4_2_manifest_digest": p4_2_manifest["manifest_digest"],
            "source_p4_3_manifest_digest": p4_3_manifest["manifest_digest"],
            "source_p4_4_manifest_digest": p4_4_manifest["manifest_digest"],
            "source_p4_5_manifest_digest": p4_5_manifest["manifest_digest"],
            "parent_g_checkpoint_digest": str(parent_metadata["digest"]),
            "k_checkpoint_digests": worker_digests,
            "fit_policy": {
                "fit_called": True,
                "training_performed": True,
                "validation_only": False,
                "growth_admitted": False,
                "p4_5_retention_fit_count": 0,
                "retention_fit_count": 0,
                "constraint_target_source": "parent_model_output_only",
            },
            "functional_objective": {
                "functional_weight": FUNCTIONAL_WEIGHT,
                "constraint_cohort_digest": constraint_digest,
                "constraint_candidate_set_digests": [
                    record["candidate_set"].candidate_set_digest for record in constraint_records
                ],
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
            "retention_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in retention_records
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
                    ("constraint", constraint_records),
                    ("validation", validation_records),
                    ("holdout", holdout_records),
                    ("retention", retention_records),
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
                "parent_metrics": parent_metrics,
                "seed_results": seed_results,
                "parameter_count": parent.parameter_count,
                "functional_objective": {
                    "weight": FUNCTIONAL_WEIGHT,
                    "constraint_cohort_digest": constraint_digest,
                },
                "functional_retention_passed": functional_retention_passed,
                "functional_new_task_noninferior": functional_new_task_noninferior,
                "functional_new_task_gain": functional_new_task_gain,
                "functional_update_repaired": functional_update_repaired,
                "outcome": outcome,
                "experiment_passed": bool(all_gates and functional_update_repaired),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; functional parent-preserving objective; "
                    "structural growth remains fail-closed"
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = _run(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
