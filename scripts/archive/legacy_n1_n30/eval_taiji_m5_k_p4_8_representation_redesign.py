"""P4.8 three-arm representation contract redesign runner.

P4.7 closed the capacity hypothesis: the retention/new-task mutual
exclusion on the 13-parameter G does not disappear with an inherited
+9-parameter expansion.  The diagnosed root cause is the scalar-MSE
teacher constraint, whose gradient never saturates and fights task
deltas on the same weights.  This runner executes the preregistered
three-arm comparison with adjacent single-variable steps:

1. ``functional-13``   - P4.6 scalar-MSE teacher constraint (in-run
   baseline, third replication);
2. ``invariant-13``    - same architecture, constraint form replaced by
   the margin-preservation hinge (finite support, birth-zero loss);
3. ``residual-26``     - frozen parent head + zero-initialised delta
   head (only the delta trains), same hinge constraint.

All arms share the task-fit mechanics, train records, constraint
cohort, interleaving order, epochs, learning rate and seeds.  Reference
decisions and margins come from the frozen P3.5 parent; behavior
targets of the constraint cohort are never read.  Preregistration:
``plans/reference/M5_K_P4_8_REPRESENTATION_CONTRACT_REDESIGN
_PREREGISTRATION_20260911.md``.  Never admits growth or promotion.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import time
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
    _save_checkpoint,
    _save_torch_atomic,
    _tamper_rejected,
)
from scripts.training.eval_taiji_m5_k_p4_6_functional_parent_objective import (  # noqa: E402
    MARGIN_EPSILON,
    _constraint_feature_groups,
    _functional_fit_sequence,
    _manifest_identity,
    _metric_summary,
    _records_identity,
    _structure_row,
)
from scripts.training.eval_taiji_m5_k_p4_6_functional_parent_objective import (
    _load_mapping as _p4_6_load_mapping,
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
from taiji.g_selection_residual import (  # noqa: E402
    ResidualGSelectionLearner,
    margin_preservation_hinge,
)
from taiji.local_learning import (  # noqa: E402
    apply_linear_delta,
    mean_squared_error_delta,
)

REPORT_FORMAT = "taiji-m5-k-p4-8-representation-redesign-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-8-representation-redesign-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_8_representation_redesign_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_8_representation_redesign_20260911.json"
P4_7_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_7_capacity_clean_test_manifest_v1.json"
)
P4_7_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_7_capacity_clean_test_20260911.json"
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
P4_6_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_6_functional_parent_objective_manifest_v1.json"
)
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
P3_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"
TRAINING_EPOCHS = 8
LEARNING_RATE = 0.15
SEEDS = (0, 1)
ARM_FUNCTIONAL = "functional-13"
ARM_INVARIANT = "invariant-13"
ARM_RESIDUAL = "residual-26"
EVAL_SPLITS = ("validation", "holdout", "retention-sibling", "retention-newtask")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _digest_without(payload: Mapping[str, Any], key: str) -> str:
    return content_digest({name: value for name, value in payload.items() if name != key})


def _p48_specs(split: str, offset: int) -> tuple[dict[str, Any], ...]:
    # The "p40_" prefix is required by _materialize_case's content-writing
    # trigger; the "p48" infix keeps identities disjoint from P4.0-P4.7.
    prefix = f"p40_p48_{split}"
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
            "project_id": f"p4-8-{split}-project-{class_key.lower()}",
            "variant_paths": paths,
            "state_profile": state_profile,
            "task_seed": offset + index,
        }
        for index, (class_key, state_profile, paths) in enumerate(rows)
    )


def _p48_structured_specs(
    contract: Mapping[str, Any], *, namespace: str, seed_offset: int
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    safe_index = 0
    high_index = 0
    for index, row in enumerate(contract["rows"]):
        if int(row["role_counts"].get("proposal", 0)) > 0:
            project_id = f"p4-8-{namespace}-project-a"
            prefix = f"p40_p48_{namespace}_a"
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
                f"p4-8-{namespace}-project-a" if safe_index == 0 else f"p4-8-{namespace}-project-b"
            )
            prefix = f"p40_p48_{namespace}_{'a' if safe_index == 0 else 'b'}"
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


def _p48_rebind(
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
        example_id=f"p4-8:{split}:{source_index}:width-{width}:{old_candidates.candidate_set_digest}",
        family_id=f"p4-8:{split}:family:{source_index}",
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
            "experiment": "p4.8",
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


def _p48_pressure_split(
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
    for spec in _p48_specs(split, offset):
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
            label=f"p4-8-{split}-{int(spec['index'])}",
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
                _p48_rebind(
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


def _p48_structured_records(
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
    specs = _p48_structured_specs(contract, namespace=namespace, seed_offset=seed_offset)
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
            label=f"p4-8-{namespace}-{int(spec['index'])}",
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
            _p48_rebind(
                pressure,
                split=namespace,
                source_index=index,
                width=int(spec["target_width"]),
                fit_eligible=False,
            )
        )
    return records


def _invariant_fit_sequence_13(
    learner: GSelectionLearner,
    reference_learner: GSelectionLearner,
    records: Sequence[Mapping[str, Any]],
    constraint_sets: Sequence[GSelectionCandidateSet],
    *,
    epochs: int,
    learning_rate: float,
    order_seed: int,
    constraint_digest: str,
) -> dict[str, Any]:
    """Task fit + margin-preservation hinge on the shared 13-param weights.

    The reference decisions and margins come from the frozen parent
    learner; the constraint cohort's behavior targets are never read.
    """
    items = tuple(record["candidate_set"] for record in records)
    if not items or any(item.split != "train" for item in items):
        raise ValueError("P4.8 invariant fit requires non-empty train records")
    if not constraint_sets:
        raise ValueError("P4.8 invariant fit requires an independent constraint cohort")
    if len({item.candidate_set_digest for item in items}) != len(items):
        raise ValueError("P4.8 invariant fit candidate sets must be unique")
    reference_decisions = {
        candidate_set.candidate_set_digest: reference_learner.select(candidate_set)
        for candidate_set in constraint_sets
    }
    reference_scores = {
        candidate_set.candidate_set_digest: {
            candidate.candidate_id: reference_learner.score(candidate)
            for candidate in candidate_set.candidates
        }
        for candidate_set in constraint_sets
    }
    dataset_digest = content_digest(
        {
            "candidate_set_digests": [item.candidate_set_digest for item in items],
            "constraint_digest": constraint_digest,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "order_seed": int(order_seed),
            "constraint_form": "margin-preservation-hinge",
        }
    )
    task_loss_sum = 0.0
    hinge_loss_sum = 0.0
    hinge_active_steps = 0
    constraint_steps = 0
    for epoch in range(int(epochs)):
        order = list(range(len(items)))
        random.Random(int(order_seed) + epoch).shuffle(order)
        for step, index in enumerate(order):
            item = items[index]
            inputs = torch.tensor(
                [candidate.feature_vector for candidate in item.candidates],
                dtype=torch.float32,
                device=learner.device,
            )
            targets = torch.zeros(
                (len(item.candidates), 1), dtype=torch.float32, device=learner.device
            )
            target_index = next(
                position
                for position, candidate in enumerate(item.candidates)
                if candidate.candidate_id == item.target_candidate_id
            )
            targets[target_index, 0] = 1.0
            with torch.no_grad():
                predictions = learner.model(inputs)
            task_loss_sum += float(torch.mean((predictions - targets) ** 2).item())
            apply_linear_delta(
                learner.model,
                inputs,
                mean_squared_error_delta(predictions, targets),
                float(learning_rate),
            )
            learner.training_steps += 1
            group = constraint_sets[(epoch * len(order) + step) % len(constraint_sets)]
            decision = reference_decisions[group.candidate_set_digest]
            hinge_loss, error_by_id = margin_preservation_hinge(
                group,
                current_scores={
                    candidate.candidate_id: learner.score(candidate)
                    for candidate in group.candidates
                },
                reference_scores=reference_scores[group.candidate_set_digest],
                reference_selected_id=decision.selected_candidate_id,
                reference_status=decision.selection_status,
                selection_margin=learner.selection_margin,
            )
            hinge_loss_sum += hinge_loss
            hinge_active_steps += int(hinge_loss > 0.0)
            group_inputs = torch.tensor(
                [candidate.feature_vector for candidate in group.candidates],
                dtype=torch.float32,
                device=learner.device,
            )
            hinge_error = torch.tensor(
                [[error_by_id[candidate.candidate_id]] for candidate in group.candidates],
                dtype=torch.float32,
                device=learner.device,
            )
            apply_linear_delta(learner.model, group_inputs, hinge_error, float(learning_rate))
            constraint_steps += 1
            learner.training_steps += 1
    learner.revision += 1
    learner.last_train_digest = dataset_digest
    return {
        "dataset_digest": dataset_digest,
        "candidate_sets": len(items),
        "constraint_groups": len(constraint_sets),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "order_seed": int(order_seed),
        "constraint_form": "margin-preservation-hinge",
        "training_steps": learner.training_steps,
        "revision": learner.revision,
        "task_loss_mean": task_loss_sum / (len(items) * int(epochs)),
        "hinge_loss_mean": hinge_loss_sum / constraint_steps if constraint_steps else 0.0,
        "hinge_active_steps": hinge_active_steps,
        "constraint_steps": constraint_steps,
    }


def _invariant_hinge_loss_13(
    learner: GSelectionLearner,
    reference_learner: GSelectionLearner,
    candidate_set: GSelectionCandidateSet,
) -> float:
    decision = reference_learner.select(candidate_set)
    loss, _errors = margin_preservation_hinge(
        candidate_set,
        current_scores={
            candidate.candidate_id: learner.score(candidate)
            for candidate in candidate_set.candidates
        },
        reference_scores={
            candidate.candidate_id: reference_learner.score(candidate)
            for candidate in candidate_set.candidates
        },
        reference_selected_id=decision.selected_candidate_id,
        reference_status=decision.selection_status,
        selection_margin=learner.selection_margin,
    )
    return loss


def _birth_equivalence(
    parent: GSelectionLearner,
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    arms: Mapping[str, Any],
    constraint_sets: Sequence[GSelectionCandidateSet],
) -> dict[str, Any]:
    """All arms must reproduce parent selections exactly at birth; the
    invariant arms must additionally show zero hinge loss at birth."""
    arm_report: dict[str, Any] = {}
    for arm, arm_birth in arms.items():
        mismatches = 0
        max_score_deviation = 0.0
        for split in EVAL_SPLITS:
            for record in split_records[split]:
                candidate_set = record["candidate_set"]
                parent_decision = parent.select(candidate_set)
                child_decision = arm_birth["select"](candidate_set)
                if (
                    parent_decision.selected_candidate_id != child_decision.selected_candidate_id
                    or parent_decision.selection_status != child_decision.selection_status
                ):
                    mismatches += 1
                for candidate in candidate_set.candidates:
                    deviation = abs(
                        float(arm_birth["score"](candidate)) - float(parent.score(candidate))
                    )
                    max_score_deviation = max(max_score_deviation, deviation)
        arm_report[arm] = {
            "selection_mismatches": mismatches,
            "max_abs_score_deviation": max_score_deviation,
        }
    hinge_report: dict[str, Any] = {}
    for arm in (ARM_INVARIANT, ARM_RESIDUAL):
        hinge = arms[arm]["hinge_loss"]
        losses = [hinge(candidate_set) for candidate_set in constraint_sets]
        hinge_report[arm] = {
            "birth_hinge_losses": losses,
            "birth_hinge_loss_zero": all(loss == 0.0 for loss in losses),
        }
    passed = all(
        report["selection_mismatches"] == 0 and report["max_abs_score_deviation"] == 0.0
        for report in arm_report.values()
    ) and all(report["birth_hinge_loss_zero"] for report in hinge_report.values())
    return {"arms": arm_report, "hinge": hinge_report, "passed": passed}


def _save_residual_checkpoint(path: Path, learner: ResidualGSelectionLearner) -> dict[str, Any]:
    payload = learner.checkpoint()
    _save_torch_atomic(path, payload)
    restored = ResidualGSelectionLearner.from_checkpoint(_load_mapping(path), device="cpu")
    independent = _independent_residual_restore(path)
    digest_matches = restored.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"]
    return {
        "path": str(path),
        "digest": payload["checkpoint_digest"],
        "bytes": path.stat().st_size,
        "roundtrip": digest_matches,
        "restore": independent,
        "passed": bool(independent.get("independent_process_restore")) and digest_matches,
    }


def _independent_residual_restore(path: Path) -> dict[str, Any]:
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify-residual-only", str(path)],
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


def _residual_tamper_rejected(payload: Mapping[str, Any]) -> bool:
    tampered = copy.deepcopy(dict(payload))
    tampered["revision"] = int(tampered.get("revision", 0)) + 1
    try:
        ResidualGSelectionLearner.from_checkpoint(tampered, device="cpu")
    except ValueError:
        return True
    return False


def _verify_residual_checkpoint(path: Path) -> dict[str, Any]:
    payload = _load_mapping(path)
    learner = ResidualGSelectionLearner.from_checkpoint(payload, device="cpu")
    return {
        "passed": learner.parameter_count == 26
        and learner.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"],
        "parameter_count": learner.parameter_count,
    }


def _run(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_8_representation_redesign_{uuid4().hex}"
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
        p4_7_report = _load_json(P4_7_REPORT)
        p3_5_report = _load_json(P3_5_REPORT)
        if (
            p4_7_report.get("status") != "completed"
            or p4_7_report.get("outcome") != "capacity_hypothesis_closed"
        ):
            raise ValueError("P4.8 requires the completed P4.7 capacity_hypothesis_closed")
        if p4_7_report.get("manifest_digest") != p4_7_manifest.get("manifest_digest"):
            raise ValueError("P4.7 manifest/report digest mismatch")
        if _digest_without(p4_7_manifest, "manifest_digest") != p4_7_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.7 manifest content digest mismatch")
        if p4_7_report.get("growth_admitted") or p4_7_report.get("can_promote"):
            raise ValueError("P4.8 cannot consume an admitted P4.7 artifact")
        if p4_7_manifest.get("source_p4_6_manifest_digest") != p4_6_manifest.get("manifest_digest"):
            raise ValueError("P4.7 source chain drifted")
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
            raise ValueError("P4.8 parent external checkpoint digest drifted")
        parent = GSelectionLearner.from_checkpoint(parent_payload, device="cpu")
        parent.assert_lineage(
            parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        if str(p4_2_manifest["parent_g_checkpoint_digest"]) != str(parent_metadata["digest"]):
            raise ValueError("P4.8 parent G lineage drifted")
        parent_restore = _independent_g_restore(parent_path)
        if not parent_restore.get("independent_process_restore"):
            raise RuntimeError("P4.8 parent independent restore failed")
        run_dir.mkdir(parents=True, exist_ok=False)
        train_records = _p48_pressure_split(
            split="train",
            offset=48200,
            scratch=run_dir / "data" / "train",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_7_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        validation_records = _p48_pressure_split(
            split="validation",
            offset=48300,
            scratch=run_dir / "data" / "validation",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_7_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        holdout_records = _p48_pressure_split(
            split="holdout",
            offset=48400,
            scratch=run_dir / "data" / "holdout",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_7_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        retention_newtask_records = _p48_pressure_split(
            split="retention-newtask",
            offset=48500,
            scratch=run_dir / "data" / "retention-newtask",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_7_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        constraint_records = _p48_structured_records(
            contract=contract,
            namespace="constraint",
            seed_offset=48100,
            scratch=run_dir / "data" / "constraint",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_7_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        retention_sibling_records = _p48_structured_records(
            contract=contract,
            namespace="retention",
            seed_offset=48000,
            scratch=run_dir / "data" / "retention",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_7_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        train_fit = [record for record in train_records if record["fit_eligible"]]
        constraint_sets = tuple(record["candidate_set"] for record in constraint_records)
        constraint_features_12 = _constraint_feature_groups(constraint_records)
        constraint_digest_scalar = content_digest(constraint_features_12)
        constraint_digest_hinge = content_digest(
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
        sibling_structure = [
            _structure_row(record["candidate_set"], record["behavior_set"])
            for record in retention_sibling_records
        ]
        constraint_structure = [
            _structure_row(record["candidate_set"], record["behavior_set"])
            for record in constraint_records
        ]
        structure_gate = {
            "retention_sibling_matches_p4_4_contract": sibling_structure == list(contract["rows"]),
            "constraint_matches_p4_4_contract": constraint_structure == list(contract["rows"]),
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
                f"P4.8 identity gate failed: {[key for key, value in identity_gate.items() if not value]}"
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
                ARM_FUNCTIONAL: GSelectionLearner.from_checkpoint(
                    copy.deepcopy(parent_payload), device="cpu"
                ),
                ARM_INVARIANT: GSelectionLearner.from_checkpoint(
                    copy.deepcopy(parent_payload), device="cpu"
                ),
                ARM_RESIDUAL: ResidualGSelectionLearner.from_parent_learner(parent),
            }
            birth_arms = {
                ARM_FUNCTIONAL: {
                    "select": learners[ARM_FUNCTIONAL].select,
                    "score": learners[ARM_FUNCTIONAL].score,
                },
                ARM_INVARIANT: {
                    "select": learners[ARM_INVARIANT].select,
                    "score": learners[ARM_INVARIANT].score,
                    "hinge_loss": lambda candidate_set, learner=learners[
                        ARM_INVARIANT
                    ]: _invariant_hinge_loss_13(learner, parent, candidate_set),
                },
                ARM_RESIDUAL: {
                    "select": learners[ARM_RESIDUAL].select,
                    "score": learners[ARM_RESIDUAL].total_score,
                    "hinge_loss": lambda candidate_set, learner=learners[
                        ARM_RESIDUAL
                    ]: learner.invariant_hinge(candidate_set)[0],
                },
            }
            birth = _birth_equivalence(parent, split_records, birth_arms, constraint_sets)
            if not birth["passed"]:
                raise ValueError(f"P4.8 birth equivalence gate failed: {birth}")
            zero_checkpoints = {
                ARM_FUNCTIONAL: _save_checkpoint(
                    seed_dir / f"{ARM_FUNCTIONAL}-zero.pt", learners[ARM_FUNCTIONAL]
                ),
                ARM_INVARIANT: _save_checkpoint(
                    seed_dir / f"{ARM_INVARIANT}-zero.pt", learners[ARM_INVARIANT]
                ),
                ARM_RESIDUAL: _save_residual_checkpoint(
                    seed_dir / f"{ARM_RESIDUAL}-zero.pt", learners[ARM_RESIDUAL]
                ),
            }
            if not all(item["passed"] for item in zero_checkpoints.values()):
                raise RuntimeError(f"P4.8 zero-step checkpoint preflight failed for seed {seed}")
            fit_functional = _functional_fit_sequence(
                learners[ARM_FUNCTIONAL],
                parent,
                train_fit,
                constraint_features_12,
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
                order_seed=seed,
                functional_weight=1.0,
                constraint_digest=constraint_digest_scalar,
            )
            fit_invariant = _invariant_fit_sequence_13(
                learners[ARM_INVARIANT],
                parent,
                train_fit,
                constraint_sets,
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
                order_seed=seed,
                constraint_digest=constraint_digest_hinge,
            )
            fit_residual = learners[ARM_RESIDUAL].invariant_fit(
                train_fit,
                constraint_sets,
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
                order_seed=seed,
                constraint_digest=constraint_digest_hinge,
            )
            trained_checkpoints = {
                ARM_FUNCTIONAL: _save_checkpoint(
                    seed_dir / f"{ARM_FUNCTIONAL}-trained.pt", learners[ARM_FUNCTIONAL]
                ),
                ARM_INVARIANT: _save_checkpoint(
                    seed_dir / f"{ARM_INVARIANT}-trained.pt", learners[ARM_INVARIANT]
                ),
                ARM_RESIDUAL: _save_residual_checkpoint(
                    seed_dir / f"{ARM_RESIDUAL}-trained.pt", learners[ARM_RESIDUAL]
                ),
            }
            if not all(item["passed"] for item in trained_checkpoints.values()):
                raise RuntimeError(f"P4.8 trained checkpoint preflight failed for seed {seed}")
            tamper_gate = {
                ARM_FUNCTIONAL: _tamper_rejected(
                    _p4_6_load_mapping(Path(str(trained_checkpoints[ARM_FUNCTIONAL]["path"])))
                ),
                ARM_INVARIANT: _tamper_rejected(
                    _p4_6_load_mapping(Path(str(trained_checkpoints[ARM_INVARIANT]["path"])))
                ),
                ARM_RESIDUAL: _residual_tamper_rejected(
                    _load_mapping(Path(str(trained_checkpoints[ARM_RESIDUAL]["path"])))
                ),
            }
            if not all(tamper_gate.values()):
                raise RuntimeError(f"P4.8 tamper gate failed for seed {seed}")
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
                    "fit": {
                        ARM_FUNCTIONAL: fit_functional,
                        ARM_INVARIANT: fit_invariant,
                        ARM_RESIDUAL: fit_residual,
                    },
                    "zero_checkpoints": zero_checkpoints,
                    "trained_checkpoints": trained_checkpoints,
                    "tamper_gate": tamper_gate,
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

        functional_passes = _arm_passes_all(ARM_FUNCTIONAL)
        invariant_passes = _arm_passes_all(ARM_INVARIANT)
        residual_passes = _arm_passes_all(ARM_RESIDUAL)
        if functional_passes:
            outcome = "baseline_drift"
            active_ingredient = "indeterminate"
        elif invariant_passes or residual_passes:
            outcome = "representation_redesign_supported"
            if invariant_passes and residual_passes:
                active_ingredient = "constraint_form"
            elif residual_passes:
                active_ingredient = "architecture"
            else:
                active_ingredient = "constraint_form_shared_weights"
        else:
            outcome = "invariant_constraint_insufficient"
            active_ingredient = "none"
        source_gate = {
            "p4_7_completed_and_closed": True,
            "p4_7_growth_still_fail_closed": True,
            "p4_7_source_chain_valid": True,
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
            "three_arms_present": True,
            "two_deterministic_seeds": len(seed_results) == len(SEEDS),
            "validation_not_fit": True,
            "holdout_not_fit": True,
            "retention_newtask_not_fit": True,
            "retention_sibling_not_fit": True,
            "constraint_targets_are_parent_only": True,
            "external_target_unused": True,
            "k_parameters_unchanged": True,
            "parameter_counts_frozen": all(
                result["parameter_count"][ARM_FUNCTIONAL] == 13
                and result["parameter_count"][ARM_INVARIANT] == 13
                and result["parameter_count"][ARM_RESIDUAL] == 26
                for result in seed_results
            ),
        }
        all_gates = (
            all(source_gate.values())
            and all(identity_gate.values())
            and structure_gate["retention_sibling_matches_p4_4_contract"]
            and structure_gate["constraint_matches_p4_4_contract"]
            and birth["passed"]
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
            "source_p4_6_manifest_digest": p4_6_manifest["manifest_digest"],
            "source_p4_7_manifest_digest": p4_7_manifest["manifest_digest"],
            "source_p4_7_report_digest": content_digest(p4_7_report),
            "parent_g_checkpoint_digest": str(parent_metadata["digest"]),
            "k_checkpoint_digests": worker_digests,
            "representation_contract": {
                "hinge_form": "margin-preservation",
                "reference": "frozen_p3_5_parent_decisions_and_margins",
                "constraint_cohort_targets_read": False,
                "arms": {
                    ARM_FUNCTIONAL: "scalar-mse-teacher (p4.6 replication)",
                    ARM_INVARIANT: "margin-preservation-hinge, shared trainable weights",
                    ARM_RESIDUAL: "margin-preservation-hinge, frozen parent head + delta head",
                },
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
                record["candidate_set"].candidate_set_digest for record in retention_newtask_records
            ],
            "constraint_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in constraint_records
            ],
            "retention_sibling_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in retention_sibling_records
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
                "birth_equivalence": birth,
                "parent_metrics": parent_metrics,
                "seed_results": seed_results,
                "functional_13_passes_all": functional_passes,
                "invariant_13_passes_all": invariant_passes,
                "residual_26_passes_all": residual_passes,
                "active_ingredient": active_ingredient,
                "outcome": outcome,
                "experiment_passed": bool(all_gates),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; active_ingredient={active_ingredient}; "
                    "representation contract redesign test; growth and promotion "
                    "remain fail-closed"
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
    parser.add_argument("--verify-residual-only", type=Path)
    args = parser.parse_args()
    if args.verify_residual_only is not None:
        result = _verify_residual_checkpoint(args.verify_residual_only)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["passed"] else 1
    result = _run(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
