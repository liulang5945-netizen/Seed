"""P4.9 feature-space factorization probe (design-time, validation-only).

P4.8 excluded constraint form, capacity and architecture as root causes
of the retention/new-task mutual exclusion, leaving the 12-dimensional
candidate feature representation itself.  This probe measures - with the
frozen P3.5 parent and zero model training - whether a feature-space
factorization can exist at all:

- M1 target-rank structure: where the new-task behavior targets sit
  relative to the frozen parent's ranking;
- M2 conflict geometry: how the "flip-needed" and "preserve" candidate
  populations relate in feature space;
- M3 joint feasibility: minimum achievable violation of the joint
  constraint system (new-task targets win on train sets AND parent
  decisions are preserved on retention-sibling sets) over the base
  12-dim space versus an extended space with frozen-parent-relative
  margin features.  The minimum violation is measured by convex
  piecewise-linear minimisation (multi-restart full-batch Adam); it is
  a probe instrument, not a Taiji learner - no checkpoint, no fit.

The probe informs the feature-space redesign preregistration; it gates
nothing by itself.  ``can_promote=false``.
"""

from __future__ import annotations

import argparse
import json
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
    _load_json,
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p4_0_capacity_pressure import (  # noqa: E402
    _materialize_case,
    _pressure_record,
)
from scripts.training.eval_taiji_m5_k_p4_6_functional_parent_objective import (  # noqa: E402
    MARGIN_EPSILON,
    _manifest_identity,
    _records_identity,
)
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidate,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-p4-9-feature-space-probe-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_9_feature_space_probe_20260911.json"
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
SELECTION_MARGIN = 0.05
SAFE_EPSILON = 1e-9
ARGMAX_EPSILON = 1e-6
SOLVER_STEPS = 20000
SOLVER_LR = 0.05
SOLVER_RESTARTS = 4
FEASIBILITY_TOLERANCE = 1e-5
EXTENDED_FEATURE_NAMES = (
    "parent_argmax_margin",
    "parent_safe_margin",
    "parent_rank_norm",
    "is_parent_pick",
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_candidate(candidates: Sequence[GSelectionCandidate]) -> GSelectionCandidate:
    safe = [
        candidate
        for candidate in candidates
        if candidate.candidate_role in {"abstain", "reobserve"}
    ]
    if not safe:
        raise ValueError("G candidate set has no safe abstain or reobserve candidate")
    return max(
        safe,
        key=lambda candidate: (
            1 if candidate.candidate_role == "abstain" else 0,
            candidate.confidence,
            candidate.candidate_id,
        ),
    )


def _p49_specs(split: str, offset: int) -> tuple[dict[str, Any], ...]:
    prefix = f"p40_p49_{split}"
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
            "project_id": f"p4-9-{split}-project-{class_key.lower()}",
            "variant_paths": paths,
            "state_profile": state_profile,
            "task_seed": offset + index,
        }
        for index, (class_key, state_profile, paths) in enumerate(rows)
    )


def _p49_structured_specs(
    contract: Mapping[str, Any], *, namespace: str, seed_offset: int
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    safe_index = 0
    high_index = 0
    for index, row in enumerate(contract["rows"]):
        if int(row["role_counts"].get("proposal", 0)) > 0:
            project_id = f"p4-9-{namespace}-project-a"
            prefix = f"p40_p49_{namespace}_a"
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
                f"p4-9-{namespace}-project-a" if safe_index == 0 else f"p4-9-{namespace}-project-b"
            )
            prefix = f"p40_p49_{namespace}_{'a' if safe_index == 0 else 'b'}"
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


def _p49_rebind(
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
        example_id=f"p4-9:{split}:{source_index}:width-{width}:{old_candidates.candidate_set_digest}",
        family_id=f"p4-9:{split}:family:{source_index}",
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
            "experiment": "p4.9-probe",
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


def _p49_pressure_split(
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
    for spec in _p49_specs(split, offset):
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
            label=f"p4-9-{split}-{int(spec['index'])}",
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
                _p49_rebind(
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


def _p49_structured_records(
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
    specs = _p49_structured_specs(contract, namespace=namespace, seed_offset=seed_offset)
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
            label=f"p4-9-{namespace}-{int(spec['index'])}",
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
            _p49_rebind(
                pressure,
                split=namespace,
                source_index=index,
                width=int(spec["target_width"]),
                fit_eligible=False,
            )
        )
    return records


def _parent_rank(
    candidate_id: str, candidate_set: GSelectionCandidateSet, parent: GSelectionLearner
) -> int:
    scored = sorted(
        candidate_set.candidates,
        key=lambda item: (
            -parent.score(item),
            -{"abstain": 2, "reobserve": 1, "proposal": 0}[item.candidate_role],
            item.candidate_id,
        ),
    )
    return [candidate.candidate_id for candidate in scored].index(candidate_id)


def _extended_features(
    candidate_set: GSelectionCandidateSet, parent: GSelectionLearner
) -> dict[str, tuple[float, ...]]:
    """Base 12 dims plus four frozen-parent-relative margin dims."""
    candidates = tuple(candidate_set.candidates)
    parent_scores = {candidate.candidate_id: parent.score(candidate) for candidate in candidates}
    safe = _safe_candidate(candidates)
    scored = sorted(
        candidates,
        key=lambda item: (
            -parent_scores[item.candidate_id],
            -{"abstain": 2, "reobserve": 1, "proposal": 0}[item.candidate_role],
            item.candidate_id,
        ),
    )
    rank_by_id = {candidate.candidate_id: index for index, candidate in enumerate(scored)}
    pick_id = scored[0].candidate_id
    features: dict[str, tuple[float, ...]] = {}
    for candidate in candidates:
        cid = candidate.candidate_id
        others = [item for item in candidates if item.candidate_id != cid]
        features[cid] = (
            *candidate.feature_vector,
            parent_scores[cid] - max(parent_scores[item.candidate_id] for item in others),
            parent_scores[cid] - parent_scores[safe.candidate_id],
            1.0 - rank_by_id[cid] / max(1, len(candidates) - 1),
            1.0 if cid == pick_id else 0.0,
        )
    return features


def _joint_constraints(
    train_fit_records: Sequence[Mapping[str, Any]],
    sibling_records: Sequence[Mapping[str, Any]],
    parent: GSelectionLearner,
    feature_fn: Any,
) -> list[tuple[tuple[float, ...], float, str]]:
    """Affine constraints ``a·w >= b`` for the joint feasibility system."""
    constraints: list[tuple[tuple[float, ...], float, str]] = []
    for record in train_fit_records:
        candidate_set = record["candidate_set"]
        behavior_set = record["behavior_set"]
        features = feature_fn(candidate_set)
        target_id = behavior_set.behavior_target_candidate_id
        target = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_id == target_id
        )
        safe = _safe_candidate(candidate_set.candidates)
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
    for record in sibling_records:
        candidate_set = record["candidate_set"]
        decision = parent.select(candidate_set)
        features = feature_fn(candidate_set)
        safe = _safe_candidate(candidate_set.candidates)
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
                        f"sibling-argmax:{candidate_set.candidate_set_digest[:16]}:{candidate.candidate_id}",
                    )
                )
            difference = tuple(
                t - o for t, o in zip(features[picked_id], features[safe.candidate_id], strict=True)
            )
            constraints.append(
                (
                    difference,
                    SELECTION_MARGIN + SAFE_EPSILON,
                    f"sibling-safe:{candidate_set.candidate_set_digest[:16]}",
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
                        f"sibling-boundary:{candidate_set.candidate_set_digest[:16]}:{candidate.candidate_id}",
                    )
                )
    return constraints


def _min_violation(
    constraints: Sequence[tuple[tuple[float, ...], float, str]],
    *,
    dim: int,
) -> dict[str, Any]:
    """Multi-restart convex minimisation of the total hinge violation.

    The objective ``sum(max(0, b - a.w))`` is convex piecewise-linear in
    ``w``; its minimum value is the factorization gap of the constraint
    system.  This is a probe instrument - no Taiji learner, no checkpoint.
    """
    matrix = torch.tensor([item[0] for item in constraints], dtype=torch.float32)
    rhs = torch.tensor([item[1] for item in constraints], dtype=torch.float32)
    best_violation = float("inf")
    best_per_class: dict[str, float] = {}
    for restart in range(SOLVER_RESTARTS):
        generator = torch.Generator().manual_seed(restart)
        variable = torch.randn(dim, generator=generator) * 0.1
        variable.requires_grad_(True)
        optimizer = torch.optim.Adam([variable], lr=SOLVER_LR)
        for _step in range(SOLVER_STEPS):
            optimizer.zero_grad()
            violations = torch.relu(rhs - matrix @ variable)
            loss = violations.sum()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            violations = torch.relu(rhs - matrix @ variable)
            total = float(violations.sum().item())
            if total < best_violation:
                best_violation = total
                per_class: dict[str, float] = {}
                for (_a, _b, label), violation in zip(constraints, violations, strict=True):
                    family = label.split(":")[0]
                    per_class[family] = per_class.get(family, 0.0) + float(violation.item())
                best_per_class = per_class
    return {
        "min_total_violation": best_violation,
        "feasible": best_violation < FEASIBILITY_TOLERANCE,
        "violation_per_constraint_family": best_per_class,
        "solver": {
            "steps": SOLVER_STEPS,
            "lr": SOLVER_LR,
            "restarts": SOLVER_RESTARTS,
            "tolerance": FEASIBILITY_TOLERANCE,
        },
    }


def _run(*, report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_9_feature_space_probe_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "report": str(report_path),
    }
    try:
        manifests = {
            name: _load_json(path)
            for name, path in (
                ("p4_1", P4_1_MANIFEST),
                ("p4_2", P4_2_MANIFEST),
                ("p4_3", P4_3_MANIFEST),
                ("p4_4", P4_4_MANIFEST),
                ("p4_5", P4_5_MANIFEST),
                ("p4_6", P4_6_MANIFEST),
                ("p4_7", P4_7_MANIFEST),
                ("p4_8", P4_8_MANIFEST),
            )
        }
        p4_8_manifest = manifests["p4_8"]
        if _load_json(P4_8_MANIFEST) is None:
            raise ValueError("P4.9 probe requires the P4.8 manifest")
        contract = manifests["p4_4"]["structure_contract"]
        if content_digest(contract["rows"]) != contract["contract_digest"]:
            raise ValueError("P4.4 structure contract digest drifted")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        p3_2_report = _load_json(P3_2_REPORT)
        p3_5_report = _load_json(P3_5_REPORT)
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        semantic_payload = _load_mapping(Path(str(worker_restore["k1"]["path"])))
        transition_payload = _load_mapping(Path(str(worker_restore["k2"]["path"])))
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        parent_metadata = p3_5_report["g_trained_checkpoint"]
        parent_path = Path(str(parent_metadata["path"]))
        parent_payload = _load_mapping(parent_path)
        if content_digest(parent_payload) != str(parent_metadata["digest"]):
            raise ValueError("P4.9 probe parent external checkpoint digest drifted")
        parent = GSelectionLearner.from_checkpoint(parent_payload, device="cpu")
        run_dir.mkdir(parents=True, exist_ok=False)
        train_records = _p49_pressure_split(
            split="train",
            offset=49200,
            scratch=run_dir / "data" / "train",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_8_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        sibling_records = _p49_structured_records(
            contract=contract,
            namespace="retention",
            seed_offset=49000,
            scratch=run_dir / "data" / "retention",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_8_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        train_fit = [record for record in train_records if record["fit_eligible"]]
        all_records = [*train_records, *sibling_records]
        old_paths: set[str] = set()
        old_projects: set[str] = set()
        for manifest in manifests.values():
            manifest_paths, manifest_projects = _manifest_identity(manifest)
            old_paths.update(manifest_paths)
            old_projects.update(manifest_projects)
        paths, projects = _records_identity(all_records)
        identity_gate = {
            "train_records": len(train_records) == 20,
            "sibling_records": len(sibling_records) == 4,
            "train_fit_positive": len(train_fit) >= 8,
            "projects_disjoint_from_historical": projects.isdisjoint(old_projects),
            "paths_disjoint_from_historical": paths.isdisjoint(old_paths),
            "all_candidate_digests_unique": len(
                {record["candidate_set"].candidate_set_digest for record in all_records}
            )
            == len(all_records),
        }
        if not all(identity_gate.values()):
            raise ValueError(
                f"P4.9 probe identity gate failed: {[k for k, v in identity_gate.items() if not v]}"
            )

        # M1: target rank structure under the frozen parent.
        target_ranks = []
        target_utility_gaps = []
        for record in train_fit:
            candidate_set = record["candidate_set"]
            behavior_set = record["behavior_set"]
            rank = _parent_rank(behavior_set.behavior_target_candidate_id, candidate_set, parent)
            target_ranks.append(rank)
            picked = parent.select(candidate_set).selected_candidate_id
            outcome_by_id = {
                outcome.candidate_id: outcome.utility for outcome in behavior_set.outcomes
            }
            target_utility_gaps.append(
                float(outcome_by_id[behavior_set.behavior_target_candidate_id])
                - float(outcome_by_id[picked])
            )
        m1 = {
            "fit_eligible_sets": len(train_fit),
            "target_rank_counts": dict(sorted(Counter(target_ranks).items())),
            "target_is_parent_pick_count": sum(1 for rank in target_ranks if rank == 0),
            "target_utility_gap_mean": sum(target_utility_gaps) / len(target_utility_gaps),
            "target_utility_gap_min": min(target_utility_gaps),
            "target_utility_gap_max": max(target_utility_gaps),
        }

        # M2: conflict geometry between flip-needed and preserve populations.
        flip_rows = []
        preserve_rows = []
        for record in train_fit:
            candidate_set = record["candidate_set"]
            behavior_set = record["behavior_set"]
            picked = parent.select(candidate_set).selected_candidate_id
            for candidate in candidate_set.candidates:
                if candidate.candidate_id == behavior_set.behavior_target_candidate_id:
                    flip_rows.append(list(candidate.feature_vector))
        for record in sibling_records:
            candidate_set = record["candidate_set"]
            decision = parent.select(candidate_set)
            for candidate in candidate_set.candidates:
                if candidate.candidate_id == decision.selected_candidate_id:
                    preserve_rows.append(list(candidate.feature_vector))
        flip_mean = [sum(row[d] for row in flip_rows) / len(flip_rows) for d in range(12)]
        preserve_mean = [
            sum(row[d] for row in preserve_rows) / len(preserve_rows) for d in range(12)
        ]
        difference = [f - p for f, p in zip(flip_mean, preserve_mean, strict=True)]
        parent_weight = parent.model.weight.detach().reshape(-1)
        norm_product = (
            sum(a * b for a, b in zip(difference, parent_weight.tolist(), strict=True)) ** 2
        ) / (sum(a * a for a in difference) * sum(a * a for a in parent_weight.tolist()) + 1e-12)
        m2 = {
            "flip_needed_count": len(flip_rows),
            "preserve_count": len(preserve_rows),
            "flip_mean": flip_mean,
            "preserve_mean": preserve_mean,
            "mean_difference": difference,
            "parent_weight": parent_weight.tolist(),
            "squared_cosine_difference_vs_parent_weight": norm_product,
        }

        # M3: joint feasibility over base vs extended feature spaces.
        m3: dict[str, Any] = {}
        for space, feature_fn in (
            (
                "base-12",
                lambda candidate_set: {
                    candidate.candidate_id: tuple(candidate.feature_vector)
                    for candidate in candidate_set.candidates
                },
            ),
            ("extended-16", lambda candidate_set: _extended_features(candidate_set, parent)),
        ):
            dim = len(next(iter(feature_fn(sibling_records[0]["candidate_set"]).values())))
            constraints = _joint_constraints(train_fit, sibling_records, parent, feature_fn)
            result = _min_violation(constraints, dim=dim)
            m3[space] = {"dim": dim, "constraint_count": len(constraints), **result}
        feasibility_verdict = (
            "both_feasible_optimization_problem"
            if m3["base-12"]["feasible"] and m3["extended-16"]["feasible"]
            else (
                "parent_relative_features_are_the_factorization"
                if m3["extended-16"]["feasible"]
                else "linear_scoring_insufficient_nonlinear_layer_required"
            )
        )
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "probe_contract": {
                    "source_p4_8_manifest_digest": p4_8_manifest["manifest_digest"],
                    "parent_g_checkpoint_digest": str(parent_metadata["digest"]),
                    "train_candidate_set_digests": [
                        record["candidate_set"].candidate_set_digest for record in train_records
                    ],
                    "sibling_candidate_set_digests": [
                        record["candidate_set"].candidate_set_digest for record in sibling_records
                    ],
                    "extended_feature_names": list(EXTENDED_FEATURE_NAMES),
                    "probe_solver": "convex violation minimisation (probe instrument, not a Taiji learner)",
                },
                "identity_gate": identity_gate,
                "m1_target_rank_structure": m1,
                "m2_conflict_geometry": m2,
                "m3_joint_feasibility": m3,
                "feasibility_verdict": feasibility_verdict,
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: feasibility_verdict={feasibility_verdict}; "
                    "design-time probe for the feature-space redesign preregistration"
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
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = _run(report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
