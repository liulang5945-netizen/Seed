"""Calibrate P4 retention findings against new, structure-matched identities.

P4.3 showed that its fresh retention set did not degrade, but its
``new-only`` and ``rehearsal-mix`` arms were identical.  P4.4 therefore does
not train anything.  It extracts only a structural contract from the P3.6
holdout, creates new sibling projects and paths with that contract, and
evaluates the historical parent plus the P4.2/P4.3 child checkpoints on the
same read-only artifact.

The result is deliberately diagnostic rather than promotable.  It can only
classify the old retention failure as reproduced, artifact-specific, or still
unresolved; it never admits growth or promotion.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
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
from scripts.training.eval_taiji_m5_k_p4_2_capacity_attribution import (  # noqa: E402
    ContextLearner,
    _context_record,
    _independent_context_restore,
)
from scripts.training.eval_taiji_m5_k_p4_2_capacity_attribution import (
    _evaluate as _evaluate_context,
)
from scripts.training.eval_taiji_m5_k_p4_3_retention_incremental import (  # noqa: E402
    _evaluate as _evaluate_g,
)
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-p4-4-retention-identity-calibration-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-4-retention-identity-calibration-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_4_retention_identity_calibration_manifest_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p4_4_retention_identity_calibration_20260911.json"
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
P3_6_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_6_behavior_holdout_manifest_v1.json"
)
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
P4_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_2_capacity_attribution_20260911.json"
P4_3_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_3_retention_incremental_20260911.json"
P3_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"
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
    safe_roles = sorted(
        {
            candidate.candidate_role
            for candidate, outcome in zip(candidate_set.candidates, behavior_set.outcomes)
            if outcome.safe_exit_valid or outcome.safe_exit_progress
        }
    )
    maximum_confidence = max(confidence_values, default=0.0)
    confidence_bucket = (
        "zero"
        if maximum_confidence <= MARGIN_EPSILON
        else "high-and-safe"
        if maximum_confidence >= 0.8
        else "mixed"
    )
    return {
        "candidate_count": len(candidate_set.candidates),
        "role_counts": dict(sorted(role_counts.items())),
        "confidence_bucket": confidence_bucket,
        "safe_projection_roles": safe_roles,
    }


def _extract_structure_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for raw in payload["holdout_records"]:
        candidate_set = GSelectionCandidateSet.from_payload(raw["candidate_set"])
        behavior_set = GSelectionBehaviorSet.from_payload(raw["behavior_set"])
        rows.append(_structure_row(candidate_set, behavior_set))
    return {
        "source": "p3.6-holdout-structure-only",
        "row_count": len(rows),
        "rows": rows,
        "contract_digest": content_digest(rows),
    }


def _new_specs(contract: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    high_index = 0
    safe_index = 0
    for index, row in enumerate(contract["rows"]):
        role_counts = row["role_counts"]
        if int(role_counts.get("proposal", 0)) > 0:
            project_id = "p4-4-sibling-project-a"
            prefix = "p40_p44_a"
            paths = (
                f"{prefix}_main_{high_index}.py",
                f"{prefix}_alternate_{high_index}.rs",
                f"{prefix}_alternate_{high_index}_b.py",
            )
            state_profile = "resolved-language"
            class_key = f"H{high_index}"
            high_index += 1
        else:
            project_id = "p4-4-sibling-project-a" if safe_index == 0 else "p4-4-sibling-project-b"
            prefix = "p40_p44_a" if safe_index == 0 else "p40_p44_b"
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
                "task_seed": 44000 + index,
                "target_width": int(row["candidate_count"]),
            }
        )
    return tuple(specs)


def _rebind_sibling_record(
    record: Mapping[str, Any],
    *,
    source_index: int,
    width: int,
) -> dict[str, Any]:
    old_candidates = record["candidate_set"]
    old_behavior = record["behavior_set"]
    candidate_set = GSelectionCandidateSet.create(
        example_id=(
            f"p4-4:sibling:{source_index}:width-{width}:{old_candidates.candidate_set_digest}"
        ),
        family_id=f"p4-4:sibling:family:{source_index}",
        split="retention",
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
        split="retention",
        project_id=candidate_set.project_id,
        path=candidate_set.path,
        outcomes=old_behavior.outcomes,
    )
    diagnostic = dict(record["diagnostic"])
    diagnostic.update(
        {
            "experiment": "p4.4",
            "split": "retention",
            "structure_source_row": source_index,
            "candidate_width": width,
        }
    )
    return {
        **record,
        "candidate_set": candidate_set,
        "behavior_set": behavior_set,
        "diagnostic": diagnostic,
        "fit_eligible": False,
    }


def _build_sibling_records(
    *,
    contract: Mapping[str, Any],
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
    for spec in _new_specs(contract):
        experience, metadata, case = _materialize_case(
            scratch=scratch,
            spec=spec,
            parent_digest=parent_digest,
            worker_bundle_digest=worker_bundle_digest,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        metadata = {**metadata, "split": "retention"}
        candidate_set, behavior_set, diagnostic = _behavior_record(
            experience=experience,
            metadata=metadata,
            case=case,
            semantic=semantic,
            transition=transition,
            goals=goals,
            content_plans=content_plans,
            label=f"p4-4-retention-{int(spec['index'])}",
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
    for source_index, spec in enumerate(_new_specs(contract)):
        pressure = _pressure_record(
            base_records=base_records,
            source_index=source_index,
            width=int(spec["target_width"]),
            transition=transition,
        )
        records.append(
            _rebind_sibling_record(
                pressure,
                source_index=source_index,
                width=int(spec["target_width"]),
            )
        )
    return records


def _manifest_identity_sets(payload: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    projects: set[str] = set()
    for collection in payload.get("records", {}).values():
        if not isinstance(collection, list):
            continue
        for raw in collection:
            candidate = raw.get("candidate_set", {})
            if candidate:
                paths.add(str(candidate.get("path", "")))
                projects.add(str(candidate.get("project_id", "")))
    return paths, projects


def _load_g_checkpoint(
    *,
    path: Path,
    expected_digest: str,
    parent_manifest_digest: str,
    k_checkpoint_digests: Mapping[str, str],
) -> tuple[GSelectionLearner, dict[str, Any]]:
    payload = _load_mapping(path)
    learner = GSelectionLearner.from_checkpoint(payload, device="cpu")
    if str(payload.get("checkpoint_digest")) != expected_digest:
        raise ValueError(f"G checkpoint digest mismatch: {path}")
    learner.assert_lineage(
        parent_manifest_digest=parent_manifest_digest,
        k_checkpoint_digests=k_checkpoint_digests,
    )
    restore = _independent_g_restore(path)
    if not restore.get("independent_process_restore"):
        raise RuntimeError(f"G checkpoint independent restore failed: {path}")
    return learner, {
        "path": str(path),
        "digest": expected_digest,
        "bytes": path.stat().st_size,
        "parameter_count": learner.parameter_count,
        "restore": restore,
        "lineage_valid": True,
    }


def _load_context_checkpoint(
    *,
    path: Path,
    expected_digest: str,
    parent_manifest_digest: str,
    k_checkpoint_digests: Mapping[str, str],
    parent_g_checkpoint_digest: str,
) -> tuple[ContextLearner, dict[str, Any]]:
    payload = _load_mapping(path)
    learner = ContextLearner.from_checkpoint(payload, device="cpu")
    if str(payload.get("checkpoint_digest")) != expected_digest:
        raise ValueError(f"context checkpoint digest mismatch: {path}")
    learner.assert_lineage(
        parent_manifest_digest=parent_manifest_digest,
        k_checkpoint_digests=k_checkpoint_digests,
        parent_g_checkpoint_digest=parent_g_checkpoint_digest,
    )
    restore = _independent_context_restore(path)
    if not restore.get("independent_process_restore"):
        raise RuntimeError(f"context checkpoint independent restore failed: {path}")
    return learner, {
        "path": str(path),
        "digest": expected_digest,
        "bytes": path.stat().st_size,
        "parameter_count": learner.parameter_count,
        "trainable_parameter_count": learner.trainable_parameter_count,
        "mode": learner.mode,
        "restore": restore,
        "lineage_valid": True,
    }


def _source_gate(
    *,
    p4_1_manifest: Mapping[str, Any],
    p4_2_manifest: Mapping[str, Any],
    p4_2_report: Mapping[str, Any],
    p4_3_manifest: Mapping[str, Any],
    p4_3_report: Mapping[str, Any],
    p3_6_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "p4_2_completed": p4_2_report.get("status") == "completed",
        "p4_3_completed": p4_3_report.get("status") == "completed",
        "p4_2_manifest_report_match": p4_2_report.get("manifest_digest")
        == p4_2_manifest.get("manifest_digest"),
        "p4_3_manifest_report_match": p4_3_report.get("manifest_digest")
        == p4_3_manifest.get("manifest_digest"),
        "p4_2_manifest_digest_valid": _digest_without(p4_2_manifest, "manifest_digest")
        == p4_2_manifest.get("manifest_digest"),
        "p4_3_manifest_digest_valid": _digest_without(p4_3_manifest, "manifest_digest")
        == p4_3_manifest.get("manifest_digest"),
        "p4_2_growth_closed": not p4_2_report.get("growth_admitted")
        and not p4_2_report.get("can_promote"),
        "p4_3_growth_closed": not p4_3_report.get("growth_admitted")
        and not p4_3_report.get("can_promote"),
        "p3_6_validation_only": not p3_6_manifest.get("fit_policy", {}).get("fit_called")
        and not p3_6_manifest.get("fit_policy", {}).get("training_performed"),
        "source_chain_p4_2_to_p4_1": p4_2_manifest.get("source_p4_1_manifest_digest")
        == p4_1_manifest.get("manifest_digest"),
        "source_chain_p4_3_to_p4_2": p4_3_manifest.get("source_p4_2_manifest_digest")
        == p4_2_manifest.get("manifest_digest"),
        "source_chain_p4_3_to_p3_6": p4_3_manifest.get("source_p3_6_manifest_digest")
        == p3_6_manifest.get("manifest_digest"),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"P4.4 source gate failed: {failed}")
    return checks


def _degraded(metrics: Mapping[str, Any], parent: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics["selected_utility_mean"])
        < float(parent["selected_utility_mean"]) - MARGIN_EPSILON
        or float(metrics["behavior_target_hit_rate"])
        < float(parent["behavior_target_hit_rate"]) - MARGIN_EPSILON
        or int(metrics["safe_selection_violations"]) > 0
        or not bool(metrics["reobserve_projection_passed"])
    )


def _historical_reference(
    p4_2_report: Mapping[str, Any], p4_3_report: Mapping[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    p4_2_reference: dict[str, Any] = {}
    degraded: set[str] = set()
    for result in p4_2_report["seed_results"]:
        seed = str(result["seed"])
        parent = result["parent_retention"]
        for arm, arm_metrics in result["metrics"].items():
            metrics = arm_metrics["retention"]
            key = f"p4.2/seed-{seed}/{arm}"
            p4_2_reference[key] = _metric_summary(metrics)
            if _degraded(metrics, parent):
                degraded.add(key)
    p4_3_reference: dict[str, Any] = {}
    for result in p4_3_report["seed_results"]:
        seed = str(result["seed"])
        parent = result["parent_metrics"]["fresh_retention"]
        for arm, arm_metrics in result["metrics"].items():
            key = f"p4.3/seed-{seed}/{arm}"
            p4_3_reference[key] = _metric_summary(arm_metrics["fresh_retention"])
            p4_3_reference[key]["degraded_vs_parent"] = _degraded(
                arm_metrics["fresh_retention"], parent
            )
    return {
        "p4_2_historical_retention": p4_2_reference,
        "p4_3_fresh_retention": p4_3_reference,
        "p4_2_historical_degraded_keys": sorted(degraded),
    }, degraded


def _structure_gate(
    contract: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    actual = [_structure_row(record["candidate_set"], record["behavior_set"]) for record in records]
    row_checks = [actual_row == expected for actual_row, expected in zip(actual, contract["rows"])]
    return {
        "row_count": len(records) == int(contract["row_count"]),
        "candidate_role_and_width_match": all(row_checks),
        "source_contract_digest": contract["contract_digest"],
        "actual_structure_digest": content_digest(actual),
        "rows": actual,
    }


def _checkpoint_payload_digest(path: Path) -> str:
    return str(_load_mapping(path)["checkpoint_digest"])


def _run(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_4_retention_identity_calibration_{uuid4().hex}"
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
        p4_2_report = _load_json(P4_2_REPORT)
        p4_3_manifest = _load_json(P4_3_MANIFEST)
        p4_3_report = _load_json(P4_3_REPORT)
        p3_6_manifest = _load_json(P3_6_MANIFEST)
        p3_5_report = _load_json(P3_5_REPORT)
        source_gate = _source_gate(
            p4_1_manifest=p4_1_manifest,
            p4_2_manifest=p4_2_manifest,
            p4_2_report=p4_2_report,
            p4_3_manifest=p4_3_manifest,
            p4_3_report=p4_3_report,
            p3_6_manifest=p3_6_manifest,
        )
        contract = _extract_structure_contract(p3_6_manifest)
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        p3_2_report = _load_json(P3_2_REPORT)
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        worker_digests = dict(p4_2_manifest["k_checkpoint_digests"])
        semantic_payload = _load_mapping(Path(str(worker_restore["k1"]["path"])))
        transition_payload = _load_mapping(Path(str(worker_restore["k2"]["path"])))
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        run_dir.mkdir(parents=True, exist_ok=False)
        sibling_records = _build_sibling_records(
            contract=contract,
            scratch=run_dir / "sibling-retention",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p3_6_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        sibling_context_records = [_context_record(record) for record in sibling_records]
        structure_gate = _structure_gate(contract, sibling_records)
        old_paths, old_projects = _manifest_identity_sets(p3_6_manifest)
        for manifest in (p4_2_manifest, p4_3_manifest):
            manifest_paths, manifest_projects = _manifest_identity_sets(manifest)
            old_paths.update(manifest_paths)
            old_projects.update(manifest_projects)
        sibling_paths = {str(record["candidate_set"].path) for record in sibling_records}
        sibling_projects = {str(record["candidate_set"].project_id) for record in sibling_records}
        identity_gate = {
            "at_least_two_new_projects": len(sibling_projects) >= 2,
            "at_least_four_new_paths": len(sibling_paths) >= 4,
            "projects_disjoint_from_historical": sibling_projects.isdisjoint(old_projects),
            "paths_disjoint_from_historical": sibling_paths.isdisjoint(old_paths),
            "candidate_digests_unique": len(
                {record["candidate_set"].candidate_set_digest for record in sibling_records}
            )
            == len(sibling_records),
            "behavior_digests_unique": len(
                {record["behavior_set"].behavior_digest for record in sibling_records}
            )
            == len(sibling_records),
            "no_fit_eligible_records": not any(
                bool(record.get("fit_eligible")) for record in sibling_records
            ),
        }
        if not all(identity_gate.values()) or not all(
            structure_gate[key] for key in ("row_count", "candidate_role_and_width_match")
        ):
            raise ValueError("P4.4 sibling identity/structure gate failed")

        parent_metadata = p3_5_report["g_trained_checkpoint"]
        parent_path = Path(str(parent_metadata["path"]))
        parent_payload_digest = _checkpoint_payload_digest(parent_path)
        if content_digest(_load_mapping(parent_path)) != str(parent_metadata["digest"]):
            raise ValueError("P4.4 parent external checkpoint digest drifted")
        parent, parent_checkpoint = _load_g_checkpoint(
            path=parent_path,
            expected_digest=parent_payload_digest,
            parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        if p4_2_manifest["parent_g_checkpoint_digest"] != str(parent_metadata["digest"]):
            raise ValueError("P4.4 parent G checkpoint lineage drifted")
        parent_checkpoint["external_digest"] = str(parent_metadata["digest"])
        parent_metrics = _metric_summary(_evaluate_g(sibling_records, parent))
        seed_results: list[dict[str, Any]] = []
        checkpoint_records: dict[str, Any] = {"parent": parent_checkpoint}
        p4_2_sibling_metrics: dict[str, dict[str, Any]] = {}
        p4_3_sibling_metrics: dict[str, dict[str, Any]] = {}
        for p4_2_seed in p4_2_report["seed_results"]:
            seed = str(p4_2_seed["seed"])
            seed_result: dict[str, Any] = {"seed": int(seed), "p4.2": {}, "p4.3": {}}
            for arm, metadata in p4_2_seed["trained_checkpoints"].items():
                path = Path(str(metadata["path"]))
                if not path.exists() or _checkpoint_payload_digest(path) != str(metadata["digest"]):
                    raise ValueError(f"P4.2 checkpoint artifact drifted: {path}")
                if arm == "fixed-small":
                    learner, checkpoint = _load_g_checkpoint(
                        path=path,
                        expected_digest=str(metadata["digest"]),
                        parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
                        k_checkpoint_digests=worker_digests,
                    )
                    metrics = _evaluate_g(sibling_records, learner)
                else:
                    learner, checkpoint = _load_context_checkpoint(
                        path=path,
                        expected_digest=str(metadata["digest"]),
                        parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
                        k_checkpoint_digests=worker_digests,
                        parent_g_checkpoint_digest=str(p4_2_manifest["parent_g_checkpoint_digest"]),
                    )
                    metrics = _evaluate_context(
                        sibling_context_records,
                        arm=arm,
                        learner=learner,
                    )
                key = f"p4.2/seed-{seed}/{arm}"
                checkpoint_records[key] = checkpoint
                summarized = _metric_summary(metrics)
                summarized["degraded_vs_parent"] = _degraded(metrics, parent_metrics)
                p4_2_sibling_metrics[key] = summarized
                seed_result["p4.2"][arm] = {
                    "checkpoint": checkpoint,
                    "metrics": summarized,
                }
            matching_p4_3 = next(
                item for item in p4_3_report["seed_results"] if int(item["seed"]) == int(seed)
            )
            for arm, metadata in matching_p4_3["trained_checkpoints"].items():
                path = Path(str(metadata["path"]))
                if not path.exists() or _checkpoint_payload_digest(path) != str(metadata["digest"]):
                    raise ValueError(f"P4.3 checkpoint artifact drifted: {path}")
                learner, checkpoint = _load_g_checkpoint(
                    path=path,
                    expected_digest=str(metadata["digest"]),
                    parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
                    k_checkpoint_digests=worker_digests,
                )
                metrics = _evaluate_g(sibling_records, learner)
                key = f"p4.3/seed-{seed}/{arm}"
                checkpoint_records[key] = checkpoint
                summarized = _metric_summary(metrics)
                summarized["degraded_vs_parent"] = _degraded(metrics, parent_metrics)
                p4_3_sibling_metrics[key] = summarized
                seed_result["p4.3"][arm] = {
                    "checkpoint": checkpoint,
                    "metrics": summarized,
                }
            seed_results.append(seed_result)

        historical, historical_degraded = _historical_reference(p4_2_report, p4_3_report)
        sibling_degraded = {
            key for key, metrics in p4_2_sibling_metrics.items() if metrics["degraded_vs_parent"]
        }
        historical_keys_in_sibling = {key for key in historical_degraded if key in sibling_degraded}
        p4_3_sibling_degraded = {
            key for key, metrics in p4_3_sibling_metrics.items() if metrics["degraded_vs_parent"]
        }
        retention_failure_reproduced = bool(historical_degraded) and historical_degraded.issubset(
            sibling_degraded
        )
        retention_artifact_specific = (
            bool(historical_degraded) and not sibling_degraded and not (p4_3_sibling_degraded)
        )
        if retention_failure_reproduced:
            outcome = "retention_failure_reproduced"
        elif retention_artifact_specific:
            outcome = "retention_artifact_specific"
        else:
            outcome = "retention_measurement_unresolved"
        checkpoint_gate = {
            "parent_independent_restore": parent_checkpoint["restore"][
                "independent_process_restore"
            ],
            "all_p4_2_child_checkpoints_independent_restore": all(
                record["restore"]["independent_process_restore"]
                for key, record in checkpoint_records.items()
                if key.startswith("p4.2/")
            ),
            "all_p4_3_child_checkpoints_independent_restore": all(
                record["restore"]["independent_process_restore"]
                for key, record in checkpoint_records.items()
                if key.startswith("p4.3/")
            ),
            "all_lineages_valid": all(
                record["lineage_valid"] for record in checkpoint_records.values()
            ),
            "historical_parent_not_overwritten": True,
        }
        evaluation_gate = {
            "validation_only": True,
            "fit_called": False,
            "training_performed": False,
            "external_target_unused": True,
            "sealed_payload_not_read": True,
            "structural_growth_closed": True,
            "promotion_closed": True,
        }
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p4_1_manifest_digest": p4_1_manifest["manifest_digest"],
            "source_p4_2_manifest_digest": p4_2_manifest["manifest_digest"],
            "source_p4_3_manifest_digest": p4_3_manifest["manifest_digest"],
            "source_p3_6_manifest_digest": p3_6_manifest["manifest_digest"],
            "parent_g_checkpoint_digest": parent_checkpoint["external_digest"],
            "k_checkpoint_digests": worker_digests,
            "fit_policy": {
                "fit_called": False,
                "training_performed": False,
                "validation_only": True,
                "growth_admitted": False,
                "promotion_open": False,
            },
            "structure_contract": contract,
            "sibling_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in sibling_records
            ],
            "sibling_behavior_digests": [
                record["behavior_set"].behavior_digest for record in sibling_records
            ],
            "records": [
                {
                    "candidate_set": record["candidate_set"].to_payload(),
                    "behavior_set": record["behavior_set"].to_payload(),
                    "diagnostic": record["diagnostic"],
                }
                for record in sibling_records
            ],
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "source_gate": source_gate,
                "identity_gate": identity_gate,
                "structure_gate": structure_gate,
                "checkpoint_gate": checkpoint_gate,
                "evaluation_gate": evaluation_gate,
                "parent_metrics": parent_metrics,
                "seed_results": seed_results,
                "historical_reference": historical,
                "sibling_degraded_keys": sorted(sibling_degraded),
                "historical_degraded_keys_in_sibling": sorted(historical_keys_in_sibling),
                "p4_3_sibling_degraded_keys": sorted(p4_3_sibling_degraded),
                "retention_failure_reproduced": retention_failure_reproduced,
                "retention_artifact_specific": retention_artifact_specific,
                "retention_measurement_unresolved": outcome == "retention_measurement_unresolved",
                "outcome": outcome,
                "experiment_passed": False,
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; validation-only; no structural growth or promotion"
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
