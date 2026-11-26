"""Run the P4.3 retention-aware incremental-learning Gate.

P4.2 did not establish a capacity benefit and exposed retention loss after an
incremental G update.  P4.3 keeps the 13-parameter fixed-small learner and
compares two child-training policies:

* ``new-only``: fit only the disjoint new-task cohort;
* ``rehearsal-mix``: fit the same new-task cohort interleaved with an
  explicitly marked rehearsal projection of the P3.6 behavior holdout.

The old P3.6 holdout is never scored after it becomes rehearsal data.  A new,
fresh retention holdout is generated and scored instead.  No structural
growth, context parameters, K-worker updates, or promotion is admitted here.
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
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
from scripts.training.eval_taiji_m5_k_p3_3_g_learning import _fresh_learners  # noqa: E402
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
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)
from taiji.local_learning import apply_linear_delta, mean_squared_error_delta  # noqa: E402

REPORT_FORMAT = "taiji-m5-k-p4-3-retention-incremental-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-3-retention-incremental-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_3_retention_incremental_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_3_retention_incremental_20260911.json"
P4_1_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_1_context_contract_manifest_v1.json"
)
P4_2_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_2_capacity_attribution_manifest_v1.json"
)
P4_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_2_capacity_attribution_20260911.json"
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
P3_6_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_6_behavior_holdout_manifest_v1.json"
)
P3_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"
WIDTHS = (2, 4, 8, 12)
TRAINING_EPOCHS = 8
LEARNING_RATE = 0.15
MARGIN_EPSILON = 1e-9
SEEDS = (0, 1)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _new_specs(split: str, offset: int) -> tuple[dict[str, Any], ...]:
    # P4.0's materializer writes only fixtures in the p40 namespace.  The
    # p43 segment keeps that tested writer while keeping all P4.3 identities
    # disjoint from P4.0/P4.2.
    prefix = f"p40_p43_{split}"
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
            "project_id": f"p4-3-{split}-project-{class_key.lower()}",
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
) -> dict[str, Any]:
    old_candidates = record["candidate_set"]
    old_behavior = record["behavior_set"]
    candidates = tuple(old_candidates.candidates)
    candidate_set = GSelectionCandidateSet.create(
        example_id=(
            f"p4-3:{split}:{source_index}:width-{width}:{old_candidates.candidate_set_digest}"
        ),
        family_id=f"p4-3:{split}:family:{source_index}",
        split=split,
        project_id=old_candidates.project_id,
        path=old_candidates.path,
        input_digest=old_candidates.input_digest,
        candidates=candidates,
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
            "experiment": "p4.3",
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
        "fit_eligible": float(behavior_set.utility_margin) > MARGIN_EPSILON,
    }


def _build_split_records(
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
            label=f"p4-3-{split}-{int(spec['index'])}",
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
        for width in WIDTHS:
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
                )
            )
    return records


def _rehearsal_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["holdout_records"]):
        source_candidates = GSelectionCandidateSet.from_payload(raw["candidate_set"])
        source_behavior = GSelectionBehaviorSet.from_payload(raw["behavior_set"])
        if source_behavior.candidate_set_digest != source_candidates.candidate_set_digest:
            raise ValueError("P4.3 rehearsal behavior/candidate binding drifted")
        candidate_set = GSelectionCandidateSet.create(
            example_id=f"p4-3:rehearsal:{index}:{source_candidates.candidate_set_digest}",
            family_id=f"p4-3:rehearsal:family:{index}",
            split="train",
            project_id=source_candidates.project_id,
            path=source_candidates.path,
            input_digest=source_candidates.input_digest,
            candidates=source_candidates.candidates,
            target_candidate_id=source_candidates.target_candidate_id,
            target_kind=source_candidates.target_kind,
        )
        behavior_set = GSelectionBehaviorSet.create(
            candidate_set_digest=candidate_set.candidate_set_digest,
            inference_digest=candidate_set.inference_digest,
            split="train",
            project_id=candidate_set.project_id,
            path=candidate_set.path,
            outcomes=source_behavior.outcomes,
        )
        records.append(
            {
                "candidate_set": candidate_set,
                "behavior_set": behavior_set,
                "fit_eligible": float(behavior_set.utility_margin) > MARGIN_EPSILON,
                "diagnostic": {
                    "experiment": "p4.3",
                    "arm": "rehearsal-source",
                    "source_manifest_digest": payload["manifest_digest"],
                    "source_split": source_candidates.split,
                    "source_candidate_set_digest": source_candidates.candidate_set_digest,
                    "class_key": raw.get("diagnostic", {}).get("class_key"),
                    "utility_margin": behavior_set.utility_margin,
                },
            }
        )
    return records


def _source_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ("train", "validation", "holdout", "retention"):
        for raw in payload["records"][split]:
            result.append(
                {
                    "candidate_set": GSelectionCandidateSet.from_payload(raw["candidate_set"]),
                    "behavior_set": GSelectionBehaviorSet.from_payload(raw["behavior_set"]),
                }
            )
    return result


def _evaluate(records: Sequence[Mapping[str, Any]], learner: GSelectionLearner) -> dict[str, Any]:
    target_hits = 0
    utility_sum = 0.0
    selected_roles: dict[str, int] = {}
    safe_selection_violations = 0
    reobserve_selected = 0
    reobserve_projection_passed = 0
    workbench_success = 0
    rows: list[dict[str, Any]] = []
    for record in records:
        candidate_set = record["candidate_set"]
        behavior_set = record["behavior_set"]
        decision = learner.select(candidate_set)
        selected_id = decision.selected_candidate_id
        selected = next(
            item for item in candidate_set.candidates if item.candidate_id == selected_id
        )
        outcome = next(item for item in behavior_set.outcomes if item.candidate_id == selected_id)
        target_hit = selected_id == behavior_set.behavior_target_candidate_id
        target_hits += int(target_hit)
        utility_sum += float(outcome.utility)
        selected_roles[selected.candidate_role] = selected_roles.get(selected.candidate_role, 0) + 1
        low_evidence_has_safe = any(
            item.safe_exit_valid or item.safe_exit_progress for item in behavior_set.outcomes
        )
        if low_evidence_has_safe and selected.candidate_role == "proposal":
            safe_selection_violations += 1
        if selected.candidate_role == "reobserve":
            reobserve_selected += 1
            if (
                not outcome.planner_accepted
                and not outcome.route_valid
                and not outcome.execution_success
            ):
                reobserve_projection_passed += 1
        workbench_success += int(outcome.execution_success)
        rows.append(
            {
                "candidate_set_digest": candidate_set.candidate_set_digest,
                "selected_candidate_id": selected_id,
                "behavior_target_candidate_id": behavior_set.behavior_target_candidate_id,
                "target_hit": target_hit,
                "selected_role": selected.candidate_role,
                "selected_utility": outcome.utility,
            }
        )
    count = len(records)
    utility_mean = utility_sum / count if count else 0.0
    return {
        "count": count,
        "behavior_target_hit_count": target_hits,
        "behavior_target_hit_rate": target_hits / count if count else 0.0,
        "selected_utility_sum": utility_sum,
        "selected_utility_mean": utility_mean,
        "selected_residual_error": 1.0 - utility_mean if count else 1.0,
        "selected_roles": selected_roles,
        "safe_selection_violations": safe_selection_violations,
        "reobserve_selected_count": reobserve_selected,
        "reobserve_projection_passed": reobserve_projection_passed == reobserve_selected,
        "workbench_success_count": workbench_success,
        "rows": rows,
    }


def _fit_sequence(
    learner: GSelectionLearner,
    records: Sequence[Mapping[str, Any]],
    *,
    epochs: int,
    learning_rate: float,
    order_seed: int,
    source_labels: Sequence[str],
) -> dict[str, Any]:
    items = tuple(record["candidate_set"] for record in records)
    if not items:
        raise ValueError("P4.3 fit requires a non-empty candidate cohort")
    if len(items) != len(set(item.candidate_set_digest for item in items)):
        raise ValueError("P4.3 fit candidate sets must be unique")
    if any(item.split != "train" for item in items):
        raise ValueError("P4.3 fit accepts train candidate sets only")
    if len(items) != len(source_labels):
        raise ValueError("P4.3 fit source labels are misaligned")
    dataset_digest = content_digest(
        {
            "candidate_set_digests": [item.candidate_set_digest for item in items],
            "source_labels": list(source_labels),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "order_seed": int(order_seed),
        }
    )
    for epoch in range(int(epochs)):
        order = list(range(len(items)))
        random.Random(int(order_seed) + epoch).shuffle(order)
        for index in order:
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
            predictions = learner.model(inputs)
            error = mean_squared_error_delta(predictions, targets)
            apply_linear_delta(learner.model, inputs, error, float(learning_rate))
            learner.training_steps += 1
    learner.revision += 1
    learner.last_train_digest = dataset_digest
    return {
        "dataset_digest": dataset_digest,
        "candidate_sets": len(items),
        "source_counts": {
            "new-task": sum(label == "new-task" for label in source_labels),
            "rehearsal": sum(label == "rehearsal" for label in source_labels),
        },
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "order_seed": int(order_seed),
        "training_steps": learner.training_steps,
        "revision": learner.revision,
    }


def _save_checkpoint(path: Path, learner: GSelectionLearner) -> dict[str, Any]:
    payload = learner.checkpoint()
    _save_torch_atomic(path, payload)
    restored = GSelectionLearner.from_checkpoint(_load_mapping(path), device="cpu")
    independent = _independent_g_restore(path)
    return {
        "path": str(path),
        "digest": payload["checkpoint_digest"],
        "bytes": path.stat().st_size,
        "roundtrip": restored.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"],
        "restore": independent,
        "passed": bool(independent.get("independent_process_restore"))
        and restored.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"],
    }


def _tamper_rejected(payload: Mapping[str, Any]) -> bool:
    tampered = copy.deepcopy(dict(payload))
    tampered["revision"] = int(tampered.get("revision", 0)) + 1
    try:
        GSelectionLearner.from_checkpoint(tampered, device="cpu")
    except ValueError:
        return True
    return False


def _source_gate(
    *,
    p4_1_manifest: Mapping[str, Any],
    p4_2_manifest: Mapping[str, Any],
    p4_2_report: Mapping[str, Any],
    p3_6_manifest: Mapping[str, Any],
) -> None:
    if p4_2_report.get("status") != "completed":
        raise ValueError("P4.3 requires a completed P4.2 report")
    if p4_2_report.get("manifest_digest") != p4_2_manifest.get("manifest_digest"):
        raise ValueError("P4.2 manifest/report digest mismatch")
    if content_digest(
        {key: value for key, value in p4_2_manifest.items() if key != "manifest_digest"}
    ) != p4_2_manifest.get("manifest_digest"):
        raise ValueError("P4.2 manifest content digest mismatch")
    if p4_2_report.get("growth_admitted") or p4_2_report.get("can_promote"):
        raise ValueError("P4.3 cannot consume an admitted P4.2 artifact")
    if p4_2_report.get("attribution", {}).get("status") != "inconclusive":
        raise ValueError("P4.3 requires the unresolved P4.2 attribution outcome")
    if not p4_2_manifest.get("parent_g_checkpoint_digest"):
        raise ValueError("P4.2 parent G digest is missing")
    if p4_1_manifest.get("manifest_digest") != p4_2_manifest.get("source_p4_1_manifest_digest"):
        raise ValueError("P4.2 source P4.1 manifest drifted")
    if p3_6_manifest.get("fit_policy", {}).get("fit_called"):
        raise ValueError("P4.3 rehearsal source must be validation-only")


def _paths_and_projects(records: Iterable[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    paths = {str(record["candidate_set"].path) for record in records}
    projects = {str(record["candidate_set"].project_id) for record in records}
    return paths, projects


def run_experiment(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_3_retention_incremental_{uuid4().hex}"
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
        p3_5_report = _load_json(P3_5_REPORT)
        p3_6_manifest = _load_json(P3_6_MANIFEST)
        _source_gate(
            p4_1_manifest=p4_1_manifest,
            p4_2_manifest=p4_2_manifest,
            p4_2_report=p4_2_report,
            p3_6_manifest=p3_6_manifest,
        )
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        trained_path = Path(str(p3_5_report["g_trained_checkpoint"]["path"]))
        parent_payload = _load_mapping(trained_path)
        parent = GSelectionLearner.from_checkpoint(parent_payload, device="cpu")
        worker_digests = dict(p4_2_manifest["k_checkpoint_digests"])
        parent.assert_lineage(
            parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        if (
            p4_2_manifest["parent_g_checkpoint_digest"]
            != p3_5_report["g_trained_checkpoint"]["digest"]
        ):
            raise ValueError("P4.3 parent G digest drifted")
        parent_restore = _independent_g_restore(trained_path)
        if not parent_restore.get("independent_process_restore"):
            raise RuntimeError("P4.3 parent G independent restore failed")
        p3_2_report = _load_json(P3_2_REPORT)
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        semantic_payload = _load_mapping(Path(str(worker_restore["k1"]["path"])))
        transition_payload = _load_mapping(Path(str(worker_restore["k2"]["path"])))
        if (
            content_digest(semantic_payload) != worker_digests["k1"]
            or content_digest(transition_payload) != worker_digests["k2"]
        ):
            raise ValueError("P4.3 K parent checkpoint digest drifted")
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        run_dir.mkdir(parents=True, exist_ok=False)
        data_root = run_dir / "data"
        split_args = {
            "parent_digest": parent_digest,
            "worker_bundle_digest": bundle.bundle_digest,
            "source_manifest_digest": str(p4_2_manifest["manifest_digest"]),
            "projector": projector,
            "semantic": semantic,
            "transition": transition,
            "semantic_payload": semantic_payload,
        }
        train_records = _build_split_records(
            split="train", offset=43000, scratch=data_root / "train", **split_args
        )
        validation_records = _build_split_records(
            split="validation", offset=43100, scratch=data_root / "validation", **split_args
        )
        holdout_records = _build_split_records(
            split="holdout", offset=43200, scratch=data_root / "holdout", **split_args
        )
        fresh_retention_records = _build_split_records(
            split="retention", offset=43300, scratch=data_root / "retention", **split_args
        )
        rehearsal_records = _rehearsal_records(p3_6_manifest)
        p4_2_source_records = _source_records(p4_2_manifest)
        p3_6_source_records = [
            {
                "candidate_set": GSelectionCandidateSet.from_payload(raw["candidate_set"]),
                "behavior_set": GSelectionBehaviorSet.from_payload(raw["behavior_set"]),
            }
            for raw in p3_6_manifest["holdout_records"]
        ]
        all_new_records = [
            *train_records,
            *validation_records,
            *holdout_records,
            *fresh_retention_records,
        ]
        train_fit = [record for record in train_records if record["fit_eligible"]]
        rehearsal_fit = [record for record in rehearsal_records if record["fit_eligible"]]
        if (
            len(train_fit) < 8
            or len({record["diagnostic"]["class_key"] for record in train_fit}) < 4
        ):
            raise ValueError("P4.3 new train cohort is too small or class-skewed")
        if len(rehearsal_fit) < 2:
            raise ValueError("P4.3 rehearsal cohort is too small")
        new_paths, new_projects = _paths_and_projects(all_new_records)
        p4_2_paths, p4_2_projects = _paths_and_projects(p4_2_source_records)
        p3_6_paths, p3_6_projects = _paths_and_projects(p3_6_source_records)
        identity_gate = {
            "train_records": len(train_records) == 20,
            "validation_records": len(validation_records) == 20,
            "holdout_records": len(holdout_records) == 20,
            "fresh_retention_records": len(fresh_retention_records) == 20,
            "train_fit_records_positive": len(train_fit) >= 8,
            "rehearsal_fit_records_positive": len(rehearsal_fit) >= 2,
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
            "new_disjoint_from_p4_2": new_paths.isdisjoint(p4_2_paths)
            and new_projects.isdisjoint(p4_2_projects),
            "new_disjoint_from_p3_6": new_paths.isdisjoint(p3_6_paths)
            and new_projects.isdisjoint(p3_6_projects),
            "fresh_retention_disjoint_from_new": _paths_and_projects(fresh_retention_records)[
                0
            ].isdisjoint(
                _paths_and_projects([*train_records, *validation_records, *holdout_records])[0]
            )
            and _paths_and_projects(fresh_retention_records)[1].isdisjoint(
                _paths_and_projects([*train_records, *validation_records, *holdout_records])[1]
            ),
            "new_candidate_digests_unique": len(
                {record["candidate_set"].candidate_set_digest for record in all_new_records}
            )
            == 80,
            "new_behavior_digests_unique": len(
                {record["behavior_set"].behavior_digest for record in all_new_records}
            )
            == 80,
        }
        seed_results: list[dict[str, Any]] = []
        for seed in SEEDS:
            seed_dir = run_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=False)
            new_only = GSelectionLearner.from_checkpoint(
                copy.deepcopy(parent_payload), device="cpu"
            )
            rehearsal_mix = GSelectionLearner.from_checkpoint(
                copy.deepcopy(parent_payload), device="cpu"
            )
            zero_checkpoints = {
                "new-only": _save_checkpoint(seed_dir / "new-only-zero.pt", new_only),
                "rehearsal-mix": _save_checkpoint(
                    seed_dir / "rehearsal-mix-zero.pt", rehearsal_mix
                ),
            }
            if not all(item["passed"] for item in zero_checkpoints.values()):
                raise RuntimeError(f"P4.3 zero-step checkpoint preflight failed for seed {seed}")
            new_only_fit = _fit_sequence(
                new_only,
                train_fit,
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
                order_seed=seed,
                source_labels=("new-task",) * len(train_fit),
            )
            mixed_records = [*train_fit, *rehearsal_fit]
            mixed_labels = ("new-task",) * len(train_fit) + ("rehearsal",) * len(rehearsal_fit)
            rehearsal_fit_result = _fit_sequence(
                rehearsal_mix,
                mixed_records,
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
                order_seed=seed,
                source_labels=mixed_labels,
            )
            trained_checkpoints = {
                "new-only": _save_checkpoint(seed_dir / "new-only-trained.pt", new_only),
                "rehearsal-mix": _save_checkpoint(
                    seed_dir / "rehearsal-mix-trained.pt", rehearsal_mix
                ),
            }
            if not all(item["passed"] for item in trained_checkpoints.values()):
                raise RuntimeError(f"P4.3 trained checkpoint preflight failed for seed {seed}")
            parent_metrics = {
                "validation": _evaluate(validation_records, parent),
                "holdout": _evaluate(holdout_records, parent),
                "fresh_retention": _evaluate(fresh_retention_records, parent),
            }
            metrics = {
                "new-only": {
                    "validation": _evaluate(validation_records, new_only),
                    "holdout": _evaluate(holdout_records, new_only),
                    "fresh_retention": _evaluate(fresh_retention_records, new_only),
                },
                "rehearsal-mix": {
                    "validation": _evaluate(validation_records, rehearsal_mix),
                    "holdout": _evaluate(holdout_records, rehearsal_mix),
                    "fresh_retention": _evaluate(fresh_retention_records, rehearsal_mix),
                },
            }
            retention_gate = {
                arm: {
                    "utility_not_below_parent": metrics[arm]["fresh_retention"][
                        "selected_utility_mean"
                    ]
                    >= parent_metrics["fresh_retention"]["selected_utility_mean"] - 1e-9,
                    "target_hit_not_below_parent": metrics[arm]["fresh_retention"][
                        "behavior_target_hit_rate"
                    ]
                    >= parent_metrics["fresh_retention"]["behavior_target_hit_rate"],
                    "safe_selection_preserved": metrics[arm]["fresh_retention"][
                        "safe_selection_violations"
                    ]
                    == 0,
                    "reobserve_projection_passed": metrics[arm]["fresh_retention"][
                        "reobserve_projection_passed"
                    ],
                }
                for arm in metrics
            }
            new_task_gate = {
                arm: {
                    "utility_not_below_new_only": metrics[arm]["holdout"]["selected_utility_mean"]
                    >= metrics["new-only"]["holdout"]["selected_utility_mean"] - 1e-9,
                    "target_hit_not_below_new_only": metrics[arm]["holdout"][
                        "behavior_target_hit_rate"
                    ]
                    >= metrics["new-only"]["holdout"]["behavior_target_hit_rate"],
                    "safe_selection_preserved": metrics[arm]["holdout"]["safe_selection_violations"]
                    == 0,
                    "reobserve_projection_passed": metrics[arm]["holdout"][
                        "reobserve_projection_passed"
                    ],
                }
                for arm in metrics
            }
            tamper_gate = {
                "new_only_tamper_rejected": _tamper_rejected(
                    _load_mapping(Path(trained_checkpoints["new-only"]["path"]))
                ),
                "rehearsal_mix_tamper_rejected": _tamper_rejected(
                    _load_mapping(Path(trained_checkpoints["rehearsal-mix"]["path"]))
                ),
                "wrong_parent_lineage_rejected": False,
                "parent_rollback_digest_equal": False,
            }
            try:
                rehearsal_mix.assert_lineage(
                    parent_manifest_digest="f" * 64,
                    k_checkpoint_digests=worker_digests,
                )
            except ValueError:
                tamper_gate["wrong_parent_lineage_rejected"] = True
            rollback = GSelectionLearner.from_checkpoint(
                copy.deepcopy(parent_payload), device="cpu"
            )
            tamper_gate["parent_rollback_digest_equal"] = (
                rollback.checkpoint()["checkpoint_digest"] == parent_payload["checkpoint_digest"]
            )
            seed_results.append(
                {
                    "seed": seed,
                    "fit": {"new-only": new_only_fit, "rehearsal-mix": rehearsal_fit_result},
                    "zero_checkpoints": zero_checkpoints,
                    "trained_checkpoints": trained_checkpoints,
                    "parent_metrics": parent_metrics,
                    "metrics": metrics,
                    "retention_gate": retention_gate,
                    "new_task_gate": new_task_gate,
                    "tamper_gate": tamper_gate,
                    "parameter_count": {
                        "new-only": new_only.parameter_count,
                        "rehearsal-mix": rehearsal_mix.parameter_count,
                    },
                }
            )
        retention_repaired = all(
            all(bool(value) for value in result["retention_gate"]["rehearsal-mix"].values())
            for result in seed_results
        )
        new_task_noninferior = all(
            all(bool(value) for value in result["new_task_gate"]["rehearsal-mix"].values())
            for result in seed_results
        )
        new_task_gain = all(
            result["metrics"]["rehearsal-mix"]["holdout"]["selected_utility_mean"]
            > result["parent_metrics"]["holdout"]["selected_utility_mean"] + 1e-9
            or result["metrics"]["rehearsal-mix"]["holdout"]["behavior_target_hit_rate"]
            > result["parent_metrics"]["holdout"]["behavior_target_hit_rate"]
            for result in seed_results
        )
        rehearsal_specific_gain = all(
            result["metrics"]["rehearsal-mix"]["fresh_retention"]["selected_utility_mean"]
            > result["metrics"]["new-only"]["fresh_retention"]["selected_utility_mean"] + 1e-9
            or result["metrics"]["rehearsal-mix"]["fresh_retention"]["behavior_target_hit_rate"]
            > result["metrics"]["new-only"]["fresh_retention"]["behavior_target_hit_rate"]
            for result in seed_results
        )
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
            "all_tamper_and_rollback_checks": all(
                all(bool(value) for value in result["tamper_gate"].values())
                for result in seed_results
            ),
            "parent_not_overwritten": True,
        }
        source_gate = {
            "p4_2_completed": True,
            "p4_2_attribution_inconclusive": True,
            "parent_g_independent_restore": bool(parent_restore.get("independent_process_restore")),
            "parent_lineage_valid": True,
            "k_parent_digests_valid": True,
            "growth_not_admitted": True,
        }
        training_gate = {
            "two_arms_present": True,
            "two_deterministic_seeds": len(seed_results) == len(SEEDS),
            "new_validation_not_fit": True,
            "new_holdout_not_fit": True,
            "fresh_retention_not_fit": True,
            "p3_6_used_only_as_rehearsal": True,
            "external_target_unused": True,
            "k_parameters_unchanged": True,
        }
        if (
            retention_repaired
            and new_task_noninferior
            and new_task_gain
            and rehearsal_specific_gain
        ):
            outcome_status = "retention_repaired"
        elif retention_repaired:
            outcome_status = "signal_insufficient"
        else:
            outcome_status = "update_rule_unresolved"
        experiment_passed = (
            all(source_gate.values())
            and all(identity_gate.values())
            and all(checkpoint_gate.values())
            and all(training_gate.values())
            and retention_repaired
            and new_task_noninferior
            and new_task_gain
            and rehearsal_specific_gain
        )
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p4_1_manifest_digest": p4_1_manifest["manifest_digest"],
            "source_p4_2_manifest_digest": p4_2_manifest["manifest_digest"],
            "source_p4_2_report_digest": content_digest(p4_2_report),
            "source_p3_6_manifest_digest": p3_6_manifest["manifest_digest"],
            "parent_g_checkpoint_digest": p4_2_manifest["parent_g_checkpoint_digest"],
            "k_checkpoint_digests": worker_digests,
            "fit_policy": {
                "fit_called": True,
                "training_performed": True,
                "validation_only": False,
                "growth_admitted": False,
                "rehearsal_source_split": "p3.6-holdout",
                "fresh_retention_is_eval_only": True,
            },
            "train_fit_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in train_fit
            ],
            "rehearsal_fit_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in rehearsal_fit
            ],
            "validation_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in validation_records
            ],
            "holdout_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in holdout_records
            ],
            "fresh_retention_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in fresh_retention_records
            ],
            "records": {
                split: [
                    {
                        "candidate_set": record["candidate_set"].to_payload(),
                        "behavior_set": record["behavior_set"].to_payload(),
                        "diagnostic": record.get("diagnostic", {}),
                    }
                    for record in records
                ]
                for split, records in (
                    ("train", train_records),
                    ("validation", validation_records),
                    ("holdout", holdout_records),
                    ("fresh_retention", fresh_retention_records),
                    ("rehearsal", rehearsal_records),
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
                "checkpoint_gate": checkpoint_gate,
                "training_gate": training_gate,
                "retention_repaired": retention_repaired,
                "new_task_noninferior": new_task_noninferior,
                "new_task_gain": new_task_gain,
                "rehearsal_specific_gain": rehearsal_specific_gain,
                "outcome": outcome_status,
                "seed_results": seed_results,
                "parameter_count": {"fixed-small": parent.parameter_count},
                "experiment_passed": experiment_passed,
                "growth_admitted": False,
                "interpretation": (
                    f"completed: outcome={outcome_status}; structural growth remains fail-closed"
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
    result = run_experiment(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
