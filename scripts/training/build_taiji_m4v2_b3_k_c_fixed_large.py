"""Build C-entry fixed-large controls on the exact target-aware courses.

Each cell owns two native K1/K2 replicas rooted at the same inherited worker
parent and trained on the same target-aware course.  Replica order is the only
declared deterministic difference (canonical versus reverse); both replicas
must pass a prefit checkpoint roundtrip before training.  The artifact is not
attached to the default runtime and cannot promote itself.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_loss_diagnostic import (  # noqa: E402
    _course_train_variants,
    _fit_tensor_digests,
)
from scripts.training.eval_taiji_m4v2_b3_k_single_step import (  # noqa: E402
    _all_course_paths,
    _build_experience,
    _load_artifact_paths,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    _parent,
    _restore_worker,
    run_preflight,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    _build_workspace,
    _episode,
    _holdout_episode_paths,
    _observe_all,
    _registry,
    _schema,
)
from taiji import (  # noqa: E402
    KWorkerManifestBundle,
    OutcomeDependencyProjector,
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)
from taiji.k_fixed_large import NativeKFixedLargeEnsemble  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-b3-k-c-fixed-large-build-v1"
VERSION = 1
SOURCE_MANIFEST_FORMAT = "taiji-k-fixed-large-c-entry-source-v1"
RESOURCE_MANIFEST_FORMAT = "taiji-c-entry-cpu-resource-manifest-v1"
ENSEMBLE_WIDTH = 2
SEMANTIC_EPOCHS = 1
SEMANTIC_LR = 2.0
TRANSITION_EPOCHS = 1
TRANSITION_LR = 0.2
MODEL_SEEDS = (17, 23, 31)
COURSE_SEEDS = (0, 1, 2)
DEFAULT_WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_fixed_large_c_entry"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_entry_evaluation_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_c_fixed_large_build_20260910.json"
)


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        return None


def _parameter_bytes(*learners) -> int:
    return sum(
        int(parameter.numel() * parameter.element_size())
        for learner in learners
        for parameter in learner.parameters()
    )


def _load_mapping(path: Path) -> dict[str, object]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


def _atomic_save(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    loaded = _load_mapping(temporary)
    if content_digest(loaded) != content_digest(dict(payload)):
        raise ValueError(f"checkpoint roundtrip changed payload: {path}")
    temporary.replace(path)
    return loaded


def _parameter_delta_norm(before, after) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for before_parameter, after_parameter in zip(
        before.parameters(), after.parameters(), strict=True
    ):
        difference = after_parameter.detach().to(dtype=torch.float64) - before_parameter.detach().to(
            dtype=torch.float64
        )
        total += torch.sum(difference * difference)
    return float(torch.sqrt(total).item())


def _parameter_delta_digest(before, after) -> str:
    payload: dict[str, torch.Tensor] = {}
    for (before_name, before_parameter), (after_name, after_parameter) in zip(
        before.named_parameters(), after.named_parameters(), strict=True
    ):
        if before_name != after_name:
            raise ValueError("fixed-large parameter names changed during fit")
        payload[before_name] = after_parameter.detach() - before_parameter.detach()
    return content_digest(payload)


def _build_course(
    *,
    temp_root: Path,
    artifact_dir: Path,
    model_seed: int,
    course_seed: int,
    parent_digest: str,
    parent_bundle: KWorkerManifestBundle,
    source_manifest_digest: str,
    projector: OutcomeDependencyProjector,
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[int, ...], tuple[tuple[str, ...], ...]]:
    _build_workspace(temp_root, task_seed=0)
    schema = _schema()
    train_observations = {
        observation.path: observation
        for observation in _observe_all(
            temp_root,
            registry=_registry(typescript_available=False),
            split="c-entry-fixed-large-train",
            paths=_all_course_paths(),
            schema=schema,
        )
    }
    holdout_paths = sorted(
        {path for episode in _holdout_episode_paths() for path in episode}
    )
    holdout_observations = {
        observation.path: observation
        for observation in _observe_all(
            temp_root,
            registry=_registry(typescript_available=True),
            split="c-entry-fixed-large-validation",
            paths=["missing_00.txt", *holdout_paths],
            schema=schema,
        )
    }
    train_indexes, train_variants = _course_train_variants(
        course_seed,
        count=3,
        strategy="target_aware",
    )
    train_experiences = tuple(
        _build_experience(
            sequence=_episode(
                train_observations["missing_00.txt"],
                train_observations,
                episode_paths,
            ),
            split="train",
            name=f"c-fixed-large-train-{index}",
            parent_digest=parent_digest,
            worker_bundle_digest=parent_bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        for index, episode_paths in zip(train_indexes, train_variants, strict=True)
    )
    holdout_anchor = holdout_observations["missing_00.txt"]
    holdout_experiences = tuple(
        _build_experience(
            sequence=_episode(holdout_anchor, holdout_observations, episode_paths),
            split="holdout",
            name=f"c-fixed-large-holdout-{index}",
            parent_digest=parent_digest,
            worker_bundle_digest=parent_bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        for index, episode_paths in enumerate(_holdout_episode_paths()[:3])
    )
    train_input_digests = {
        digest
        for experience in train_experiences
        for digest in (
            experience.semantic_example.input_digest,
            experience.transition_example.input_digest,
        )
    }
    holdout_input_digests = {
        digest
        for experience in holdout_experiences
        for digest in (
            experience.semantic_example.input_digest,
            experience.transition_example.input_digest,
        )
    }
    if train_input_digests & holdout_input_digests:
        raise ValueError("C-entry fixed-large course overlaps validation input")
    return train_experiences, holdout_experiences, train_indexes, train_variants


def _fit_replica(
    *,
    semantic_parent,
    transition_parent,
    train_experiences: Sequence[object],
    reverse_order: bool,
    output_dir: Path,
    replica_index: int,
) -> tuple[dict[str, object], StructuredSemanticLearner, StructuredSemanticTransitionLearner]:
    semantic = StructuredSemanticLearner.from_checkpoint(
        copy.deepcopy(semantic_parent.checkpoint()), device="cpu"
    )
    transition = StructuredSemanticTransitionLearner.from_checkpoint(
        copy.deepcopy(transition_parent.checkpoint()), device="cpu"
    )
    prefit_dir = output_dir / f"replica_{replica_index}" / "prefit"
    prefit_dir.mkdir(parents=True, exist_ok=True)
    prefit_semantic = _atomic_save(
        prefit_dir / "k1_semantic.pt", semantic.checkpoint()
    )
    prefit_transition = _atomic_save(
        prefit_dir / "k2_transition.pt", transition.checkpoint()
    )
    semantic = StructuredSemanticLearner.from_checkpoint(prefit_semantic, device="cpu")
    transition = StructuredSemanticTransitionLearner.from_checkpoint(
        prefit_transition, device="cpu"
    )
    ordered = tuple(reversed(train_experiences)) if reverse_order else tuple(train_experiences)
    semantic_before = StructuredSemanticLearner.from_checkpoint(
        copy.deepcopy(semantic.checkpoint()), device="cpu"
    )
    transition_before = StructuredSemanticTransitionLearner.from_checkpoint(
        copy.deepcopy(transition.checkpoint()), device="cpu"
    )
    semantic_losses: dict[str, float] = {}
    transition_losses: dict[str, float] = {}
    for experience in ordered:
        semantic_losses = semantic.fit(
            (experience.semantic_example,),
            epochs=SEMANTIC_EPOCHS,
            learning_rate=SEMANTIC_LR,
        )
        transition_losses = transition.fit(
            (experience.transition_example,),
            epochs=TRANSITION_EPOCHS,
            learning_rate=TRANSITION_LR,
        )
    semantic_checkpoint = _atomic_save(
        output_dir / f"replica_{replica_index}" / "k1_semantic.pt",
        semantic.checkpoint(),
    )
    transition_checkpoint = _atomic_save(
        output_dir / f"replica_{replica_index}" / "k2_transition.pt",
        transition.checkpoint(),
    )
    restored_semantic = StructuredSemanticLearner.from_checkpoint(
        semantic_checkpoint, device="cpu"
    )
    restored_transition = StructuredSemanticTransitionLearner.from_checkpoint(
        transition_checkpoint, device="cpu"
    )
    report = {
        "replica_index": replica_index,
        "reverse_order": reverse_order,
        "prefit_checkpoint_gate": {
            "k1.semantic": content_digest(prefit_semantic)
            == content_digest(semantic_before.checkpoint()),
            "k2.transition": content_digest(prefit_transition)
            == content_digest(transition_before.checkpoint()),
        },
        "postfit_restore_gate": {
            "k1.semantic": restored_semantic.owner_digests()
            == semantic.owner_digests(),
            "k2.transition": restored_transition.owner_digests()
            == transition.owner_digests(),
        },
        "k1_checkpoint_path": str(output_dir / f"replica_{replica_index}" / "k1_semantic.pt"),
        "k2_checkpoint_path": str(output_dir / f"replica_{replica_index}" / "k2_transition.pt"),
        "k1_checkpoint_digest": content_digest(semantic_checkpoint),
        "k2_checkpoint_digest": content_digest(transition_checkpoint),
        "k1_parameter_delta_digest": _parameter_delta_digest(
            semantic_before, restored_semantic
        ),
        "k2_parameter_delta_digest": _parameter_delta_digest(
            transition_before, restored_transition
        ),
        "k1_parameter_delta_norm": _parameter_delta_norm(
            semantic_before, restored_semantic
        ),
        "k2_parameter_delta_norm": _parameter_delta_norm(
            transition_before, restored_transition
        ),
        "k1_training_steps": restored_semantic.training_steps,
        "k2_training_steps": restored_transition.training_steps,
        "losses": {"k1.semantic": semantic_losses, "k2.transition": transition_losses},
    }
    return report, restored_semantic, restored_transition


def run_cell(
    *,
    worker_root: Path,
    output_root: Path,
    manifest: Mapping[str, object],
    model_seed: int,
    course_seed: int,
) -> dict[str, object]:
    started = time.perf_counter()
    artifact_dir = worker_root / f"model_{model_seed}"
    artifacts = _load_artifact_paths(artifact_dir)
    parent_payload = _parent(model_seed)
    parent_digest = content_digest(parent_payload)
    candidate_namespace = str(artifacts["k1.semantic"]["candidate_namespace"])
    preflight = run_preflight(
        semantic_checkpoint=str(artifact_dir / "taiji_r6_k1_semantic.pt"),
        transition_checkpoint=str(artifact_dir / "taiji_r6_k2_transition.pt"),
        projection_checkpoint=str(artifact_dir / "taiji_r6_k3_outcome_projection.pt"),
        model_seed=model_seed,
        course_seed=course_seed,
        candidate_namespace=candidate_namespace,
    )
    if preflight.get("status") != "passed":
        raise RuntimeError("C-entry fixed-large requires passed worker preflight")
    manifests = []
    for worker_id in ("k1.semantic", "k2.transition", "k3.outcome_projection"):
        manifest_item, _ = _restore_worker(
            worker_id,
            artifacts[worker_id],
            parent_digest=parent_digest,
        )
        manifests.append(manifest_item)
    parent_bundle = KWorkerManifestBundle.create(
        parent_checkpoint_digest=parent_digest,
        source_manifest_digest=str(artifacts["k1.semantic"]["source_manifest_digest"]),
        resource_manifest_digest=str(artifacts["k1.semantic"]["resource_manifest_digest"]),
        candidate_namespace=candidate_namespace,
        workers=manifests,
    )
    source_manifest_digest = str(artifacts["k1.semantic"]["source_manifest_digest"])
    projector = OutcomeDependencyProjector.from_checkpoint(
        copy.deepcopy(artifacts["k3.outcome_projection"]["checkpoint"])
    )
    temp_parent = PROJECT_ROOT / ".tmp-m4v2-b3-k-c-fixed-large"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / uuid4().hex
    temp_root.mkdir()
    artifact_dir_out = output_root / f"model_{model_seed}" / f"course_{course_seed}"
    try:
        train, holdout, indexes, variants = _build_course(
            temp_root=temp_root,
            artifact_dir=artifact_dir,
            model_seed=model_seed,
            course_seed=course_seed,
            parent_digest=parent_digest,
            parent_bundle=parent_bundle,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        parent_semantic = StructuredSemanticLearner.from_checkpoint(
            copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
        )
        parent_transition = StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
        )
        fit_digests = [
            {
                "experience_digest": experience.experience_digest,
                **_fit_tensor_digests(parent_semantic, parent_transition, experience),
            }
            for experience in train
        ]
        target_digests = [
            content_digest(
                {
                    "k1.semantic": item["k1.semantic.target_tensor_digest"],
                    "k2.transition": item["k2.transition.target_tensor_digest"],
                }
            )
            for item in fit_digests
        ]
        manifest_course = manifest["course_contract"]
        source_manifest = {
            "format": SOURCE_MANIFEST_FORMAT,
            "version": 1,
            "parent_checkpoint_digest": parent_digest,
            "course_contract_digest": str(manifest["course_contract_digest"]),
            "model_seed": int(model_seed),
            "course_seed": int(course_seed),
            "episode_indexes": list(indexes),
            "episode_paths": [list(paths) for paths in variants],
            "train_experience_digests": [
                experience.experience_digest for experience in train
            ],
            "holdout_experience_digests": [
                experience.experience_digest for experience in holdout
            ],
            "train_target_digests": target_digests,
            "train_target_multiset_digest": content_digest(sorted(target_digests)),
            "validation_paths": sorted(
                {
                    path
                    for episode in manifest["validation_split"]["episodes"]
                    for path in episode
                }
            ),
            "path_sets": {
                "training": sorted(
                    {
                        path
                        for episode_paths in variants
                        for path in episode_paths
                    }
                ),
                "formal_holdout": sorted(
                    {
                        path
                        for episode in manifest["validation_split"]["episodes"]
                        for path in episode
                    }
                ),
            },
            "sealed_test_artifact_digest": str(
                manifest["sealed_test_split"]["artifact_digest"]
            ),
            "replica_update_order": ["canonical", "reverse"],
            "training": {
                "update_schedule": "single-example-sequential",
                "semantic_epochs": SEMANTIC_EPOCHS,
                "semantic_learning_rate": SEMANTIC_LR,
                "transition_epochs": TRANSITION_EPOCHS,
                "transition_learning_rate": TRANSITION_LR,
            },
            "course_contract_snapshot_digest": content_digest(manifest_course),
        }
        source_manifest_digest = content_digest(source_manifest)
        replica_reports = []
        semantic_replicas = []
        transition_replicas = []
        training_before_rss = _rss_bytes()
        training_started = time.perf_counter()
        for replica_index, reverse_order in enumerate((False, True)):
            replica_report, semantic_replica, transition_replica = _fit_replica(
                semantic_parent=parent_semantic,
                transition_parent=parent_transition,
                train_experiences=train,
                reverse_order=reverse_order,
                output_dir=artifact_dir_out,
                replica_index=replica_index,
            )
            replica_reports.append(replica_report)
            semantic_replicas.append(semantic_replica)
            transition_replicas.append(transition_replica)
        training_elapsed = time.perf_counter() - training_started
        training_after_rss = _rss_bytes()
        ensemble = NativeKFixedLargeEnsemble(semantic_replicas, transition_replicas)
        ensemble_checkpoint = ensemble.checkpoint()
        restored_ensemble = NativeKFixedLargeEnsemble.from_checkpoint(
            ensemble_checkpoint
        )
        resource_manifest = {
            "format": RESOURCE_MANIFEST_FORMAT,
            "version": 1,
            "device": "cpu",
            "cuda_required": False,
            "optimizer_state_present": False,
            "ensemble_width": ENSEMBLE_WIDTH,
            "semantic_epochs_per_replica": SEMANTIC_EPOCHS,
            "transition_epochs_per_replica": TRANSITION_EPOCHS,
        }
        checkpoint_path = artifact_dir_out / "taiji_c_entry_k_fixed_large_ensemble.pt"
        checkpoint_paths = tuple(
            sorted(
                path
                for path in artifact_dir_out.rglob("*.pt")
                if path != checkpoint_path
            )
        )
        checkpoint_write_bytes = sum(int(path.stat().st_size) for path in checkpoint_paths)
        training_steps = sum(
            int(item["k1_training_steps"]) + int(item["k2_training_steps"])
            for item in replica_reports
        )
        rss_values = tuple(
            value for value in (training_before_rss, training_after_rss) if value is not None
        )
        resource_measurement = {
            "device": "cpu",
            "resource_manifest_digest": content_digest(resource_manifest),
            "training_wall_clock_seconds": float(training_elapsed),
            "inference_wall_clock_seconds": None,
            "training_peak_working_set_bytes": (
                max(training_before_rss, training_after_rss)
                if training_before_rss is not None and training_after_rss is not None
                else None
            ),
            "inference_peak_working_set_bytes": None,
            "peak_working_set_bytes": max(rss_values) if rss_values else None,
            "peak_working_set_method": "process_rss_before_after_lower_bound",
            "training_update_steps": training_steps,
            "worker_parameter_count": int(restored_ensemble.parameter_count),
            "worker_parameter_bytes": _parameter_bytes(
                *semantic_replicas, *transition_replicas
            ),
            "candidate_parameter_bytes": _parameter_bytes(
                *semantic_replicas, *transition_replicas
            ),
            "parameter_bytes": _parameter_bytes(
                *semantic_replicas, *transition_replicas
            ),
            "checkpoint_write_bytes": checkpoint_write_bytes,
            "checkpoint_write_paths": [str(path) for path in checkpoint_paths],
            "inference_trace_count": None,
            "measurement_complete": False,
        }
        k3_checkpoint = copy.deepcopy(artifacts["k3.outcome_projection"]["checkpoint"])
        unsigned = {
            "format": "taiji-k-fixed-large-c-entry-ensemble-v1",
            "version": 1,
            "parent_checkpoint_digest": parent_digest,
            "candidate_namespace": (
                f"taiji:k:fixed-large:c-entry:model-{model_seed}:course-{course_seed}"
            ),
            "model_seed": int(model_seed),
            "course_seed": int(course_seed),
            "ensemble_width": ENSEMBLE_WIDTH,
            "source_manifest": source_manifest,
            "source_manifest_digest": source_manifest_digest,
            "resource_manifest": resource_manifest,
            "resource_manifest_digest": content_digest(resource_manifest),
            "replica_reports": replica_reports,
            "resource": resource_measurement,
            "ensemble_checkpoint": ensemble_checkpoint,
            "ensemble_checkpoint_digest": content_digest(ensemble_checkpoint),
            "k3_projection_checkpoint": k3_checkpoint,
            "k3_projection_checkpoint_digest": content_digest(k3_checkpoint),
            "optimizer_state_present": False,
            "default_runtime_attached": False,
        }
        payload = {**unsigned, "artifact_digest": content_digest(unsigned)}
        artifact_path = checkpoint_path
        saved = _atomic_save(artifact_path, payload)
        restored_payload = _load_mapping(artifact_path)
        restored_ensemble_from_disk = NativeKFixedLargeEnsemble.from_checkpoint(
            restored_payload["ensemble_checkpoint"]
        )
        replica_distinct = len(
            {
                (item["k1_checkpoint_digest"], item["k2_checkpoint_digest"])
                for item in replica_reports
            }
        ) == ENSEMBLE_WIDTH
        report_resource = dict(resource_measurement)
        report_resource["checkpoint_write_bytes"] = int(
            resource_measurement["checkpoint_write_bytes"]
        ) + int(artifact_path.stat().st_size)
        report_resource["checkpoint_write_paths"] = [
            *resource_measurement["checkpoint_write_paths"],
            str(artifact_path),
        ]
        report_resource["measurement_complete"] = bool(
            report_resource["peak_working_set_bytes"] is not None
        )
        report = {
            "report_format": REPORT_FORMAT,
            "version": VERSION,
            "status": "passed" if replica_distinct else "failed",
            "run_kind": "c-entry-fixed-large-build",
            "model_seed": int(model_seed),
            "course_seed": int(course_seed),
            "parent_checkpoint_digest": parent_digest,
            "course_contract_digest": str(manifest["course_contract_digest"]),
            "target_multiset_digest": source_manifest["train_target_multiset_digest"],
            "train_episode_indexes": list(indexes),
            "artifact_path": str(artifact_path),
            "artifact_digest": str(saved["artifact_digest"]),
            "source_manifest_digest": source_manifest_digest,
            "ensemble_checkpoint_digest": content_digest(ensemble_checkpoint),
            "ensemble_fresh_restore": content_digest(restored_ensemble.checkpoint())
            == content_digest(ensemble_checkpoint),
            "disk_ensemble_fresh_restore": content_digest(
                restored_ensemble_from_disk.checkpoint()
            )
            == content_digest(ensemble_checkpoint),
            "replica_updates_distinct": replica_distinct,
            "replica_reports": replica_reports,
            "resource": report_resource,
            "worker_attachment_preflight": preflight,
            "default_runtime_attached": False,
            "training_performed": True,
            "cuda_used": False,
            "sealed_test_scored": False,
            "can_start_formal": False,
            "can_promote": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        if temp_parent.exists() and not any(temp_parent.iterdir()):
            temp_parent.rmdir()


def run_matrix(
    *,
    worker_root: Path = DEFAULT_WORKER_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    model_seeds: tuple[int, ...] = MODEL_SEEDS,
    course_seeds: tuple[int, ...] = COURSE_SEEDS,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = []
    for model_seed in model_seeds:
        for course_seed in course_seeds:
            cells.append(
                run_cell(
                    worker_root=worker_root,
                    output_root=output_root,
                    manifest=manifest,
                    model_seed=model_seed,
                    course_seed=course_seed,
                )
            )
    passed = all(cell["status"] == "passed" for cell in cells)
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if passed else "failed",
        "run_kind": "c-entry-fixed-large-build-matrix",
        "model_seeds": list(model_seeds),
        "course_seeds": list(course_seeds),
        "cell_count": len(cells),
        "cells": cells,
        "course_contract_digest": str(manifest["course_contract_digest"]),
        "sealed_test_artifact_digest": str(
            manifest["sealed_test_split"]["artifact_digest"]
        ),
        "training_performed": True,
        "sealed_test_scored": False,
        "can_start_formal": False,
        "can_promote": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    worker_root = args.worker_root if args.worker_root.is_absolute() else PROJECT_ROOT / args.worker_root
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    manifest = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    report = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    result = run_matrix(
        worker_root=worker_root,
        output_root=output_root,
        manifest_path=manifest,
        report_path=report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
