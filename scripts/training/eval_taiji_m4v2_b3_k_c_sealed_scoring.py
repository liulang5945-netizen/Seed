"""Run a read-only C-entry validation/sealed scoring canary.

The canary consumes already-built parent, continuation-candidate and C-entry
fixed-large artifacts.  It does not train, mutate, attach, or promote any
artifact.  Resource measurement remains a separate Gate before formal.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_loss_diagnostic import (  # noqa: E402
    _loss_score,
)
from scripts.training.eval_taiji_m4v2_b3_k_single_step import (  # noqa: E402
    _build_experience,
    _load_artifact_paths,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    _parent,
    _restore_worker,
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
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    WorkbenchObservation,
    content_digest,
)
from taiji.k_fixed_large import NativeKFixedLargeEnsemble  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-b3-k-c-sealed-scoring-canary-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
COURSE_SEEDS = (0, 1, 2)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_entry_evaluation_v1.json"
)
DEFAULT_TARGET_AWARE_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_target_aware_model_seeds_v2_20260910.json"
)
DEFAULT_WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers"
DEFAULT_FIXED_LARGE_ROOT = (
    PROJECT_ROOT / "checkpoints" / "taiji_k_fixed_large_c_entry"
)
DEFAULT_SEALED_TEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_sealed_test_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_c_sealed_scoring_20260910.json"
)


def _load_mapping(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"artifact is not a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


def _mse(scores: Mapping[str, float], target: Mapping[str, float]) -> float:
    if set(scores) != set(target):
        raise ValueError(
            f"fixed-large score keys drifted: scores={sorted(scores)} target={sorted(target)}"
        )
    return sum((float(scores[key]) - float(target[key])) ** 2 for key in target) / len(
        target
    )


def _fixed_large_loss_score(ensemble, experiences) -> dict[str, float]:
    rows = []
    for experience in experiences:
        semantic_example = experience.semantic_example
        semantic_result = ensemble.predict_semantic(semantic_example.percept)
        semantic_fact_target = {
            key: float(key in semantic_example.fact_keys)
            for key in ensemble.semantic_fact_keys
        }
        semantic_goal_target = {
            key: float(key == semantic_example.goal.goal_id)
            for key in ensemble.semantic_goal_ids
        }
        semantic_content_target = {
            key: float(key == semantic_example.content.content_id)
            for key in ensemble.semantic_content_ids
        }

        transition_example = experience.transition_example
        transition_result = ensemble.predict_transition(
            transition_example.before,
            transition_example.event,
        )
        representative = ensemble.transition_replicas[0]
        current = representative._fact_vector(transition_example.before)
        target_next = representative._fact_vector(transition_example.after)
        transition_delta_target = {
            key: float(value)
            for key, value in zip(
                ensemble.transition_fact_keys,
                target_next - current,
                strict=True,
            )
        }
        transition_goal_target = {
            key: float(key == transition_example.goal.goal_id)
            for key in ensemble.transition_goal_ids
        }
        transition_content_target = {
            key: float(key == transition_example.content.content_id)
            for key in ensemble.transition_content_ids
        }
        rows.append(
            {
                "k1.fact_mse": _mse(
                    semantic_result.fact_scores, semantic_fact_target
                ),
                "k1.goal_mse": _mse(
                    semantic_result.goal_scores, semantic_goal_target
                ),
                "k1.content_mse": _mse(
                    semantic_result.content_scores, semantic_content_target
                ),
                "k2.transition_mse": _mse(
                    transition_result.delta_scores, transition_delta_target
                ),
                "k2.goal_mse": _mse(
                    transition_result.goal_scores, transition_goal_target
                ),
                "k2.content_mse": _mse(
                    transition_result.content_scores, transition_content_target
                ),
            }
        )
    means = {
        key: sum(row[key] for row in rows) / len(rows) for key in rows[0]
    }
    means["combined_mse"] = sum(means.values()) / len(means)
    return means


def _delta(after: Mapping[str, float], before: Mapping[str, float]) -> dict[str, float]:
    return {key: float(after[key]) - float(before[key]) for key in before}


def _context(
    *,
    worker_root: Path,
    model_seed: int,
) -> tuple[dict[str, Any], str, KWorkerManifestBundle, Any]:
    artifact_dir = worker_root / f"model_{model_seed}"
    artifacts = _load_artifact_paths(artifact_dir)
    parent_digest = content_digest(_parent(model_seed))
    manifests = []
    for worker_id in ("k1.semantic", "k2.transition", "k3.outcome_projection"):
        manifest, _ = _restore_worker(
            worker_id,
            artifacts[worker_id],
            parent_digest=parent_digest,
        )
        manifests.append(manifest)
    bundle = KWorkerManifestBundle.create(
        parent_checkpoint_digest=parent_digest,
        source_manifest_digest=str(artifacts["k1.semantic"]["source_manifest_digest"]),
        resource_manifest_digest=str(artifacts["k1.semantic"]["resource_manifest_digest"]),
        candidate_namespace=str(artifacts["k1.semantic"]["candidate_namespace"]),
        workers=manifests,
    )
    from taiji import OutcomeDependencyProjector

    projector = OutcomeDependencyProjector.from_checkpoint(
        copy.deepcopy(artifacts["k3.outcome_projection"]["checkpoint"])
    )
    return artifacts, parent_digest, bundle, projector


def _validation_experiences(
    *,
    temp_root: Path,
    artifacts: Mapping[str, Any],
    parent_digest: str,
    bundle: KWorkerManifestBundle,
    projector: Any,
) -> tuple[Any, ...]:
    _build_workspace(temp_root, task_seed=0)
    schema = _schema()
    paths = sorted({path for episode in _holdout_episode_paths() for path in episode})
    observations = {
        observation.path: observation
        for observation in _observe_all(
            temp_root,
            registry=_registry(typescript_available=True),
            split="c-entry-validation",
            paths=["missing_00.txt", *paths],
            schema=schema,
        )
    }
    return tuple(
        _build_experience(
            sequence=_episode(
                observations["missing_00.txt"], observations, episode_paths
            ),
            split="holdout",
            name=f"loss-holdout-{index}",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(
                artifacts["k1.semantic"]["source_manifest_digest"]
            ),
            projector=projector,
        )
        for index, episode_paths in enumerate(_holdout_episode_paths()[:3])
    )


def _sealed_experiences(
    *,
    sealed: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    parent_digest: str,
    bundle: KWorkerManifestBundle,
    projector: Any,
) -> tuple[Any, ...]:
    anchor = WorkbenchObservation.from_payload(sealed["anchor_payload"])
    observations = {
        item["path"]: WorkbenchObservation.from_payload(item)
        for item in sealed["observation_payloads"]
    }
    return tuple(
        _build_experience(
            sequence=_episode(anchor, observations, episode["paths"]),
            split="holdout",
            name=str(episode["episode_id"]),
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(
                artifacts["k1.semantic"]["source_manifest_digest"]
            ),
            projector=projector,
        )
        for episode in sealed["episodes"]
    )


def _candidate_paths(target_report: Mapping[str, Any], model_seed: int, course_seed: int) -> dict[str, Path]:
    model_report = next(
        item["report"]
        for item in target_report["model_reports"]
        if int(item["model_seed"]) == int(model_seed)
    )
    cell = next(
        item["report"]
        for item in model_report["cells"]
        if int(item["course_seed"]) == int(course_seed)
    )
    return {
        key: Path(value)
        for key, value in cell["candidate_checkpoint_paths"].items()
    }


def _arm_scores(
    *,
    artifacts: Mapping[str, Any],
    candidate_paths: Mapping[str, Path],
    fixed_large_path: Path,
    validation: tuple[Any, ...],
    sealed: tuple[Any, ...],
) -> dict[str, Any]:
    parent_semantic = StructuredSemanticLearner.from_checkpoint(
        copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
    )
    parent_transition = StructuredSemanticTransitionLearner.from_checkpoint(
        copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
    )
    candidate_semantic = StructuredSemanticLearner.from_checkpoint(
        _load_mapping(candidate_paths["k1.semantic"]), device="cpu"
    )
    candidate_transition = StructuredSemanticTransitionLearner.from_checkpoint(
        _load_mapping(candidate_paths["k2.transition"]), device="cpu"
    )
    fixed_large_payload = _load_mapping(fixed_large_path)
    fixed_large = NativeKFixedLargeEnsemble.from_checkpoint(
        fixed_large_payload["ensemble_checkpoint"]
    )
    scores = {}
    for split, experiences in (("validation", validation), ("sealed", sealed)):
        frozen_loss = _loss_score(parent_semantic, parent_transition, experiences)
        candidate_loss = _loss_score(
            candidate_semantic, candidate_transition, experiences
        )
        fixed_large_loss = _fixed_large_loss_score(fixed_large, experiences)
        scores[split] = {
            "frozen_parent": frozen_loss,
            "candidate_continuation": candidate_loss,
            "fixed_large": fixed_large_loss,
            "candidate_delta_vs_frozen": _delta(candidate_loss, frozen_loss),
            "fixed_large_delta_vs_frozen": _delta(fixed_large_loss, frozen_loss),
            "candidate_delta_vs_fixed_large": _delta(
                candidate_loss, fixed_large_loss
            ),
        }
    return scores


def run_canary(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    target_aware_report_path: Path = DEFAULT_TARGET_AWARE_REPORT,
    worker_root: Path = DEFAULT_WORKER_ROOT,
    fixed_large_root: Path = DEFAULT_FIXED_LARGE_ROOT,
    sealed_test_path: Path = DEFAULT_SEALED_TEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_report = json.loads(target_aware_report_path.read_text(encoding="utf-8"))
    sealed = json.loads(sealed_test_path.read_text(encoding="utf-8"))
    if sealed["artifact_digest"] != manifest["sealed_test_split"]["artifact_digest"]:
        raise ValueError("sealed-test artifact does not match frozen C-entry manifest")
    temp_parent = PROJECT_ROOT / ".tmp-m4v2-b3-k-c-scoring"
    temp_parent.mkdir(parents=True, exist_ok=True)
    cells = []
    try:
        for model_seed in MODEL_SEEDS:
            artifacts, parent_digest, bundle, projector = _context(
                worker_root=worker_root,
                model_seed=model_seed,
            )
            temp_root = temp_parent / f"model_{model_seed}_{uuid4().hex}"
            temp_root.mkdir()
            try:
                validation = _validation_experiences(
                    temp_root=temp_root,
                    artifacts=artifacts,
                    parent_digest=parent_digest,
                    bundle=bundle,
                    projector=projector,
                )
                sealed_experiences = _sealed_experiences(
                    sealed=sealed,
                    artifacts=artifacts,
                    parent_digest=parent_digest,
                    bundle=bundle,
                    projector=projector,
                )
                for course_seed in COURSE_SEEDS:
                    candidate_paths = _candidate_paths(
                        target_report, model_seed, course_seed
                    )
                    fixed_large_path = (
                        fixed_large_root
                        / f"model_{model_seed}"
                        / f"course_{course_seed}"
                        / "taiji_c_entry_k_fixed_large_ensemble.pt"
                    )
                    scores = _arm_scores(
                        artifacts=artifacts,
                        candidate_paths=candidate_paths,
                        fixed_large_path=fixed_large_path,
                        validation=validation,
                        sealed=sealed_experiences,
                    )
                    cells.append(
                        {
                            "model_seed": model_seed,
                            "course_seed": course_seed,
                            "parent_checkpoint_digest": parent_digest,
                            "candidate_checkpoint_paths": {
                                key: str(value) for key, value in candidate_paths.items()
                            },
                            "fixed_large_path": str(fixed_large_path),
                            "validation": scores["validation"],
                            "sealed": scores["sealed"],
                            "candidate_artifacts_read_only": True,
                            "sealed_test_scored": True,
                        }
                    )
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)
    finally:
        if temp_parent.exists() and not any(temp_parent.iterdir()):
            temp_parent.rmdir()

    sealed_candidate_deltas = [
        float(cell["sealed"]["candidate_delta_vs_frozen"]["combined_mse"])
        for cell in cells
    ]
    sealed_fixed_large_deltas = [
        float(cell["sealed"]["fixed_large_delta_vs_frozen"]["combined_mse"])
        for cell in cells
    ]
    candidate_beats_fixed_large = [
        candidate <= fixed
        for candidate, fixed in zip(
            sealed_candidate_deltas, sealed_fixed_large_deltas, strict=True
        )
    ]
    validation_candidate_deltas = [
        float(cell["validation"]["candidate_delta_vs_frozen"]["combined_mse"])
        for cell in cells
    ]
    resource_measurement_complete = False
    technical_gate_passed = all(
        cell["candidate_artifacts_read_only"]
        and cell["sealed_test_scored"]
        for cell in cells
    )
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if technical_gate_passed else "failed",
        "run_kind": "c-entry-sealed-scoring-canary",
        "model_seeds": list(MODEL_SEEDS),
        "course_seeds": list(COURSE_SEEDS),
        "cell_count": len(cells),
        "cells": cells,
        "validation_candidate_combined_delta_mean": sum(validation_candidate_deltas)
        / len(validation_candidate_deltas),
        "sealed_candidate_combined_delta_mean": sum(sealed_candidate_deltas)
        / len(sealed_candidate_deltas),
        "sealed_candidate_combined_delta_worst": max(sealed_candidate_deltas),
        "sealed_fixed_large_combined_delta_mean": sum(sealed_fixed_large_deltas)
        / len(sealed_fixed_large_deltas),
        "candidate_beats_fixed_large_count": sum(candidate_beats_fixed_large),
        "candidate_beats_fixed_large_all": all(candidate_beats_fixed_large),
        "technical_gate_passed": technical_gate_passed,
        "resource_measurement_complete": resource_measurement_complete,
        "formal_gate_passed": False,
        "sealed_test_scored": True,
        "training_performed": False,
        "can_start_formal": False,
        "can_promote": False,
        "promotion_blocking_reason": "resource ledger is not yet attached to this read-only scoring canary",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-aware-report", type=Path, default=DEFAULT_TARGET_AWARE_REPORT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--fixed-large-root", type=Path, default=DEFAULT_FIXED_LARGE_ROOT)
    parser.add_argument("--sealed-test", type=Path, default=DEFAULT_SEALED_TEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    values = {
        "manifest_path": args.manifest,
        "target_aware_report_path": args.target_aware_report,
        "worker_root": args.worker_root,
        "fixed_large_root": args.fixed_large_root,
        "sealed_test_path": args.sealed_test,
        "report_path": args.report,
    }
    for key, value in tuple(values.items()):
        if isinstance(value, Path) and not value.is_absolute():
            values[key] = PROJECT_ROOT / value
    result = run_canary(**values)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
