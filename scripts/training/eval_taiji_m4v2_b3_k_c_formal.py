"""Execute the frozen C-entry K continuation comparison on CPU.

This runner deliberately separates three conclusions:

* candidate continuation quality against the frozen parent;
* paired CPU resource measurement for candidate and fixed-large;
* the stronger fixed-large comparison used for promotion.

The sealed split is consumed only after the unchanged manifest, candidate
training, checkpoint roundtrip, and resource ledger have been materialized.
No artifact is attached to the default runtime and no promotion is performed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m4v2_b3_k_c_fixed_large import (  # noqa: E402
    DEFAULT_REPORT as DEFAULT_FIXED_LARGE_REPORT,
)
from scripts.training.eval_taiji_m4v2_b3_k_c_sealed_scoring import (  # noqa: E402
    COURSE_SEEDS,
    DEFAULT_FIXED_LARGE_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_SEALED_TEST,
    DEFAULT_WORKER_ROOT,
    _context,
    _delta,
    _fixed_large_loss_score,
    _load_mapping,
    _sealed_experiences,
    _validation_experiences,
)
from scripts.training.eval_taiji_m4v2_b3_k_loss_diagnostic import (  # noqa: E402
    _loss_score,
    run_diagnostic,
)
from taiji import (  # noqa: E402
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-c-formal-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "taiji_m4v2_b3_k_c_formal_20260910"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_c_formal_20260910.json"


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        return None


def _parameter_bytes(*learners: Any) -> int:
    return sum(
        int(parameter.numel() * parameter.element_size())
        for learner in learners
        for parameter in learner.parameters()
    )


def _timed_score(
    scorer: Callable[[], dict[str, float]],
    *,
    trace_count: int,
    parameter_bytes: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    before_rss = _rss_bytes()
    started = time.perf_counter()
    scores = scorer()
    elapsed = time.perf_counter() - started
    after_rss = _rss_bytes()
    rss_values = tuple(value for value in (before_rss, after_rss) if value is not None)
    return scores, {
        "device": "cpu",
        "wall_clock_seconds": float(elapsed),
        "peak_working_set_bytes": max(rss_values) if rss_values else None,
        "peak_working_set_method": "process_rss_before_after_lower_bound",
        "checkpoint_write_bytes": 0,
        "parameter_bytes": int(parameter_bytes),
        "inference_trace_count": int(trace_count),
        "measurement_complete": bool(before_rss is not None and after_rss is not None),
    }


def _combined_inference_resource(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values = tuple(rows.values())
    return {
        "device": "cpu",
        "wall_clock_seconds": sum(float(row["wall_clock_seconds"]) for row in values),
        "peak_working_set_bytes": (
            max(
                int(row["peak_working_set_bytes"])
                for row in values
                if row["peak_working_set_bytes"] is not None
            )
            if all(row["peak_working_set_bytes"] is not None for row in values)
            else None
        ),
        "peak_working_set_method": "process_rss_before_after_lower_bound",
        "checkpoint_write_bytes": 0,
        "parameter_bytes": int(max(row["parameter_bytes"] for row in values)),
        "inference_trace_count": sum(int(row["inference_trace_count"]) for row in values),
        "measurement_complete": all(bool(row["measurement_complete"]) for row in values),
        "by_split": {key: dict(value) for key, value in rows.items()},
    }


def _resource_complete(resource: Mapping[str, Any], *, training: bool) -> bool:
    required = ("peak_working_set_bytes", "checkpoint_write_bytes", "parameter_bytes")
    if resource.get("device") != "cpu":
        return False
    if training:
        required = (*required, "training_wall_clock_seconds", "training_update_steps")
    else:
        required = (*required, "wall_clock_seconds", "inference_trace_count")
    return bool(
        resource.get("measurement_complete") is True
        and all(
            isinstance(resource.get(key), (int, float))
            and not isinstance(resource.get(key), bool)
            and float(resource[key]) >= 0.0
            for key in required
        )
    )


def _load_fixed_large_cell(
    build_report: Mapping[str, Any], model_seed: int, course_seed: int
) -> Mapping[str, Any]:
    for cell in build_report.get("cells", []):
        if int(cell.get("model_seed", -1)) == int(model_seed) and int(
            cell.get("course_seed", -1)
        ) == int(course_seed):
            return cell
    raise ValueError(f"missing fixed-large build cell: model={model_seed} course={course_seed}")


def _score_cell(
    *,
    artifacts: Mapping[str, Any],
    candidate_paths: Mapping[str, Path],
    fixed_large_path: Path,
    validation: tuple[Any, ...],
    sealed: tuple[Any, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    from taiji.k_fixed_large import NativeKFixedLargeEnsemble

    fixed_large = NativeKFixedLargeEnsemble.from_checkpoint(
        fixed_large_payload["ensemble_checkpoint"]
    )
    parent_parameter_bytes = _parameter_bytes(parent_semantic, parent_transition)
    candidate_parameter_bytes = _parameter_bytes(candidate_semantic, candidate_transition)
    fixed_parameter_bytes = int(
        sum(
            _parameter_bytes(replica)
            for replica in (
                *fixed_large.semantic_replicas,
                *fixed_large.transition_replicas,
            )
        )
    )
    scores: dict[str, Any] = {}
    resources: dict[str, Any] = {}
    for split, experiences in (("validation", validation), ("sealed", sealed)):
        frozen, frozen_resource = _timed_score(
            lambda e=experiences: _loss_score(parent_semantic, parent_transition, e),
            trace_count=len(experiences),
            parameter_bytes=parent_parameter_bytes,
        )
        candidate, candidate_resource = _timed_score(
            lambda e=experiences: _loss_score(candidate_semantic, candidate_transition, e),
            trace_count=len(experiences),
            parameter_bytes=candidate_parameter_bytes,
        )
        fixed_large_scores, fixed_resource = _timed_score(
            lambda e=experiences: _fixed_large_loss_score(fixed_large, e),
            trace_count=len(experiences),
            parameter_bytes=fixed_parameter_bytes,
        )
        scores[split] = {
            "frozen_parent": frozen,
            "candidate_continuation": candidate,
            "fixed_large": fixed_large_scores,
            "candidate_delta_vs_frozen": _delta(candidate, frozen),
            "fixed_large_delta_vs_frozen": _delta(fixed_large_scores, frozen),
            "candidate_delta_vs_fixed_large": _delta(candidate, fixed_large_scores),
        }
        resources[split] = {
            "frozen_parent": frozen_resource,
            "candidate_continuation": candidate_resource,
            "fixed_large": fixed_resource,
        }
    return scores, resources


def run_formal(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    worker_root: Path = DEFAULT_WORKER_ROOT,
    fixed_large_root: Path = DEFAULT_FIXED_LARGE_ROOT,
    fixed_large_report_path: Path = DEFAULT_FIXED_LARGE_REPORT,
    sealed_test_path: Path = DEFAULT_SEALED_TEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed = json.loads(sealed_test_path.read_text(encoding="utf-8"))
    fixed_large_build = json.loads(fixed_large_report_path.read_text(encoding="utf-8"))
    if sealed["artifact_digest"] != manifest["sealed_test_split"]["artifact_digest"]:
        raise ValueError("sealed-test artifact does not match frozen C-entry manifest")
    if fixed_large_build.get("status") != "passed":
        raise ValueError("fixed-large resource build is not passed")

    temp_parent = PROJECT_ROOT / ".tmp-m4v2-b3-k-c-formal"
    temp_parent.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
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
                    candidate_dir = output_root / f"model_{model_seed}" / f"course_{course_seed}"
                    candidate = run_diagnostic(
                        artifact_dir=worker_root / f"model_{model_seed}",
                        candidate_dir=candidate_dir,
                        model_seed=model_seed,
                        course_seed=course_seed,
                        train_episode_count=3,
                        train_variant_strategy="target_aware",
                        measure_resources=True,
                    )
                    if candidate.get("status") != "passed":
                        raise RuntimeError(
                            f"C-entry candidate cell failed: model={model_seed} "
                            f"course={course_seed}: {candidate.get('blocking_reason')}"
                        )
                    candidate_paths = {
                        key: Path(value)
                        for key, value in candidate["candidate_checkpoint_paths"].items()
                    }
                    fixed_large_path = (
                        fixed_large_root
                        / f"model_{model_seed}"
                        / f"course_{course_seed}"
                        / "taiji_c_entry_k_fixed_large_ensemble.pt"
                    )
                    scores, inference_resources = _score_cell(
                        artifacts=artifacts,
                        candidate_paths=candidate_paths,
                        fixed_large_path=fixed_large_path,
                        validation=validation,
                        sealed=sealed_experiences,
                    )
                    fixed_cell = _load_fixed_large_cell(fixed_large_build, model_seed, course_seed)
                    candidate_training = dict(candidate["resource"])
                    fixed_training = dict(fixed_cell["resource"])
                    candidate_inference = _combined_inference_resource(
                        {
                            split: inference_resources[split]["candidate_continuation"]
                            for split in ("validation", "sealed")
                        }
                    )
                    fixed_inference = _combined_inference_resource(
                        {
                            split: inference_resources[split]["fixed_large"]
                            for split in ("validation", "sealed")
                        }
                    )
                    frozen_inference = _combined_inference_resource(
                        {
                            split: inference_resources[split]["frozen_parent"]
                            for split in ("validation", "sealed")
                        }
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
                            "candidate_training": candidate,
                            "validation": scores["validation"],
                            "sealed": scores["sealed"],
                            "resource": {
                                "candidate": {
                                    "training": candidate_training,
                                    "inference": candidate_inference,
                                },
                                "fixed_large": {
                                    "training": fixed_training,
                                    "inference": fixed_inference,
                                },
                                "frozen_parent": {"inference": frozen_inference},
                            },
                            "technical_checks": {
                                "candidate_status_passed": candidate["status"] == "passed",
                                "candidate_checkpoint_fresh_restore": bool(
                                    candidate["checks"]["candidate_checkpoint_fresh_restore"]
                                ),
                                "parent_checkpoint_unchanged": bool(
                                    candidate["checks"]["parent_checkpoint_unchanged"]
                                ),
                                "k3_owner_unchanged": bool(
                                    candidate["checks"]["k3_owner_unchanged"]
                                ),
                                "rollback_restored": bool(
                                    candidate["checks"]["adapter_rollback_restored"]
                                ),
                                "fixed_large_restore": bool(
                                    fixed_cell["disk_ensemble_fresh_restore"]
                                ),
                                "fixed_large_replica_distinct": bool(
                                    fixed_cell["replica_updates_distinct"]
                                ),
                                "target_multiset_matches_manifest": (
                                    candidate["train_target_multiset_digest"]
                                    == next(
                                        item["target_multiset_digest"]
                                        for item in manifest["course_contract"]["variants"]
                                        if int(item["course_seed"]) == int(course_seed)
                                    )
                                ),
                            },
                        }
                    )
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)
    finally:
        if temp_parent.exists() and not any(temp_parent.iterdir()):
            temp_parent.rmdir()

    sealed_candidate_deltas = [
        float(cell["sealed"]["candidate_delta_vs_frozen"]["combined_mse"]) for cell in cells
    ]
    sealed_fixed_large_deltas = [
        float(cell["sealed"]["fixed_large_delta_vs_frozen"]["combined_mse"]) for cell in cells
    ]
    validation_candidate_deltas = [
        float(cell["validation"]["candidate_delta_vs_frozen"]["combined_mse"]) for cell in cells
    ]
    candidate_beats_fixed_large = [
        candidate <= fixed
        for candidate, fixed in zip(sealed_candidate_deltas, sealed_fixed_large_deltas, strict=True)
    ]
    technical_gate_passed = bool(cells) and all(
        all(cell["technical_checks"].values()) for cell in cells
    )
    candidate_training_complete = all(
        _resource_complete(cell["resource"]["candidate"]["training"], training=True)
        for cell in cells
    )
    fixed_training_complete = all(
        _resource_complete(cell["resource"]["fixed_large"]["training"], training=True)
        for cell in cells
    )
    inference_complete = all(
        _resource_complete(cell["resource"]["candidate"]["inference"], training=False)
        and _resource_complete(cell["resource"]["fixed_large"]["inference"], training=False)
        and _resource_complete(cell["resource"]["frozen_parent"]["inference"], training=False)
        for cell in cells
    )
    resource_gate_passed = (
        candidate_training_complete and fixed_training_complete and inference_complete
    )
    candidate_updates = [
        cell["candidate_training"]["candidate_worker_bundle_digest"] for cell in cells
    ]
    candidate_updates_distinct = len(candidate_updates) == len(set(candidate_updates))
    candidate_quality_gate_passed = bool(
        sealed_candidate_deltas
        and all(delta <= 0.0 for delta in sealed_candidate_deltas)
        and sum(sealed_candidate_deltas) / len(sealed_candidate_deltas) <= -0.0001
        and all(delta <= 0.0 for delta in validation_candidate_deltas)
        and candidate_updates_distinct
    )
    candidate_beats_fixed_large_all = all(candidate_beats_fixed_large)
    formal_gate_passed = (
        technical_gate_passed and resource_gate_passed and candidate_quality_gate_passed
    )
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if technical_gate_passed and resource_gate_passed else "failed",
        "run_kind": "c-entry-formal-execution",
        "manifest_path": str(manifest_path),
        "manifest_course_contract_digest": manifest["course_contract_digest"],
        "model_seeds": list(MODEL_SEEDS),
        "course_seeds": list(COURSE_SEEDS),
        "cell_count": len(cells),
        "cells": cells,
        "training_performed": True,
        "sealed_test_scored": True,
        "technical_gate_passed": technical_gate_passed,
        "candidate_quality_gate_passed": candidate_quality_gate_passed,
        "candidate_updates_distinct": candidate_updates_distinct,
        "resource_gate_passed": resource_gate_passed,
        "candidate_training_resource_complete": candidate_training_complete,
        "fixed_large_training_resource_complete": fixed_training_complete,
        "inference_resource_complete": inference_complete,
        "validation_candidate_combined_delta_mean": sum(validation_candidate_deltas)
        / len(validation_candidate_deltas),
        "validation_candidate_combined_delta_worst": max(validation_candidate_deltas),
        "sealed_candidate_combined_delta_mean": sum(sealed_candidate_deltas)
        / len(sealed_candidate_deltas),
        "sealed_candidate_combined_delta_worst": max(sealed_candidate_deltas),
        "sealed_fixed_large_combined_delta_mean": sum(sealed_fixed_large_deltas)
        / len(sealed_fixed_large_deltas),
        "candidate_beats_fixed_large_count": sum(candidate_beats_fixed_large),
        "candidate_beats_fixed_large_all": candidate_beats_fixed_large_all,
        "formal_gate_passed": formal_gate_passed,
        "strong_fixed_large_comparison_passed": candidate_beats_fixed_large_all,
        "formal_started": True,
        "can_start_formal": False,
        "can_promote": bool(formal_gate_passed and candidate_beats_fixed_large_all),
        "candidate_promoted": False,
        "promotion_blocking_reason": (
            None
            if formal_gate_passed and candidate_beats_fixed_large_all
            else (
                "candidate does not beat fixed-large in every sealed cell"
                if formal_gate_passed
                else "C-entry technical, resource, or candidate-quality Gate failed"
            )
        ),
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
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--fixed-large-root", type=Path, default=DEFAULT_FIXED_LARGE_ROOT)
    parser.add_argument("--fixed-large-report", type=Path, default=DEFAULT_FIXED_LARGE_REPORT)
    parser.add_argument("--sealed-test", type=Path, default=DEFAULT_SEALED_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    values = {
        "manifest_path": args.manifest,
        "worker_root": args.worker_root,
        "fixed_large_root": args.fixed_large_root,
        "fixed_large_report_path": args.fixed_large_report,
        "sealed_test_path": args.sealed_test,
        "output_root": args.output_root,
        "report_path": args.report,
    }
    for key, value in tuple(values.items()):
        if isinstance(value, Path) and not value.is_absolute():
            values[key] = PROJECT_ROOT / value
    result = run_formal(**values)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
