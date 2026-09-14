"""C-stage formal runner: fast/slow+replay vs continuation, class-decomposed.

Preregistration: ``plans/reference/M4V2_C_STAGE_FORMAL_PREREGISTRATION
_20260910.md``.  Four arms with zero retraining of the base learners:
F frozen / C continuation (300 new steps) / FS candidate (300 wake
steps + 100 replay steps costed separately) / XL fixed-large ensemble
(2x capacity, secondary bar).  Two-stage discipline: the pre-sealed
artifact (validation scores, epsilon_cat and epsilon_ni derivations) is
written before any sealed byte is read.  Statistics unit = course
(n=3); the model dimension is a replicate.  ``can_promote=false``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m4v2_b3_k_c_fixed_large import (  # noqa: E402
    SEMANTIC_EPOCHS,
    SEMANTIC_LR,
    TRANSITION_EPOCHS,
    TRANSITION_LR,
)
from scripts.training.build_taiji_m5_k_v4_parity import (  # noqa: E402
    _build_course_v4,
)
from scripts.training.eval_taiji_m4v2_b3_k_c_parity_formal import (  # noqa: E402
    _load_mapping,
)
from scripts.training.eval_taiji_m4v2_b3_k_c_sealed_scoring import (  # noqa: E402
    _context,
    _delta,
    _loss_score,
    _sealed_experiences,
    _validation_experiences,
)
from scripts.training.eval_taiji_m5_k_v4_parity_formal import (  # noqa: E402
    _v4_ensemble_loss_score,
)
from taiji import (  # noqa: E402
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)
from taiji.k_fast_slow import (  # noqa: E402
    FastSlowKInstance,
    replay_sample_indices,
)

REPORT_FORMAT = "taiji-m4v2-c-stage-formal-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
COURSE_SEEDS = (0, 1, 2)
N_NEW = 150
REPLAY_SAMPLE = 50
V4_BUILD_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_v4_parity_build_20260910.json"
SEALED_TEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_c_stage_sealed_test_v4.json"
WIDENED_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_candidate_c_entry_parity_v4"
FIXED_LARGE_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_fixed_large_c_entry_v4"
WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers_v4"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_c_stage_formal_20260910.json"
PRESEALED_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_c_stage_formal_presealed_20260910.json"
)
EPISODE_CLASSES = ("D", "R", "A")
WEAK_CLASSES = ("D", "R")


def _average_state_dicts(
    a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(a) != set(b):
        raise ValueError("widened channel state dict keys drifted")
    return {key: (a[key] + b[key]) / 2.0 for key in a}


def _verify_inputs() -> dict[str, Any]:
    sealed_bytes = SEALED_TEST.read_bytes()
    sealed = json.loads(sealed_bytes.decode("utf-8"))
    internal_digest = str(sealed.get("artifact_digest", ""))
    recomputed = content_digest(
        {
            key: value
            for key, value in sealed.items()
            if key not in ("artifact_digest", "materializer_sha256")
        }
    )
    if internal_digest != recomputed:
        raise ValueError("sealed v4 internal artifact_digest mismatch")
    if sealed.get("status") != "materialized-unread":
        raise ValueError("sealed v4 artifact is not in materialized-unread state")
    if sealed.get("episode_classes") != ["D", "R", "A"]:
        raise ValueError("sealed v4 must cover the D/R/A episode classes")
    build = json.loads(V4_BUILD_REPORT.read_text(encoding="utf-8"))
    if build.get("status") != "passed":
        raise ValueError("v4 parity build report is not passed")
    if not build.get("course_independence", {}).get("gate_passed"):
        raise ValueError("v4 build course independence gate not passed")
    build_cells = {
        (int(cell["model_seed"]), int(cell["course_seed"])): cell for cell in build["cells"]
    }
    ensemble_digests: set[str] = set()
    for model_seed in MODEL_SEEDS:
        for course_seed in COURSE_SEEDS:
            cell = build_cells[(model_seed, course_seed)]
            ensemble_path = (
                FIXED_LARGE_ROOT
                / f"model_{model_seed}"
                / f"course_{course_seed}"
                / "taiji_c_entry_parity_v4_ensemble.pt"
            )
            ensemble_digest = content_digest(_load_mapping(ensemble_path))
            if ensemble_digest != cell["artifact_digests"]["fixed_large_ensemble"]:
                raise ValueError(
                    f"fixed-large artifact digest mismatch: {model_seed}x{course_seed}"
                )
            ensemble_digests.add(ensemble_digest)
    if len(ensemble_digests) != len(COURSE_SEEDS):
        raise ValueError(
            "fixed-large ensemble digests must be exactly one per course; "
            f"got {len(ensemble_digests)}"
        )
    return {
        "sealed_v4_sha256": hashlib.sha256(sealed_bytes).hexdigest(),
        "sealed_v4_internal_digest": internal_digest,
        "sealed_v4_task_seed": sealed.get("task_seed"),
        "v4_report_status": build["status"],
        "v4_course_independence": build["course_independence"]["gate_passed"],
        "ensemble_digest_count": len(ensemble_digests),
    }


def _per_experience_combined(
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
    experiences,
) -> list[float]:
    """Per-experience combined MSE (mean of the six structured components)."""

    return [
        float(_loss_score(semantic, transition, [experience])["combined_mse"])
        for experience in experiences
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--presealed-report", type=Path, default=PRESEALED_REPORT)
    args = parser.parse_args()

    started = time.perf_counter()
    input_verification = _verify_inputs()

    # -- Pre-sealed stage: validation scoring + epsilon derivations -------
    validation_rows: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="c_stage_formal_"))
    try:
        for model_seed in MODEL_SEEDS:
            artifacts, parent_digest, bundle, projector = _context(
                worker_root=WORKER_ROOT, model_seed=model_seed
            )
            validation = _validation_experiences(
                temp_root=temp_root,
                artifacts=artifacts,
                parent_digest=parent_digest,
                bundle=bundle,
                projector=projector,
            )
            for course_seed in COURSE_SEEDS:
                experiences, _counts, _keys, _block = _build_course_v4(
                    temp_root=temp_root,
                    model_seed=model_seed,
                    course_seed=course_seed,
                    parent_digest=parent_digest,
                    bundle=bundle,
                    projector=projector,
                    source_manifest_digest=content_digest(
                        {
                            "c_stage_formal": True,
                            "model_seed": model_seed,
                            "course_seed": course_seed,
                        }
                    ),
                )
                frozen_k1 = StructuredSemanticLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
                )
                frozen_k2 = StructuredSemanticTransitionLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
                )
                # C continuation arm.
                c_k1 = StructuredSemanticLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
                )
                c_k2 = StructuredSemanticTransitionLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
                )
                for experience in experiences:
                    c_k1.fit(
                        (experience.semantic_example,),
                        epochs=SEMANTIC_EPOCHS,
                        learning_rate=SEMANTIC_LR,
                    )
                    c_k2.fit(
                        (experience.transition_example,),
                        epochs=TRANSITION_EPOCHS,
                        learning_rate=TRANSITION_LR,
                    )
                # FS fast/slow + replay arm.
                instance = FastSlowKInstance(
                    semantic_parent_checkpoint=dict(
                        artifacts["k1.semantic"]["checkpoint"]
                    ),
                    transition_parent_checkpoint=dict(
                        artifacts["k2.transition"]["checkpoint"]
                    ),
                    parent_worker_bundle_digest=bundle.bundle_digest,
                    source_manifest_digest=content_digest(
                        {
                            "c_stage_formal": True,
                            "model_seed": model_seed,
                            "course_seed": course_seed,
                        }
                    ),
                    parent_checkpoint_digest=parent_digest,
                )
                for experience in experiences:
                    instance.wake_experience(
                        experience,
                        semantic_epochs=SEMANTIC_EPOCHS,
                        semantic_lr=SEMANTIC_LR,
                        transition_epochs=TRANSITION_EPOCHS,
                        transition_lr=TRANSITION_LR,
                    )
                replay_indices = replay_sample_indices(
                    buffer_size=len(experiences),
                    sample_count=REPLAY_SAMPLE,
                    digest=content_digest(
                        {"experiences": [item.experience_digest for item in experiences]}
                    ),
                )
                for index in replay_indices:
                    instance.replay_experience(
                        experiences[index],
                        semantic_epochs=SEMANTIC_EPOCHS,
                        semantic_lr=SEMANTIC_LR,
                        transition_epochs=TRANSITION_EPOCHS,
                        transition_lr=TRANSITION_LR,
                    )
                instance.consolidate()
                fs_k1, fs_k2 = instance.effective_learners()
                # XL fixed-large ensemble.
                ensemble_payload = _load_mapping(
                    FIXED_LARGE_ROOT
                    / f"model_{model_seed}"
                    / f"course_{course_seed}"
                    / "taiji_c_entry_parity_v4_ensemble.pt"
                )
                replicas = [
                    (
                        StructuredSemanticLearner.from_checkpoint(
                            replica["k1.semantic"], device="cpu"
                        ),
                        StructuredSemanticTransitionLearner.from_checkpoint(
                            replica["k2.transition"], device="cpu"
                        ),
                    )
                    for replica in ensemble_payload["replicas"]
                ]
                frozen_loss = _loss_score(frozen_k1, frozen_k2, validation)
                c_loss = _loss_score(c_k1, c_k2, validation)
                fs_loss = _loss_score(fs_k1, fs_k2, validation)
                xl_loss = _v4_ensemble_loss_score(replicas, validation)
                validation_rows.append(
                    {
                        "model_seed": model_seed,
                        "course_seed": course_seed,
                        "frozen_combined": float(frozen_loss["combined_mse"]),
                        "c_combined": float(c_loss["combined_mse"]),
                        "c_delta": float(
                            c_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                        "fs_combined": float(fs_loss["combined_mse"]),
                        "fs_delta": float(
                            fs_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                        "paired_fs_minus_c": float(
                            fs_loss["combined_mse"] - c_loss["combined_mse"]
                        ),
                        "xl_combined": float(xl_loss["combined_mse"]),
                        "xl_delta": float(
                            xl_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                    }
                )
    finally:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)

    fs_deltas = [row["fs_delta"] for row in validation_rows]
    fs_mean = sum(fs_deltas) / len(fs_deltas)
    fs_std = (sum((value - fs_mean) ** 2 for value in fs_deltas) / len(fs_deltas)) ** 0.5
    epsilon_cat = max(0.01, 3.0 * fs_std)
    paired = [row["paired_fs_minus_c"] for row in validation_rows]
    paired_mean = sum(paired) / len(paired)
    paired_std = (sum((value - paired_mean) ** 2 for value in paired) / len(paired)) ** 0.5
    epsilon_ni = max(0.002, 3.0 * paired_std)

    pre_sealed = {
        "format": "taiji-m4v2-c-stage-formal-presealed-v1",
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "input_verification": input_verification,
        "validation_rows": validation_rows,
        "epsilon_derivation": {
            "epsilon_cat": {
                "rule": "max(0.01, 3 x population std of the 9 validation FS combined deltas)",
                "deltas": fs_deltas,
                "population_std": fs_std,
                "epsilon_cat": epsilon_cat,
            },
            "epsilon_ni": {
                "rule": "max(0.002, 3 x population std of the 9 validation paired (FS - C) combined deltas)",
                "paired_deltas": paired,
                "population_std": paired_std,
                "epsilon_ni": epsilon_ni,
            },
            "frozen_before_sealed_read": True,
        },
        "sealed_read": "pending",
    }
    args.presealed_report.parent.mkdir(parents=True, exist_ok=True)
    args.presealed_report.write_text(
        json.dumps(pre_sealed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # -- Sealed stage: read sealed v4 only after the pre-sealed freeze ----
    sealed = json.loads(SEALED_TEST.read_text(encoding="utf-8"))
    episode_classes = [str(episode["episode_class"]) for episode in sealed["episodes"]]
    sealed_rows: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="c_stage_sealed_"))
    try:
        for model_seed in MODEL_SEEDS:
            artifacts, parent_digest, bundle, projector = _context(
                worker_root=WORKER_ROOT, model_seed=model_seed
            )
            sealed_experiences = _sealed_experiences(
                sealed=sealed,
                artifacts=artifacts,
                parent_digest=parent_digest,
                bundle=bundle,
                projector=projector,
            )
            for course_seed in COURSE_SEEDS:
                experiences, _counts, _keys, _block = _build_course_v4(
                    temp_root=temp_root,
                    model_seed=model_seed,
                    course_seed=course_seed,
                    parent_digest=parent_digest,
                    bundle=bundle,
                    projector=projector,
                    source_manifest_digest=content_digest(
                        {
                            "c_stage_formal": True,
                            "model_seed": model_seed,
                            "course_seed": course_seed,
                        }
                    ),
                )
                frozen_k1 = StructuredSemanticLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
                )
                frozen_k2 = StructuredSemanticTransitionLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
                )
                c_k1 = StructuredSemanticLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
                )
                c_k2 = StructuredSemanticTransitionLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
                )
                for experience in experiences:
                    c_k1.fit(
                        (experience.semantic_example,),
                        epochs=SEMANTIC_EPOCHS,
                        learning_rate=SEMANTIC_LR,
                    )
                    c_k2.fit(
                        (experience.transition_example,),
                        epochs=TRANSITION_EPOCHS,
                        learning_rate=TRANSITION_LR,
                    )
                instance = FastSlowKInstance(
                    semantic_parent_checkpoint=dict(
                        artifacts["k1.semantic"]["checkpoint"]
                    ),
                    transition_parent_checkpoint=dict(
                        artifacts["k2.transition"]["checkpoint"]
                    ),
                    parent_worker_bundle_digest=bundle.bundle_digest,
                    source_manifest_digest=content_digest(
                        {
                            "c_stage_formal": True,
                            "model_seed": model_seed,
                            "course_seed": course_seed,
                        }
                    ),
                    parent_checkpoint_digest=parent_digest,
                )
                for experience in experiences:
                    instance.wake_experience(
                        experience,
                        semantic_epochs=SEMANTIC_EPOCHS,
                        semantic_lr=SEMANTIC_LR,
                        transition_epochs=TRANSITION_EPOCHS,
                        transition_lr=TRANSITION_LR,
                    )
                replay_indices = replay_sample_indices(
                    buffer_size=len(experiences),
                    sample_count=REPLAY_SAMPLE,
                    digest=content_digest(
                        {"experiences": [item.experience_digest for item in experiences]}
                    ),
                )
                for index in replay_indices:
                    instance.replay_experience(
                        experiences[index],
                        semantic_epochs=SEMANTIC_EPOCHS,
                        semantic_lr=SEMANTIC_LR,
                        transition_epochs=TRANSITION_EPOCHS,
                        transition_lr=TRANSITION_LR,
                    )
                instance.consolidate()
                fs_k1, fs_k2 = instance.effective_learners()
                ensemble_payload = _load_mapping(
                    FIXED_LARGE_ROOT
                    / f"model_{model_seed}"
                    / f"course_{course_seed}"
                    / "taiji_c_entry_parity_v4_ensemble.pt"
                )
                replicas = [
                    (
                        StructuredSemanticLearner.from_checkpoint(
                            replica["k1.semantic"], device="cpu"
                        ),
                        StructuredSemanticTransitionLearner.from_checkpoint(
                            replica["k2.transition"], device="cpu"
                        ),
                    )
                    for replica in ensemble_payload["replicas"]
                ]
                frozen_loss = _loss_score(frozen_k1, frozen_k2, sealed_experiences)
                c_loss = _loss_score(c_k1, c_k2, sealed_experiences)
                fs_loss = _loss_score(fs_k1, fs_k2, sealed_experiences)
                xl_loss = _v4_ensemble_loss_score(replicas, sealed_experiences)
                c_per = _per_experience_combined(c_k1, c_k2, sealed_experiences)
                fs_per = _per_experience_combined(fs_k1, fs_k2, sealed_experiences)
                frozen_per = _per_experience_combined(frozen_k1, frozen_k2, sealed_experiences)
                per_class = {}
                for class_name, _experience, c_combined, fs_combined, frozen_combined in zip(
                    episode_classes, sealed_experiences, c_per, fs_per, frozen_per, strict=True
                ):
                    per_class[class_name] = {
                        "c_delta": c_combined - frozen_combined,
                        "fs_delta": fs_combined - frozen_combined,
                    }
                sealed_rows.append(
                    {
                        "model_seed": model_seed,
                        "course_seed": course_seed,
                        "frozen_combined": float(frozen_loss["combined_mse"]),
                        "c_delta": float(
                            c_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                        "fs_delta": float(
                            fs_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                        "xl_delta": float(
                            xl_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                        "weak_class_delta": {
                            "c": sum(
                                per_class[name]["c_delta"] for name in WEAK_CLASSES
                            )
                            / len(WEAK_CLASSES),
                            "fs": sum(
                                per_class[name]["fs_delta"] for name in WEAK_CLASSES
                            )
                            / len(WEAK_CLASSES),
                        },
                        "per_class": per_class,
                        "c_component_delta": _delta(c_loss, frozen_loss),
                        "fs_component_delta": _delta(fs_loss, frozen_loss),
                        "xl_component_delta": _delta(xl_loss, frozen_loss),
                    }
                )
    finally:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)

    def course_means(rows: list[dict[str, Any]], key: str) -> dict[int, float]:
        result: dict[int, float] = {}
        for course_seed in COURSE_SEEDS:
            values = [
                float(row[key]) for row in rows if row["course_seed"] == course_seed
            ]
            result[course_seed] = sum(values) / len(values)
        return result

    val_c_course = course_means(validation_rows, "c_delta")
    val_fs_course = course_means(validation_rows, "fs_delta")
    sealed_c_course = course_means(sealed_rows, "c_delta")
    sealed_fs_course = course_means(sealed_rows, "fs_delta")
    sealed_xl_course = course_means(sealed_rows, "xl_delta")
    weak_c_means = {
        course: sum(
            row["weak_class_delta"]["c"] for row in sealed_rows if row["course_seed"] == course
        )
        / len(MODEL_SEEDS)
        for course in COURSE_SEEDS
    }
    weak_fs_means = {
        course: sum(
            row["weak_class_delta"]["fs"] for row in sealed_rows if row["course_seed"] == course
        )
        / len(MODEL_SEEDS)
        for course in COURSE_SEEDS
    }
    weak_wins = [
        course for course in COURSE_SEEDS if weak_fs_means[course] < weak_c_means[course]
    ]
    ni_courses = [
        course
        for course in COURSE_SEEDS
        if sealed_fs_course[course] <= sealed_c_course[course] + epsilon_ni
    ]

    g1_courses = [
        course
        for course in COURSE_SEEDS
        if val_c_course[course] < 0.0 and val_fs_course[course] < 0.0
    ]
    g2_c_courses = [
        course for course in COURSE_SEEDS if sealed_c_course[course] <= epsilon_cat
    ]
    g2_fs_courses = [
        course for course in COURSE_SEEDS if sealed_fs_course[course] <= epsilon_cat
    ]
    gates = {
        "G1_learning_floor": {
            "rule": "C and FS validation combined deltas < 0 in >= 2/3 courses",
            "course_c_deltas": val_c_course,
            "course_fs_deltas": val_fs_course,
            "passed": len(g1_courses) >= 2,
        },
        "G2_catastrophe_bound": {
            "rule": "every course-level sealed delta <= +epsilon_cat for both learning arms",
            "epsilon_cat": epsilon_cat,
            "course_c_deltas": sealed_c_course,
            "course_fs_deltas": sealed_fs_course,
            "passed": len(g2_c_courses) == len(COURSE_SEEDS)
            and len(g2_fs_courses) == len(COURSE_SEEDS),
        },
        "G3_weak_class_primary": {
            "rule": "D/R weak-class sealed delta: FS < C in >= 2/3 courses",
            "course_c_weak_deltas": weak_c_means,
            "course_fs_weak_deltas": weak_fs_means,
            "courses_won": len(weak_wins),
            "passed": len(weak_wins) >= 2,
        },
        "G4_overall_non_inferiority": {
            "rule": "FS overall sealed delta <= C + epsilon_ni in >= 2/3 courses",
            "epsilon_ni": epsilon_ni,
            "course_fs_deltas": sealed_fs_course,
            "course_c_deltas": sealed_c_course,
            "courses_passed": len(ni_courses),
            "passed": len(ni_courses) >= 2,
        },
    }
    formal_passed = all(gate["passed"] for gate in gates.values())

    payload = {
        "format": "taiji-m4v2-c-stage-formal-v1",
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if formal_passed else "failed",
        "can_promote": False,
        "preregistration": (
            "plans/reference/M4V2_C_STAGE_FORMAL_PREREGISTRATION_20260910.md"
        ),
        "input_verification": input_verification,
        "pre_sealed_artifact": str(args.presealed_report),
        "pre_sealed": {
            "validation_rows": validation_rows,
            "epsilon_derivation": pre_sealed["epsilon_derivation"],
        },
        "sealed_rows": sealed_rows,
        "course_aggregation": {
            "statistics_unit": "course (n=3; model dimension is a replicate)",
            "validation_c_course_means": val_c_course,
            "validation_fs_course_means": val_fs_course,
            "sealed_c_course_means": sealed_c_course,
            "sealed_fs_course_means": sealed_fs_course,
            "sealed_xl_course_means": sealed_xl_course,
            "weak_class_course_means": {
                "c": weak_c_means,
                "fs": weak_fs_means,
            },
            "courses_won_by_fs_weak_class": weak_wins,
        },
        "gates": gates,
        "verdict": {
            "formal_passed": formal_passed,
            "interpretation": (
                "class-decomposed learning-mechanism comparison: FS "
                "(fast/slow+replay) vs C (direct continuation) with the XL "
                "ensemble as a capacity-labeled secondary bar"
            ),
        },
        "sealed_read_count": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "boundary": (
            "C-stage formal only; no promotion, no default runtime "
            "attachment, no provider/MCP/client/CUDA; can_promote=false."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": payload["status"],
                "epsilon_cat": epsilon_cat,
                "epsilon_ni": epsilon_ni,
                "weak_class_course_means": weak_c_means | weak_fs_means,
                "weak_class_courses_won": weak_wins,
                "gates": {key: value["passed"] for key, value in gates.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if formal_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
