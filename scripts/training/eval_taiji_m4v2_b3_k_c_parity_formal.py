"""C-entry parity formal runner (preregistration: M4V2_B3_K_C_PARITY_FORMAL_PREREGISTRATION_20260910.md).

Two-stage auditable execution:

1. **Pre-sealed**: verify every frozen input digest, score the validation
   split, derive the catastrophe epsilon ``max(0.01, 3 x popstd)`` from the
   nine validation candidate deltas, and write the epsilon to a standalone
   pre-sealed artifact BEFORE any sealed byte is read.
2. **Sealed**: score the sealed artifact for both arms, aggregate at the
   course level (n=3 independent courses; the model dimension is a
   replicate), and evaluate the frozen gates G1-G3 (G4 descriptive).

Criteria are frozen in the preregistration; this runner never adjusts them.
``can_promote=false`` regardless of outcome.
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

from scripts.training.eval_taiji_m4v2_b3_k_c_sealed_scoring import (  # noqa: E402
    _context,
    _delta,
    _loss_score,
    _mse,
    _sealed_experiences,
    _validation_experiences,
)
from taiji import (  # noqa: E402
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-c-parity-formal-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
COURSE_SEEDS = (0, 1, 2)
SEALED_SHA256 = "89896556d1d3dc86fb21a8c9303d7fdf1b83ecc5a7f22777586c7fe22539d78f"
SEALED_INTERNAL_DIGEST_PREFIX = "1c9e3853"
V2_BUILD_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_c_parity_v2_build_20260910.json"
SEALED_TEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_sealed_test_v1.json"
WIDENED_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_candidate_c_entry_parity_v2"
FIXED_LARGE_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_fixed_large_c_entry_v2"
WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_c_parity_formal_20260910.json"
PRESEALED_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_c_parity_formal_presealed_20260910.json"
LOSS_KEYS = (
    "k1.fact_mse",
    "k1.goal_mse",
    "k1.content_mse",
    "k2.transition_mse",
    "k2.goal_mse",
    "k2.content_mse",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"artifact is not a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


def _sum_state_dicts(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if set(a) != set(b):
        raise ValueError("widened channel state dict keys drifted")
    return {key: a[key] + b[key] for key in a}


def _widened_candidate_learners(
    widened_payload: Mapping[str, Any], parent_artifacts: Mapping[str, Any]
) -> tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]:
    """Weight-space addition of the two channels (frozen readout semantic)."""

    channels = widened_payload["channels"]
    k1 = StructuredSemanticLearner.from_checkpoint(
        copy.deepcopy(parent_artifacts["k1.semantic"]["checkpoint"]), device="cpu"
    )
    k1.load_state_dict(
        _sum_state_dicts(
            channels["forward"]["k1.semantic"]["state_dict"],
            channels["anchored"]["k1.semantic"]["state_dict"],
        )
    )
    k1._apply_fact_feature_masks()
    k2 = StructuredSemanticTransitionLearner.from_checkpoint(
        copy.deepcopy(parent_artifacts["k2.transition"]["checkpoint"]), device="cpu"
    )
    k2.load_state_dict(
        _sum_state_dicts(
            channels["forward"]["k2.transition"]["state_dict"],
            channels["anchored"]["k2.transition"]["state_dict"],
        )
    )
    k2._apply_transition_input_masks()
    return k1, k2


def _v2_ensemble_loss_score(
    replicas: list[tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]],
    experiences,
) -> dict[str, float]:
    """Probability arithmetic mean over same-shape replicas (frozen semantic)."""

    rows = []
    for experience in experiences:
        averaged: dict[str, dict[str, float]] = {}
        for replica_index, (semantic, transition) in enumerate(replicas):
            semantic_result = semantic.predict(experience.semantic_example.percept)
            transition_result = transition.predict(
                experience.transition_example.before,
                experience.transition_example.event,
            )
            if replica_index == 0:
                averaged = {
                    "k1.fact": dict(semantic_result.fact_scores),
                    "k1.goal": dict(semantic_result.goal_scores),
                    "k1.content": dict(semantic_result.content_scores),
                    "k2.transition": dict(transition_result.delta_scores),
                    "k2.goal": dict(transition_result.goal_scores),
                    "k2.content": dict(transition_result.content_scores),
                }
            else:
                for group, scores in (
                    ("k1.fact", semantic_result.fact_scores),
                    ("k1.goal", semantic_result.goal_scores),
                    ("k1.content", semantic_result.content_scores),
                    ("k2.transition", transition_result.delta_scores),
                    ("k2.goal", transition_result.goal_scores),
                    ("k2.content", transition_result.content_scores),
                ):
                    if set(scores) != set(averaged[group]):
                        raise ValueError("replica score keys drifted")
                    for key, value in scores.items():
                        averaged[group][key] += float(value)
        scale = 1.0 / len(replicas)
        for group in averaged:
            averaged[group] = {key: value * scale for key, value in averaged[group].items()}

        semantic_example = experience.semantic_example
        transition_example = experience.transition_example
        representative = replicas[0][1]
        current = representative._fact_vector(transition_example.before)
        target_next = representative._fact_vector(transition_example.after)
        rows.append(
            {
                "k1.fact_mse": _mse(
                    averaged["k1.fact"],
                    {key: float(key in semantic_example.fact_keys) for key in averaged["k1.fact"]},
                ),
                "k1.goal_mse": _mse(
                    averaged["k1.goal"],
                    {
                        key: float(key == semantic_example.goal.goal_id)
                        for key in averaged["k1.goal"]
                    },
                ),
                "k1.content_mse": _mse(
                    averaged["k1.content"],
                    {
                        key: float(key == semantic_example.content.content_id)
                        for key in averaged["k1.content"]
                    },
                ),
                "k2.transition_mse": _mse(
                    averaged["k2.transition"],
                    {
                        key: float(value)
                        for key, value in zip(
                            representative.fact_keys,
                            target_next - current,
                            strict=True,
                        )
                    },
                ),
                "k2.goal_mse": _mse(
                    averaged["k2.goal"],
                    {
                        key: float(key == transition_example.goal.goal_id)
                        for key in averaged["k2.goal"]
                    },
                ),
                "k2.content_mse": _mse(
                    averaged["k2.content"],
                    {
                        key: float(key == transition_example.content.content_id)
                        for key in averaged["k2.content"]
                    },
                ),
            }
        )
    means = {key: sum(row[key] for row in rows) / len(rows) for key in LOSS_KEYS}
    means["combined_mse"] = sum(means[key] for key in LOSS_KEYS) / len(LOSS_KEYS)
    return means


def _verify_inputs() -> dict[str, Any]:
    sealed_bytes = SEALED_TEST.read_bytes()
    sealed_sha = hashlib.sha256(sealed_bytes).hexdigest()
    if sealed_sha != SEALED_SHA256:
        raise ValueError(f"sealed artifact digest drifted: {sealed_sha}")
    sealed = json.loads(sealed_bytes.decode("utf-8"))
    if not str(sealed.get("artifact_digest", "")).startswith(
        SEALED_INTERNAL_DIGEST_PREFIX
    ):
        raise ValueError("sealed internal artifact_digest mismatch")
    build = json.loads(V2_BUILD_REPORT.read_text(encoding="utf-8"))
    if build.get("status") != "passed":
        raise ValueError("v2 build report is not passed")
    if not build.get("course_independence", {}).get("gate_passed"):
        raise ValueError("v2 build course independence gate not passed")
    build_cells = {
        (int(cell["model_seed"]), int(cell["course_seed"])): cell for cell in build["cells"]
    }
    for model_seed in MODEL_SEEDS:
        for course_seed in COURSE_SEEDS:
            cell = build_cells[(model_seed, course_seed)]
            widened_path = (
                WIDENED_ROOT
                / f"model_{model_seed}"
                / f"course_{course_seed}"
                / "taiji_c_entry_parity_v2_widened.pt"
            )
            widened_digest = content_digest(_load_mapping(widened_path))
            if widened_digest != cell["artifact_digests"]["widened"]:
                raise ValueError(f"widened artifact digest mismatch: {model_seed}x{course_seed}")
    ensemble_digests = {
        cell["artifact_digests"]["fixed_large_ensemble"] for cell in build["cells"]
    }
    if len(ensemble_digests) != len(COURSE_SEEDS):
        raise ValueError(
            "fixed-large ensemble digests must be exactly one per course; "
            f"got {len(ensemble_digests)}"
        )
    return {
        "sealed_sha256_ok": True,
        "sealed_internal_digest_ok": True,
        "v2_report_status": build["status"],
        "v2_course_independence": build["course_independence"]["gate_passed"],
        "widened_digests_verified": len(build_cells),
        "ensemble_digest_count": len(ensemble_digests),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--presealed-report", type=Path, default=PRESEALED_REPORT)
    args = parser.parse_args()

    started = time.perf_counter()
    input_verification = _verify_inputs()

    validation_rows: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="parity_formal_"))
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
                widened_payload = _load_mapping(
                    WIDENED_ROOT
                    / f"model_{model_seed}"
                    / f"course_{course_seed}"
                    / "taiji_c_entry_parity_v2_widened.pt"
                )
                candidate_k1, candidate_k2 = _widened_candidate_learners(
                    widened_payload, artifacts
                )
                frozen_k1 = StructuredSemanticLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
                )
                frozen_k2 = StructuredSemanticTransitionLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
                )
                ensemble_payload = _load_mapping(
                    FIXED_LARGE_ROOT
                    / f"model_{model_seed}"
                    / f"course_{course_seed}"
                    / "taiji_c_entry_parity_v2_ensemble.pt"
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
                candidate_loss = _loss_score(candidate_k1, candidate_k2, validation)
                fixed_loss = _v2_ensemble_loss_score(replicas, validation)
                validation_rows.append(
                    {
                        "model_seed": model_seed,
                        "course_seed": course_seed,
                        "frozen_combined": float(frozen_loss["combined_mse"]),
                        "candidate_combined": float(candidate_loss["combined_mse"]),
                        "candidate_delta": float(
                            candidate_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                        "fixed_large_combined": float(fixed_loss["combined_mse"]),
                        "fixed_large_delta": float(
                            fixed_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                        "candidate_component_delta": _delta(candidate_loss, frozen_loss),
                        "fixed_large_component_delta": _delta(fixed_loss, frozen_loss),
                    }
                )
    finally:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)

    deltas = [row["candidate_delta"] for row in validation_rows]
    mean_delta = sum(deltas) / len(deltas)
    population_std = (
        sum((value - mean_delta) ** 2 for value in deltas) / len(deltas)
    ) ** 0.5
    epsilon_cat = max(0.01, 3.0 * population_std)

    pre_sealed = {
        "format": "taiji-m4v2-b3-k-c-parity-formal-presealed-v1",
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "input_verification": input_verification,
        "validation_rows": validation_rows,
        "epsilon_derivation": {
            "rule": "max(0.01, 3 x population std of the 9 validation candidate combined deltas)",
            "deltas": deltas,
            "mean": mean_delta,
            "population_std": population_std,
            "epsilon_cat": epsilon_cat,
            "frozen_before_sealed_read": True,
        },
        "sealed_read": "pending",
    }
    args.presealed_report.parent.mkdir(parents=True, exist_ok=True)
    args.presealed_report.write_text(
        json.dumps(pre_sealed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Sealed bytes are read only after the pre-sealed artifact (frozen
    # epsilon) is on disk -- the auditable two-stage discipline.
    sealed = json.loads(SEALED_TEST.read_text(encoding="utf-8"))
    sealed_rows: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="parity_formal_sealed_"))
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
                widened_payload = _load_mapping(
                    WIDENED_ROOT
                    / f"model_{model_seed}"
                    / f"course_{course_seed}"
                    / "taiji_c_entry_parity_v2_widened.pt"
                )
                candidate_k1, candidate_k2 = _widened_candidate_learners(
                    widened_payload, artifacts
                )
                frozen_k1 = StructuredSemanticLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
                )
                frozen_k2 = StructuredSemanticTransitionLearner.from_checkpoint(
                    copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
                )
                ensemble_payload = _load_mapping(
                    FIXED_LARGE_ROOT
                    / f"model_{model_seed}"
                    / f"course_{course_seed}"
                    / "taiji_c_entry_parity_v2_ensemble.pt"
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
                candidate_loss = _loss_score(candidate_k1, candidate_k2, sealed_experiences)
                fixed_loss = _v2_ensemble_loss_score(replicas, sealed_experiences)
                sealed_rows.append(
                    {
                        "model_seed": model_seed,
                        "course_seed": course_seed,
                        "frozen_combined": float(frozen_loss["combined_mse"]),
                        "candidate_combined": float(candidate_loss["combined_mse"]),
                        "candidate_delta": float(
                            candidate_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                        "fixed_large_combined": float(fixed_loss["combined_mse"]),
                        "fixed_large_delta": float(
                            fixed_loss["combined_mse"] - frozen_loss["combined_mse"]
                        ),
                        "candidate_component_delta": _delta(candidate_loss, frozen_loss),
                        "fixed_large_component_delta": _delta(fixed_loss, frozen_loss),
                        "candidate_minus_fixed_large": float(
                            candidate_loss["combined_mse"] - fixed_loss["combined_mse"]
                        ),
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

    val_candidate_course = course_means(validation_rows, "candidate_delta")
    sealed_candidate_course = course_means(sealed_rows, "candidate_delta")
    sealed_fixed_course = course_means(sealed_rows, "fixed_large_delta")
    wins = [
        course
        for course in COURSE_SEEDS
        if sealed_candidate_course[course] < sealed_fixed_course[course]
    ]
    candidate_course_mean = sum(sealed_candidate_course.values()) / len(COURSE_SEEDS)
    fixed_course_mean = sum(sealed_fixed_course.values()) / len(COURSE_SEEDS)

    g1_courses = [
        course for course in COURSE_SEEDS if val_candidate_course[course] < 0.0
    ]
    g2_courses = [
        course
        for course in COURSE_SEEDS
        if sealed_candidate_course[course] <= epsilon_cat
    ]
    gates = {
        "G1_quality_floor": {
            "rule": "candidate validation combined delta < 0 in >= 2/3 courses",
            "course_deltas": val_candidate_course,
            "courses_passed": len(g1_courses),
            "passed": len(g1_courses) >= 2,
        },
        "G2_catastrophe_bound": {
            "rule": "every course-level sealed candidate delta <= +epsilon_cat",
            "epsilon_cat": epsilon_cat,
            "course_deltas": sealed_candidate_course,
            "courses_passed": len(g2_courses),
            "passed": len(g2_courses) == len(COURSE_SEEDS),
        },
        "G3_primary": {
            "rule": "sealed candidate < fixed-large in >= 2/3 courses AND course-mean lower",
            "course_candidate_deltas": sealed_candidate_course,
            "course_fixed_large_deltas": sealed_fixed_course,
            "courses_won": len(wins),
            "candidate_course_mean": candidate_course_mean,
            "fixed_large_course_mean": fixed_course_mean,
            "passed": len(wins) >= 2 and candidate_course_mean < fixed_course_mean,
        },
    }
    formal_passed = all(gate["passed"] for gate in gates.values())

    payload = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if formal_passed else "failed",
        "can_promote": False,
        "preregistration": (
            "plans/reference/M4V2_B3_K_C_PARITY_FORMAL_PREREGISTRATION_20260910.md"
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
            "validation_candidate_course_means": val_candidate_course,
            "sealed_candidate_course_means": sealed_candidate_course,
            "sealed_fixed_large_course_means": sealed_fixed_course,
            "courses_won_by_candidate": wins,
        },
        "gates": gates,
        "verdict": {
            "formal_passed": formal_passed,
            "interpretation": (
                "equal-budget learning-rule comparison: candidate (widened "
                "dual-channel, weight-addition readout) vs strong control "
                "(fixed-large probability-mean ensemble)"
            ),
        },
        "sealed_read_count": 2,
        "elapsed_seconds": time.perf_counter() - started,
        "boundary": (
            "C-entry parity formal only; no promotion, no default runtime "
            "attachment, no provider/MCP/client/CUDA; can_promote=false until "
            "an aggregate scorecard computes it."
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
                "courses_won": wins,
                "sealed_candidate_course_means": sealed_candidate_course,
                "sealed_fixed_large_course_means": sealed_fixed_course,
                "gates": {key: value["passed"] for key, value in gates.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if formal_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
