"""Expanded-space (v4) parity formal runner.

Preregistration: ``plans/reference/M5_K_SIGNAL_SPACE_EXPANSION
_PREREGISTRATION_20260910.md`` §6.  Same two-stage discipline and frozen
G1-G3 gate structure as the v3 formal; the candidate readout is weight
averaging (frozen in v3), both arms reuse the v4 artifacts (no
retraining), and the sealed split is the freshly materialized v3
artifact covering the expanded 5-class space.  ``can_promote=false``.
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
from taiji import (  # noqa: E402
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)

LOSS_KEYS = (
    "k1.fact_mse",
    "k1.goal_mse",
    "k1.content_mse",
    "k2.transition_mse",
    "k2.goal_mse",
    "k2.content_mse",
)


def _one_hot(index: int, width: int) -> torch.Tensor:
    result = torch.zeros(width, dtype=torch.float32)
    result[index] = 1.0
    return result


def _v4_ensemble_loss_score(
    replicas: list[tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]],
    experiences,
) -> dict[str, float]:
    """Probability-mean ensemble scoring, teacher-forced.

    Mirrors ``_semantic_structured_loss`` / ``_transition_structured_loss``
    exactly (head outputs vs one-hot targets over the full vocabulary) so
    the ensemble arm is scored on the same path as the candidate and
    frozen arms; the only composition difference is that head
    probabilities are averaged across the two replicas instead of coming
    from a single learner.
    """

    rows = []
    for experience in experiences:
        k1_fact, k1_goal, k1_content = [], [], []
        k2_delta, k2_goal, k2_content = [], [], []
        for semantic, transition in replicas:
            with torch.no_grad():
                semantic_example = experience.semantic_example
                inputs = semantic._percept_input(semantic_example.percept).reshape(1, -1)
                fact_probabilities = torch.sigmoid(semantic.fact_head(inputs))
                readout = semantic._masked_readout_input(fact_probabilities)
                goal_probabilities = torch.softmax(semantic.goal_head(readout), dim=-1)
                content_probabilities = torch.softmax(
                    semantic.content_head(torch.cat((readout, goal_probabilities), dim=1)),
                    dim=-1,
                )
                k1_fact.append(fact_probabilities.reshape(-1))
                k1_goal.append(goal_probabilities.reshape(-1))
                k1_content.append(content_probabilities.reshape(-1))

                transition_example = experience.transition_example
                current = transition._fact_vector(transition_example.before)
                event_context = transition._event_vector(transition_example.event)
                tinputs = transition._transition_input(
                    current.reshape(1, -1), event_context.reshape(1, -1)
                )
                predicted_delta = transition.transition_head(tinputs)
                next_values = torch.clamp(current + predicted_delta.reshape(-1), 0.0, 1.0)
                tgoal_probabilities = torch.softmax(
                    transition.goal_head(next_values.reshape(1, -1)), dim=-1
                )
                tcontent_probabilities = torch.softmax(
                    transition.content_head(
                        torch.cat((next_values.reshape(1, -1), tgoal_probabilities), dim=1)
                    ),
                    dim=-1,
                )
                k2_delta.append(predicted_delta.reshape(-1))
                k2_goal.append(tgoal_probabilities.reshape(-1))
                k2_content.append(tcontent_probabilities.reshape(-1))

        def average(tensors):
            return torch.stack(tensors).mean(dim=0)

        semantic0, transition0 = replicas[0]
        semantic_example = experience.semantic_example
        transition_example = experience.transition_example
        fact_target = torch.zeros(len(semantic0.fact_keys), dtype=torch.float32)
        fact_index = {key: index for index, key in enumerate(semantic0.fact_keys)}
        for key in semantic_example.fact_keys:
            fact_target[fact_index[key]] = 1.0
        target_delta = transition0._fact_vector(transition_example.after) - transition0._fact_vector(
            transition_example.before
        )
        rows.append(
            {
                "k1.fact_mse": float(
                    torch.mean((average(k1_fact) - fact_target) ** 2)
                ),
                "k1.goal_mse": float(
                    torch.mean(
                        (average(k1_goal) - _one_hot(
                            semantic0.goal_ids.index(semantic_example.goal.goal_id),
                            len(semantic0.goal_ids),
                        ))
                        ** 2
                    )
                ),
                "k1.content_mse": float(
                    torch.mean(
                        (average(k1_content) - _one_hot(
                            semantic0.content_ids.index(semantic_example.content.content_id),
                            len(semantic0.content_ids),
                        ))
                        ** 2
                    )
                ),
                "k2.transition_mse": float(
                    torch.mean((average(k2_delta) - target_delta) ** 2)
                ),
                "k2.goal_mse": float(
                    torch.mean(
                        (average(k2_goal) - _one_hot(
                            transition0.goal_ids.index(transition_example.goal.goal_id),
                            len(transition0.goal_ids),
                        ))
                        ** 2
                    )
                ),
                "k2.content_mse": float(
                    torch.mean(
                        (average(k2_content) - _one_hot(
                            transition0.content_ids.index(transition_example.content.content_id),
                            len(transition0.content_ids),
                        ))
                        ** 2
                    )
                ),
            }
        )
    means = {key: sum(row[key] for row in rows) / len(rows) for key in LOSS_KEYS}
    means["combined_mse"] = sum(means[key] for key in LOSS_KEYS) / len(LOSS_KEYS)
    return means

REPORT_FORMAT = "taiji-m5-k-v4-parity-formal-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
COURSE_SEEDS = (0, 1, 2)
V4_BUILD_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_v4_parity_build_20260910.json"
SEALED_TEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_v4_sealed_test_v3.json"
WIDENED_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_candidate_c_entry_parity_v4"
FIXED_LARGE_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_fixed_large_c_entry_v4"
WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers_v4"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_v4_parity_formal_20260910.json"
PRESEALED_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_v4_parity_formal_presealed_20260910.json"
)


def _average_state_dicts(
    a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(a) != set(b):
        raise ValueError("widened channel state dict keys drifted")
    return {key: (a[key] + b[key]) / 2.0 for key in a}


def _widened_average_learners(
    widened_payload: Mapping[str, Any], parent_artifacts: Mapping[str, Any]
) -> tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]:
    channels = widened_payload["channels"]
    k1 = StructuredSemanticLearner.from_checkpoint(
        copy.deepcopy(parent_artifacts["k1.semantic"]["checkpoint"]), device="cpu"
    )
    k1.load_state_dict(
        _average_state_dicts(
            channels["forward"]["k1.semantic"]["state_dict"],
            channels["anchored"]["k1.semantic"]["state_dict"],
        )
    )
    k1._apply_fact_feature_masks()
    k2 = StructuredSemanticTransitionLearner.from_checkpoint(
        copy.deepcopy(parent_artifacts["k2.transition"]["checkpoint"]), device="cpu"
    )
    k2.load_state_dict(
        _average_state_dicts(
            channels["forward"]["k2.transition"]["state_dict"],
            channels["anchored"]["k2.transition"]["state_dict"],
        )
    )
    k2._apply_transition_input_masks()
    return k1, k2


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
        raise ValueError("sealed v3 internal artifact_digest mismatch")
    if sealed.get("status") != "materialized-unread":
        raise ValueError("sealed v3 artifact is not in materialized-unread state")
    build = json.loads(V4_BUILD_REPORT.read_text(encoding="utf-8"))
    if build.get("status") != "passed":
        raise ValueError("v4 parity build report is not passed")
    if not build.get("course_independence", {}).get("gate_passed"):
        raise ValueError("v4 build course independence gate not passed")
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
                / "taiji_c_entry_parity_v4_widened.pt"
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
        "sealed_v3_sha256": hashlib.sha256(sealed_bytes).hexdigest(),
        "sealed_v3_internal_digest": internal_digest,
        "sealed_v3_task_seed": sealed.get("task_seed"),
        "v4_report_status": build["status"],
        "v4_course_independence": build["course_independence"]["gate_passed"],
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
    temp_root = Path(tempfile.mkdtemp(prefix="parity_v4_formal_"))
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
                    / "taiji_c_entry_parity_v4_widened.pt"
                )
                candidate_k1, candidate_k2 = _widened_average_learners(
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
                candidate_loss = _loss_score(candidate_k1, candidate_k2, validation)
                fixed_loss = _v4_ensemble_loss_score(replicas, validation)
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
        "format": "taiji-m5-k-v4-parity-formal-presealed-v1",
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

    # Sealed v3 bytes are read only after the pre-sealed artifact (frozen
    # epsilon) is on disk.
    sealed = json.loads(SEALED_TEST.read_text(encoding="utf-8"))
    sealed_rows: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="parity_v4_sealed_"))
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
                    / "taiji_c_entry_parity_v4_widened.pt"
                )
                candidate_k1, candidate_k2 = _widened_average_learners(
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
                candidate_loss = _loss_score(candidate_k1, candidate_k2, sealed_experiences)
                fixed_loss = _v4_ensemble_loss_score(replicas, sealed_experiences)
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
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if formal_passed else "failed",
        "can_promote": False,
        "preregistration": (
            "plans/reference/M5_K_SIGNAL_SPACE_EXPANSION_PREREGISTRATION_20260910.md"
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
                "expanded-space (5-class) equal-budget learning-rule "
                "comparison: weight-averaged order-diverse candidate vs "
                "probability-mean ensemble"
            ),
        },
        "sealed_read_count": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "boundary": (
            "expanded-space parity formal only; no promotion, no default "
            "runtime attachment, no provider/MCP/client/CUDA; "
            "can_promote=false until an aggregate scorecard computes it."
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
