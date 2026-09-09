"""Measure a non-saturated K continuation update on three holdout records.

The preceding B3-K pilot proved that K1/K2 can be updated and rolled back, but
its 0/1 holdout accuracy was already 1.0 before the update.  This diagnostic
keeps the same parent, one-record train update, worker ownership and rollback
contract, while measuring continuous fact/goal/content probability MSE for
four independent holdout experiences.  It is still a CPU diagnostic, not a
formal promotion run.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_single_step import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    _all_course_paths,
    _build_experience,
    _candidate_manifest,
    _load_artifact_paths,
    _save_checkpoint,
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
    _train_episode_paths,
)
from taiji import (  # noqa: E402
    KContinualAdapter,
    KContinuationCourse,
    KContinuationUpdateReceipt,
    KWorkerManifestBundle,
    OutcomeDependencyProjector,
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-loss-diagnostic-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_loss_diagnostic_20260910.json"
DEFAULT_CANDIDATE_DIR = (
    PROJECT_ROOT / "output" / "taiji_m4v2_b3_k_loss_diagnostic_20260910" / "model_17"
)
SEMANTIC_SINGLE_STEP_LR = 2.0
TRANSITION_SINGLE_STEP_LR = 0.2


def _course_train_variant(course_seed: int) -> tuple[int, tuple[str, ...]]:
    """Select a real train episode; the seed must change the consumed input."""

    variants = _train_episode_paths()
    index = int(course_seed) % len(variants)
    return index, variants[index]


def _course_train_variants(
    course_seed: int, *, count: int, strategy: str = "contiguous"
) -> tuple[tuple[int, ...], tuple[tuple[str, ...], ...]]:
    variants = _train_episode_paths()
    if not 1 <= int(count) <= len(variants):
        raise ValueError("bounded K course train count must fit the available variants")
    if strategy == "contiguous":
        start = int(course_seed) % len(variants)
        indexes = tuple((start + offset) % len(variants) for offset in range(int(count)))
    elif strategy == "non_sliding":
        if int(count) != 3:
            raise ValueError("non_sliding K course variants require exactly three examples")
        fixed = ((0, 2, 4), (1, 3, 5), (0, 3, 5))
        indexes = fixed[int(course_seed) % len(fixed)]
    else:
        raise ValueError(f"unknown K course train variant strategy: {strategy}")
    return indexes, tuple(variants[index] for index in indexes)


def _one_hot(index: int, width: int) -> torch.Tensor:
    values = torch.zeros((1, int(width)), dtype=torch.float32)
    values[0, int(index)] = 1.0
    return values


def _semantic_structured_loss(
    learner: StructuredSemanticLearner,
    experience,
) -> dict[str, float]:
    example = experience.semantic_example
    with torch.no_grad():
        inputs = learner._percept_input(example.percept).reshape(1, -1)
        targets = torch.zeros((1, len(learner.fact_keys)), dtype=torch.float32)
        fact_index = {key: index for index, key in enumerate(learner.fact_keys)}
        for key in example.fact_keys:
            targets[0, fact_index[key]] = 1.0
        fact_probabilities = torch.sigmoid(learner.fact_head(inputs))
        readout = learner._masked_readout_input(fact_probabilities)
        goal_index = learner.goal_ids.index(example.goal.goal_id)
        goal_probabilities = torch.softmax(learner.goal_head(readout), dim=-1)
        content_inputs = torch.cat((readout, goal_probabilities), dim=1)
        content_probabilities = torch.softmax(
            learner.content_head(content_inputs), dim=-1
        )
        content_index = learner.content_ids.index(example.content.content_id)
        return {
            "fact_mse": float(torch.mean((fact_probabilities - targets) ** 2)),
            "goal_mse": float(
                torch.mean((goal_probabilities - _one_hot(goal_index, len(learner.goal_ids))) ** 2)
            ),
            "content_mse": float(
                torch.mean(
                    (content_probabilities - _one_hot(content_index, len(learner.content_ids)))
                    ** 2
                )
            ),
        }


def _transition_structured_loss(
    learner: StructuredSemanticTransitionLearner,
    experience,
) -> dict[str, float]:
    example = experience.transition_example
    with torch.no_grad():
        current = learner._fact_vector(example.before)
        event_context = learner._event_vector(example.event)
        inputs = learner._transition_input(
            current.reshape(1, -1), event_context.reshape(1, -1)
        )
        predicted_delta = learner.transition_head(inputs)
        target_next = learner._fact_vector(example.after)
        target_delta = target_next - current
        next_values = torch.clamp(current + predicted_delta.reshape(-1), 0.0, 1.0)
        goal_index = learner.goal_ids.index(example.goal.goal_id)
        goal_probabilities = torch.softmax(
            learner.goal_head(next_values.reshape(1, -1)), dim=-1
        )
        content_probabilities = torch.softmax(
            learner.content_head(torch.cat((next_values.reshape(1, -1), goal_probabilities), dim=1)),
            dim=-1,
        )
        content_index = learner.content_ids.index(example.content.content_id)
        return {
            "transition_mse": float(torch.mean((predicted_delta - target_delta) ** 2)),
            "goal_mse": float(
                torch.mean((goal_probabilities - _one_hot(goal_index, len(learner.goal_ids))) ** 2)
            ),
            "content_mse": float(
                torch.mean(
                    (content_probabilities - _one_hot(content_index, len(learner.content_ids)))
                    ** 2
                )
            ),
        }


def _loss_score(
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
    experiences,
) -> dict[str, float]:
    rows = []
    for experience in experiences:
        semantic_loss = _semantic_structured_loss(semantic, experience)
        transition_loss = _transition_structured_loss(transition, experience)
        rows.append({
            "k1.fact_mse": semantic_loss["fact_mse"],
            "k1.goal_mse": semantic_loss["goal_mse"],
            "k1.content_mse": semantic_loss["content_mse"],
            "k2.transition_mse": transition_loss["transition_mse"],
            "k2.goal_mse": transition_loss["goal_mse"],
            "k2.content_mse": transition_loss["content_mse"],
        })
    means = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }
    means["combined_mse"] = sum(means.values()) / len(means)
    return means


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


def run_diagnostic(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR,
    model_seed: int = 17,
    course_seed: int = 0,
    train_episode_count: int = 1,
    train_variant_strategy: str = "contiguous",
    candidate_namespace: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    temp_parent = PROJECT_ROOT / ".tmp-m4v2-b3-k-loss-diagnostic"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / f"course-{uuid4().hex}"
    temp_root.mkdir()
    report: dict[str, Any] = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "run_kind": "learning-diagnostic",
        "training_performed": False,
        "research_course_executed": False,
        "candidate_promoted": False,
        "can_start_r6_formal": False,
        "can_promote": False,
    }
    try:
        parent = _parent(model_seed)
        parent_digest = content_digest(parent)
        artifacts = _load_artifact_paths(artifact_dir)
        if candidate_namespace is None:
            candidate_namespace = str(artifacts["k1.semantic"]["candidate_namespace"])
        if not candidate_namespace:
            raise ValueError("K loss diagnostic candidate namespace is missing")
        paths = {
            worker_id: artifact_dir / path.name
            for worker_id, path in {
                "k1.semantic": artifact_dir / "taiji_r6_k1_semantic.pt",
                "k2.transition": artifact_dir / "taiji_r6_k2_transition.pt",
                "k3.outcome_projection": artifact_dir / "taiji_r6_k3_outcome_projection.pt",
            }.items()
        }
        preflight = run_preflight(
            semantic_checkpoint=str(paths["k1.semantic"]),
            transition_checkpoint=str(paths["k2.transition"]),
            projection_checkpoint=str(paths["k3.outcome_projection"]),
            model_seed=model_seed,
            course_seed=course_seed,
            candidate_namespace=candidate_namespace,
        )
        if preflight.get("status") != "passed":
            raise RuntimeError("K loss diagnostic requires a passed worker attachment preflight")
        manifests = []
        for worker_id in ("k1.semantic", "k2.transition", "k3.outcome_projection"):
            manifest, _ = _restore_worker(
                worker_id, artifacts[worker_id], parent_digest=parent_digest
            )
            manifests.append(manifest)
        source_manifest_digest = str(artifacts["k1.semantic"]["source_manifest_digest"])
        resource_manifest_digest = str(artifacts["k1.semantic"]["resource_manifest_digest"])
        parent_bundle = KWorkerManifestBundle.create(
            parent_checkpoint_digest=parent_digest,
            source_manifest_digest=source_manifest_digest,
            resource_manifest_digest=resource_manifest_digest,
            candidate_namespace=candidate_namespace,
            workers=manifests,
        )
        semantic_parent_payload = copy.deepcopy(artifacts["k1.semantic"]["checkpoint"])
        transition_parent_payload = copy.deepcopy(artifacts["k2.transition"]["checkpoint"])
        k3_parent_payload = copy.deepcopy(artifacts["k3.outcome_projection"]["checkpoint"])
        semantic_parent = StructuredSemanticLearner.from_checkpoint(
            semantic_parent_payload, device="cpu"
        )
        transition_parent = StructuredSemanticTransitionLearner.from_checkpoint(
            transition_parent_payload, device="cpu"
        )
        projector = OutcomeDependencyProjector.from_checkpoint(k3_parent_payload)

        # Keep the observed holdout fixed across course seeds.  Course
        # stability is varied by the actual train episode below, not by a
        # label-only seed or by changing the evaluation data.
        workspace_seed = 0
        _build_workspace(temp_root, task_seed=workspace_seed)
        schema = _schema()
        train_observations = {
            observation.path: observation
            for observation in _observe_all(
                temp_root,
                registry=_registry(typescript_available=False),
                split="b3-k-loss-train",
                paths=_all_course_paths(),
                schema=schema,
            )
        }
        holdout_paths = sorted(
            {path for paths_for_episode in _holdout_episode_paths() for path in paths_for_episode}
        )
        holdout_observations = {
            observation.path: observation
            for observation in _observe_all(
                temp_root,
                registry=_registry(typescript_available=True),
                split="b3-k-loss-holdout",
                paths=["missing_00.txt", *holdout_paths],
                schema=schema,
            )
        }
        train_episode_indexes, train_episode_variants = _course_train_variants(
            course_seed, count=train_episode_count, strategy=train_variant_strategy
        )
        train_experiences = tuple(
            _build_experience(
                sequence=_episode(
                    train_observations["missing_00.txt"],
                    train_observations,
                    episode_paths,
                ),
                split="train",
                name=f"loss-train-{index}",
                parent_digest=parent_digest,
                worker_bundle_digest=parent_bundle.bundle_digest,
                source_manifest_digest=source_manifest_digest,
                projector=projector,
            )
            for index, episode_paths in zip(
                train_episode_indexes, train_episode_variants, strict=True
            )
        )
        holdout_anchor = holdout_observations["missing_00.txt"]
        # The fourth M5.K2 cross-combination reuses the same raw observation
        # as holdout-0/1/2.  Keep the continuation contract strict and use
        # the three genuinely record-disjoint holdout observations instead of
        # weakening it merely to count four episodes.
        holdout_experiences = tuple(
            _build_experience(
                sequence=_episode(holdout_anchor, holdout_observations, episode_paths),
                split="holdout",
                name=f"loss-holdout-{index}",
                parent_digest=parent_digest,
                worker_bundle_digest=parent_bundle.bundle_digest,
                source_manifest_digest=source_manifest_digest,
                projector=projector,
            )
            for index, episode_paths in enumerate(_holdout_episode_paths()[:3])
        )
        course = KContinuationCourse(
            course_id=f"b3-k-loss-model-{model_seed}-course-{course_seed}",
            parent_checkpoint_digest=parent_digest,
            worker_bundle_digest=parent_bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            train=train_experiences,
            holdout=holdout_experiences,
        )
        loss_before = _loss_score(semantic_parent, transition_parent, course.holdout)
        train_loss_before = _loss_score(
            semantic_parent, transition_parent, course.train
        )
        control_semantic = StructuredSemanticLearner.from_checkpoint(
            copy.deepcopy(semantic_parent_payload), device="cpu"
        )
        control_transition = StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(transition_parent_payload), device="cpu"
        )
        control_loss = _loss_score(control_semantic, control_transition, course.holdout)
        no_update_control_delta = {
            key: control_loss[key] - loss_before[key] for key in loss_before
        }

        semantic_candidate = StructuredSemanticLearner.from_checkpoint(
            copy.deepcopy(semantic_parent_payload), device="cpu"
        )
        transition_candidate = StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(transition_parent_payload), device="cpu"
        )
        semantic_losses = semantic_candidate.fit(
            tuple(item.semantic_example for item in course.train),
            epochs=1,
            learning_rate=SEMANTIC_SINGLE_STEP_LR,
        )
        transition_losses = transition_candidate.fit(
            tuple(item.transition_example for item in course.train),
            epochs=1,
            learning_rate=TRANSITION_SINGLE_STEP_LR,
        )
        report["training_performed"] = True
        report["research_course_executed"] = True
        semantic_checkpoint = semantic_candidate.checkpoint()
        transition_checkpoint = transition_candidate.checkpoint()
        candidate_paths = {
            "k1.semantic": candidate_dir / "taiji_b3_k1_semantic_loss_step1.pt",
            "k2.transition": candidate_dir / "taiji_b3_k2_transition_loss_step1.pt",
        }
        saved_semantic = _save_checkpoint(candidate_paths["k1.semantic"], semantic_checkpoint)
        saved_transition = _save_checkpoint(candidate_paths["k2.transition"], transition_checkpoint)
        semantic_fresh = StructuredSemanticLearner.from_checkpoint(saved_semantic, device="cpu")
        transition_fresh = StructuredSemanticTransitionLearner.from_checkpoint(
            saved_transition, device="cpu"
        )
        loss_after = _loss_score(semantic_fresh, transition_fresh, course.holdout)
        train_loss_after = _loss_score(
            semantic_fresh, transition_fresh, course.train
        )
        train_loss_delta = {
            key: train_loss_after[key] - train_loss_before[key]
            for key in train_loss_before
        }

        candidate_manifests = [
            _candidate_manifest(
                worker_id="k1.semantic",
                artifact=artifacts["k1.semantic"],
                checkpoint=semantic_checkpoint,
                owner_digests=semantic_candidate.owner_digests(),
                parent_digest=parent_digest,
                candidate_namespace=candidate_namespace,
            ),
            _candidate_manifest(
                worker_id="k2.transition",
                artifact=artifacts["k2.transition"],
                checkpoint=transition_checkpoint,
                owner_digests=transition_candidate.owner_digests(),
                parent_digest=parent_digest,
                candidate_namespace=candidate_namespace,
            ),
            manifests[2],
        ]
        candidate_bundle = KWorkerManifestBundle.create(
            parent_checkpoint_digest=parent_digest,
            source_manifest_digest=source_manifest_digest,
            resource_manifest_digest=resource_manifest_digest,
            candidate_namespace=candidate_namespace,
            workers=candidate_manifests,
        )
        parent_pairs = tuple(
            (worker_id, str(artifacts[worker_id]["worker_checkpoint_digest"]))
            for worker_id in ("k1.semantic", "k2.transition", "k3.outcome_projection")
        )
        candidate_pairs = (
            ("k1.semantic", content_digest(semantic_checkpoint)),
            ("k2.transition", content_digest(transition_checkpoint)),
            ("k3.outcome_projection", content_digest(k3_parent_payload)),
        )
        adapter = KContinualAdapter(
            parent_checkpoint_digest=parent_digest,
            owner_graph_digest=parent_bundle.owner_graph_digest,
            source_manifest_digest=source_manifest_digest,
            resource_manifest_digest=resource_manifest_digest,
            dependency_scope_id=course.train[0].projection.scope_id,
            candidate_namespace=candidate_namespace,
        )
        adapter.bind_worker_bundle(parent_bundle)
        adapter.bind_dependency_projection(course.train[0].projection)
        adapter.record_exchange(course.train[0].exchange)
        rollback_token = adapter.stage_candidate(
            candidate_checkpoint_digest=candidate_bundle.bundle_digest,
            candidate_owner_graph_digest=candidate_bundle.owner_graph_digest,
            candidate_source_manifest_digest=source_manifest_digest,
            candidate_parent_checkpoint_digest=parent_digest,
        )
        staged = KContinualAdapter.from_checkpoint(adapter.checkpoint())
        rollback_record = adapter.rollback(rollback_token)
        rollback = KContinualAdapter.from_checkpoint(adapter.checkpoint())
        receipt = KContinuationUpdateReceipt(
            course_digest=course.course_digest,
            parent_checkpoint_digest=parent_digest,
            parent_worker_bundle_digest=parent_bundle.bundle_digest,
            candidate_worker_bundle_digest=candidate_bundle.bundle_digest,
            parent_worker_checkpoint_digests=parent_pairs,
            candidate_worker_checkpoint_digests=candidate_pairs,
            train_experience_digests=course.train_experience_digests,
            holdout_experience_digests=course.holdout_experience_digests,
            updated_workers=("k1.semantic", "k2.transition"),
            frozen_workers=("k3.outcome_projection",),
            candidate_namespace=candidate_namespace,
            training_steps=(
                semantic_candidate.training_steps - semantic_parent.training_steps
                + transition_candidate.training_steps - transition_parent.training_steps
            ),
            optimizer_state_present=False,
            fresh_restore_verified=(
                content_digest(saved_semantic) == content_digest(semantic_checkpoint)
                and content_digest(saved_transition) == content_digest(transition_checkpoint)
                and semantic_fresh.owner_digests() == semantic_candidate.owner_digests()
                and transition_fresh.owner_digests() == transition_candidate.owner_digests()
            ),
            parent_unchanged=(
                content_digest(semantic_parent_payload)
                == str(artifacts["k1.semantic"]["worker_checkpoint_digest"])
                and content_digest(transition_parent_payload)
                == str(artifacts["k2.transition"]["worker_checkpoint_digest"])
                and content_digest(k3_parent_payload)
                == str(artifacts["k3.outcome_projection"]["worker_checkpoint_digest"])
            ),
            holdout_untrained=(
                set(course.train_experience_digests).isdisjoint(
                    course.holdout_experience_digests
                )
                and all(
                    item.target_digest
                    not in {experience.target_digest for experience in course.train}
                    for item in course.holdout
                )
            ),
            rollback_restored=(
                rollback.active_namespace == rollback.parent_namespace
                and rollback.worker_bundle == parent_bundle
                and rollback.last_exchange == course.train[0].exchange
                and rollback_record.status == "rolled_back"
            ),
        )
        checks = {
            "worker_attachment_preflight_passed": preflight.get("status") == "passed",
            "course_roundtrip": KContinuationCourse.from_payload(course.to_payload()).course_digest
            == course.course_digest,
            "three_disjoint_holdout_records": len(course.holdout) == 3,
            "train_holdout_disjoint": receipt.holdout_untrained,
            "candidate_checkpoint_fresh_restore": receipt.fresh_restore_verified,
            "parent_checkpoint_unchanged": receipt.parent_unchanged,
            "k1_owner_changed": dict(parent_pairs)["k1.semantic"]
            != dict(candidate_pairs)["k1.semantic"],
            "k2_owner_changed": dict(parent_pairs)["k2.transition"]
            != dict(candidate_pairs)["k2.transition"],
            "k3_owner_unchanged": dict(parent_pairs)["k3.outcome_projection"]
            == dict(candidate_pairs)["k3.outcome_projection"],
            "adapter_candidate_stage_roundtrip": staged.active_namespace
            == staged.candidate_namespace,
            "adapter_rollback_restored": receipt.rollback_restored,
            "no_optimizer_state": not receipt.optimizer_state_present,
            "no_update_control": all(
                abs(value) <= 1e-12 for value in no_update_control_delta.values()
            ),
        }
        loss_delta = {key: loss_after[key] - loss_before[key] for key in loss_before}
        report.update(
            {
                "status": "passed" if all(checks.values()) and receipt.passed else "failed",
                "checks": checks,
                "parent_checkpoint_digest": parent_digest,
                "candidate_namespace": candidate_namespace,
                "worker_bundle_digest": parent_bundle.bundle_digest,
                "candidate_worker_bundle_digest": candidate_bundle.bundle_digest,
                "course_digest": course.course_digest,
                "course_seed": int(course_seed),
                "workspace_seed": workspace_seed,
                "train_episode_count": len(train_episode_indexes),
                "train_variant_strategy": train_variant_strategy,
                "train_episode_indexes": list(train_episode_indexes),
                "train_episode_paths": [
                    list(paths) for paths in train_episode_variants
                ],
                "train_experience_digests": list(course.train_experience_digests),
                "train_course_digest": content_digest(
                    list(course.train_experience_digests)
                ),
                "train_fit_input_digests": [
                    {
                        "experience_digest": experience.experience_digest,
                        "semantic_input_digest": experience.semantic_example.input_digest,
                        "transition_input_digest": experience.transition_example.input_digest,
                        "target_digest": experience.target_digest,
                    }
                    for experience in course.train
                ],
                "holdout_count": len(course.holdout),
                "training_update_steps": receipt.training_steps,
                "updated_workers": list(receipt.updated_workers),
                "frozen_workers": list(receipt.frozen_workers),
                "optimizer_state_present": receipt.optimizer_state_present,
                "semantic_losses": semantic_losses,
                "transition_losses": transition_losses,
                "train_structured_loss_before": train_loss_before,
                "train_structured_loss_after": train_loss_after,
                "train_structured_loss_delta": train_loss_delta,
                "holdout_structured_loss_before": loss_before,
                "holdout_structured_loss_after": loss_after,
                "holdout_structured_loss_delta": loss_delta,
                "no_update_control": {
                    "holdout_structured_loss_before": loss_before,
                    "holdout_structured_loss_after": control_loss,
                    "holdout_structured_loss_delta": no_update_control_delta,
                    "training_update_steps": 0,
                    "passed": checks["no_update_control"],
                },
                "parameter_delta_norm": {
                    "k1.semantic": _parameter_delta_norm(
                        semantic_parent, semantic_candidate
                    ),
                    "k2.transition": _parameter_delta_norm(
                        transition_parent, transition_candidate
                    ),
                },
                "holdout_loss_improved": loss_after["combined_mse"] < loss_before["combined_mse"],
                "candidate_checkpoint_paths": {
                    key: str(path) for key, path in candidate_paths.items()
                },
                "receipt": receipt.to_payload(),
                "candidate_promoted": False,
                "can_start_r6_formal": False,
                "can_promote": False,
                "blocking_reason": (
                    None
                    if all(checks.values()) and receipt.passed
                    else "B3-K continuous-loss diagnostic technical Gate failed"
                ),
            }
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        report["blocking_reason"] = str(exc)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        report["elapsed_seconds"] = time.perf_counter() - started
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--course-seed", type=int, default=0)
    parser.add_argument("--candidate-namespace", default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else PROJECT_ROOT / args.artifact_dir
    candidate_dir = args.candidate_dir if args.candidate_dir.is_absolute() else PROJECT_ROOT / args.candidate_dir
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report = run_diagnostic(
        artifact_dir=artifact_dir,
        candidate_dir=candidate_dir,
        model_seed=args.model_seed,
        course_seed=args.course_seed,
        candidate_namespace=args.candidate_namespace,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
