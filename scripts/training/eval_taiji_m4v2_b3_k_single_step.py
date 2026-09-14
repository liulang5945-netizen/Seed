"""Run one real native K continuation-learning step on CPU.

This pilot is intentionally narrower than the R6 formal matrix.  It restores
the existing model-17 K1/K2/K3 artifacts, materializes one train and one
family-disjoint holdout Workbench experience, updates only K1 and K2 with one
native detached-local-delta step, saves and fresh-restores the candidate, and
rolls the adapter back to the parent namespace.  The holdout is scored only
after the update and is never passed to ``fit``.

The report is a learning-pilot technical result, not a promotion result.  A
non-positive holdout delta is still useful evidence; it means the update path
is live but the one-record course did not yet demonstrate generalization.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    _parent,
    _restore_worker,
    run_preflight,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_controlled_canary import (  # noqa: E402
    _artifact_paths,
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    FILES_PER_LANG,
    LANGS,
    _build_workspace,
    _episode,
    _holdout_episode_paths,
    _observe_all,
    _path,
    _registry,
    _schema,
    _semantic_example,
    _train_episode_paths,
    _transition_examples,
)
from taiji import (  # noqa: E402
    KAdapterExchange,
    KAdapterInput,
    KAdapterOutput,
    KContinualAdapter,
    KContinuationCourse,
    KContinuationExperience,
    KContinuationUpdateReceipt,
    KWorkerManifest,
    KWorkerManifestBundle,
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    WorldEvent,
    content_digest,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-single-step-pilot-v1"
VERSION = 1
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "checkpoints" / "taiji_k_workers" / "model_17"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_single_step_20260910.json"
DEFAULT_CANDIDATE_DIR = (
    PROJECT_ROOT / "output" / "taiji_m4v2_b3_k_single_step_20260910" / "model_17"
)
DEFAULT_CANDIDATE_NAMESPACE = "taiji:k:candidate"
SEMANTIC_SINGLE_STEP_LR = 2.0
TRANSITION_SINGLE_STEP_LR = 0.2


def _load_artifact_paths(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    paths = _artifact_paths(artifact_dir)
    artifacts = {worker_id: _load_mapping(path) for worker_id, path in paths.items()}
    return artifacts


def _all_course_paths() -> list[str]:
    paths = ["missing_00.txt"]
    for index in range(FILES_PER_LANG):
        paths.extend(_path(language, index) for language in LANGS)
    return paths


def _build_experience(
    *,
    sequence: tuple[Any, ...],
    split: str,
    name: str,
    parent_digest: str,
    worker_bundle_digest: str,
    source_manifest_digest: str,
    projector: OutcomeDependencyProjector,
) -> KContinuationExperience:
    current = sequence[1]
    semantic_example = _semantic_example(current, split=split, tick=1)
    transition_example = _transition_examples(sequence, split=split)[0]
    outcome_event = WorldEvent(
        event_id=f"b3-k-outcome-{name}",
        kind="workbench.observed",
        tick=transition_example.after.tick,
        subject_id=current.path,
        attributes=(
            ("capability_id", "workspace.read"),
            ("success", bool(current.read_success)),
            ("observation_digest", current.observation_digest),
        ),
        provenance="workbench-observation",
    )
    dependency_spec = OutcomeDependencySpec(
        dependency_id=f"b3-k-dependency-{name}",
        next_task_id=f"b3-k-follow-up-{name}",
        capability_id="workspace.read",
        required_outcome="success" if current.read_success else "failure",
    )
    projection = projector.project(transition_example.after, outcome_event, dependency_spec)
    if not projection.accepted:
        raise ValueError(f"B3-K {name} projection rejected: {projection.reason_code}")
    input_item = KAdapterInput(
        episode_id=f"b3-k-episode-{name}",
        parent_checkpoint_digest=parent_digest,
        observation_digest=current.observation_digest,
        world_digest=content_digest(transition_example.before.to_payload()),
        goal_digest=content_digest(semantic_example.goal.to_payload()),
        content_plan_digest=content_digest(semantic_example.content.to_payload()),
        source_manifest_digest=source_manifest_digest,
        tick=transition_example.event.observation_tick,
    )
    output_item = KAdapterOutput(
        parent_checkpoint_digest=parent_digest,
        input_digest=input_item.input_digest,
        action_digest=content_digest(
            {"kind": "workspace.read", "path": current.path, "observation": name}
        ),
        outcome_signature=projection.outcome_signature,
        dependency_digest=projection.dependency_digest,
        dependency_projection_digest=projection.projection_digest,
        success=bool(current.read_success),
        lineage=(
            projection.scope_id,
            input_item.input_digest,
            projection.projection_digest,
            projection.dependency_digest,
            worker_bundle_digest,
        ),
    )
    exchange = KAdapterExchange.create(
        scope_id=projection.scope_id,
        input=input_item,
        output=output_item,
    )
    return KContinuationExperience(
        experience_id=f"b3-k-experience-{name}",
        family_id=f"b3-k-family-{name}",
        split=split,
        parent_checkpoint_digest=parent_digest,
        worker_bundle_digest=worker_bundle_digest,
        source_manifest_digest=source_manifest_digest,
        observation_digest=current.observation_digest,
        semantic_example=semantic_example,
        transition_example=transition_example,
        projection=projection,
        exchange=exchange,
    )


def _prediction_score(
    semantic: Any, transition: Any, experience: KContinuationExperience
) -> dict[str, float]:
    semantic_result = semantic.predict(experience.semantic_example.percept)
    transition_result = transition.predict(
        experience.transition_example.before,
        experience.transition_example.event,
    )
    semantic_goal = float(
        semantic_result.goal is not None
        and semantic_result.goal.goal_id == experience.semantic_example.goal.goal_id
    )
    semantic_content = float(
        semantic_result.content_plan is not None
        and semantic_result.content_plan.content_id
        == experience.semantic_example.content.content_id
    )
    transition_goal = float(
        transition_result.goal is not None
        and transition_result.goal.goal_id == experience.transition_example.goal.goal_id
    )
    transition_content = float(
        transition_result.content_plan is not None
        and transition_result.content_plan.content_id
        == experience.transition_example.content.content_id
    )
    return {
        "semantic_goal_accuracy": semantic_goal,
        "semantic_content_accuracy": semantic_content,
        "transition_goal_accuracy": transition_goal,
        "transition_content_accuracy": transition_content,
        "combined_accuracy": (
            semantic_goal + semantic_content + transition_goal + transition_content
        )
        / 4.0,
    }


def _score_course(
    semantic: Any,
    transition: Any,
    experiences: tuple[KContinuationExperience, ...],
) -> dict[str, float]:
    rows = [_prediction_score(semantic, transition, item) for item in experiences]
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def _save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(checkpoint), temporary)
    loaded = _load_mapping(temporary)
    expected = content_digest(dict(checkpoint))
    if content_digest(loaded) != expected:
        raise ValueError("B3-K candidate checkpoint changed during disk roundtrip")
    temporary.replace(path)
    return loaded


def _candidate_manifest(
    *,
    worker_id: str,
    artifact: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    owner_digests: Mapping[str, str],
    parent_digest: str,
    candidate_namespace: str,
) -> KWorkerManifest:
    return KWorkerManifest.create(
        worker_id=worker_id,
        checkpoint_format=str(checkpoint["format"]),
        checkpoint_version=int(checkpoint["version"]),
        worker_checkpoint_digest=content_digest(dict(checkpoint)),
        owner_digests=tuple(sorted((str(key), str(value)) for key, value in owner_digests.items())),
        source_digest=str(artifact["source_digest"]),
        input_contract_digest=str(artifact["input_contract_digest"]),
        output_contract_digest=str(artifact["output_contract_digest"]),
        parent_checkpoint_digest=parent_digest,
        candidate_namespace=candidate_namespace,
        training_steps=int(checkpoint.get("training_steps", 0)),
        optimizer_state_present=False,
    )


def run_pilot(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR,
    model_seed: int = 17,
    course_seed: int = 0,
    candidate_namespace: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    temp_parent = PROJECT_ROOT / ".tmp-m4v2-b3-k-single-step"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / f"course-{uuid4().hex}"
    temp_root.mkdir()
    report: dict[str, Any] = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "run_kind": "learning-pilot",
        "training_performed": False,
        "research_course_executed": False,
        "candidate_promoted": False,
        "default_runtime_attached": False,
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
            raise ValueError("B3-K candidate namespace is missing from the worker artifact")
        paths = _artifact_paths(artifact_dir)
        preflight = run_preflight(
            semantic_checkpoint=str(paths["k1.semantic"]),
            transition_checkpoint=str(paths["k2.transition"]),
            projection_checkpoint=str(paths["k3.outcome_projection"]),
            model_seed=model_seed,
            course_seed=course_seed,
            candidate_namespace=candidate_namespace,
        )
        if preflight.get("status") != "passed":
            raise RuntimeError("B3-K requires a passed worker attachment preflight")

        manifests = []
        for worker_id in ("k1.semantic", "k2.transition", "k3.outcome_projection"):
            manifest, _ = _restore_worker(
                worker_id,
                artifacts[worker_id],
                parent_digest=parent_digest,
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

        _build_workspace(temp_root, task_seed=course_seed)
        schema = _schema()
        train_registry = _registry(typescript_available=False)
        train_observations = {
            observation.path: observation
            for observation in _observe_all(
                temp_root,
                registry=train_registry,
                split="b3-k-train",
                paths=_all_course_paths(),
                schema=schema,
            )
        }
        holdout_paths = sorted(
            {path for episode_paths in _holdout_episode_paths() for path in episode_paths}
        )
        holdout_registry = _registry(typescript_available=True)
        holdout_observations = {
            observation.path: observation
            for observation in _observe_all(
                temp_root,
                registry=holdout_registry,
                split="b3-k-holdout",
                paths=["missing_00.txt", *holdout_paths],
                schema=schema,
            )
        }
        train_anchor = train_observations["missing_00.txt"]
        holdout_anchor = holdout_observations["missing_00.txt"]
        train_sequence = _episode(
            train_anchor,
            train_observations,
            _train_episode_paths()[0],
        )
        holdout_sequence = _episode(
            holdout_anchor,
            holdout_observations,
            _holdout_episode_paths()[0],
        )
        train_experience = _build_experience(
            sequence=train_sequence,
            split="train",
            name="train",
            parent_digest=parent_digest,
            worker_bundle_digest=parent_bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        holdout_experience = _build_experience(
            sequence=holdout_sequence,
            split="holdout",
            name="holdout",
            parent_digest=parent_digest,
            worker_bundle_digest=parent_bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        course = KContinuationCourse(
            course_id=f"b3-k-single-step-model-{model_seed}-course-{course_seed}",
            parent_checkpoint_digest=parent_digest,
            worker_bundle_digest=parent_bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            train=(train_experience,),
            holdout=(holdout_experience,),
        )
        holdout_before = _score_course(
            semantic_parent,
            transition_parent,
            course.holdout,
        )

        semantic_candidate = StructuredSemanticLearner.from_checkpoint(
            copy.deepcopy(semantic_parent_payload), device="cpu"
        )
        transition_candidate = StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(transition_parent_payload), device="cpu"
        )
        semantic_losses = semantic_candidate.fit(
            (course.train[0].semantic_example,),
            epochs=1,
            learning_rate=SEMANTIC_SINGLE_STEP_LR,
        )
        transition_losses = transition_candidate.fit(
            (course.train[0].transition_example,),
            epochs=1,
            learning_rate=TRANSITION_SINGLE_STEP_LR,
        )
        report["training_performed"] = True
        report["research_course_executed"] = True

        semantic_candidate_checkpoint = semantic_candidate.checkpoint()
        transition_candidate_checkpoint = transition_candidate.checkpoint()
        candidate_paths = {
            "k1.semantic": candidate_dir / "taiji_b3_k1_semantic_step1.pt",
            "k2.transition": candidate_dir / "taiji_b3_k2_transition_step1.pt",
        }
        saved_semantic = _save_checkpoint(
            candidate_paths["k1.semantic"], semantic_candidate_checkpoint
        )
        saved_transition = _save_checkpoint(
            candidate_paths["k2.transition"], transition_candidate_checkpoint
        )
        semantic_fresh = StructuredSemanticLearner.from_checkpoint(saved_semantic, device="cpu")
        transition_fresh = StructuredSemanticTransitionLearner.from_checkpoint(
            saved_transition, device="cpu"
        )
        holdout_after = _score_course(semantic_fresh, transition_fresh, course.holdout)

        candidate_manifests = [
            _candidate_manifest(
                worker_id="k1.semantic",
                artifact=artifacts["k1.semantic"],
                checkpoint=semantic_candidate_checkpoint,
                owner_digests=semantic_candidate.owner_digests(),
                parent_digest=parent_digest,
                candidate_namespace=candidate_namespace,
            ),
            _candidate_manifest(
                worker_id="k2.transition",
                artifact=artifacts["k2.transition"],
                checkpoint=transition_candidate_checkpoint,
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
        parent_worker_checkpoint_digests = tuple(
            (worker_id, str(artifacts[worker_id]["worker_checkpoint_digest"]))
            for worker_id in ("k1.semantic", "k2.transition", "k3.outcome_projection")
        )
        candidate_worker_checkpoint_digests = (
            ("k1.semantic", content_digest(semantic_candidate_checkpoint)),
            ("k2.transition", content_digest(transition_candidate_checkpoint)),
            ("k3.outcome_projection", content_digest(k3_parent_payload)),
        )

        adapter = KContinualAdapter(
            parent_checkpoint_digest=parent_digest,
            owner_graph_digest=parent_bundle.owner_graph_digest,
            source_manifest_digest=source_manifest_digest,
            resource_manifest_digest=resource_manifest_digest,
            dependency_scope_id=train_experience.projection.scope_id,
            candidate_namespace=candidate_namespace,
        )
        adapter.bind_worker_bundle(parent_bundle)
        adapter.bind_dependency_projection(train_experience.projection)
        adapter.record_exchange(train_experience.exchange)
        adapter_before = adapter.checkpoint()
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
            parent_worker_checkpoint_digests=parent_worker_checkpoint_digests,
            candidate_worker_checkpoint_digests=candidate_worker_checkpoint_digests,
            train_experience_digests=course.train_experience_digests,
            holdout_experience_digests=course.holdout_experience_digests,
            updated_workers=("k1.semantic", "k2.transition"),
            frozen_workers=("k3.outcome_projection",),
            candidate_namespace=candidate_namespace,
            training_steps=(
                int(semantic_candidate.training_steps - semantic_parent.training_steps)
                + int(transition_candidate.training_steps - transition_parent.training_steps)
            ),
            optimizer_state_present=False,
            fresh_restore_verified=(
                content_digest(saved_semantic) == content_digest(semantic_candidate_checkpoint)
                and content_digest(saved_transition)
                == content_digest(transition_candidate_checkpoint)
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
                set(course.train_experience_digests).isdisjoint(course.holdout_experience_digests)
                and course.holdout[0].target_digest != course.train[0].target_digest
            ),
            rollback_restored=(
                rollback.active_namespace == rollback.parent_namespace
                and rollback.worker_bundle == parent_bundle
                and rollback.last_exchange == train_experience.exchange
                and rollback_record.status == "rolled_back"
            ),
        )
        checks = {
            "worker_attachment_preflight_passed": preflight.get("status") == "passed",
            "course_roundtrip": KContinuationCourse.from_payload(course.to_payload()).course_digest
            == course.course_digest,
            "train_holdout_disjoint": receipt.holdout_untrained,
            "k1_owner_changed": dict(parent_worker_checkpoint_digests)["k1.semantic"]
            != dict(candidate_worker_checkpoint_digests)["k1.semantic"],
            "k2_owner_changed": dict(parent_worker_checkpoint_digests)["k2.transition"]
            != dict(candidate_worker_checkpoint_digests)["k2.transition"],
            "k3_owner_unchanged": dict(parent_worker_checkpoint_digests)["k3.outcome_projection"]
            == dict(candidate_worker_checkpoint_digests)["k3.outcome_projection"],
            "candidate_bundle_changed": candidate_bundle.bundle_digest
            != parent_bundle.bundle_digest,
            "candidate_checkpoint_fresh_restore": receipt.fresh_restore_verified,
            "parent_checkpoint_unchanged": receipt.parent_unchanged,
            "adapter_candidate_stage_roundtrip": (
                staged.active_namespace == staged.candidate_namespace
                and staged.worker_bundle == parent_bundle
            ),
            "adapter_rollback_restored": receipt.rollback_restored,
            "no_optimizer_state": not receipt.optimizer_state_present,
            "no_default_runtime": True,
            "no_external_integrations": True,
        }
        report.update(
            {
                "status": "passed" if all(checks.values()) and receipt.passed else "failed",
                "checks": checks,
                "parent_checkpoint_digest": parent_digest,
                "worker_bundle_digest": parent_bundle.bundle_digest,
                "candidate_worker_bundle_digest": candidate_bundle.bundle_digest,
                "candidate_namespace": candidate_namespace,
                "course_digest": course.course_digest,
                "train_experience_digests": list(course.train_experience_digests),
                "holdout_experience_digests": list(course.holdout_experience_digests),
                "training_update_steps": receipt.training_steps,
                "updated_workers": list(receipt.updated_workers),
                "frozen_workers": list(receipt.frozen_workers),
                "optimizer_state_present": receipt.optimizer_state_present,
                "semantic_losses": semantic_losses,
                "transition_losses": transition_losses,
                "holdout_before": holdout_before,
                "holdout_after": holdout_after,
                "holdout_delta": {
                    key: holdout_after[key] - holdout_before[key] for key in holdout_before
                },
                "holdout_improved": holdout_after["combined_accuracy"]
                > holdout_before["combined_accuracy"],
                "candidate_checkpoint_paths": {
                    key: str(path) for key, path in candidate_paths.items()
                },
                "candidate_checkpoint_digests": dict(candidate_worker_checkpoint_digests),
                "receipt": receipt.to_payload(),
                "adapter_before_digest": content_digest(adapter_before),
                "adapter_rollback_digest": content_digest(rollback.checkpoint()),
                "candidate_promoted": False,
                "can_start_r6_formal": False,
                "can_promote": False,
                "blocking_reason": (
                    None
                    if all(checks.values()) and receipt.passed
                    else "B3-K single-step technical Gate failed"
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
    artifact_dir = (
        args.artifact_dir if args.artifact_dir.is_absolute() else PROJECT_ROOT / args.artifact_dir
    )
    candidate_dir = (
        args.candidate_dir
        if args.candidate_dir.is_absolute()
        else PROJECT_ROOT / args.candidate_dir
    )
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report = run_pilot(
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
