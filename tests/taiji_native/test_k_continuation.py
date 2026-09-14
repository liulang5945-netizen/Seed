from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from taiji import (
    ContentPlan,
    Goal,
    KAdapterExchange,
    KAdapterInput,
    KAdapterOutput,
    KContinuationCourse,
    KContinuationExperience,
    KContinuationUpdateReceipt,
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    PerceptEvent,
    StructuredSemanticExample,
    StructuredSemanticTransitionExample,
    WorldEvent,
    WorldState,
    content_digest,
)

PARENT_DIGEST = "a" * 64
BUNDLE_DIGEST = "b" * 64
SOURCE_DIGEST = "c" * 64


def _experience(name: str, split: str) -> KContinuationExperience:
    feature = torch.tensor([1.0, 0.0]) if name == "train" else torch.tensor([0.0, 1.0])
    event = PerceptEvent(
        event_id=f"percept-{name}",
        observation_tick=1,
        modality="workbench",
        features=feature,
        assembly_id=f"assembly-{name}",
        confidence=1.0,
    )
    before = WorldState(
        tick=0,
        entities=("workbench",),
        relations=(("workbench", "read", "pending"),),
        uncertainty=0.0,
    )
    after = WorldState(
        tick=1,
        entities=("workbench",),
        relations=(("workbench", "read", "success"),),
        uncertainty=0.0,
        percept_event_id=event.event_id,
        percept_assembly_id=event.assembly_id,
    )
    goal = Goal(goal_id=f"goal:{name}", description="inspect the workbench", priority=0.9)
    content = ContentPlan(
        content_id=f"content:{name}",
        intent_id=f"intent:{name}",
        intent_kind="report",
        semantic_slots={"name": name},
        source_goal_id=goal.goal_id,
        expected_outcome="verified report",
        tick=1,
    )
    semantic = StructuredSemanticExample(
        example_id=f"semantic:{name}",
        family_id=f"semantic-family:{name}",
        percept=event,
        world=after,
        goal=goal,
        content=content,
    )
    transition = StructuredSemanticTransitionExample(
        example_id=f"transition:{name}",
        family_id=f"transition-family:{name}",
        before=before,
        event=event,
        after=after,
        goal=goal,
        content=content,
    )
    outcome_event = WorldEvent(
        event_id=f"outcome:{name}",
        kind="workbench.execution",
        tick=1,
        subject_id="workbench",
        attributes=(
            ("capability_id", "workspace.read"),
            ("success", True),
        ),
        provenance="test-workbench",
    )
    projection = OutcomeDependencyProjector("k-test-scope").project(
        after,
        outcome_event,
        OutcomeDependencySpec(
            dependency_id=f"dependency:{name}",
            next_task_id=f"follow-up:{name}",
            capability_id="workspace.read",
            required_outcome="success",
        ),
    )
    input_item = KAdapterInput(
        episode_id=f"episode:{name}",
        parent_checkpoint_digest=PARENT_DIGEST,
        observation_digest=content_digest(event.to_payload()),
        world_digest=content_digest(after.to_payload()),
        goal_digest=content_digest(goal.to_payload()),
        content_plan_digest=content_digest(content.to_payload()),
        source_manifest_digest=SOURCE_DIGEST,
        tick=1,
    )
    output_item = KAdapterOutput(
        parent_checkpoint_digest=PARENT_DIGEST,
        input_digest=input_item.input_digest,
        action_digest=content_digest({"action": "workspace.read", "name": name}),
        outcome_signature=projection.outcome_signature,
        dependency_digest=projection.dependency_digest,
        dependency_projection_digest=projection.projection_digest,
        success=True,
        lineage=(
            "k-test-scope",
            input_item.input_digest,
            projection.projection_digest,
            projection.dependency_digest,
        ),
    )
    return KContinuationExperience(
        experience_id=f"experience:{name}",
        family_id=f"episode-family:{name}",
        split=split,
        parent_checkpoint_digest=PARENT_DIGEST,
        worker_bundle_digest=BUNDLE_DIGEST,
        source_manifest_digest=SOURCE_DIGEST,
        observation_digest=input_item.observation_digest,
        semantic_example=semantic,
        transition_example=transition,
        projection=projection,
        exchange=KAdapterExchange.create(
            scope_id="k-test-scope",
            input=input_item,
            output=output_item,
        ),
    )


def _course() -> KContinuationCourse:
    return KContinuationCourse(
        course_id="k-course-test",
        parent_checkpoint_digest=PARENT_DIGEST,
        worker_bundle_digest=BUNDLE_DIGEST,
        source_manifest_digest=SOURCE_DIGEST,
        train=(_experience("train", "train"),),
        holdout=(_experience("holdout", "holdout"),),
    )


def test_k_continuation_course_roundtrip_seals_disjoint_records() -> None:
    course = _course()
    restored = KContinuationCourse.from_payload(course.to_payload())

    assert restored.course_digest == course.course_digest
    assert restored.train[0].experience_digest == course.train[0].experience_digest
    assert restored.holdout[0].experience_digest == course.holdout[0].experience_digest
    assert set(restored.train_experience_digests).isdisjoint(restored.holdout_experience_digests)


def test_k_continuation_course_rejects_family_leakage() -> None:
    course = _course()
    leaked = replace(course.holdout[0], family_id=course.train[0].family_id)

    with pytest.raises(ValueError, match="family leakage"):
        replace(course, holdout=(leaked,))


def test_k_continuation_update_receipt_allows_only_k1_k2_and_roundtrips() -> None:
    course = _course()
    receipt = KContinuationUpdateReceipt(
        course_digest=course.course_digest,
        parent_checkpoint_digest=PARENT_DIGEST,
        parent_worker_bundle_digest="d" * 64,
        candidate_worker_bundle_digest="e" * 64,
        parent_worker_checkpoint_digests=(
            ("k1.semantic", "1" * 64),
            ("k2.transition", "2" * 64),
            ("k3.outcome_projection", "3" * 64),
        ),
        candidate_worker_checkpoint_digests=(
            ("k1.semantic", "4" * 64),
            ("k2.transition", "5" * 64),
            ("k3.outcome_projection", "3" * 64),
        ),
        train_experience_digests=course.train_experience_digests,
        holdout_experience_digests=course.holdout_experience_digests,
        updated_workers=("k1.semantic", "k2.transition"),
        frozen_workers=("k3.outcome_projection",),
        candidate_namespace="taiji:k:candidate",
        training_steps=1,
        optimizer_state_present=False,
        fresh_restore_verified=True,
        parent_unchanged=True,
        holdout_untrained=True,
        rollback_restored=True,
    )

    restored = KContinuationUpdateReceipt.from_payload(receipt.to_payload())
    assert restored == receipt
    assert restored.passed

    with pytest.raises(ValueError, match="changed deterministic K3"):
        replace(
            receipt,
            candidate_worker_checkpoint_digests=(
                ("k1.semantic", "4" * 64),
                ("k2.transition", "5" * 64),
                ("k3.outcome_projection", "9" * 64),
            ),
        )
