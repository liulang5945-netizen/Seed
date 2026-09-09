from __future__ import annotations

import copy

import pytest

from taiji import (
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    WorldEvent,
    WorldState,
)


def _world(*, tick: int = 0, events: tuple[WorldEvent, ...] = ()) -> WorldState:
    return WorldState(
        tick=tick,
        entities=("workbench",),
        relations=(("workbench", "anchor", "ready"),),
        events=events,
        uncertainty=0.0,
    )


def _event(*, tick: int = 0, success: bool = True, event_id: str = "event-0") -> WorldEvent:
    return WorldEvent(
        event_id=event_id,
        kind="workbench.evidence",
        tick=tick,
        subject_id="workspace.read",
        attributes=(
            ("capability_id", "workspace.read"),
            ("success", success),
            ("after_state_digest", "a" * 64),
        ),
        provenance="workbench-observed",
    )


def _spec(*, required_outcome: str = "success") -> OutcomeDependencySpec:
    return OutcomeDependencySpec(
        dependency_id="dep-0",
        next_task_id="task-follow-up",
        capability_id="workspace.programming_language.resolve",
        required_outcome=required_outcome,
    )


def test_projection_apply_and_checkpoint_roundtrip() -> None:
    projector = OutcomeDependencyProjector("episode-0")
    world = _world()
    projection = projector.project(world, _event(), _spec())

    assert projection.accepted is True
    assert projection.reason_code == "accepted"
    assert len(projection.lineage) == 4
    enriched = projector.apply(world, projection)
    assert enriched.events == (_event(),)
    assert ("dependency", "digest", projection.dependency_digest) in enriched.relations

    restored_projection = type(projection).from_payload(projection.to_payload())
    assert restored_projection.projection_digest == projection.projection_digest
    restored_projector = OutcomeDependencyProjector.from_checkpoint(projector.checkpoint())
    assert restored_projector.checkpoint() == projector.checkpoint()


def test_projection_allows_runtime_recorded_event_but_rejects_duplicate_dependency() -> None:
    projector = OutcomeDependencyProjector("episode-0")
    event = _event()
    recorded = _world(events=(event,))
    projection = projector.project(recorded, event, _spec())

    enriched = projector.apply(recorded, projection)
    assert enriched.events == (event,)
    with pytest.raises(ValueError, match="already applied"):
        projector.apply(enriched, projection)


@pytest.mark.parametrize(
    ("world", "event", "spec", "reason"),
    (
        (_world(), _event(tick=1), _spec(), "stale_event_tick"),
        (_world(), _event(success=False), _spec(), "dependency_outcome_mismatch"),
    ),
)
def test_projection_rejects_stale_or_wrong_outcome(
    world: WorldState,
    event: WorldEvent,
    spec: OutcomeDependencySpec,
    reason: str,
) -> None:
    projection = OutcomeDependencyProjector("episode-0").project(world, event, spec)
    assert projection.accepted is False
    assert projection.reason_code == reason


def test_lesion_and_cross_scope_are_fail_closed() -> None:
    world = _world()
    event = _event()
    spec = _spec()
    lesioned = OutcomeDependencyProjector("episode-0", lesioned=True).project(world, event, spec)
    assert lesioned.accepted is False
    assert lesioned.reason_code == "outcome_feedback_lesioned"

    projection = OutcomeDependencyProjector("episode-0").project(world, event, spec)
    with pytest.raises(ValueError, match="scope mismatch"):
        OutcomeDependencyProjector("episode-1").apply(world, projection)


def test_projection_payload_digest_is_tamper_evident() -> None:
    projector = OutcomeDependencyProjector("episode-0")
    payload = copy.deepcopy(projector.project(_world(), _event(), _spec()).to_payload())
    payload["spec"]["next_task_id"] = "other-task"

    with pytest.raises(ValueError, match="projection digest mismatch"):
        type(projector.project(_world(), _event(), _spec())).from_payload(payload)


def test_projector_checkpoint_digest_is_tamper_evident() -> None:
    payload = OutcomeDependencyProjector("episode-0").checkpoint()
    payload["lesioned"] = True

    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        OutcomeDependencyProjector.from_checkpoint(payload)
