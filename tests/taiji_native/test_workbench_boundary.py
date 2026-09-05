from __future__ import annotations

import pytest

from taiji import (
    Taiji,
    TaijiConfig,
    WorkbenchBoundaryAuthorization,
    WorkbenchTaskBoundary,
    select_readout_generation,
)


def _authorization(
    boundary: WorkbenchTaskBoundary,
    *,
    project_id: str | None = None,
    current_tick: int = 12,
    usage: str = "execute",
    active_boundary_digest: str | None = None,
    authorized_capability_ids: tuple[str, ...] = ("workspace.read",),
) -> WorkbenchBoundaryAuthorization:
    return WorkbenchBoundaryAuthorization(
        project_id=boundary.project_id if project_id is None else project_id,
        task_id=boundary.task_id,
        session_id=boundary.session_id,
        capability_snapshot_id=boundary.capability_snapshot_id,
        authorized_capability_ids=authorized_capability_ids,
        active_boundary_digest=(
            boundary.token_digest
            if active_boundary_digest is None
            else active_boundary_digest
        ),
        current_tick=current_tick,
        usage=usage,
    )


def _boundary(*, task_id: str = "task:alpha") -> WorkbenchTaskBoundary:
    return WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id=task_id,
        session_id="session:one",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="protected" if task_id == "task:alpha" else "active",
        issued_tick=10,
        ttl_ticks=20,
    )


def test_same_task_boundary_is_content_addressed_and_selects_generation() -> None:
    first = _boundary()
    second = _boundary()

    assert first.token_digest == second.token_digest
    assert select_readout_generation(first, _authorization(first)) == "protected"
    restored = WorkbenchTaskBoundary.from_payload(first.to_payload())
    assert restored == first


def test_task_switch_issues_new_generation_with_lineage() -> None:
    old = _boundary()
    new = old.successor(
        task_id="task:beta",
        generation_scope="active",
        issued_tick=12,
        ttl_ticks=20,
    )

    assert new.token_digest != old.token_digest
    assert new.parent_token_digest == old.token_digest
    assert new.generation_scope == "active"
    assert select_readout_generation(new, _authorization(new, current_tick=12)) == "active"


def test_closed_old_task_can_only_be_replayed_read_only() -> None:
    old = _boundary().close(closed_tick=12)

    replay = _authorization(old, current_tick=12, usage="read_only_replay")
    assert select_readout_generation(old, replay) == "protected"

    execute = _authorization(old, current_tick=12, usage="execute")
    decision = old.authorize(execute)
    assert decision.accepted is False
    assert decision.reason_code == "closed_boundary_not_executable"


@pytest.mark.parametrize(
    ("kwargs", "reason_code"),
    [
        ({"project_id": "project:other"}, "cross_project_boundary"),
        ({"current_tick": 31}, "expired_boundary"),
        ({"authorized_capability_ids": ()}, "unauthorized_capability"),
    ],
)
def test_boundary_rejects_unsafe_context(kwargs: dict[str, object], reason_code: str) -> None:
    boundary = _boundary()
    decision = boundary.authorize(_authorization(boundary, **kwargs))

    assert decision.accepted is False
    assert decision.reason_code == reason_code
    with pytest.raises(PermissionError, match=reason_code):
        select_readout_generation(boundary, _authorization(boundary, **kwargs))


def test_stale_task_generation_is_rejected() -> None:
    old = _boundary()
    new = old.successor(
        task_id="task:beta",
        generation_scope="active",
        issued_tick=12,
        ttl_ticks=20,
    )
    context = _authorization(old, active_boundary_digest=new.token_digest)

    decision = old.authorize(context)
    assert decision.accepted is False
    assert decision.reason_code == "stale_task_generation"


def test_boundary_schema_rejects_covert_target_or_phase_fields() -> None:
    boundary = _boundary()
    payload = boundary.to_payload()
    payload["phase"] = "phase-B"
    payload["target_bytes"] = "answer"

    with pytest.raises(ValueError, match="unknown fields"):
        WorkbenchTaskBoundary.from_payload(payload)


def test_read_only_replay_does_not_mutate_boundary_payload() -> None:
    boundary = _boundary().close(closed_tick=12)
    before = boundary.to_payload()

    decision = boundary.authorize(
        _authorization(boundary, current_tick=12, usage="read_only_replay")
    )

    assert decision.read_only is True
    assert boundary.to_payload() == before


def test_taiji_generation_consumes_boundary_and_rejects_missing_readout() -> None:
    model = Taiji(TaijiConfig(seed=7), episode_id="workbench-generation")
    protected = _boundary()
    protected_context = _authorization(protected, current_tick=10)

    generated = model.generate(
        b"hi",
        2,
        boundary=protected,
        authorization=protected_context,
    )

    assert isinstance(generated, bytes)
    assert model.last_generation_route == {
        "boundary_digest": protected.token_digest,
        "generation_scope": "protected",
        "readout_owner": "predictive_readout",
        "read_only_replay": False,
    }

    active = protected.successor(
        task_id="task:beta",
        generation_scope="active",
        issued_tick=12,
        ttl_ticks=20,
    )
    with pytest.raises(RuntimeError, match="not attached"):
        model.generate(
            b"hi",
            2,
            boundary=active,
            authorization=_authorization(active, current_tick=12),
        )
