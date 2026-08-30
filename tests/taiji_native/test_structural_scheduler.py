from __future__ import annotations

from taiji.structural_scheduler import StructuralGrowthScheduleState


def test_structural_scheduler_cooldown_is_scoped_to_evidence_stream() -> None:
    state = StructuralGrowthScheduleState().advance(
        last_evaluated_tick=12,
        window_digests=("docs-window",),
        stream_key="workbench:workbench.docs",
    )

    assert state.last_evaluated_tick == 12
    assert state.last_evaluated_tick_for("workbench:workbench.docs") == 12
    assert state.last_evaluated_tick_for("workbench:workbench.code") == 0

    state = state.advance(
        last_evaluated_tick=9,
        window_digests=("code-window",),
        stream_key="workbench:workbench.code",
    )
    restored = StructuralGrowthScheduleState.from_payload(state.to_payload())

    assert restored.last_evaluated_tick_for("workbench:workbench.docs") == 12
    assert restored.last_evaluated_tick_for("workbench:workbench.code") == 9


def test_legacy_scheduler_checkpoint_uses_aggregate_cursor_conservatively() -> None:
    restored = StructuralGrowthScheduleState.from_payload(
        {
            "format": "taiji-structural-growth-scheduler-v1",
            "window_interval_ticks": 3,
            "last_evaluated_tick": 12,
            "evaluated_window_digests": ["old-window"],
            "revision": 4,
        }
    )

    assert restored.last_evaluated_tick_for("workbench:workbench.code") == 12
