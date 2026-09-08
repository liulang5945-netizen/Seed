from __future__ import annotations

import pytest

from taiji import (
    AdaptiveResidualGrowthPolicy,
    AdaptiveResidualGrowthPressure,
    AdaptiveResidualGrowthTrigger,
)

PARENT_DIGEST = "p" * 64


def _pressure(
    tick: int,
    *,
    conflict: float = 0.9,
    resource: float = 0.9,
    parent_digest: str = PARENT_DIGEST,
) -> AdaptiveResidualGrowthPressure:
    return AdaptiveResidualGrowthPressure.create(
        bridge_id="predictive_residual.bridge",
        tick=tick,
        residual_error=0.9,
        fast_slow_conflict=conflict,
        activity_saturation=0.9,
        utility_gap=0.9,
        resource_state=resource,
        evidence_id=f"pressure:{tick}",
        parent_checkpoint_digest=parent_digest,
    )


def _policy(**overrides: object) -> AdaptiveResidualGrowthPolicy:
    values: dict[str, object] = {
        "ema_rate": 1.0,
        "minimum_pressure": 0.5,
        "minimum_residual_error": 0.5,
        "minimum_fast_slow_conflict": 0.5,
        "minimum_activity_saturation": 0.5,
        "minimum_utility_gap": 0.5,
        "minimum_resource_state": 0.5,
        "required_pressure_steps": 3,
        "growth_resource_cost": 1,
    }
    values.update(overrides)
    return AdaptiveResidualGrowthPolicy(**values)


def test_r4_pressure_is_content_addressed_and_round_trips() -> None:
    observation = _pressure(1)
    restored = AdaptiveResidualGrowthPressure.from_payload(observation.to_payload())

    assert restored == observation
    assert restored.pressure == pytest.approx(0.9)
    tampered = observation.to_payload()
    tampered["utility_gap"] = 0.1
    with pytest.raises(ValueError, match="digest mismatch"):
        AdaptiveResidualGrowthPressure.from_payload(tampered)


def test_r4_trigger_requires_native_pressure_persistence_and_emits_no_task_route() -> None:
    trigger = AdaptiveResidualGrowthTrigger(
        bridge_id="predictive_residual.bridge",
        policy=_policy(),
    )

    low_conflict = trigger.observe(_pressure(1, conflict=0.1), structural_budget=1)
    assert low_conflict.should_propose is False
    assert low_conflict.reasons == ("pressure_below_threshold", "pressure_persistence_below_threshold")
    assert "task_id" not in low_conflict.to_payload()

    assert trigger.observe(_pressure(2), structural_budget=1).should_propose is False
    assert trigger.observe(_pressure(3), structural_budget=1).should_propose is False
    decision = trigger.observe(_pressure(4), structural_budget=1)

    assert decision.should_propose is True
    assert decision.proposal_ordinal == 1
    assert decision.evidence_ids == ("pressure:2", "pressure:3", "pressure:4")
    assert "persistent_native_pressure" in decision.reasons


def test_r4_trigger_is_budget_gated_parent_bound_and_checkpointable() -> None:
    trigger = AdaptiveResidualGrowthTrigger(
        bridge_id="predictive_residual.bridge",
        policy=_policy(),
    )
    for tick in range(1, 4):
        decision = trigger.observe(_pressure(tick), structural_budget=0)
    assert decision.should_propose is False
    assert "structural_budget_insufficient" in decision.reasons

    checkpoint = trigger.checkpoint()
    restored = AdaptiveResidualGrowthTrigger.from_checkpoint(checkpoint)
    assert restored.checkpoint() == checkpoint

    with pytest.raises(ValueError, match="already observed"):
        restored.observe(_pressure(3), structural_budget=1)
    with pytest.raises(ValueError, match="parent checkpoint changed"):
        restored.observe(_pressure(4, parent_digest="q" * 64), structural_budget=1)
