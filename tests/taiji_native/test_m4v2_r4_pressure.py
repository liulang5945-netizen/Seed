from __future__ import annotations

import pytest

from taiji import (
    AdaptiveResidualGrowthDecision,
    AdaptiveResidualGrowthPolicy,
    AdaptiveResidualGrowthPressure,
    AdaptiveResidualGrowthTrigger,
    Taiji,
    TaijiConfig,
)
from taiji.internalization import content_digest

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
    assert AdaptiveResidualGrowthDecision.from_payload(decision.to_payload()) == decision


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


def _taiji_config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(16,),
        synapse_fan_in=4,
        motor_fan_in=8,
        predictive_context_fan_in=4,
        memory_units=16,
        memory_fan_in=4,
        memory_meta_dim=8,
        memory_readout_fan_in=4,
        identity_organ_enabled=False,
        seed=321,
    )


def test_r4_live_pressure_is_emitted_by_bridge_tick_and_restores_exactly() -> None:
    model = Taiji(_taiji_config(), episode_id="r4-live")
    model.enable_adaptive_residual_bridge(gate=1.0, residual_gain=1.0)
    policy = AdaptiveResidualGrowthPolicy(
        ema_rate=1.0,
        minimum_pressure=0.0,
        minimum_residual_error=0.0,
        minimum_fast_slow_conflict=0.0,
        minimum_activity_saturation=0.0,
        minimum_utility_gap=0.0,
        minimum_resource_state=0.0,
        required_pressure_steps=1,
        growth_resource_cost=1,
    )
    metadata = model.enable_adaptive_residual_growth(policy=policy)
    parent_digest = str(metadata["parent_checkpoint_digest"])
    unit_count = model.adaptive_residual_bridge.unit_count

    model.reset_dynamics(episode_id="r4-live-probe")
    for symbol in (97, 98):
        model.observe(
            symbol,
            learn=True,
            learn_fabric=False,
            learn_predictive_context=False,
            learn_predictive_readout=False,
            learn_adaptive_residual_bridge=True,
            readout="predictive",
        )

    trigger = model.adaptive_residual_growth_trigger
    assert trigger is not None
    assert trigger.last_pressure is not None
    assert trigger.last_pressure.parent_checkpoint_digest == parent_digest
    assert trigger.last_pressure.bridge_id == "predictive_residual.bridge"
    assert "task_id" not in trigger.last_pressure.to_payload()
    decision = model.adaptive_residual_growth_decision
    assert decision is not None
    assert decision.should_propose is True
    assert AdaptiveResidualGrowthDecision.from_payload(decision.to_payload()) == decision
    assert model.adaptive_residual_bridge.unit_count == unit_count

    checkpoint = model.checkpoint()
    restored = Taiji.from_checkpoint(checkpoint)
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)
    assert restored.adaptive_residual_growth_decision == decision
    assert restored.adaptive_residual_growth_trigger.last_pressure == trigger.last_pressure
