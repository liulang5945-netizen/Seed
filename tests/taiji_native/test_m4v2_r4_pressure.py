from __future__ import annotations

import pytest
import torch

from taiji import (
    AdaptiveResidualGrowthDecision,
    AdaptiveResidualGrowthPolicy,
    AdaptiveResidualGrowthPressure,
    AdaptiveResidualGrowthTrigger,
    AdaptiveResidualShadow,
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
    bridge = model.adaptive_residual_bridge
    assert bridge is not None
    bridge_before = content_digest(bridge.to_payload())
    source_checkpoint_digest = content_digest(model.checkpoint())

    result = model.propose_adaptive_residual_growth_candidate()
    candidate = model.adaptive_residual_growth_candidate
    assert candidate is not None
    assert result["candidate_id"] == candidate.candidate_id
    assert candidate.source_checkpoint_digest == source_checkpoint_digest
    assert candidate.parent_checkpoint_digest == parent_digest
    assert candidate.proposal.status == "pending"
    assert candidate.proposal.evidence_ids == decision.evidence_ids
    assert candidate.unit_id not in bridge.region.unit_ids
    assert candidate.proposed_unit_count == candidate.parent_unit_count + 1
    assert candidate.proposed_edge_count > candidate.parent_edge_count
    assert content_digest(bridge.to_payload()) == bridge_before
    assert bridge.unit_count == unit_count

    with pytest.raises(RuntimeError, match="already pending"):
        model.propose_adaptive_residual_growth_candidate()

    model_before_shadow = content_digest(model.checkpoint())
    parent_region = bridge.region
    parent_region_payload = parent_region.to_payload()
    shadow = model.materialize_adaptive_residual_shadow()
    assert shadow.gate == 0.0
    assert shadow.unit_count == candidate.proposed_unit_count
    assert candidate.unit_id in shadow.region.unit_ids
    assert shadow.birth_anchor_unit_id in parent_region.unit_ids
    assert shadow.region.unit_ids[: candidate.parent_unit_count] == parent_region.unit_ids
    assert torch.equal(
        shadow.region.incoming.pre_index[: candidate.parent_unit_count],
        parent_region.incoming.pre_index,
    )
    assert torch.equal(
        shadow.region.incoming.edge_weight[: candidate.parent_unit_count],
        parent_region.incoming.edge_weight,
    )
    assert torch.equal(
        shadow.region.membrane[: candidate.parent_unit_count],
        parent_region.membrane,
    )
    assert torch.equal(
        shadow.region.activity[: candidate.parent_unit_count],
        parent_region.activity,
    )
    assert shadow.region.to_payload()["unit_ids"][-1] == candidate.unit_id
    candidate_index = shadow.region.unit_index(candidate.unit_id)
    candidate_projection = shadow.output_projection.pre_index == candidate_index
    assert bool(candidate_projection.any())
    assert torch.equal(
        shadow.output_projection.edge_weight[candidate_projection],
        torch.zeros_like(shadow.output_projection.edge_weight[candidate_projection]),
    )
    shadow_before_gate = content_digest(shadow.region.to_payload())
    context = torch.zeros(model.config.motor_context_dim)
    assert torch.equal(shadow.forward(context), context)
    assert content_digest(shadow.region.to_payload()) == shadow_before_gate
    assert content_digest(model.checkpoint()) == model_before_shadow

    shadow_checkpoint = shadow.to_payload()
    restored_shadow = AdaptiveResidualShadow.from_checkpoint(model.config, shadow_checkpoint)
    assert content_digest(restored_shadow.to_payload()) == content_digest(shadow_checkpoint)
    assert restored_shadow.birth_anchor_unit_id == shadow.birth_anchor_unit_id
    assert content_digest(parent_region_payload) == content_digest(parent_region.to_payload())

    bare_shadow_checkpoint = restored_shadow.to_payload()
    bare_shadow_digest = content_digest(bare_shadow_checkpoint)
    parent_incoming = restored_shadow.region.incoming.edge_weight[: candidate.parent_unit_count].clone()
    parent_recurrent = (
        None
        if restored_shadow.region.recurrent is None
        else restored_shadow.region.recurrent.edge_weight[: candidate.parent_unit_count].clone()
    )
    candidate_index = restored_shadow.region.unit_index(candidate.unit_id)
    parent_projection_mask = restored_shadow.output_projection.pre_index != candidate_index
    parent_projection = restored_shadow.output_projection.edge_weight[parent_projection_mask].clone()

    selected_context = None
    selected_feedback = None
    for symbol in range(99, 163):
        before = model.snapshot()
        model.observe(symbol, learn=False, readout="predictive")
        causal_error = model.predictive_readout.prediction_error(before.motor_probabilities, symbol)
        causal_feedback = model.predictive_readout.context_feedback(causal_error)
        current_context = bridge.last_input
        probe_shadow = AdaptiveResidualShadow.from_checkpoint(
            model.config,
            bare_shadow_checkpoint,
        )
        probe_shadow.set_gate(1.0)
        probe_shadow.forward(current_context)
        if probe_shadow.candidate_activity > 1e-8:
            selected_context = current_context
            selected_feedback = causal_feedback
            break
    assert selected_context is not None
    assert selected_feedback is not None
    model_after_parent_tick = content_digest(model.checkpoint())
    parent_bridge_after_tick = content_digest(bridge.to_payload())
    restored_shadow = AdaptiveResidualShadow.from_checkpoint(
        model.config,
        bare_shadow_checkpoint,
    )
    restored_shadow.set_gate(1.0)
    restored_shadow.forward(selected_context)
    candidate_index = restored_shadow.region.unit_index(candidate.unit_id)
    assert restored_shadow.candidate_activity > 1e-8
    assert restored_shadow.candidate_eligibility_norm > 1e-8
    # The candidate projection must keep learning from the causal eligibility
    # trace even when the current tick is silent.
    restored_shadow._last_activity[candidate_index] = 0.0
    restored_shadow.region.activity[candidate_index] = 0.0
    candidate_projection_before = restored_shadow.output_projection.edge_weight[
        ~parent_projection_mask
    ].clone()
    restored_shadow.learn(selected_feedback)
    candidate_projection_after = restored_shadow.output_projection.edge_weight[
        ~parent_projection_mask
    ]
    assert not torch.equal(candidate_projection_after, candidate_projection_before)
    assert torch.equal(
        restored_shadow.region.incoming.edge_weight[: candidate.parent_unit_count],
        parent_incoming,
    )
    if parent_recurrent is not None:
        assert restored_shadow.region.recurrent is not None
        assert torch.equal(
            restored_shadow.region.recurrent.edge_weight[: candidate.parent_unit_count],
            parent_recurrent,
        )
    assert torch.equal(
        restored_shadow.output_projection.edge_weight[parent_projection_mask],
        parent_projection,
    )
    trained_shadow_checkpoint = restored_shadow.to_payload()
    trained_shadow = AdaptiveResidualShadow.from_checkpoint(
        model.config,
        trained_shadow_checkpoint,
    )
    assert content_digest(trained_shadow.to_payload()) == content_digest(trained_shadow_checkpoint)
    continued = AdaptiveResidualShadow.from_checkpoint(model.config, trained_shadow_checkpoint)
    continued.forward(selected_context)
    continued.learn(selected_feedback)
    assert content_digest(continued.to_payload()) != content_digest(trained_shadow_checkpoint)
    rolled_back_shadow = AdaptiveResidualShadow.from_checkpoint(
        model.config,
        bare_shadow_checkpoint,
    )
    assert content_digest(rolled_back_shadow.to_payload()) == bare_shadow_digest
    assert content_digest(model.checkpoint()) == model_after_parent_tick
    assert content_digest(bridge.to_payload()) == parent_bridge_after_tick

    checkpoint = model.checkpoint()
    restored = Taiji.from_checkpoint(checkpoint)
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)
    assert restored.adaptive_residual_growth_decision == decision
    assert restored.adaptive_residual_growth_trigger.last_pressure == trigger.last_pressure
    assert restored.adaptive_residual_growth_candidate == candidate
