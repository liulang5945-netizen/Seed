from __future__ import annotations

import pytest
import torch

from taiji import AdaptiveNeuronRegion, NeuronRegionDynamics


def _region() -> AdaptiveNeuronRegion:
    return AdaptiveNeuronRegion(
        region_id="cortex.test",
        input_dim=7,
        unit_ids=("u0", "u1"),
        fan_in=3,
        input_source_id="fabric.region.0",
        dynamics=NeuronRegionDynamics(
            learning_rate=0.8,
            weight_decay=0.0,
            recurrent_gain=0.0,
        ),
        generator=torch.Generator().manual_seed(71),
    )


def _proposal(region: AdaptiveNeuronRegion):
    return region.propose_unit_add(
        unit_id="u2",
        evidence_ids=("holdout:novel-unit",),
        parent_checkpoint_id="parent:region",
    )


def test_neuron_growth_preserves_identity_support_and_state_dimensions() -> None:
    region = _region()
    old_ids = region.unit_ids
    old_incoming_index = region.incoming.pre_index.clone()
    old_incoming_weight = region.incoming.edge_weight.clone()
    old_recurrent_index = None if region.recurrent is None else region.recurrent.pre_index.clone()
    old_recurrent_weight = None if region.recurrent is None else region.recurrent.edge_weight.clone()
    proposal = _proposal(region)

    assert region.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    ) is True
    assert region.unit_ids == (*old_ids, "u2")
    assert region.unit_count == 3
    assert region.membrane.shape == (3,)
    assert region.activity.shape == (3,)
    assert region.trace.shape == (3,)
    assert region.threshold.shape == (3,)
    assert torch.equal(region.incoming.pre_index[:2], old_incoming_index)
    assert torch.equal(region.incoming.edge_weight[:2], old_incoming_weight)
    assert region.recurrent is not None
    assert old_recurrent_index is not None and old_recurrent_weight is not None
    assert torch.equal(region.recurrent.pre_index[:2], old_recurrent_index)
    assert torch.equal(region.recurrent.edge_weight[:2], old_recurrent_weight)


def test_new_unit_can_learn_a_holdout_pattern_and_lesion_is_causal() -> None:
    region = _region()
    proposal = _proposal(region)
    region.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    )
    holdout = torch.zeros(region.input_dim)
    holdout[region.incoming.pre_index[2].long()] = 1.0
    region.incoming.edge_weight[2].zero_()
    before = float(region.incoming.forward(holdout)[2].item())
    region.learn(holdout, torch.tensor([0.0, 0.0, 1.0]))
    after = float(region.incoming.forward(holdout)[2].item())
    assert before == 0.0
    assert after > 0.0

    region.lesion_topology_proposal(proposal)
    assert "u2" in region.lesioned_unit_ids
    assert float(region.step(torch.ones(region.input_dim))[2].item()) == 0.0


def test_neuron_region_checkpoint_roundtrip_preserves_growth_and_lesion() -> None:
    region = _region()
    proposal = _proposal(region)
    region.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    )
    region.step(torch.ones(region.input_dim))
    region.lesion_topology_proposal(proposal)
    restored = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(123),
    )

    assert restored.unit_ids == region.unit_ids
    assert restored.lesioned_unit_ids == region.lesioned_unit_ids
    assert torch.equal(restored.incoming.pre_index, region.incoming.pre_index)
    assert torch.equal(restored.incoming.edge_weight, region.incoming.edge_weight)
    assert torch.allclose(restored.activity, region.activity)
    assert torch.allclose(restored.threshold, region.threshold)


def test_neuron_proposal_rejects_parent_drift_and_missing_evidence() -> None:
    region = _region()
    no_evidence = region.propose_unit_add(unit_id="u2", evidence_ids=())
    with pytest.raises(ValueError, match="evidence_ids"):
        region.apply_topology_proposal(
            no_evidence,
            generator=torch.Generator().manual_seed(99),
        )

    proposal = _proposal(region)
    region.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    )
    with pytest.raises(ValueError, match="already exists"):
        region.apply_topology_proposal(
            proposal,
            generator=torch.Generator().manual_seed(100),
        )
