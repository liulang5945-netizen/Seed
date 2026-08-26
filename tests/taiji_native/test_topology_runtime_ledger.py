from __future__ import annotations

from taiji import TaijiConfig, TSKV8Adapter


def _config(*, budget: int) -> TaijiConfig:
    return TaijiConfig(
        alphabet_size=257,
        boundary_symbol=256,
        region_sizes=(8, 6),
        synapse_fan_in=3,
        motor_fan_in=4,
        lateral_fan_in=3,
        memory_units=12,
        memory_fan_in=3,
        memory_readout_fan_in=4,
        memory_meta_dim=4,
        memory_time_dim=4,
        memory_episode_dim=4,
        development_structural_budget=budget,
        seed=71,
    )


def _proposal(model: TSKV8Adapter):
    bank = model.fabric.decoders[0]
    row = bank.pre_index[0].long()
    replacement = next(index for index in range(bank.in_features) if index not in row)
    return model.propose_synapse_rewire(
        substrate_id="fabric.decoder.0",
        post_index=0,
        slot_index=0,
        replacement_pre_index=replacement,
        evidence_ids=("runtime:topology-holdout",),
    )


def test_runtime_topology_ledger_consumes_budget_and_native_rollback() -> None:
    model = TSKV8Adapter(_config(budget=1), episode_id="topology-ledger")
    original = model.fabric.decoders[0].pre_index.clone()
    proposal = _proposal(model)

    assert model.commit_synapse_rewire(proposal) is True
    accepted = model.topology_proposals[-1]
    assert accepted.status == "accepted"
    assert accepted.validation_score == 1.0
    assert model.cognitive_snapshot().development.structural_budget == 0
    assert model.cognitive_snapshot().development.last_update_source == "synapse-topology-rewire"

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.topology_proposals[-1].status == "accepted"
    assert restored.cognitive_snapshot().development.structural_budget == 0
    assert restored.rollback_synapse_rewire(accepted.proposal_id) is True
    assert restored.cognitive_snapshot().development.structural_budget == 1
    assert restored.topology_proposals[-1].status == "rolled_back"
    assert restored.cognitive_snapshot().development.growth_count == 0
    assert restored.fabric.decoders[0].pre_index.equal(original)


def test_runtime_topology_ledger_rejects_growth_without_budget() -> None:
    model = TSKV8Adapter(_config(budget=0), episode_id="topology-no-budget")
    original = model.fabric.decoders[0].pre_index.clone()
    proposal = _proposal(model)

    assert model.commit_synapse_rewire(proposal) is False
    assert model.topology_proposals[-1].status == "rejected"
    assert model.cognitive_snapshot().development.structural_budget == 0
    assert model.fabric.decoders[0].pre_index.equal(original)
