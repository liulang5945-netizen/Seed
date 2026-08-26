from __future__ import annotations

import pytest
import torch

from taiji import SparseSynapses, StructuralTopologyProposal


def _synapses() -> SparseSynapses:
    return SparseSynapses(
        out_features=4,
        in_features=9,
        fan_in=3,
        generator=torch.Generator().manual_seed(71),
        init_scale=0.45,
        max_weight_norm=2.5,
    )


def _proposal(synapses: SparseSynapses) -> StructuralTopologyProposal:
    row = synapses.pre_index[0].long()
    replacement = next(index for index in range(synapses.in_features) if index not in row)
    return synapses.propose_topology_rewire(
        substrate_id="test.synapse",
        post_index=0,
        slot_index=0,
        replacement_pre_index=replacement,
        evidence_ids=("holdout:donor",),
        parent_checkpoint_id="parent:0",
    )


def test_topology_proposal_roundtrip_rewire_and_lesion_gate() -> None:
    synapses = _synapses()
    proposal = _proposal(synapses)
    restored_proposal = StructuralTopologyProposal.from_payload(proposal.to_payload())
    donor = int(dict(proposal.specification)["replacement_pre_index"])
    holdout = torch.zeros(synapses.in_features)
    holdout[donor] = 0.8

    assert float(synapses.forward(holdout)[0].item()) == 0.0
    assert synapses.apply_topology_proposal(restored_proposal) is True
    synapses.local_update(
        torch.ones(synapses.out_features),
        holdout,
        learning_rate=0.8,
        weight_decay=0.0,
    )
    learned_score = float(synapses.forward(holdout)[0].item())
    assert learned_score > 0.0

    payload = synapses.to_payload()
    checkpoint = _synapses()
    checkpoint.load_payload(payload)
    assert torch.equal(checkpoint.pre_index, synapses.pre_index)
    assert torch.equal(checkpoint.edge_weight, synapses.edge_weight)
    assert torch.allclose(checkpoint.forward(holdout), synapses.forward(holdout))

    assert checkpoint.lesion_topology_proposal(restored_proposal) is True
    assert float(checkpoint.forward(holdout)[0].item()) < learned_score


def test_topology_proposal_rejects_missing_evidence_and_parent_drift() -> None:
    synapses = _synapses()
    proposal = _proposal(synapses)
    no_evidence = synapses.propose_topology_rewire(
        substrate_id="test.synapse",
        post_index=0,
        slot_index=0,
        replacement_pre_index=int(dict(proposal.specification)["replacement_pre_index"]),
        evidence_ids=(),
    )
    with pytest.raises(ValueError, match="evidence_ids"):
        synapses.apply_topology_proposal(no_evidence)

    assert synapses.apply_topology_proposal(proposal) is True
    with pytest.raises(ValueError, match="parent edge"):
        synapses.apply_topology_proposal(proposal)
