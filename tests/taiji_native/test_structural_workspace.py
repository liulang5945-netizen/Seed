from __future__ import annotations

import pytest
import torch

from taiji import (
    AdaptiveNeuronRegion,
    StructuralWorkspaceRouter,
    WorkspaceCandidate,
    WorkspaceRouter,
)


def _region() -> AdaptiveNeuronRegion:
    return AdaptiveNeuronRegion(
        region_id="workspace.cortex",
        input_dim=3,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(7),
    )


def test_structural_workspace_capacity_tracks_live_neuron_count() -> None:
    region = _region()
    binding = StructuralWorkspaceRouter(WorkspaceRouter(2, capacity=2, seed=11), region)
    candidates = tuple(
        WorkspaceCandidate(f"candidate-{index}", torch.tensor([float(index), 1.0]))
        for index in range(3)
    )

    before = binding.route(candidates, tick=1, mode="random", random_seed=3)
    proposal = region.propose_unit_add(unit_id="u2", evidence_ids=("evidence",))
    region.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(13))
    after = binding.route(candidates, tick=2, mode="random", random_seed=3)
    region.lesion_topology_proposal(proposal)
    lesioned = binding.route(candidates, tick=3, mode="random", random_seed=3)

    assert before.capacity == 2
    assert len(before.selected_ids) == 2
    assert after.capacity == 3
    assert len(after.selected_ids) == 3
    assert lesioned.capacity == 2
    assert len(lesioned.selected_ids) == 2


def test_structural_workspace_checkpoint_binds_neuron_identity() -> None:
    region = _region()
    binding = StructuralWorkspaceRouter(WorkspaceRouter(2, capacity=2, seed=11), region)
    restored = StructuralWorkspaceRouter.from_checkpoint(binding.checkpoint(), region=region)

    assert restored.capacity == 2
    assert restored.checkpoint()["checkpoint_digest"] == binding.checkpoint()["checkpoint_digest"]

    grown = _region()
    proposal = grown.propose_unit_add(unit_id="u2", evidence_ids=("evidence",))
    grown.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(13))
    with pytest.raises(ValueError, match="neuron identities drifted"):
        StructuralWorkspaceRouter.from_checkpoint(binding.checkpoint(), region=grown)


def test_structural_workspace_rebind_tracks_region_replacement() -> None:
    region = _region()
    binding = StructuralWorkspaceRouter(WorkspaceRouter(2, capacity=2, seed=11), region)
    grown = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(17),
    )
    proposal = grown.propose_unit_add(unit_id="u2", evidence_ids=("evidence",))
    grown.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(13))
    binding.rebind(grown)
    assert binding.capacity == 3

    replaced = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(19),
    )
    binding.rebind(replaced)
    assert binding.capacity == 2
    with pytest.raises(ValueError, match="region mismatch"):
        binding.rebind(
            AdaptiveNeuronRegion(
                region_id="other",
                input_dim=3,
                unit_ids=("u0", "u1"),
                fan_in=2,
                generator=torch.Generator().manual_seed(23),
            )
        )
