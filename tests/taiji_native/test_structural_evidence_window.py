from __future__ import annotations

import pytest
import torch

from taiji import (
    AdaptiveNeuronRegion,
    StructuralEvidenceLedger,
    StructuralRuntimeObservation,
    TaijiConfig,
    TSKV8Adapter,
)


def _observation(tick: int, *, usage: float = 0.4) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id="network:demo",
        region_id="region:demo",
        tick=tick,
        usage=usage,
        resource_pressure=0.2,
        prediction_error=0.7,
        learning_gain=0.1,
        holdout_transfer=0.8,
        evidence_id=f"evidence:{tick}",
    )


def test_structural_evidence_window_is_content_addressed_and_idempotent() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=2)

    first = ledger.append(_observation(1))
    duplicate = ledger.append(_observation(1))
    assert first.status == "accepted"
    assert duplicate.status == "duplicate"
    assert ledger.observed_count == 1

    conflicting = _observation(1, usage=0.9)
    with pytest.raises(ValueError, match="different content"):
        ledger.append(conflicting)
    assert ledger.observed_count == 1

    sealed = ledger.append(_observation(2))
    assert sealed.status == "window_sealed"
    assert len(ledger.open_windows) == 0
    assert len(ledger.sealed_summaries) == 1
    summary = ledger.sealed_summaries[0]
    assert summary.observation_count == 2
    assert summary.prediction_observation_count == 2
    assert summary.first_tick == 1
    assert summary.last_tick == 2
    assert summary.window_digest == sealed.sealed_window_digest

    restored = StructuralEvidenceLedger.from_payload(ledger.to_payload())
    assert restored.digest == ledger.digest
    assert restored.sealed_summaries == ledger.sealed_summaries
    assert restored.append(_observation(1)).status == "duplicate"


def test_structural_evidence_capacity_failure_is_atomic() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=4, max_evidence_index=1)
    ledger.append(_observation(1))
    before = ledger.to_payload()

    with pytest.raises(OverflowError, match="index capacity"):
        ledger.append(_observation(2))

    assert ledger.to_payload() == before


def test_adapter_checkpoints_long_horizon_structural_evidence() -> None:
    config = TaijiConfig(
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
        seed=71,
    )
    model = TSKV8Adapter(config, episode_id="evidence-checkpoint")
    region = AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(7),
    )
    model.attach_adaptive_neuron_region(region)
    model.step_adaptive_neuron_region(region.region_id, torch.ones(5))
    model.step_adaptive_neuron_region(region.region_id, torch.ones(5))

    assert model.structural_evidence_ledger.observed_count == 2
    assert len(model.structural_evidence_ledger.open_windows) == 1
    before_digest = model.structural_evidence_ledger.digest

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.structural_evidence_ledger.digest == before_digest
    assert restored.structural_evidence_ledger.observed_count == 2
    assert restored.structural_runtime_observations == model.structural_runtime_observations
