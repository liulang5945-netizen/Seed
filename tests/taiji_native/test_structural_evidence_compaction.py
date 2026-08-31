from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from scripts.training.eval_taiji_structural_bridge import _build_model
from taiji import (
    StructuralEvidenceCompactionResult,
    StructuralEvidenceConsumptionAudit,
    StructuralEvidenceLedger,
    StructuralEvidencePressureSnapshot,
    StructuralRuntimeObservation,
    TaijiConfig,
    TSKV8Adapter,
)
from taiji.structural_pressure import project_structural_growth_pressure


def _observation(
    tick: int,
    evidence_id: str,
    *,
    network_id: str = "network:demo",
    region_id: str = "region:demo",
) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id=network_id,
        region_id=region_id,
        tick=tick,
        usage=0.4,
        resource_pressure=0.2,
        prediction_error=0.7,
        learning_gain=0.1,
        holdout_transfer=0.8,
        task_slice_id="task:train",
        partition="train",
        evidence_id=evidence_id,
    )


def test_consumption_audit_and_compaction_preserve_cross_round_lineage() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=2, max_compacted_windows=4)
    for tick in (1, 2, 3, 4, 5, 6):
        ledger.append(_observation(tick, f"evidence:{tick}"))
    ledger.append(
        _observation(
            1,
            "other:1",
            network_id="network:other",
            region_id="region:other",
        )
    )
    ledger.append(
        _observation(
            2,
            "other:2",
            network_id="network:other",
            region_id="region:other",
        )
    )

    summaries = ledger.sealed_summaries
    first_digest = summaries[0].window_digest
    second_digest = summaries[1].window_digest
    third_digest = summaries[2].window_digest
    other_digest = summaries[3].window_digest
    audit = ledger.audit_consumption(
        evaluated_window_digests=(first_digest, second_digest, "missing:digest"),
        scheduler_revision=2,
    )
    assert isinstance(audit, StructuralEvidenceConsumptionAudit)
    assert audit.consumed_window_digests == (first_digest, second_digest)
    assert audit.unconsumed_window_digests == (third_digest, other_digest)
    assert audit.orphaned_evaluated_window_digests == ("missing:digest",)
    assert audit.stream_status[0][0] == "network:demo:region:demo"
    assert StructuralEvidenceConsumptionAudit.from_payload(audit.to_payload()) == audit

    result = ledger.compact_consumed_windows(
        evaluated_window_digests=(first_digest, second_digest),
        scheduler_revision=2,
        keep_latest_per_stream=1,
    )
    assert isinstance(result, StructuralEvidenceCompactionResult)
    assert result.status == "compacted"
    assert result.compacted_window_digests == (first_digest,)
    assert result.retained_window_digests == (second_digest, third_digest, other_digest)
    assert ledger.active_observed_count == 6
    assert ledger.observed_count == 8
    assert ledger.compacted_windows[0].task_slice_ids == ("task:train",)
    assert ledger.compacted_windows[0].partition_counts == (("train", 2),)
    assert ledger.compacted_windows[0].consumed_scheduler_revision == 2
    assert ledger.append(_observation(1, "evidence:1")).status == "duplicate"

    restored = StructuralEvidenceLedger.from_payload(ledger.to_payload())
    assert restored.digest == ledger.digest
    assert restored.compacted_windows == ledger.compacted_windows
    assert StructuralEvidenceCompactionResult.from_payload(result.to_payload()) == result


def test_compaction_keeps_unconsumed_windows_and_rejects_tampered_provenance() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=1, max_compacted_windows=2)
    ledger.append(_observation(1, "evidence:1"))
    ledger.append(_observation(2, "evidence:2"))
    first_digest, second_digest = [item.window_digest for item in ledger.sealed_summaries]

    result = ledger.compact_consumed_windows(
        evaluated_window_digests=(first_digest,),
        keep_latest_per_stream=1,
    )
    assert result.status == "nothing_to_compact"
    assert tuple(item.window_digest for item in ledger.sealed_summaries) == (
        first_digest,
        second_digest,
    )

    tampered = copy.deepcopy(ledger.to_payload())
    tampered["sealed_windows"][0]["window_digest"] = "0" * 64
    with pytest.raises(ValueError, match="ledger digest mismatch"):
        StructuralEvidenceLedger.from_payload(tampered)


def test_compacted_history_remains_checkpointable_without_resurrection() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=1, max_compacted_windows=2)
    ledger.append(_observation(1, "evidence:1"))
    digest = ledger.sealed_summaries[0].window_digest
    ledger.compact_consumed_windows(evaluated_window_digests=(digest,), keep_latest_per_stream=0)

    payload = ledger.to_payload()
    tampered = copy.deepcopy(payload)
    tampered["compacted_windows"][0]["provenance_digest"] = "1" * 64
    with pytest.raises(ValueError, match="ledger digest mismatch"):
        StructuralEvidenceLedger.from_payload(tampered)

    with pytest.raises(ValueError, match="different content"):
        ledger.append(_observation(1, "evidence:1", network_id="network:changed"))


def test_adapter_exposes_audit_and_persists_compaction_through_native_checkpoint() -> None:
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
        seed=73,
    )
    model = TSKV8Adapter(config, episode_id="evidence-compaction")
    for tick in range(1, 17):
        model.record_structural_runtime_observation(
            _observation(tick, f"evidence:{tick}")
        )
    first, second = model.structural_evidence_summaries
    scheduler = model.structural_growth_scheduler_state.advance(
        last_evaluated_tick=first.last_tick,
        window_digests=(first.window_digest,),
        stream_key="network:demo:region:demo",
    )
    model._structural_growth_scheduler_state = scheduler

    before = model.structural_evidence_consumption_audit
    assert before.consumed_window_digests == (first.window_digest,)
    assert before.unconsumed_window_digests == (second.window_digest,)
    result = model.compact_structural_evidence_history(keep_latest_per_stream=0)
    assert result.status == "compacted"
    assert result.compacted_window_digests == (first.window_digest,)
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    assert restored.structural_evidence_ledger.digest == model.structural_evidence_ledger.digest
    assert restored.structural_evidence_consumption_audit == model.structural_evidence_consumption_audit
    assert restored.structural_evidence_ledger.compacted_windows == model.structural_evidence_ledger.compacted_windows


def test_pressure_projection_identity_survives_consumed_window_compaction() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=1, max_compacted_windows=4)
    train_one = _observation(1, "evidence:1")
    train_two = _observation(2, "evidence:2")
    holdout = _observation(3, "evidence:3")
    train_one = replace(train_one, task_slice_id="task:one")
    train_two = replace(train_two, task_slice_id="task:two")
    holdout = replace(holdout, task_slice_id="task:holdout", partition="holdout")
    for item in (train_one, train_two, holdout):
        ledger.append(item)
    before = project_structural_growth_pressure(
        ledger.sealed_summaries,
        minimum_train_task_slices=2,
        minimum_train_windows=2,
    )
    train_digests = tuple(item.window_digest for item in ledger.sealed_summaries[:2])
    ledger.compact_consumed_windows(
        evaluated_window_digests=train_digests,
        keep_latest_per_stream=0,
    )
    assert len(ledger.pressure_snapshots) == 1
    assert isinstance(ledger.pressure_snapshots[0], StructuralEvidencePressureSnapshot)
    after = project_structural_growth_pressure(
        ledger.sealed_summaries,
        minimum_train_task_slices=2,
        minimum_train_windows=2,
        historical_snapshots=ledger.pressure_snapshots,
    )
    assert after == before


def test_scheduler_does_not_recreate_candidate_after_compaction() -> None:
    model, region = _build_model()
    model._structural_evidence_ledger = StructuralEvidenceLedger(window_capacity=1)
    observations = (
        replace(
            _observation(1, "schedule:train:one"),
            network_id="standalone:adaptive.cortex",
            region_id="region:pressure-canary",
            task_slice_id="task:one",
        ),
        replace(
            _observation(2, "schedule:train:two"),
            network_id="standalone:adaptive.cortex",
            region_id="region:pressure-canary",
            task_slice_id="task:two",
        ),
        replace(
            _observation(3, "schedule:holdout"),
            network_id="standalone:adaptive.cortex",
            region_id="region:pressure-canary",
            task_slice_id="task:holdout",
            partition="holdout",
        ),
    )
    for observation in observations:
        model.record_structural_runtime_observation(observation)

    scheduled = model.schedule_structural_growth_from_evidence(
        network_id="standalone:adaptive.cortex",
        region_id="region:pressure-canary",
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
        minimum_train_task_slices=2,
        minimum_train_windows=2,
    )
    assert scheduled.status == "candidate_created"
    candidate_count = len(model.structural_proposal_candidates)
    compacted = model.compact_structural_evidence_history(keep_latest_per_stream=1)
    assert compacted.status == "compacted"
    repeated = model.schedule_structural_growth_from_evidence(
        network_id="standalone:adaptive.cortex",
        region_id="region:pressure-canary",
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
        minimum_train_task_slices=2,
        minimum_train_windows=2,
    )
    assert repeated.status == "waiting"
    assert len(model.structural_proposal_candidates) == candidate_count
    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert len(restored.structural_proposal_candidates) == candidate_count
    assert restored.structural_evidence_ledger.pressure_snapshots == (
        model.structural_evidence_ledger.pressure_snapshots
    )
