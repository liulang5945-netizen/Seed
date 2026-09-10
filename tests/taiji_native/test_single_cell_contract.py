from __future__ import annotations

import random

import pytest

from taiji import (
    SingleCellEventCursor,
    SingleCellOwnerContract,
    TaijiSingleCellCheckpoint,
    TaijiSingleCellEvent,
    TaijiSingleCellManifest,
)


def _digest(letter: str) -> str:
    return letter * 64


def _owners() -> tuple[SingleCellOwnerContract, ...]:
    return (
        SingleCellOwnerContract.create(
            owner_id="S",
            owner_kind="runtime_evidence",
            availability="active",
            schema="observation+percept-v1",
            read_scopes=("external.observation",),
            write_scopes=("evidence.observation", "evidence.percept"),
            learning_owner="none",
        ),
        SingleCellOwnerContract.create(
            owner_id="G",
            owner_kind="runtime_selection",
            availability="active",
            schema="goal-selection-v1",
            read_scopes=("evidence.percept", "external.goal_target"),
            write_scopes=("selection.goal", "selection.content"),
            learning_owner="none",
        ),
        SingleCellOwnerContract.create(
            owner_id="K",
            owner_kind="learned_worker",
            availability="active",
            schema="k1.semantic+k2.transition-v1",
            read_scopes=("evidence.percept", "evidence.world"),
            write_scopes=("readout.k1", "readout.k2"),
            learning_owner="k1.semantic+k2.transition",
        ),
    )


def _manifest() -> TaijiSingleCellManifest:
    return TaijiSingleCellManifest.create(
        base_continuation_checkpoint_digest=_digest("a"),
        source_cohort_digest=_digest("b"),
        owner_contracts=_owners(),
        rollback_parent_digest=_digest("a"),
    )


def _checkpoint() -> TaijiSingleCellCheckpoint:
    manifest = _manifest()
    event = TaijiSingleCellEvent.create(
        event_index=0,
        event_type="observation",
        input_digest=_digest("f"),
        state_before={"S": None, "G": None, "K": None},
        state_after={"S": _digest("c"), "G": None, "K": None},
        output_digest=_digest("c"),
        confidence=0.9,
        status="observed",
        attributes={"source": "test"},
    )
    rng = random.Random(17)
    return TaijiSingleCellCheckpoint.create(
        base_continuation_checkpoint_digest=_digest("a"),
        base_continuation_ref="base.json",
        manifest_digest=manifest.manifest_digest,
        worker_checkpoint_digests={"k1": _digest("d"), "k2": _digest("e")},
        worker_checkpoint_refs={"k1": "k1.pt", "k2": "k2.pt"},
        owner_state_digests={"S": _digest("c"), "G": _digest("d"), "K": _digest("e")},
        owner_state_refs={"S": "s.pt", "G": "g.pt", "K": "k.pt"},
        event_digests=(event.event_digest,),
        event_refs=("event-0.pt",),
        cursor=SingleCellEventCursor(
            event_index=1,
            event_total=1,
            case_index=1,
            stage="complete",
        ),
        rng_state=rng.getstate(),
        budget_counts={"events": 1, "k1_readouts": 0, "k2_readouts": 0, "actions": 0},
        lineage_chain=(_digest("a"), manifest.manifest_digest),
    )


def test_single_cell_manifest_and_event_bind_owners() -> None:
    manifest = _manifest()
    restored_manifest = TaijiSingleCellManifest.from_payload(manifest.to_payload())
    assert restored_manifest.manifest_digest == manifest.manifest_digest

    event = TaijiSingleCellEvent.create(
        event_index=1,
        event_type="g_select",
        input_digest=_digest("a"),
        state_before={"S": _digest("b"), "G": None, "K": None},
        state_after={"S": _digest("b"), "G": _digest("c"), "K": None},
        output_digest=_digest("c"),
        confidence=1.0,
        status="control-only",
        attributes={"learned": False},
    )
    assert TaijiSingleCellEvent.from_payload(event.to_payload()).event_digest == event.event_digest

    with pytest.raises(ValueError, match="owner mask"):
        TaijiSingleCellEvent(
            **{
                **event.to_payload(),
                "read_owners": ["K"],
            }
        )


def test_single_cell_checkpoint_roundtrip_is_path_independent() -> None:
    checkpoint = _checkpoint()
    restored = TaijiSingleCellCheckpoint.from_payload(checkpoint.to_payload())
    assert restored.checkpoint_digest == checkpoint.checkpoint_digest
    assert restored.logical_digest == checkpoint.logical_digest
    restored.assert_base(_digest("a"))
    restored.assert_manifest(checkpoint.manifest_digest)


def test_single_cell_checkpoint_rejects_tampering_and_wrong_base() -> None:
    checkpoint = _checkpoint()
    tampered = checkpoint.to_payload()
    tampered["cursor"]["event_index"] = 0
    with pytest.raises(ValueError, match="digest"):
        TaijiSingleCellCheckpoint.from_payload(tampered)
    with pytest.raises(ValueError, match="boundary"):
        checkpoint.assert_base(_digest("0"))
