from __future__ import annotations

import random

import pytest

from taiji import (
    ContentPlan,
    Goal,
    GSelectionState,
    OwnerTransferCursor,
    StructuredSemanticResult,
    TaijiOwnerTransferCheckpoint,
    TaijiOwnerTransferEvent,
    TaijiOwnerTransferManifest,
)


def _digest(letter: str) -> str:
    return letter * 64


def _result() -> StructuredSemanticResult:
    return StructuredSemanticResult(
        status="resolved",
        world=None,
        goal=Goal("goal-1", "inspect the language", priority=1.0),
        content_plan=ContentPlan(
            content_id="content:inspect-language",
            intent_id="inspect-language",
            intent_kind="request_information",
            source_goal_id="goal-1",
            confidence=0.9,
        ),
        fact_scores={"fact": 0.9},
        goal_scores={"goal-1": 0.9},
        content_scores={"content:inspect-language": 0.9},
        confidence=0.9,
        ambiguity=0.1,
    )


def _manifest() -> TaijiOwnerTransferManifest:
    return TaijiOwnerTransferManifest.create(
        base_single_cell_manifest_digest=_digest("a"),
        base_continuation_checkpoint_digest=_digest("b"),
        worker_checkpoint_digests={"k1": _digest("c"), "k2": _digest("d")},
    )


def _checkpoint() -> TaijiOwnerTransferCheckpoint:
    manifest = _manifest()
    event = TaijiOwnerTransferEvent.create(
        event_index=0,
        event_type="g_selection",
        input_payload={"candidate": _digest("e")},
        state_before={"S": _digest("f"), "G": None, "K": _digest("a")},
        state_after={"S": _digest("f"), "G": _digest("b"), "K": _digest("a")},
        output_payload={"selection": _digest("b")},
        confidence=0.9,
        status="selected",
        attributes={"external_target_used": False},
    )
    rng = random.Random(11)
    return TaijiOwnerTransferCheckpoint.create(
        base_continuation_checkpoint_digest=_digest("b"),
        base_continuation_ref="base.json",
        base_single_cell_manifest_digest=_digest("a"),
        base_single_cell_manifest_ref="p3-1.json",
        manifest_digest=manifest.manifest_digest,
        worker_checkpoint_digests={"k1": _digest("c"), "k2": _digest("d")},
        worker_checkpoint_refs={"k1": "k1.pt", "k2": "k2.pt"},
        owner_state_digests={"S": _digest("f"), "G": _digest("b"), "K": _digest("a")},
        owner_state_refs={"S": "s.pt", "G": "g.pt", "K": "k.pt"},
        event_digests=(event.event_digest,),
        event_refs=("event-0.pt",),
        cursor=OwnerTransferCursor(1, 1, 1, "complete"),
        rng_state=rng.getstate(),
        budget_counts={"events": 1, "selections": 1, "actions": 0},
        lineage_chain=(_digest("b"), _digest("a"), manifest.manifest_digest),
    )


def test_g_selection_state_rebinds_k_candidate_without_external_target() -> None:
    selection = GSelectionState.from_k1_result(_result())
    restored = GSelectionState.from_payload(selection.to_payload())
    assert restored.selection_digest == selection.selection_digest
    assert restored.selected_goal is not None
    assert restored.selected_content is not None
    assert not restored.external_target_used

    with pytest.raises(ValueError, match="external target"):
        GSelectionState(
            **{
                **selection.__dict__,
                "external_target_used": True,
            }
        )


def test_owner_transfer_manifest_event_and_checkpoint_roundtrip() -> None:
    manifest = _manifest()
    assert TaijiOwnerTransferManifest.from_payload(manifest.to_payload()).manifest_digest == (
        manifest.manifest_digest
    )
    checkpoint = _checkpoint()
    restored = TaijiOwnerTransferCheckpoint.from_payload(checkpoint.to_payload())
    assert restored.checkpoint_digest == checkpoint.checkpoint_digest
    assert restored.logical_digest == checkpoint.logical_digest
    restored.assert_base(_digest("b"))
    restored.assert_single_cell(_digest("a"))
    restored.assert_manifest(checkpoint.manifest_digest)


def test_owner_transfer_rejects_wrong_masks_and_tampering() -> None:
    event = TaijiOwnerTransferEvent.create(
        event_index=0,
        event_type="k_candidate",
        input_payload={"evidence": _digest("a")},
        state_before={"S": _digest("b"), "G": None, "K": None},
        state_after={"S": _digest("b"), "G": None, "K": _digest("c")},
        output_payload={"candidate": _digest("c")},
        confidence=0.8,
        status="candidate",
        attributes={},
    )
    with pytest.raises(ValueError, match="owner mask"):
        TaijiOwnerTransferEvent(
            **{
                **event.to_payload(),
                "read_owners": ["G"],
            }
        )

    checkpoint = _checkpoint()
    tampered = checkpoint.to_payload()
    tampered["cursor"]["event_index"] = 0
    with pytest.raises(ValueError, match="digest"):
        TaijiOwnerTransferCheckpoint.from_payload(tampered)
    with pytest.raises(ValueError, match="P3.0"):
        checkpoint.assert_base(_digest("0"))
