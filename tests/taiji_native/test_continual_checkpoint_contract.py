from __future__ import annotations

import pytest

from taiji import ContinuationPhaseCursor, TaijiContinuationCheckpoint


def _checkpoint() -> TaijiContinuationCheckpoint:
    parent = "a" * 64
    return TaijiContinuationCheckpoint.create(
        parent_checkpoint_digest=parent,
        worker_checkpoint_digests={"k1": "b" * 64, "k2": "c" * 64},
        worker_checkpoint_refs={"k1": "k1.pt", "k2": "k2.pt"},
        phase_cursor=ContinuationPhaseCursor(phase="wake", index=1, total=2),
        experience_digests=("d" * 64, "e" * 64),
        stream_digest="f" * 64,
        rng_state=[3, [7, 8], None],
        budget_counts={"k1": 1, "k2": 1},
        origin_parent_digest=parent,
        attached_parent_digest=parent,
        lineage_chain=(parent,),
    )


def test_continuation_checkpoint_roundtrip_is_content_addressed() -> None:
    checkpoint = _checkpoint()
    restored = TaijiContinuationCheckpoint.from_payload(checkpoint.to_payload())

    assert restored == checkpoint
    restored.assert_parent("a" * 64)


def test_continuation_checkpoint_rejects_tamper_and_wrong_parent() -> None:
    payload = _checkpoint().to_payload()
    payload["phase_cursor"]["index"] = 2

    with pytest.raises(ValueError, match="digest mismatch"):
        TaijiContinuationCheckpoint.from_payload(payload)

    with pytest.raises(ValueError, match="crosses the expected parent"):
        _checkpoint().assert_parent("9" * 64)


def test_continuation_checkpoint_rejects_broken_lineage() -> None:
    payload = _checkpoint().to_payload()
    payload.pop("lineage_chain")
    payload["checkpoint_digest"] = "0" * 64

    with pytest.raises((KeyError, ValueError)):
        TaijiContinuationCheckpoint.from_payload(payload)
