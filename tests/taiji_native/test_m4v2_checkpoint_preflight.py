from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.training.check_taiji_m4v2_checkpoint_preflight import (
    COURSE_CURSOR,
    COURSE_ID,
    _envelope,
    _verify_envelope,
)
from taiji import content_digest

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_checkpoint_preflight_20260910.json"
)


def test_checkpoint_envelope_roundtrips_and_rejects_cursor_tampering() -> None:
    checkpoint = {
        "format": "test-checkpoint",
        "rng_state": torch.arange(8, dtype=torch.uint8),
        "payload": {"phase": "G"},
    }
    envelope = _envelope(
        parent_digest="p" * 64,
        checkpoint=checkpoint,
        owner_graph_digest="o" * 64,
        replay_event_count=2,
    )

    restored = _verify_envelope(
        envelope,
        expected_parent_digest="p" * 64,
        expected_owner_graph_digest="o" * 64,
    )
    assert content_digest(restored) == content_digest(checkpoint)
    assert envelope["training_state"]["course_id"] == COURSE_ID
    assert envelope["training_state"]["course_cursor"] == COURSE_CURSOR

    tampered = copy.deepcopy(envelope)
    tampered["training_state"]["course_cursor"]["step"] = 2
    with pytest.raises(ValueError, match="envelope digest is invalid"):
        _verify_envelope(
            tampered,
            expected_parent_digest="p" * 64,
            expected_owner_graph_digest="o" * 64,
        )


def test_checkpoint_preflight_report_allows_pilot_but_not_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["preflight_only"] is True
    assert report["research_course_executed"] is False
    assert report["preflight_only_training_step"] is True
    assert report["can_start_training_pilot"] is True
    assert report["can_promote"] is False
    assert report["temporary_checkpoint_removed"] is True
    assert all(value is True for value in report["checks"].values())
