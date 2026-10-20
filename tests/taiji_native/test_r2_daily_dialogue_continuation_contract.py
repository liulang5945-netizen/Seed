from __future__ import annotations

from scripts.training.run_taiji_r2_daily_dialogue_continuation import (
    _generation_progress_outputs,
)


def test_generation_progress_maps_every_endpoint_to_its_row_field() -> None:
    row = {
        "parent_with_history": {
            "generated_text": "parent history",
            "generated_bytes_length": 14,
            "raw_generated_sha256": "a" * 64,
            "response_boundary_present": True,
            "pii_pattern_redacted": False,
        },
        "parent_single_turn": {
            "generated_text": "parent single",
            "generated_bytes_length": 13,
            "raw_generated_sha256": "b" * 64,
            "response_boundary_present": True,
            "pii_pattern_redacted": False,
        },
        "old_dialogue_context": {
            "generated_text": "old history",
            "generated_bytes_length": 11,
            "raw_generated_sha256": "c" * 64,
            "response_boundary_present": True,
            "pii_pattern_redacted": False,
        },
        "old_single_turn": {
            "generated_text": "old single",
            "generated_bytes_length": 10,
            "raw_generated_sha256": "d" * 64,
            "response_boundary_present": True,
            "pii_pattern_redacted": False,
        },
        "continued_dialogue_context": {
            "generated_text": "continued history",
            "generated_bytes_length": 17,
            "raw_generated_sha256": "e" * 64,
            "response_boundary_present": True,
            "pii_pattern_redacted": False,
        },
        "continued_single_turn": {
            "generated_text": "continued single",
            "generated_bytes_length": 16,
            "raw_generated_sha256": "f" * 64,
            "response_boundary_present": True,
            "pii_pattern_redacted": False,
        },
    }

    progress = _generation_progress_outputs(row)

    assert set(progress) == {
        "parent_dialogue_context",
        "parent_single_turn",
        "old_dialogue_context",
        "old_single_turn",
        "continued_dialogue_context",
        "continued_single_turn",
    }
    assert progress["parent_dialogue_context"]["generated_text"] == "parent history"
    assert progress["continued_single_turn"]["generated_text"] == "continued single"
