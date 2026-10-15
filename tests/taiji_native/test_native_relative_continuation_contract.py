"""Continuation preserves source optimizer, RNG and all-seed heldout preconditions."""

from copy import deepcopy

import pytest
import torch

from scripts.training import run_taiji_n1_relative_continuation as continuation
from scripts.training import run_taiji_n1_relative_paired as source
from tests.taiji_native.test_native_relative_paired_contract import readiness


def test_continuation_requires_final_cumulative_updates_and_all_original_gates():
    training, rows, restores = readiness()
    assert not continuation.training_ready(training, rows, restores)
    for entry in training:
        entry["updates"] = 9216
    assert continuation.training_ready(training, rows, restores)
    training[3]["train_summary"]["free_by_kind"]["transition"] = 39
    assert not continuation.training_ready(training, rows, restores)
    assert continuation.PROTOCOL["total_updates"] == 27648
    assert continuation.PROTOCOL["cpu_limit_seconds"] == 600
    assert continuation.PROTOCOL["cpu_includes_engineering_preflight"]


def test_restore_source_preserves_adam_rng_and_exact_next_update(tmp_path):
    torch.set_num_threads(1)
    item = {
        "split": "train",
        "observations": ["entity uses abcdef."],
        "question": "material?",
        "answer": "abcdef",
    }
    model, optimizer = source.fresh(11, [item])
    source.train_step(model, optimizer, item, "first", "position")
    path = tmp_path / "source.pt"
    payload = source.save(model, optimizer, path, 11, "position", 6144)
    restored, other_optimizer = continuation.restore_input(payload, 11, "position")
    assert torch.equal(torch.get_rng_state(), payload["candidate"]["rng"])
    assert continuation.pilot.signature(restored, other_optimizer) == payload["expected_signature"]
    source.train_step(model, optimizer, item, "next", "position")
    continuation.train_step(restored, other_optimizer, item, "next", "position")
    assert all(
        torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()
    )
    new_path = tmp_path / "new.pt"
    saved = continuation.save(restored, other_optimizer, new_path, 11, "position", 6145)
    assert continuation.restore(new_path)["signature"] == saved["expected_signature"]
    with pytest.raises(ValueError, match="protocol"):
        source.restore(new_path)
    with pytest.raises(ValueError, match="mismatch"):
        continuation.restore_input(payload, 12, "position")
    bad = deepcopy(payload)
    bad["expected_signature"]["second_logits"] = "tampered"
    with pytest.raises(ValueError, match="signature"):
        continuation.restore_input(bad, 11, "position")
