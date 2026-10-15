"""Frozen calibration dose, input boundary and full zero restoration."""

import pytest
import torch

from scripts.training import run_taiji_n1_residual_question as runner


def test_recipe_and_restore(tmp_path):
    torch.set_num_threads(1)
    assert runner.PROTOCOL["updates"] == 9216
    assert runner.PROTOCOL["milestones"] == [3072, 6144, 9216]
    assert runner.PROTOCOL["dev_queries"] == runner.PROTOCOL["holdout_queries"] == 0
    train = runner.checked_train()
    model, optimizer = runner.fresh(train)
    assert runner.baseline_initial_match(model)["passed"]
    before = runner.paired.fingerprint(model.bundle(optimizer))
    saved = runner.save(model, optimizer, tmp_path / "zero.pt", 0)
    assert runner.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
    assert runner.paired.fingerprint(model.bundle(optimizer)) == before
    assert "question_alignment" in saved["expected_signature"]
    assert "conditioned_values" in saved["expected_signature"]


def test_refuse_nontrain_without_observing():
    model, optimizer = runner.fresh(runner.checked_train())
    before = runner.paired.fingerprint(model.bundle(optimizer))
    for split in ("dev", "holdout"):
        with pytest.raises(ValueError, match="train"):
            runner.train_step(model, optimizer, {"split": split}, "forbidden")
        with pytest.raises(ValueError, match="dev or holdout"):
            runner.measure(model, {"split": split}, "position", "forbidden")
    assert runner.paired.fingerprint(model.bundle(optimizer)) == before
