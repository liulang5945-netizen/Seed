"""Artifact-free fresh training recovery, phase and exact cycle contracts."""

from copy import deepcopy

import pytest

from scripts.training import run_taiji_n1_fresh_full_history_calibration as train


def test_real_fresh_zero_independent_recovery_and_wrong_marker_rejection(tmp_path):
    train.torch.set_num_threads(1)
    model, optimizer = train.engineering.fresh(11)
    before = train.paired.fingerprint(model.bundle(optimizer))
    path = tmp_path / "zero.pt"
    saved = train.save(model, optimizer, path, 0)
    assert train.restore(path)["signature"] == saved["expected_signature"]
    restored = train.independent(
        path, saved["expected_signature"], train.paired.CpuBudget(1200, clock=lambda: 0)
    )
    assert restored["equal"] and restored["signature_clone_updates"] == 2
    assert train.paired.fingerprint(model.bundle(optimizer)) == before
    for changed in (
        dict(saved, phase="partial"),
        dict(saved, completed_updates=1),
        dict(saved, protocol=train.engineering.PROTOCOL),
        dict(saved, seed=13),
        dict(saved, update_policy=dict(train.update.POLICY, relative_limit=0.01)),
    ):
        with pytest.raises(ValueError, match="marker"):
            train.load(changed)
    with pytest.raises(ValueError, match="marker"):
        train.engineering.load(saved)


def test_synthetic_final_layout_full_recovery_and_counter_history_rejection(tmp_path):
    """Synthetic layout only: not evidence that 24576 real updates were run."""
    train.torch.set_num_threads(1)
    model, optimizer = train.engineering.fresh(11)
    item = train.engineering.SYNTHETIC
    train.update.train_step(
        model,
        optimizer,
        item,
        "layout75-0",
        "position",
        train.runner.core.renaming.second_question(item),
    )
    observations = model._observations["layout75-0"]
    for i in range(1, train.UPDATES):
        model._observations[f"layout75-{i}"] = observations
    for state in optimizer.state.values():
        state["step"].fill_(train.UPDATES)
    train.validate_state(model, optimizer, train.UPDATES)
    path = tmp_path / "final.pt"
    saved = train.save(model, optimizer, path, train.UPDATES)
    assert train.restore(path)["signature"] == saved["expected_signature"]
    original = deepcopy(next(iter(optimizer.state.values()))["step"])
    next(iter(optimizer.state.values()))["step"].sub_(1)
    with pytest.raises(ValueError, match="Adam"):
        train.validate_state(model, optimizer, train.UPDATES)
    next(iter(optimizer.state.values()))["step"].copy_(original)
    model._observations.pop("layout75-1")
    with pytest.raises(ValueError, match="session"):
        train.validate_state(model, optimizer, train.UPDATES)


def test_exact_four_cycles_and_fixed_no_holdout_scope(monkeypatch):
    calls = []

    def record(step, items, curriculum):
        calls.append((step, items, curriculum))
        return step, step

    monkeypatch.setattr(train.runner.core.renaming, "train_item", record)
    for step in (0, 6143, 6144, 12288, 18432, 24575):
        assert train.scheduled(step, "items", "curriculum") == (step % 6144, step % 6144)
    assert [c[0] for c in calls] == [0, 6143, 0, 0, 0, 6143]
    for step in (-1, 24576, 1.0, True):
        with pytest.raises(ValueError, match="schedule"):
            train.scheduled(step, "items", "curriculum")
    assert train.PROTOCOL["updates"] == 24576
    assert train.PROTOCOL["qa_sequences"] == 36864
    assert train.PROTOCOL["known_score_rows"] == 2880
    assert train.PROTOCOL["signature_clone_updates"] == 8
    assert train.PROTOCOL["synthetic_qa_sequences"] == 16
    assert train.PROTOCOL["cpu_limit_seconds"] == 1200
    assert train.PROTOCOL["confirmation_queries"] == 0
