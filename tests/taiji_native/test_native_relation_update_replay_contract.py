"""The update observer must not alter gradients, Adam, RNG, history or recovery."""

import math
from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_relation_update_replay as audit


def test_update_observer_is_bitwise_neutral_and_recoverable(tmp_path):
    runner = audit.runner
    runner.torch.set_num_threads(1)
    ready, _ = runner.readiness()
    _, source, _ = runner.transfer.source_payload()
    initial, initial_optimizer = runner.fresh(source, 13, ready)
    payload = deepcopy(initial.bundle(initial_optimizer))
    item = dict(
        id="synthetic66",
        kind="lookup",
        split="train",
        observations=["alpha uses quartz.", "bravo uses copper."],
        question="material of alpha?",
        answer="quartz",
    )
    second = runner.core.renaming.second_question(item)

    def restored():
        model = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(payload)
        optimizer = runner.torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(deepcopy(payload["optimizer"]))
        return model, optimizer

    reference, reference_optimizer = restored()
    expected = [
        runner.train_step(
            reference, reference_optimizer, item, f"synthetic66-{i}", "position", second
        )
        for i in range(2)
    ]
    reference_bundle = runner.paired.fingerprint(reference.bundle(reference_optimizer))
    measured, measured_optimizer = restored()
    zero = runner.save(measured, measured_optimizer, tmp_path / "zero.pt", 13, "position", 0)
    assert runner.restore(tmp_path / "zero.pt")["signature"] == zero["expected_signature"]
    for i in range(2):
        loss, stats = audit.instrumented_step(
            measured, measured_optimizer, item, f"synthetic66-{i}", second
        )
        assert loss == expected[i]
        assert set(stats["groups"]) == {"reader", "local", "context", "relation", "writer"}
        assert stats["groups"]["writer"]["gradient_l2"] > 0
        assert stats["groups"]["relation"]["update_l2"] > 0
        assert all(
            math.isfinite(v) and v >= 0
            for group in stats["groups"].values()
            for v in group.values()
        )
    assert runner.paired.fingerprint(measured.bundle(measured_optimizer)) == reference_bundle
    final = runner.save(measured, measured_optimizer, tmp_path / "final.pt", 13, "position", 2)
    assert runner.restore(tmp_path / "final.pt")["signature"] == final["expected_signature"]
    original_step = measured_optimizer.step
    with pytest.raises(ValueError, match="train"):
        audit.instrumented_step(
            measured, measured_optimizer, dict(item, split="holdout"), "forbidden", second
        )
    assert measured_optimizer.step == original_step


def test_exact_loss_tolerance_and_fixed_scope():
    expected = dict(total=1.0, qa=0.2, observation=0.1, qa_sequences=2)
    assert audit.loss_matches(expected, expected)
    assert not audit.loss_matches(dict(expected, qa=0.21), expected)
    assert not audit.loss_matches(dict(expected, qa=float("nan")), expected)
    assert not audit.loss_matches(dict(expected, qa_sequences=1), expected)
    small = dict(expected, qa=1e-9)
    assert not audit.loss_matches(dict(small, qa=2e-9), small)
    assert audit.PROTOCOL["updates"] == 6144
    assert audit.PROTOCOL["cpu_limit_seconds"] == 300
    assert audit.PROTOCOL["teacher_free_groups"] == 1728
    assert audit.PROTOCOL["confirmation_queries"] == 0
