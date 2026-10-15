"""Actual-step boundaries retain unmodified moments, native credit and recovery."""

from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_group_relative_adam as audit
from tests.taiji_native._native_relation_fixtures import relation_zero

runner = audit.runner
bounded = audit.bounded


@pytest.fixture(scope="module")
def source():
    runner.torch.set_num_threads(1)
    return relation_zero()


def restored(source):
    candidate = deepcopy(source)
    model = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(candidate)
    optimizer = runner.torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(candidate["optimizer"])
    return model, optimizer


def test_inactive_bound_preserves_two_complete_updates_bitwise(source):
    item = audit.SYNTHETIC
    second = runner.core.renaming.second_question(item)
    reference, refopt = restored(source)
    losses = [
        runner.train_step(reference, refopt, item, f"inactive67-{i}", "position", second)
        for i in range(2)
    ]
    expected = runner.paired.fingerprint(reference.bundle(refopt))
    model, optimizer = restored(source)
    for i in range(2):
        loss, stats = bounded.train_step(
            model, optimizer, item, f"inactive67-{i}", second, relative_limit=1e9, audit=True
        )
        assert loss == losses[i]
        assert not any(s["clipped"] for s in stats.values())
    assert runner.paired.fingerprint(model.bundle(optimizer)) == expected


def test_active_bound_preserves_gradients_and_adam_moments(source):
    item = audit.SYNTHETIC
    second = runner.core.renaming.second_question(item)
    raw, rawopt = restored(source)
    expected_loss = runner.train_step(raw, rawopt, item, "active67", "position", second)
    moments = runner.paired.fingerprint(rawopt.state_dict())
    gradients = runner.paired.fingerprint({n: p.grad for n, p in raw.named_parameters()})
    model, optimizer = restored(source)
    old_head = model.relation_projection.weight.detach().clone()
    loss, stats = bounded.train_step(model, optimizer, item, "active67", second, audit=True)
    assert loss == expected_loss
    assert runner.paired.fingerprint(optimizer.state_dict()) == moments
    assert runner.paired.fingerprint({n: p.grad for n, p in model.named_parameters()}) == gradients
    assert (
        len({n for values in bounded.grouped_parameters(model).values() for n, _ in values}) == 41
    )
    assert any(s["clipped"] for s in stats.values())
    assert all(s["actual_l2"] <= s["cap"] * 1.0001 + 1e-6 for s in stats.values())
    assert stats["writer"]["gradient_l2"] > 0
    assert not runner.torch.equal(old_head, model.relation_projection.weight)
    with pytest.raises(ValueError, match="positive finite"):
        bounded.bounded_step(model, optimizer, optimizer.step, relative_limit=0)


def test_rule_marker_and_full_next_step_recovery(source, tmp_path):
    model, optimizer = restored(source)
    rng = runner.torch.get_rng_state().clone()
    zero = audit.save(model, optimizer, tmp_path / "zero.pt", 0)
    assert runner.torch.equal(rng, runner.torch.get_rng_state())
    assert audit.restore(tmp_path / "zero.pt")["signature"] == zero["expected_signature"]
    for step in range(2):
        bounded.train_step(
            model,
            optimizer,
            audit.SYNTHETIC,
            f"recover67-{step}",
            runner.core.renaming.second_question(audit.SYNTHETIC),
            audit=True,
        )
    final = audit.save(model, optimizer, tmp_path / "final.pt", 2)
    assert audit.restore(tmp_path / "final.pt")["signature"] == final["expected_signature"]
    for marker in (None, dict(bounded.POLICY, relative_limit=0.01)):
        changed = dict(final, update_policy=marker)
        with pytest.raises(ValueError, match="marker"):
            audit.load(changed)
