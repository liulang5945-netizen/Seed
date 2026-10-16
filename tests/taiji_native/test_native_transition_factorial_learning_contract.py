"""Synthetic-only quartet weighting, same-arm credit and full checkpoint recovery."""

from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_transition_factorial_learning as audit

learning = audit.learning
torch = audit.torch


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("arm", ["position", "terminal", "cut"])
def test_quartet_one_proposal_four_histories_and_mean_loss(monkeypatch, arm):
    model, optimizer = audit.engineering.fresh(13)
    observed = []
    original = learning.train.runner.core.paired_question_loss
    steps = []
    original_step = optimizer.step

    def loss(*args, **kwargs):
        value = original(*args, **kwargs)
        observed.append(
            (
                float(value.total.detach()),
                float(value.qa.detach()),
                float(value.observation.detach()),
            )
        )
        return value

    def step():
        steps.append(1)
        return original_step()

    monkeypatch.setattr(learning.train.runner.core, "paired_question_loss", loss)
    optimizer.step = step
    result, groups = learning.train_step(model, optimizer, audit.SYNTHETIC, "test", arm, audit=True)
    assert len(steps) == 1 and len(observed) == 4
    assert result["total"] == pytest.approx(sum(x[0] for x in observed) / 4)
    assert result["qa"] == pytest.approx(sum(x[1] for x in observed) / 4)
    assert result["observation"] == pytest.approx(sum(x[2] for x in observed) / 4)
    assert result["qa_sequences"] == len(model._observations) == 4
    assert len(optimizer.state) == (37 if arm == "cut" else 41)
    assert all(int(s["step"]) == 1 for s in optimizer.state.values())
    assert (groups["writer"]["gradient_l2"] == 0) == (arm == "cut")
    for group in groups.values():
        assert group["actual_l2"] <= group["cap"] * 1.0001 + 1e-6
    with pytest.raises(ValueError, match="unique"):
        learning.train_step(model, optimizer, audit.SYNTHETIC, "test", arm)


def test_nontransition_is_bitwise_old_objective():
    item = audit.engineering.SYNTHETIC
    second = audit.train.runner.core.renaming.second_question(item)
    model, optimizer = audit.engineering.fresh(13)
    bundle = deepcopy(model.bundle(optimizer))
    loss, measurement = learning.train_step(model, optimizer, item, "same", "position", second)
    expected = audit.paired.fingerprint(model.bundle(optimizer))
    clone = audit.engineering.LocalRelationEvidenceCandidate.restore_bundle(bundle)
    opt = torch.optim.Adam(clone.parameters(), lr=0.001)
    opt.load_state_dict(bundle["optimizer"])
    old_loss, old_measurement = audit.train.update.train_step(
        clone, opt, item, "same", "position", second
    )
    assert loss == old_loss and measurement == old_measurement
    assert audit.paired.fingerprint(clone.bundle(opt)) == expected


def test_backward_matches_equal_weight_sum_gradient(monkeypatch):
    model, optimizer = audit.engineering.fresh(13)
    bundle = deepcopy(model.bundle(optimizer))
    captured = {}

    def no_step(model, optimizer, original_step, **kwargs):
        captured.update(
            {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        )
        return None, {}

    monkeypatch.setattr(learning.train.update.bounded, "bounded_step", no_step)
    learning.train_step(model, optimizer, audit.SYNTHETIC, "weighted", "position")
    clone = audit.engineering.LocalRelationEvidenceCandidate.restore_bundle(bundle)
    losses = []
    for entry in learning.factorial.variants(audit.SYNTHETIC):
        item = entry["item"]
        session = entry["condition"]
        audit.paired.observe(clone, audit.paired.project(item), session)
        losses.append(
            audit.train.runner.core.paired_question_loss(
                clone, session, item["question"].encode(), item["answer"].encode()
            ).total
        )
    torch.stack(losses).mean().backward()
    for name, parameter in clone.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(captured[name], parameter.grad, rtol=1e-5, atol=1e-7)


def test_full_zero_roundtrip_signature_and_parent_neutrality(tmp_path):
    model, optimizer = audit.engineering.fresh(13)
    before = audit.paired.fingerprint(model.bundle(optimizer))
    expected = audit.signature(model, optimizer)
    assert audit.paired.fingerprint(model.bundle(optimizer)) == before
    assert len(expected["trajectory"]) == 2
    path = tmp_path / "zero.pt"
    with path.open("xb") as stream:
        torch.save(
            dict(
                schema="n1-transition-factorial-zero-v1",
                protocol=audit.PROTOCOL,
                completed_updates=0,
                candidate=model.bundle(optimizer),
                expected_signature=expected,
            ),
            stream,
        )
    restored = audit.restore(path)
    assert restored["signature"] == expected
    assert restored["signature_clone_updates"] == 2
    with pytest.raises(ValueError):
        audit.load(dict(schema="other"))


@pytest.mark.parametrize("change", [dict(split="holdout"), dict(answer="wrong")])
def test_bad_training_item_fails_before_history_or_adam(change):
    model, optimizer = audit.engineering.fresh(13)
    with pytest.raises(ValueError):
        learning.train_step(model, optimizer, dict(audit.SYNTHETIC, **change), "bad", "position")
    assert not optimizer.state and not model._observations


def test_attempt_prevents_retry_before_reading_evidence(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(audit, "ATTEMPT", attempt)
    with pytest.raises(FileExistsError):
        audit.audit()
