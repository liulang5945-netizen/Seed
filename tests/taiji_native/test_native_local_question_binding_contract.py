"""Local byte matching, explicit migration and unchanged initial decoder behavior."""

from dataclasses import replace

import pytest
import torch

from scripts.training import audit_taiji_n1_local_question_binding as audit


def test_shared_windows_preserve_local_identity_and_source_boundaries():
    torch.set_num_threads(1)
    model = audit.transfer.LocalQuestionEvidenceCandidate()
    assert torch.equal(model.local_features(b"abcde")[2], model.local_features(b"xxabcdeyy")[4])
    split = torch.cat([model.local_features(b"abc"), model.local_features(b"def")])
    assert not torch.equal(split, model.local_features(b"abcdef"))
    encoded = "太极".encode()
    assert model.local_features(encoded).shape == (len(encoded), 16)
    with pytest.raises(ValueError, match="nonempty"):
        model.local_features(b"")


def test_migration_moments_zero_behavior_and_restore(tmp_path):
    _, payload, _ = audit.parent.transfer_source()
    old = audit.parent.ResidualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    model, optimizer = audit.transfer.migrate(payload)
    assert sum(p.numel() for p in model.parameters()) == 47480
    assert len(optimizer.state) == len(list(old.parameters()))
    assert all(int(v["step"]) == 6144 for v in optimizer.state.values())
    audit.prepare(old)
    audit.prepare(model)
    with torch.no_grad():
        expected, addresses = old.teacher_logits(
            old.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, audit.TARGET
        )
        actual, weights = model.teacher_logits(
            model.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, audit.TARGET
        )
        assert torch.equal(actual, expected) and torch.equal(weights, addresses)
        tape = model.evidence(audit.SESSION)
        state = model.question_state(audit.QUESTION)
        bound, alignment, salience = model.bind_question(tape, state)
        assert torch.equal(bound.hidden, state.hidden)
        assert torch.equal(bound.question_states, state.question_states)
        assert torch.allclose(alignment.sum(-1), torch.ones(len(audit.QUESTION)))
        assert torch.isclose(salience.sum(), torch.tensor(1.0))
        with pytest.raises(ValueError, match="cross sessions"):
            model.step(tape, "wrong", state, 256)
        with pytest.raises(ValueError, match="cross sessions"):
            model.step(tape, audit.SESSION, replace(state, input_tape=replace(tape)), 256)
    saved = audit.save(model, optimizer, tmp_path / "zero.pt")
    assert audit.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
    with pytest.raises(ValueError, match="explicit"):
        audit.transfer.LocalQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    broken = model.bundle(optimizer)
    del broken["state"]["local_projection.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        audit.transfer.LocalQuestionEvidenceCandidate.restore_bundle(broken)


def test_local_credit_and_cut_writer():
    _, payload, _ = audit.parent.transfer_source()
    model, optimizer = audit.transfer.migrate(payload)
    audit.prepare(model)
    logits, _ = model.teacher_logits(
        model.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, audit.TARGET
    )
    logits.square().mean().backward()
    assert model.local_fusion.weight.grad.abs().sum() > 0
    assert model.local_projection.weight.grad.abs().sum() == 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.local_fusion.weight.copy_(
            torch.linspace(-0.02, 0.02, model.local_fusion.weight.numel()).reshape_as(
                model.local_fusion.weight
            )
        )
    logits, _ = model.teacher_logits(
        model.evidence(audit.SESSION), audit.SESSION, audit.QUESTION, audit.TARGET
    )
    logits.square().mean().backward()
    for parameter in (
        model.local_projection.weight,
        model.local_salience.weight,
        model.local_log_temperature,
        model.writer.weight_ih,
        model.question.weight_ih,
    ):
        assert (
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
        )
    model.zero_grad(set_to_none=True)
    logits, _ = model.teacher_logits(
        model.evidence(audit.SESSION, cut_credit=True), audit.SESSION, audit.QUESTION, audit.TARGET
    )
    logits.square().mean().backward()
    assert model.writer.weight_ih.grad is None or model.writer.weight_ih.grad.abs().sum() == 0
    assert all(int(v["step"]) == 6144 for v in optimizer.state.values())
