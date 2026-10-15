"""Residual initialization equivalence and available learned alignment credit."""

import pytest
import torch

from taiji.contracts import Observation
from taiji.native_evidence_fidelity import learning_loss
from taiji.native_question_aligned_evidence_candidate import QuestionAlignedEvidenceCandidate
from taiji.native_relative_evidence_candidate import RelativeEvidenceCandidate
from taiji.native_residual_question_evidence_candidate import ResidualQuestionEvidenceCandidate


def candidate(cls=ResidualQuestionEvidenceCandidate):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = cls()
    model.observe("probe", Observation("text", "abc", 1, "source-1"))
    model.observe("probe", Observation("text", "de", 2, "source-2"))
    return model


def test_exact_old_shared_parameters_and_multistep_native_content():
    model, old = candidate(), candidate(RelativeEvidenceCandidate)
    assert sum(p.numel() for p in model.parameters()) == 44085
    assert all(torch.equal(model.state_dict()[k], v) for k, v in old.state_dict().items())
    for arm in ("position", "terminal", "cut"):
        kwargs = dict(cut_credit=arm == "cut", terminal_per_source=arm == "terminal")
        tape, original = model.evidence("probe", **kwargs), old.evidence("probe", **kwargs)
        state, hidden = model.question_state(b"which?"), old.question_state(b"which?")
        for previous in (256, *b"abc"):
            logits, state, attention = model.step(tape, "probe", state, previous)
            expected, hidden, weights = old.step(original, "probe", hidden, previous)
            assert torch.equal(logits, expected)
            assert torch.equal(attention, weights)
        assert torch.equal(
            model.teacher_logits(tape, "probe", b"which?", b"abc")[0],
            old.teacher_logits(original, "probe", b"which?", b"abc")[0],
        )


def test_residual_initial_gradient_and_nonzero_alignment_credit():
    model = candidate()
    loss = learning_loss(model, "probe", b"which?", b"abc")
    loss.total.backward()
    assert model.question_fusion.weight.grad.abs().sum() > 0
    assert model.question_key.weight.grad.abs().sum() == 0
    assert model.question_value.weight.grad.abs().sum() == 0
    model.zero_grad(set_to_none=True)
    # Synthetic path test only; no optimizer update or curriculum model.
    with torch.no_grad():
        model.question_fusion.weight.fill_(0.01)
    full = learning_loss(model, "probe", b"which?", b"abc")
    full.total.backward()
    assert model.question_key.weight.grad.abs().sum() > 0
    assert model.question_value.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    cut = learning_loss(model, "probe", b"which?", b"abc", arm="cut")
    assert torch.equal(full.total, cut.total)
    cut.total.backward()
    assert all(p.grad is None for p in model.writer.parameters())
    assert all(s.grad is None for s in cut.tape.states)


def test_own_format_and_scope():
    model = candidate()
    tape = model.evidence("probe")
    _, state, _ = model.step(tape, "probe", model.question_state(b"?"), 256)
    with pytest.raises(ValueError, match="cannot cross"):
        model.step(model.evidence("probe"), "probe", state, ord("a"))
    restored = ResidualQuestionEvidenceCandidate.restore_bundle(model.bundle())
    assert restored.answer("probe", b"?", 8) == model.answer("probe", b"?", 8)
    with pytest.raises(ValueError):
        QuestionAlignedEvidenceCandidate.restore_bundle(model.bundle())


def test_teacher_free_same_residual_trajectory(monkeypatch):
    model = candidate()
    tape = model.evidence("probe")
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()
        model.readout.bias[ord("a")] = 1
    step = model.step
    seen = []

    def record(*args):
        output = step(*args)
        seen.append(output[1].address.detach().clone())
        return output

    monkeypatch.setattr(model, "step", record)
    model.teacher_logits(tape, "probe", b"?", b"aaa")
    teacher = seen[:3]
    seen.clear()
    assert model.generate_trace(tape, "probe", b"?", 3).raw == b"aaa"
    assert all(torch.equal(a, b) for a, b in zip(teacher, seen, strict=True))
