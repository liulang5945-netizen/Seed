"""Question alignment semantics, credit and native byte execution contract."""

import pytest
import torch

from taiji.contracts import Observation
from taiji.native_evidence_fidelity import learning_loss
from taiji.native_question_aligned_evidence_candidate import QuestionAlignedEvidenceCandidate
from taiji.native_relative_evidence_candidate import RelativeEvidenceCandidate


def candidate():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = QuestionAlignedEvidenceCandidate()
    model.observe("a", Observation("text", "abc", 1, "source-1"))
    model.observe("a", Observation("text", "de", 2, "source-2"))
    return model


def test_parameter_count_and_question_conditioning():
    model = candidate()
    assert sum(p.numel() for p in model.parameters()) == 44085
    tape = model.evidence("a")
    first = model.question_state(b"ab?")
    assert first.question_states.shape == (3, 32)
    _, state, _ = model.step(tape, "a", first, 256)
    assert state.alignment.shape == (5, 3)
    assert torch.allclose(state.alignment.sum(-1), torch.ones(5))
    _, changed, _ = model.step(tape, "a", model.question_state(b"de?"), 256)
    assert not torch.equal(state.conditioned_tape.values, changed.conditioned_tape.values)
    with pytest.raises(ValueError, match="nonempty"):
        model.question_state(b"")


def test_cut_blocks_both_losses_and_terminal_uses_only_own_last_state():
    model = candidate()
    full = learning_loss(model, "a", b"abc?", b"abc")
    cut = learning_loss(model, "a", b"abc?", b"abc", arm="cut")
    terminal = learning_loss(model, "a", b"abc?", b"abc", arm="terminal")
    assert torch.equal(full.total, cut.total)
    assert torch.equal(full.observation, terminal.observation)
    full.total.backward()
    assert full.tape.states[0].grad.norm() > 0
    for layer in (model.question_key, model.question_value, model.question_fusion):
        assert layer.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    cut.total.backward()
    assert all(s.grad is None for s in cut.tape.states)
    assert all(p.grad is None for p in model.writer.parameters())
    model.zero_grad(set_to_none=True)
    terminal.qa.backward()
    assert all(terminal.tape.states[i].grad.norm() > 0 for i in (2, 4))
    # Earlier recurrent states may receive credit through each final state.
    conditioned, _ = model.condition(terminal.tape, model.question_state(b"abc?").question_states)
    assert torch.equal(conditioned.values[0], conditioned.values[2])
    assert torch.equal(conditioned.values[3], conditioned.values[4])
    assert not torch.equal(conditioned.values[2], conditioned.values[4])


def test_shared_teacher_free_and_scope(monkeypatch):
    model = candidate()
    tape = model.evidence("a")
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()
        model.readout.bias[ord("a")] = 1
    original = model.step
    trajectories = []

    def counted(*args):
        result = original(*args)
        trajectories.append(
            (result[1].address.detach().clone(), result[1].alignment.detach().clone())
        )
        return result

    monkeypatch.setattr(model, "step", counted)
    before = model.sources("a")
    model.teacher_logits(tape, "a", b"?", b"aaa")
    teacher = trajectories[:3]
    trajectories.clear()
    trace = model.generate_trace(tape, "a", b"?", 3)
    assert trace.raw == b"aaa"
    assert all(
        torch.equal(a, b) and torch.equal(c, d)
        for (a, c), (b, d) in zip(teacher, trajectories, strict=True)
    )
    assert model.sources("a") == before
    _, state, _ = original(tape, "a", model.question_state(b"?"), 256)
    with pytest.raises(ValueError, match="cannot cross"):
        original(model.evidence("a"), "a", state, ord("a"))
    with pytest.raises(ValueError, match="cannot cross"):
        original(tape, "b", state, ord("a"))
    assert model.question_state(b"?").conditioned_tape is None


def test_format_and_restore():
    model = candidate()
    payload = model.bundle()
    restored = QuestionAlignedEvidenceCandidate.restore_bundle(payload)
    assert restored.answer("a", b"?", 8) == model.answer("a", b"?", 8)
    with pytest.raises(ValueError):
        RelativeEvidenceCandidate.restore_bundle(payload)
    payload.pop("question_mechanism")
    with pytest.raises(ValueError, match="question alignment"):
        QuestionAlignedEvidenceCandidate.restore_bundle(payload)
