"""Learned address transport preserves source scope, credit and runtime identity."""

import pytest
import torch

from scripts.training import run_taiji_n1_native_evidence_paired as paired
from taiji.contracts import Observation
from taiji.native_evidence_fidelity import learning_loss
from taiji.native_iterative_evidence_candidate import IterativeEvidenceCandidate
from taiji.native_relative_evidence_candidate import RelativeEvidenceCandidate


def candidate():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = RelativeEvidenceCandidate()
    model.observe("a", Observation("text", "abc", 1, "source-1"))
    model.observe("a", Observation("text", "de", 2, "source-2"))
    return model


def test_old_parameters_first_step_and_zero_gate_match_two_hop():
    model = candidate()
    torch.manual_seed(11)
    old = IterativeEvidenceCandidate()
    old.observe("a", Observation("text", "abc", 1, "source-1"))
    old.observe("a", Observation("text", "de", 2, "source-2"))
    assert sum(p.numel() for p in model.parameters()) == 40469
    assert all(torch.equal(model.state_dict()[k], v) for k, v in old.state_dict().items())
    tape, old_tape = model.evidence("a"), old.evidence("a")
    state, hidden = model.question_state(b"?"), old.question_state(b"?")
    # First step ignores the new head even if its gate is forced open.
    with torch.no_grad():
        model.location.bias[3] = 1000
        actual, state, _ = model.step(tape, "a", state, 256)
        expected, hidden, _ = old.step(old_tape, "a", hidden, 256)
        assert torch.equal(actual, expected)
        model.location.bias[3] = -1000
        for previous in b"abc":
            actual, state, _ = model.step(tape, "a", state, previous)
            expected, hidden, _ = old.step(old_tape, "a", hidden, previous)
            assert torch.equal(actual, expected)


def test_transport_clamps_at_each_source_and_preserves_mass_and_gradient():
    model = candidate()
    tape = model.evidence("a")
    address = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.25], requires_grad=True)
    left = model.transport(tape, address, torch.tensor([1.0, 0, 0]))
    right = model.transport(tape, address, torch.tensor([0.0, 0, 1]))
    assert torch.allclose(left, torch.tensor([0.3, 0.3, 0, 0.4, 0]))
    assert torch.allclose(right, torch.tensor([0.0, 0.1, 0.5, 0, 0.4]))
    for shifted in (left, right):
        assert torch.allclose(shifted.sum(), address.sum())
        assert torch.allclose(shifted[:3].sum(), address[:3].sum())
    right.square().sum().backward()
    assert address.grad is not None and address.grad.abs().sum() > 0


def test_address_state_is_answer_local_and_tape_scoped():
    model = candidate()
    tape = model.evidence("a")
    initial = model.question_state(b"?")
    first, state, attention = model.step(tape, "a", initial, 256)
    assert state.address is attention and initial.address is None
    again, reset, _ = model.step(tape, "a", model.question_state(b"?"), 256)
    assert torch.equal(first, again) and reset.address is not None
    with pytest.raises(ValueError, match="cannot cross"):
        model.step(model.evidence("a"), "a", state, ord("a"))
    with pytest.raises(ValueError, match="cannot cross"):
        model.step(tape, "wrong", state, ord("a"))


def test_teacher_and_free_share_address_trajectory(monkeypatch):
    model = candidate()
    tape = model.evidence("a")
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()
        model.readout.bias[ord("a")] = 1
    original = model.step
    addresses = []

    def counted(*args):
        output = original(*args)
        addresses.append(output[1].address.detach().clone())
        return output

    monkeypatch.setattr(model, "step", counted)
    model.teacher_logits(tape, "a", b"?", b"aaa")
    teacher = addresses[:3]
    addresses.clear()
    trace = model.generate_trace(tape, "a", b"?", 3)
    assert trace.raw == b"aaa" and trace.generated_steps == 3
    assert all(torch.equal(a, b) for a, b in zip(teacher, addresses, strict=True))


def test_combined_credit_cut_and_location_head_gradient():
    model = candidate()
    before = paired.fingerprint(model.state_dict())
    full = learning_loss(model, "a", b"?", b"abc")
    cut = learning_loss(model, "a", b"?", b"abc", arm="cut")
    terminal = learning_loss(model, "a", b"?", b"abc", arm="terminal")
    assert torch.equal(full.total, cut.total)
    assert torch.equal(full.observation, terminal.observation)
    full.total.backward()
    assert full.tape.states[0].grad.norm() > 0
    assert model.location.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    cut.total.backward()
    assert all(s.grad is None for s in cut.tape.states)
    assert all(p.grad is None for p in model.writer.parameters())
    assert before == paired.fingerprint(model.state_dict())


def test_relative_bundle_owns_format_and_rejects_old_semantics():
    model = candidate()
    saved = model.bundle()
    recovered = RelativeEvidenceCandidate.restore_bundle(saved)
    assert recovered.answer("a", b"?", 8) == model.answer("a", b"?", 8)
    with pytest.raises(ValueError):
        IterativeEvidenceCandidate.restore_bundle(saved)
    saved.pop("address_mechanism")
    with pytest.raises(ValueError, match="address mechanism"):
        RelativeEvidenceCandidate.restore_bundle(saved)
