"""Internal read hops change computation, not emission count or provenance."""

from copy import deepcopy

import pytest
import torch

from scripts.training import run_taiji_n1_native_evidence_paired as paired
from taiji.contracts import Observation
from taiji.native_evidence_candidate import NativeEvidenceCandidate
from taiji.native_evidence_fidelity import learning_loss
from taiji.native_iterative_evidence_candidate import IterativeEvidenceCandidate


def candidate():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = IterativeEvidenceCandidate()
    model.observe("a", Observation("text", "entity uses abcdef.", 1, "source-1"))
    return model


def test_two_hops_match_manual_read_and_have_no_extra_parameters():
    model = candidate()
    tape = model.evidence("a")
    hidden = model.question_state(b"material?")
    _, intermediate, _ = NativeEvidenceCandidate.step(model, tape, "a", hidden, 256)
    expected = NativeEvidenceCandidate.step(model, tape, "a", intermediate, 256)
    actual = model.step(tape, "a", hidden, 256)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))
    assert sum(p.numel() for p in model.parameters()) == 40337


def test_teacher_and_free_generation_only_emit_after_two_hops(monkeypatch):
    model = candidate()
    before = deepcopy(model.bundle()["sessions"])
    original = NativeEvidenceCandidate.step
    calls = []

    def counted(self, tape, session, hidden, previous):
        calls.append(previous)
        return original(self, tape, session, hidden, previous)

    monkeypatch.setattr(NativeEvidenceCandidate, "step", counted)
    logits, _ = model.teacher_logits(model.evidence("a"), "a", b"?", b"ab")
    assert len(logits) == 3 and calls == [256, 256, 97, 97, 98, 98]
    calls.clear()
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()
        model.readout.bias[256] = 100
    trace = model.generate_trace(model.evidence("a"), "a", b"?", 64)
    assert trace.raw == b"" and trace.stop_reason == "boundary"
    assert trace.generated_steps == 1 and calls == [256, 256]
    assert model.bundle()["sessions"] == before


def test_new_bundle_cannot_silently_change_old_runtime_semantics():
    model = candidate()
    saved = model.bundle()
    recovered = IterativeEvidenceCandidate.restore_bundle(saved)
    assert recovered.answer("a", b"?", 8) == model.answer("a", b"?", 8)
    with pytest.raises(ValueError):
        NativeEvidenceCandidate.restore_bundle(saved)
    saved["read_hops"] = 1
    with pytest.raises(ValueError, match="two read hops"):
        IterativeEvidenceCandidate.restore_bundle(saved)


def test_two_hop_combined_credit_cut_remains_forward_identical():
    model = candidate()
    full = learning_loss(model, "a", b"?", b"abcdef")
    cut = learning_loss(model, "a", b"?", b"abcdef", arm="cut")
    assert torch.equal(full.total, cut.total)
    full.total.backward()
    assert full.tape.states[0].grad.norm() > 0
    model.zero_grad(set_to_none=True)
    cut.total.backward()
    assert all(state.grad is None for state in cut.tape.states)
    assert all(parameter.grad is None for parameter in model.writer.parameters())
    assert paired.fingerprint(model.state_dict()) == paired.fingerprint(model.bundle()["state"])
