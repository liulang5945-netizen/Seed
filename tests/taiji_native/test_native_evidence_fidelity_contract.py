"""Auxiliary learning preserves provenance, inference and effective credit cuts."""

from dataclasses import replace

import pytest
import torch

from scripts.training import run_taiji_n1_native_evidence_paired as paired
from taiji.contracts import Observation
from taiji.native_evidence_candidate import NativeEvidenceCandidate
from taiji.native_evidence_fidelity import learning_loss, observation_logits


def candidate():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = NativeEvidenceCandidate()
    model.observe("a", Observation("text", "alpha uses abcdef.", 1, "first"))
    model.observe("a", Observation("text", "alpha now uses ghijkl.", 2, "second"))
    model.observe("b", Observation("text", "other uses mnopqr.", 1, "first"))
    return model


def test_combined_credit_cut_is_forward_equal_and_blocks_all_writer_paths():
    model = candidate()
    full = learning_loss(model, "a", b"material of alpha?", b"ghijkl")
    cut = learning_loss(model, "a", b"material of alpha?", b"ghijkl", arm="cut")
    assert torch.equal(full.total, cut.total)
    assert torch.equal(full.observation, cut.observation)
    full.total.backward()
    assert full.tape.states[0].grad.norm() > 0
    assert model.writer.weight_ih.grad.norm() > 0
    model.zero_grad(set_to_none=True)
    cut.total.backward()
    assert all(state.grad is None for state in cut.tape.states)
    assert all(parameter.grad is None for parameter in model.writer.parameters())
    assert model.value.weight.grad.norm() > 0
    assert model.readout.weight.grad.norm() > 0


def test_auxiliary_cuts_itself_and_terminal_gets_same_observation_target():
    model = candidate()
    full = model.evidence("a")
    terminal = model.evidence("a", terminal_per_source=True)
    logits, labels = observation_logits(model, full, "a")
    terminal_logits, terminal_labels = observation_logits(model, terminal, "a")
    assert torch.equal(logits, terminal_logits)
    assert torch.equal(labels, terminal_labels)
    assert labels.tolist() == list(b"alpha uses abcdef.alpha now uses ghijkl.")
    cut_logits, cut_labels = observation_logits(model, full, "a", cut_credit=True)
    torch.nn.functional.cross_entropy(cut_logits, cut_labels).backward()
    assert all(state.grad is None for state in full.states)
    assert all(parameter.grad is None for parameter in model.writer.parameters())


def test_no_parameter_or_inference_change_and_old_bundle_restores():
    model = candidate()
    before = paired.fingerprint(model.state_dict())
    count = sum(p.numel() for p in model.parameters())
    output = model.answer("a", b"material of alpha?", max_length=8)
    loss = learning_loss(model, "a", b"material of alpha?", b"ghijkl")
    loss.total.backward()
    assert paired.fingerprint(model.state_dict()) == before
    assert sum(p.numel() for p in model.parameters()) == count
    assert model.answer("a", b"material of alpha?", max_length=8) == output
    recovered = NativeEvidenceCandidate.restore_bundle(model.bundle())
    assert recovered.answer("a", b"material of alpha?", max_length=8) == output


def test_fidelity_rejects_cross_session_or_misaligned_tape():
    model = candidate()
    tape = model.evidence("a")
    with pytest.raises(ValueError, match="session"):
        observation_logits(model, tape, "b")
    with pytest.raises(ValueError, match="align"):
        observation_logits(model, replace(tape, positions=tape.positions[::-1]), "a")
    with pytest.raises(ValueError, match="unknown"):
        learning_loss(model, "a", b"?", b"abcdef", arm="invalid")
