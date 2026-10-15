"""Native interface ownership does not establish native task capability."""

from copy import deepcopy

import pytest
import torch

from taiji.contracts import Observation
from taiji.native_evidence_candidate import NativeEvidenceCandidate


def candidate():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = NativeEvidenceCandidate()
    model.observe("a", Observation("text", "copper", 1, "source"))
    model.observe("b", Observation("text", "silver", 1, "source"))
    return model


def test_observation_isolation_and_read_only_answer():
    model = candidate()
    before = deepcopy(model.bundle()["sessions"])
    a = model.evidence("a")
    b = model.evidence("b")
    assert not torch.equal(a.values, b.values)
    model.answer("a", b"material?", max_length=8).decode("utf-8")
    assert model.bundle()["sessions"] == before
    with pytest.raises(ValueError):
        model.answer("missing", b"?")
    for provenance in ("query", "generated", "imagined"):
        with pytest.raises(ValueError):
            model.observe("a", Observation("text", "fake", 2, "new", provenance))
    for observation in (
        Observation("text", "late", 0, "new"),
        Observation("text", "duplicate", 2, "source"),
        Observation("audio", b"x", 2, "new"),
    ):
        with pytest.raises(ValueError):
            model.observe("a", observation)


def test_terminal_control_preserves_each_source_and_credit_cut():
    model = candidate()
    model.observe("a", Observation("text", "zinc", 2, "second"))
    full = model.evidence("a")
    terminal = model.evidence("a", terminal_per_source=True)
    assert full.positions == terminal.positions
    # Batched GEMM and vector GEMV differ by rounding (observed max 2.98e-8).
    torch.testing.assert_close(terminal.values[:6], model.value(full.states[5]).expand(6, -1))
    torch.testing.assert_close(terminal.values[6:], model.value(full.states[-1]).expand(4, -1))
    assert torch.equal(terminal.values[:6], terminal.values[:1].expand(6, -1))
    assert torch.equal(terminal.values[6:], terminal.values[6:7].expand(4, -1))
    assert not torch.equal(terminal.values[0], terminal.values[-1])
    cut = model.evidence("a", cut_credit=True)
    logits, _ = model.teacher_logits(full, "a", b"?", b"zinc")
    cut_logits, _ = model.teacher_logits(cut, "a", b"?", b"zinc")
    assert torch.equal(logits, cut_logits)
    logits[-1, 256].backward()
    assert full.states[0].grad.norm() > 0
    model.zero_grad(set_to_none=True)
    cut_logits[-1, 256].backward()
    assert all(state.grad is None for state in cut.states)


def test_self_contained_multisession_bundle(tmp_path):
    model = candidate()
    path = tmp_path / "candidate.pt"
    torch.save(model.bundle(), path)
    loaded = torch.load(path, weights_only=True)
    recovered = model.restore_bundle(loaded)
    assert recovered.bundle()["sessions"] == loaded["sessions"]
    for session in ("a", "b"):
        original, _ = model.teacher_logits(model.evidence(session), session, b"?", b"metal")
        restored, _ = recovered.teacher_logits(recovered.evidence(session), session, b"?", b"metal")
        assert torch.equal(original, restored)
    loaded["sessions"]["a"].append(deepcopy(loaded["sessions"]["a"][0]))
    with pytest.raises(ValueError):
        model.restore_bundle(loaded)
