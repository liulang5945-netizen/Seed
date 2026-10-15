"""Engineering contracts only; no task fitting or capability evidence."""

import pytest
import torch

from taiji.position_evidence_reference import ObservedSource, PositionEvidenceReference


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = PositionEvidenceReference()
    sources = (ObservedSource("session", "observation", b"agent uses copper"),)
    return model, sources


def test_position_credit_and_forward_identical_negative_control():
    model, sources = fixture()
    full = model.write(sources, "session")
    cut = model.write(sources, "session", cut_credit=True)
    logits, attention = model.teacher_logits(full, "session", b"material?", b"copper")
    cut_logits, _ = model.teacher_logits(cut, "session", b"material?", b"copper")
    assert torch.equal(logits, cut_logits)
    assert not torch.equal(full.keys[0], full.keys[-1])
    assert not torch.equal(attention[0], attention[-1])
    torch.nn.functional.cross_entropy(logits[-1:], torch.tensor([256])).backward()
    for state in (full.states[0], full.states[-1]):
        assert state.grad is not None
        assert torch.isfinite(state.grad).all() and state.grad.norm() > 0
    model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(cut_logits[-1:], torch.tensor([256])).backward()
    assert all(state.grad is None for state in cut.states)
    assert full.positions == tuple(("observation", i) for i in range(len(sources[0].content)))


def test_source_and_session_boundaries():
    model, sources = fixture()
    for provenance in ("generated", "query"):
        with pytest.raises(ValueError):
            ObservedSource("session", "x", b"x", provenance)
    with pytest.raises(ValueError):
        ObservedSource("session", "x", bytearray(b"mutable"))
    for pool, session in (((), "session"), (sources, "other"), (sources * 2, "session")):
        with pytest.raises(ValueError):
            model.write(pool, session)
    with pytest.raises(ValueError):
        model.generate(model.write(sources, "session"), "other", b"?")


def test_utf8_boundary_and_truncated_character():
    model, sources = fixture()
    tape = model.write(sources, "session")
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.fill_(-100)
        model.readout.bias[256] = 100
    assert model.generate(tape, "session", b"?") == b""
    with torch.no_grad():
        model.readout.bias[256] = -100
        model.readout.bias[0xE4] = 100
        model.readout.bias[0xB8] = 50
    assert model.generate(tape, "session", b"?", max_length=1) == b""
    model.generate(tape, "session", b"?", max_length=9).decode("utf-8")


def test_checkpoint_weights_only_roundtrip_and_schema(tmp_path):
    model, sources = fixture()
    payload = model.checkpoint(sources)
    path = tmp_path / "zero.pt"
    torch.save(payload, path)
    restored, recovered = model.restore(torch.load(path, weights_only=True))
    assert recovered == sources
    assert torch.equal(torch.get_rng_state(), payload["rng"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    payload["format"] = "unknown"
    with pytest.raises(ValueError):
        model.restore(payload)
