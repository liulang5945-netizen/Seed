"""Oracle diagnostics must isolate addresses, preserve native control and scope."""

import pytest
import torch

from scripts.training import audit_taiji_n1_address_intervention as audit
from taiji.contracts import Observation
from taiji.native_evidence_candidate import NativeEvidenceCandidate
from taiji.native_iterative_evidence_candidate import IterativeEvidenceCandidate


def fixture(cls):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = cls()
    model.observe("a", Observation("text", "entity uses abcdef.", 1, "source-1"))
    model.observe("a", Observation("text", "other uses ghijkl.", 2, "source-2"))
    tape = model.evidence("a")
    item = {
        "split": "train",
        "kind": "lookup",
        "answer": "abcdef",
        "observations": [s.content.decode() for s in tape.sources],
    }
    source, positions = audit.addresses(item, tape)
    return model, tape, source, positions


@pytest.mark.parametrize("cls", [NativeEvidenceCandidate, IterativeEvidenceCandidate])
def test_native_manual_path_matches_unmodified_candidate(cls):
    model, tape, source, positions = fixture(cls)
    with torch.no_grad():
        expected, _ = model.teacher_logits(tape, "a", b"?", b"abcdef")
        actual = audit.teacher(model, tape, "a", b"?", b"abcdef", "native", source, positions)
    assert torch.equal(actual, expected)


def test_source_mask_and_one_hot_change_only_read_address():
    model, tape, source, positions = fixture(IterativeEvidenceCandidate)
    hidden = model.question_state(b"?")
    native = audit.weights(model, tape, hidden, "native", source, positions[0])
    masked = audit.weights(model, tape, hidden, "source", source, positions[0])
    mask = torch.tensor([s == source for s, _ in tape.positions])
    assert torch.equal(masked[~mask], torch.zeros_like(masked[~mask]))
    assert torch.allclose(masked[mask], native[mask] / native[mask].sum())
    point = audit.weights(model, tape, hidden, "position", source, positions[0])
    assert point.sum() == 1 and point[positions[0]] == 1
    assert torch.equal(point @ tape.values, tape.values[positions[0]])


@pytest.mark.parametrize("cls", [NativeEvidenceCandidate, IterativeEvidenceCandidate])
def test_both_hops_intervened_but_boundary_native_and_previous_gold(cls, monkeypatch):
    model, tape, source, positions = fixture(cls)
    original = audit.weights
    calls = []
    previous_symbols = []
    boundary_calls = []
    original_symbol = model.symbol
    original_step = model.step

    def counted(*args):
        calls.append((args[3], args[5]))
        return original(*args)

    def symbol(value):
        previous_symbols.append(value)
        return original_symbol(value)

    def step(*args):
        boundary_calls.append(args[-1])
        return original_step(*args)

    monkeypatch.setattr(audit, "weights", counted)
    monkeypatch.setattr(model, "symbol", symbol)
    monkeypatch.setattr(model, "step", step)
    before = model.bundle()["sessions"]
    audit.teacher(model, tape, "a", b"?", b"abcdef", "position", source, positions)
    hops = getattr(model, "READ_HOPS", 1)
    assert calls == [("position", p) for p in positions for _ in range(hops)]
    assert boundary_calls == [ord("f")]
    assert previous_symbols[1:] == [v for v in (256, *b"abcdef") for _ in range(hops)]
    assert model.bundle()["sessions"] == before


def test_heldout_rejected_before_tape_access():
    with pytest.raises(ValueError, match="material train"):
        audit.addresses({"split": "holdout"}, None)
