"""Geometry audit stays train-only and never calls answer generators."""

import pytest
import torch

from scripts.training import audit_taiji_n1_question_separation as audit
from taiji.native_evidence_candidate import Observation


def test_native_capture_repeat_and_no_scoring():
    torch.set_num_threads(1)
    model = audit.source.ContextualQuestionEvidenceCandidate()
    model.observe("history", Observation("text", "saved history", 1, "history"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = audit.paired.fingerprint(model.bundle(optimizer))

    def forbidden(*args, **kwargs):
        raise AssertionError("no scoring allowed")

    model.teacher_logits = forbidden
    model.generate_trace = forbidden
    item = dict(
        id="synthetic",
        split="train",
        observations=["alpha uses quartz.", "bravo uses copper."],
        question="material of alpha?",
    )
    a = audit.capture(model, item, "test56-a")
    b = audit.capture(model, item, "test56-b")
    assert torch.equal(a["hidden"], b["hidden"]) and torch.equal(a["states"], b["states"])
    assert abs(audit.distance(a["hidden"], b["hidden"])) < 1e-6
    assert before == audit.paired.fingerprint(model.bundle(optimizer))
    with pytest.raises(ValueError, match="train"):
        audit.capture(model, dict(item, split="holdout"), "bad")
    assert audit.ratio(1, 0) is None
    assert audit.ratio(1, 2) == 0.5


def test_unique_full_binding_denominator_and_shapes():
    parents = [f"parent-{i}" for i in range(96)]
    keys = [(n, p, v) for n in ("52", "55") for p in parents for v in audit.VIEWS]
    audit.validate_keys(keys, parents)
    for bad in (keys[:-1], keys[:-1] + [keys[0]]):
        with pytest.raises(ValueError, match="768"):
            audit.validate_keys(bad, parents)
    with pytest.raises(ValueError, match="aligned"):
        audit.distance(torch.ones(2), torch.ones(3))
