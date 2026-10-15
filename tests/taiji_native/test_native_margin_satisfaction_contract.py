"""Geometry audit stays train-only and never calls answer generators."""

import pytest
import torch

from scripts.training import audit_taiji_n1_margin_satisfaction as audit
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
    keys = [(n, p, v) for n in ("52", "58") for p in parents for v in audit.VIEWS]
    audit.validate_keys(keys, parents)
    for bad in (keys[:-1], keys[:-1] + [keys[0]]):
        with pytest.raises(ValueError, match="1152"):
            audit.validate_keys(bad, parents)
    with pytest.raises(ValueError, match="aligned"):
        audit.distance(torch.ones(2), torch.ones(3))


def test_relative_gap_formula_and_two_faces():
    a = dict(hidden=torch.tensor([1.0, 0.0]), states=torch.tensor([[1.0, 0.0]]))
    b = dict(hidden=torch.tensor([0.0, 1.0]), states=torch.tensor([[0.0, 1.0]]))
    captured = {
        "original": a,
        "pair": b,
        "rename": a,
        "rename_pair": b,
        "train_rename": b,
        "train_pair": a,
    }
    metrics = audit.margin_metrics(captured)
    assert metrics["probe_0_joint_gap"] == pytest.approx(1.0)
    assert metrics["probe_0_joint_hinge"] == 0
    assert metrics["probe_1_joint_satisfied"] == 1
    assert metrics["train_0_joint_hinge"] == pytest.approx(1.2)
    assert metrics["train_1_joint_satisfied"] == 0
