"""Teacher capture neutrality and full fixed train-derived denominator."""

import pytest
import torch

from scripts.training import audit_taiji_n1_read_equivariance as audit


def test_divergence_support_and_position_pairing():
    p, q = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert audit.js(p, p) == 0
    assert audit.js(p, q) == pytest.approx(0.69314718)
    assert audit.js(p, q) == audit.js(q, p)
    assert audit.relative_l2(p, p) == 0
    with pytest.raises(ValueError, match="positions"):
        audit.compare(
            dict(positions=[1], target_symbols=[1]), dict(positions=[2], target_symbols=[1])
        )


def test_complete_fixed_unique_denominator():
    views = audit.checked_views()
    keys = [
        (name, view, item["id"])
        for name in ("41", "43")
        for view, items in views.items()
        for item in items
    ]
    audit.validate_denominator(keys)
    with pytest.raises(ValueError, match="1152"):
        audit.validate_denominator(keys[:-1])
    with pytest.raises(ValueError, match="1152"):
        audit.validate_denominator(keys[:-1] + [keys[0]])
    for original, renamed in (("original", "rename"), ("pair", "rename_pair")):
        for a, b in zip(views[original], views[renamed], strict=True):
            assert a["answer"] == b["answer"] and a["id"] == b["id"]
            assert list(map(len, a["observations"])) == list(map(len, b["observations"]))


def test_capture_preserves_teacher_and_full_state():
    torch.set_num_threads(1)
    item = audit.checked_views()["original"][0]
    model = audit.source.ResidualQuestionEvidenceCandidate()
    audit.paired.observe(model, audit.paired.project(item), "control")
    tape = model.evidence("control")
    with torch.no_grad():
        expected, _ = model.teacher_logits(
            tape, "control", item["question"].encode(), item["answer"].encode()
        )
    before = audit.paired.fingerprint(model.bundle())
    trace = audit.capture(model, item, "capture-test")
    assert torch.equal(expected, torch.stack([s["logits"] for s in trace["steps"]]))
    assert before == audit.paired.fingerprint(model.bundle())
    assert "step" not in model.__dict__
    assert all(b["address_js"] == 0 for b in audit.compare(trace, trace))
    density = audit.alignment_density(trace, item)
    assert 0 <= density["entity_mass"] <= 1
    with pytest.raises(ValueError, match="train"):
        audit.capture(model, dict(item, split="holdout"), "forbidden")
    with pytest.raises(ValueError, match="collides"):
        audit.capture(model, item, "control")
