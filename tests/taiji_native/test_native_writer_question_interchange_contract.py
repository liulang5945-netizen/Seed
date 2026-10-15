"""Native capture and matched activation donor controls."""

from dataclasses import replace

import pytest
import torch

from scripts.training import audit_taiji_n1_writer_question_interchange as audit


def test_native_and_two_donors_preserve_original_logits():
    torch.set_num_threads(1)
    model, optimizer = audit.source.fresh(audit.source.checked_train())
    before = audit.paired.fingerprint(model.bundle(optimizer))
    views = audit.previous.prior.checked_views()
    for i in range(4):
        original, renamed = views["original"][i], views["rename"][i]
        reference = audit.measure(model, original, f"original-{i}")
        native = audit.measure(model, renamed, f"rename-{i}")
        both = audit.measure(model, renamed, f"both-{i}", reference, "both")
        assert torch.equal(reference["logits"], both["logits"])
        for condition in ("writer", "question"):
            row = audit.measure(model, renamed, f"single-{i}-{condition}", reference, condition)
            assert row["logits"].shape == native["logits"].shape
        for condition in audit.CONDITIONS:
            row = audit.measure(model, original, f"identity-{i}-{condition}", reference, condition)
            assert torch.equal(reference["logits"], row["logits"])
    assert before == audit.paired.fingerprint(model.bundle(optimizer))
    assert "question_state" not in model.__dict__


def test_train_and_position_guards_and_provenance():
    model = audit.source.ResidualQuestionEvidenceCandidate()
    item = audit.source.checked_train()[0]
    reference = audit.measure(model, item, "reference")
    with pytest.raises(ValueError, match="train"):
        audit.measure(model, dict(item, split="holdout"), "bad")
    bad = dict(reference, tape=replace(reference["tape"], positions=()))
    with pytest.raises(ValueError, match="positions"):
        audit.measure(model, item, "bad-positions", bad, "writer")
    receiver = replace(reference["tape"], session="receiver")
    hybrid = audit.donor_tape(receiver, reference["tape"])
    assert hybrid.session == "receiver" and hybrid.sources is receiver.sources
    assert hybrid.states is reference["tape"].states
    assert not model._observations and "question_state" not in model.__dict__


def test_complete_unique_denominator():
    keys = [
        (model, view, item["id"], condition)
        for model in ("41", "46")
        for view in ("original", "pair")
        for item in audit.previous.prior.checked_views()[view]
        for condition in ("reference", *audit.CONDITIONS)
    ]
    audit.validate_keys(keys)
    with pytest.raises(ValueError, match="2880"):
        audit.validate_keys(keys[:-1])
    with pytest.raises(ValueError, match="2880"):
        audit.validate_keys(keys[:-1] + [keys[0]])
