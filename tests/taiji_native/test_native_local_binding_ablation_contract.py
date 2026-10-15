"""Diagnostic neutrality, genuine local bypass and offline absent-entity handling."""

import pytest
import torch

from scripts.training import audit_taiji_n1_local_binding_ablation as audit
from taiji.native_residual_question_evidence_candidate import ResidualQuestionEvidenceCandidate


def synthetic_item():
    return dict(
        id="synthetic-ablation",
        kind="lookup",
        split="train",
        entities=["alpha", "bravo", "absent"],
        observations=["alpha uses quartz.", "bravo uses copper."],
        question="material of alpha?",
        answer="quartz",
    )


def test_native_neutrality_and_local_only_bypass():
    torch.set_num_threads(1)
    model = audit.source.LocalQuestionEvidenceCandidate()
    with torch.no_grad():
        model.local_fusion.weight.fill_(0.01)
        model.local_salience.weight.zero_()
        model.local_salience.bias.zero_()
    item = synthetic_item()
    audit.paired.observe(model, audit.paired.project(item), "control")
    with torch.no_grad():
        expected, _ = model.teacher_logits(
            model.evidence("control"), "control", item["question"].encode(), item["answer"].encode()
        )
    before = audit.paired.fingerprint(model.bundle())
    native = audit.measure(model, item, "native", "native")
    bypass = audit.measure(model, item, "bypass", "bypass")
    assert torch.equal(native["logits"], expected)
    assert not torch.equal(native["logits"], bypass["logits"])
    assert before == audit.paired.fingerprint(model.bundle())
    statistics = native["alignment"]
    assert statistics["question_entity_observed"]
    assert statistics["entity_salience_mass"] == pytest.approx(
        statistics["entity_salience_uniform"]
    )
    assert 0 <= statistics["same_entity_alignment_mass"] <= 1
    original_step = model.step
    model.step = lambda tape, session, state, previous: ResidualQuestionEvidenceCandidate.step(
        model, tape, session, state, previous
    )
    try:
        with torch.no_grad():
            old_logits, _ = model.teacher_logits(
                model.evidence("control"),
                "control",
                item["question"].encode(),
                item["answer"].encode(),
            )
        assert torch.equal(old_logits, bypass["logits"])
    finally:
        model.__dict__.pop("step", None)
    assert model.step == original_step and "bind_question" not in model.__dict__


def test_absent_alignment_and_train_only():
    model = audit.source.LocalQuestionEvidenceCandidate()
    item = dict(synthetic_item(), question="material of absent?", answer="unknown", kind="unknown")
    native = audit.measure(model, item, "absent", "native")
    assert not native["alignment"]["question_entity_observed"]
    assert native["alignment"]["same_entity_alignment_mass"] is None
    assert native["alignment"]["same_entity_uniform"] is None
    with pytest.raises(ValueError, match="train"):
        audit.measure(model, dict(item, split="holdout"), "forbidden", "native")
    assert not model._observations


def test_complete_unique_denominator():
    keys = [
        (view, item["id"], condition)
        for view, items in audit.views_source.checked_views().items()
        for item in items
        for condition in audit.CONDITIONS
    ]
    audit.validate_keys(keys)
    with pytest.raises(ValueError, match="1152"):
        audit.validate_keys(keys[:-1])
    with pytest.raises(ValueError, match="1152"):
        audit.validate_keys(keys[:-1] + [keys[0]])
