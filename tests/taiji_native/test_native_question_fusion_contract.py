"""Frozen ablation controls, scopes and equal paired denominator."""

from types import MethodType

import pytest
import torch

from scripts.training import audit_taiji_n1_question_fusion as audit
from taiji.contracts import Observation
from taiji.native_residual_question_evidence_candidate import ResidualQuestionEvidenceCandidate


def test_masks_controls_and_train_scope():
    item = audit.residual.checked_train()[0]
    mask = audit.entity_mask(item)
    assert int(mask.sum()) == 10
    assert (
        item["question"]
        .encode()[mask.nonzero().flatten()[0] : mask.nonzero().flatten()[-1] + 1]
        .decode()
        in item["entities"]
    )
    row = dict(raw_hex="61", stop_reason="boundary", teacher_tokens_correct=2, teacher_ce=0.5)
    assert audit.control_equal(row, row)
    assert not audit.control_equal(dict(row, teacher_ce=0.51), row)
    assert not audit.control_equal(dict(row, raw_hex="62"), row)
    for split in ("dev", "holdout"):
        with pytest.raises(ValueError, match="train"):
            audit.entity_mask(dict(item, split=split))


def test_complete_unique_pairs():
    rows = [
        dict(model=name, condition=condition, id=f"train-{i:03d}")
        for name in ("replacing", "residual")
        for condition in audit.CONDITIONS
        for i in range(192)
    ]
    assert audit.complete_rows(rows)
    assert not audit.complete_rows(rows[:-1])
    rows[-1] = rows[0]
    assert not audit.complete_rows(rows)


def test_bypass_uses_native_values_without_updates_and_shared_execution(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = ResidualQuestionEvidenceCandidate()
    model.observe("p", Observation("text", "abc", 1, "source-1"))
    with torch.no_grad():
        model.question_fusion.weight.fill_(0.1)
        model.readout.weight.zero_()
        model.readout.bias.zero_()
        model.readout.bias[ord("a")] = 1
    before = audit.paired.fingerprint(model.bundle())
    native = model.condition
    calls = []

    def condition(self, tape, states):
        conditioned, alignment = native(tape, states)
        result, _ = audit.bypass(tape, alignment)
        assert torch.equal(result.keys, tape.keys)
        assert torch.equal(result.values, tape.values)
        assert not torch.equal(conditioned.values, tape.values)
        calls.append(result.values.detach().clone())
        return result, alignment

    monkeypatch.setattr(model, "condition", MethodType(condition, model))
    tape = model.evidence("p")
    model.teacher_logits(tape, "p", b"?", b"aa")
    assert model.generate_trace(tape, "p", b"?", 2).raw == b"aa"
    assert len(calls) == 2 and torch.equal(calls[0], calls[1])
    assert before == audit.paired.fingerprint(model.bundle())
