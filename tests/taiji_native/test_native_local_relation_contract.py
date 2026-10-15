"""Zero residual, state reuse and source/credit boundaries for PLAN60."""

from dataclasses import replace

import pytest
import torch

from scripts.training import _taiji_n1_local_relation_transfer as transfer
from taiji.native_evidence_candidate import Observation
from taiji.native_local_relation_evidence_candidate import LocalRelationEvidenceCandidate


def prepare(model):
    model.observe("relation60", Observation("text", "alpha uses quartz.", 1, "one"))
    model.observe("relation60", Observation("text", "bravo uses copper.", 2, "two"))


def test_source_migration_zero_behavior_and_restore():
    torch.set_num_threads(1)
    _, payload, _ = transfer.source_payload()
    old = transfer.source.ContextualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    model, optimizer = transfer.migrate(payload)
    assert sum(p.numel() for p in model.parameters()) == 62056
    prepare(old)
    prepare(model)
    question = b"material of alpha?"
    old_tape, tape = old.evidence("relation60"), model.evidence("relation60")
    a, aw = old.teacher_logits(old_tape, "relation60", question, b"quartz")
    b, bw = model.teacher_logits(tape, "relation60", question, b"quartz")
    assert torch.equal(a, b) and torch.equal(aw, bw)
    _, s, _ = old.step(old_tape, "relation60", old.question_state(question), 256)
    _, t, _ = model.step(tape, "relation60", model.question_state(question), 256)
    assert torch.equal(s.conditioned_tape.keys, t.conditioned_tape.keys)
    assert torch.equal(s.conditioned_tape.values, t.conditioned_tape.values)
    restored = LocalRelationEvidenceCandidate.restore_bundle(model.bundle(optimizer))
    assert torch.equal(
        restored.teacher_logits(restored.evidence("relation60"), "relation60", question, b"quartz")[
            0
        ],
        b,
    )
    with pytest.raises(ValueError, match="mechanism"):
        LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    bad = model.bundle(optimizer)
    del bad["state"]["relation_projection.weight"]
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        LocalRelationEvidenceCandidate.restore_bundle(bad)


def test_relation_credit_cut_terminal_sources_and_reuse():
    torch.set_num_threads(1)
    model = LocalRelationEvidenceCandidate()
    prepare(model)
    q = b"material of alpha?"
    tape = model.evidence("relation60")
    logits, _ = model.teacher_logits(tape, "relation60", q, b"quartz")
    logits.square().mean().backward()
    assert model.relation_projection.weight.grad.abs().sum() > 0
    assert model.relation_sequence.weight_ih_l0.grad.abs().sum() == 0
    with torch.no_grad():
        model.relation_projection.weight.normal_(0, 0.03)
    model.zero_grad(set_to_none=True)
    tape = model.evidence("relation60")
    logits, _ = model.teacher_logits(tape, "relation60", q, b"quartz")
    logits.square().mean().backward()
    assert model.relation_sequence.weight_ih_l0.grad.abs().sum() > 0
    assert model.local_projection.weight.grad.abs().sum() > 0
    blocks = model.relation_inputs(model.evidence("relation60", terminal_per_source=True), q)
    assert len(blocks) == 2
    assert all(torch.equal(b, b[:1].expand_as(b)) for b in blocks)
    calls = []
    hook = model.relation_sequence.register_forward_pre_hook(
        lambda module, args: calls.append(len(args))
    )
    _, state, _ = model.step(tape, "relation60", model.question_state(q), 256)
    assert calls == [1, 1]
    model.step(tape, "relation60", state, ord("q"))
    assert calls == [1, 1]
    hook.remove()
    with pytest.raises(ValueError, match="sessions"):
        model.step(replace(tape, session="wrong"), "wrong", state, 256)
    captured = []

    def keep(module, args, out):
        if args[0].ndim == 1 and len(args[0]) == len(b"alpha uses quartz."):
            out.retain_grad()
            captured.append(out)

    hook = model.embedding.register_forward_hook(keep)
    for cut in (False, True):
        captured.clear()
        model.zero_grad(set_to_none=True)
        inputs = model.relation_inputs(model.evidence("relation60", cut_credit=cut), q)
        torch.cat(inputs).square().mean().backward()
        assert len(captured) == 2
        assert (
            all(t.grad is None for t in captured)
            if cut
            else all(t.grad is not None and t.grad.abs().sum() > 0 for t in captured)
        )
    hook.remove()


def test_nonzero_engineering_probe_preserves_full_rng_and_bundle():
    from scripts.training import audit_taiji_n1_local_relation as audit

    model = LocalRelationEvidenceCandidate()
    audit.prepare(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = audit.paired.fingerprint(model.bundle(optimizer))
    checks = {}
    audit.credit_probe(model, optimizer, checks)
    assert all(checks.values())
    assert before == audit.paired.fingerprint(model.bundle(optimizer))
