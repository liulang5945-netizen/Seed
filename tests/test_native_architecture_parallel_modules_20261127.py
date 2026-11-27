"""Tests for the 4 parallel authoritative-architecture modules (2026-11-27)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from taiji.action_lifecycle import (
    ActionLifecycleError,
    ActionLifecycleLedger,
    DuplicateSettlementError,
)
from taiji.conditional_value import (
    ActionCandidate,
    ConditionalValueHead,
    TaskDecisionContext,
)
from taiji.development_learning import (
    LearningPermissionPolicy,
    OutcomeRouter,
    VerifiedOutcomeEvent,
)
from taiji.dynamic_byte_chunker import (
    BOUNDARY_ROLE,
    DynamicByteChunkEncoder,
    DynamicByteVocab,
    EntropyBoundaryChunker,
)


def test_dynamic_byte_chunker_lossless_and_neutral_birth() -> None:
    vocab = DynamicByteVocab()
    sample = "态极原生字节架构：支持动态宏词元与OOV无损回退！ABC123".encode()
    macro_b = "原生字节".encode()
    sid = vocab.register_macro(macro_b, birth_step=10)
    assert sid == vocab.base_vocab_size

    encoded = vocab.encode_bytes(sample)
    assert sid in encoded
    assert vocab.decode_to_bytes(encoded) == sample

    # Capsule roundtrip
    vocab2 = DynamicByteVocab.from_dict(vocab.to_dict())
    assert vocab2.encode_bytes(sample) == encoded
    assert vocab2.decode_to_bytes(encoded) == sample

    # Neutral macro birth: exact 0.0 perturbation relative to raw bytes + learnable delta grad
    torch.manual_seed(5)
    enc = DynamicByteChunkEncoder(d_model=64, byte_emb_dim=32)
    enc.eval()
    raw_ids = list(sample)
    with torch.no_grad():
        raw_out = enc.encode_symbol_stream(raw_ids)
        ev = enc.grow_macro_symbol_neutral_(macro_b, birth_step=10)
        assert ev["vocab_size"] == vocab.base_vocab_size + 1
        grown_ids = enc.vocab.encode_bytes(sample)
        macro_out = enc.encode_symbol_stream(grown_ids)
    assert float((macro_out - raw_out).abs().max().item()) == 0.0

    # Gradient must flow to the newly grown macro_deltas row
    enc.train()
    enc.zero_grad(set_to_none=True)
    out = enc.encode_symbol_stream(grown_ids)
    loss = out.pow(2).mean()
    loss.backward()
    assert enc.macro_deltas.grad is not None
    assert float(enc.macro_deltas.grad.abs().sum().item()) > 0.0

    # Entropy boundary chunker strict causality & boundary role tag
    chunker = EntropyBoundaryChunker(entropy_threshold=2.5, min_chunk=2, max_chunk=6)
    ent = [0.5, 3.0, 0.4, 0.5, 0.6, 3.2]
    spans = chunker.segment(ent)
    assert spans == [(0, 2), (2, 6)]
    pooled, meta = chunker.pool_features(raw_out[:, :6, :], ent)
    assert pooled.shape == (1, 2, 64)
    assert meta["boundary_role"] == BOUNDARY_ROLE


def test_conditional_value_state_flips_ranking_and_wait_guards() -> None:
    torch.manual_seed(3)
    head = ConditionalValueHead(state_dim=4, action_dim=4, hidden_dim=16)
    # Construct weights where state flips preference between c_continue and c_verify
    with torch.no_grad():
        head.state_proj.weight.copy_(torch.eye(4))
        head.state_proj.bias.zero_()
        head.mlp[0].weight.zero_()
        head.mlp[0].bias.zero_()
        # Hidden unit 0 reads interaction dims 0..3 equally
        head.mlp[0].weight[0, 8:12] = 1.0
        head.mlp[2].weight.zero_()
        head.mlp[2].bias.zero_()
        head.mlp[2].weight[0, 0] = 1.0

    c_continue = ActionCandidate(
        "c1",
        "continue_task",
        cost=1.0,
        precondition_met=True,
        feature_vector=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        parent_goal_id="g1",
    )
    c_verify = ActionCandidate(
        "c2",
        "verify_evidence",
        cost=1.0,
        precondition_met=True,
        feature_vector=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        parent_goal_id="g1",
    )

    ctx1 = TaskDecisionContext(
        "g1",
        goal_active=True,
        goal_completed=False,
        budget_remaining=5.0,
        uncertainty=0.1,
        state_vector=torch.tensor([1.0, -1.0, 0.0, 0.0]),
    )
    ctx2 = TaskDecisionContext(
        "g1",
        goal_active=True,
        goal_completed=False,
        budget_remaining=5.0,
        uncertainty=0.8,
        state_vector=torch.tensor([-1.0, 1.0, 0.0, 0.0]),
    )
    t1 = head.score_candidates(ctx1, [c_continue, c_verify])
    t2 = head.score_candidates(ctx2, [c_continue, c_verify])
    assert t1.selected_action == "continue_task"
    assert t2.selected_action == "verify_evidence"

    # Completed goal or insufficient budget forces ask_wait
    ctx_done = TaskDecisionContext(
        "g1",
        goal_active=True,
        goal_completed=True,
        budget_remaining=5.0,
        uncertainty=0.1,
        state_vector=torch.zeros(4),
    )
    assert head.score_candidates(ctx_done, [c_continue, c_verify]).waited_due_to_guard is True
    assert (
        head.propose_subgoals(ctx_done, gap_evidence=[{"evidence_ref": "e1", "expected_gain": 1.0}])
        == []
    )

    # Active goal bounded subgoal proposal respects budget
    subs = head.propose_subgoals(
        ctx1,
        gap_evidence=[
            {"evidence_ref": "ev_a", "expected_gain": 0.4, "estimated_cost": 3.0},
            {"evidence_ref": "ev_b", "expected_gain": 0.5, "estimated_cost": 3.0},
        ],
    )
    assert len(subs) == 1 and subs[0].evidence_ref == "ev_a"


def test_action_lifecycle_transitions_idempotency_and_4_boundary_recovery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "action_ledger.json"
    ledger = ActionLifecycleLedger(path)

    # Boundary 1: prepared; Boundary 2: dispatched -> unknown; Boundary 3: observed; Boundary 4: settled
    r1 = ledger.propose(
        "a1", idempotency_key="k1", intent_digest="d1", goal_revision=1, belief_revision=1
    )
    assert (
        ledger.propose(
            "a1_dup", idempotency_key="k1", intent_digest="d1", goal_revision=1, belief_revision=1
        ).action_id
        == "a1"
    )
    assert r1.state == "proposed"
    ledger.prepare("a1")

    ledger.propose(
        "a2", idempotency_key="k2", intent_digest="d2", goal_revision=1, belief_revision=1
    )
    ledger.prepare("a2")
    ledger.dispatch("a2")

    ledger.propose(
        "a3", idempotency_key="k3", intent_digest="d3", goal_revision=1, belief_revision=1
    )
    ledger.prepare("a3")
    ledger.dispatch("a3")
    ledger.observe("a3", receipt_id="rc3", outcome_payload={"ok": True})

    ledger.propose(
        "a4", idempotency_key="k4", intent_digest="d4", goal_revision=1, belief_revision=1
    )
    ledger.prepare("a4")
    ledger.dispatch("a4")
    ledger.observe("a4", receipt_id="rc4", outcome_payload={"ok": True})
    ledger.settle("a4")

    with pytest.raises(ActionLifecycleError):
        ledger.settle("a1")
    with pytest.raises(DuplicateSettlementError):
        ledger.settle("a4")

    # Reload from disk and verify 4-boundary crash recovery
    reloaded = ActionLifecycleLedger.open(path)
    summary = reloaded.recover_in_flight()
    assert summary["prepared_ready_or_cancel"] == ["a1"]
    assert summary["dispatched_moved_to_unknown"] == ["a2"]
    assert summary["observed_ready_to_settle"] == ["a3"]
    assert summary["already_settled"] == ["a4"]

    # Model rollback must preserve external dispatched/unknown/observed/settled facts
    note = reloaded.record_model_rollback("shadow_rejected")
    assert set(note["preserved_external_actions"]) == {"a2", "a3", "a4"}


def test_development_learning_outcome_router_provenance_and_retention() -> None:
    router = OutcomeRouter(LearningPermissionPolicy(max_retention_drop=0.02, min_gain=0.01))

    # Block imagined, unknown, self_generated, unauthorized
    for prov, auth, eid in [
        ("imagined", True, "e1"),
        ("unknown", True, "e2"),
        ("self_generated", True, "e3"),
        ("real", False, "e4"),
    ]:
        ev = VerifiedOutcomeEvent(
            event_id=eid,
            action_ref="act1",
            provenance=prov,  # type: ignore[arg-type]
            authorized=auth,
            reward=1.0,
            prediction_error=0.2,
            cost=0.1,
        )
        rec = router.route_event(
            ev, apply_fn=lambda _p: None, eval_old_fn=lambda: 0.9, eval_new_fn=lambda: 0.9
        )
        assert rec.accepted is False

    # Valid real authorized event with positive gain & safe retention
    step = {"done": False}

    def _apply(_planes: list[str]) -> None:
        step["done"] = True

    ev_ok = VerifiedOutcomeEvent(
        event_id="e_real",
        action_ref="act2",
        provenance="real",
        authorized=True,
        reward=1.0,
        prediction_error=0.1,
        cost=0.2,
    )
    rec_ok = router.route_event(
        ev_ok,
        apply_fn=_apply,
        eval_old_fn=lambda: 0.95,
        eval_new_fn=lambda: 0.80 if step["done"] else 0.50,
    )
    assert rec_ok.accepted is True
    assert "fast_f" in rec_ok.updated_planes

    # Duplicate real event rejected fail-closed
    rec_dup = router.route_event(
        ev_ok, apply_fn=_apply, eval_old_fn=lambda: 0.95, eval_new_fn=lambda: 0.80
    )
    assert rec_dup.accepted is False
    assert rec_dup.reject_reason == "duplicate_event"
