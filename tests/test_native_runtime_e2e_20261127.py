"""End-to-end tests for unified `TaijiNativeRuntime` wiring all architecture layers."""

from __future__ import annotations

from pathlib import Path

import torch

from taiji.action_lifecycle import ActionLifecycleLedger
from taiji.conditional_value import ActionCandidate
from taiji.development_learning import LearningPermissionPolicy, VerifiedOutcomeEvent
from taiji.dynamic_byte_chunker import BOUNDARY_ROLE
from taiji.native_cognitive_backbone import NATIVE_PRESETS
from taiji.native_growth_controller import NativeGrowthPolicy, save_growth_policy
from taiji.native_runtime import SCHEMA, TaijiNativeRuntime


def test_native_runtime_e2e_rollout_value_action_growth_and_persistence(tmp_path: Path) -> None:
    torch.manual_seed(31)
    state_dir = tmp_path / "runtime_state"
    rt = TaijiNativeRuntime.from_preset(
        NATIVE_PRESETS["native_byte_tiny"],
        state_dir=state_dir,
        growth_policy=NativeGrowthPolicy(
            auto_grow_enabled=True, delta_units=8, min_gain=0.0, max_retention_drop=1.0
        ),
        learning_policy=LearningPermissionPolicy(max_retention_drop=0.02, min_gain=0.0),
    )

    # 1. Native rollout with & without copy records + causal chunk stats
    out_plain = rt.observe_and_respond("今天天气不错", max_new_bytes=12)
    assert out_plain["schema"] == SCHEMA
    assert out_plain["used_copy_head"] is False
    assert out_plain["boundary_stats"]["boundary_role"] == BOUNDARY_ROLE

    out_copy = rt.observe_and_respond(
        "请回答：",
        experience_records=["蓝色小球在右边"],
        max_new_bytes=12,
    )
    assert out_copy["used_copy_head"] is True
    assert out_copy["copy_meta"]["n_records"] == 1

    # 2. ConditionalValueHead scoring + wait guard
    d_model = rt.backbone.cfg.d_model
    cand = ActionCandidate(
        candidate_id="act_1",
        action_kind="verify_evidence",
        cost=1.0,
        precondition_met=True,
        feature_vector=torch.ones(d_model) * 0.1,
        parent_goal_id="g_e2e",
    )
    trace_ok = rt.evaluate_goal_action([cand], wait_budget_ok=True)
    assert trace_ok.waited_due_to_guard is False
    assert trace_ok.selected_action == "verify_evidence"

    trace_wait = rt.evaluate_goal_action([cand], wait_budget_ok=False)
    assert trace_wait.waited_due_to_guard is True
    assert trace_wait.selected_action == "ask_wait"

    # 3. 9-state external action lifecycle + idempotency + OutcomeRouter
    act_res = rt.propose_and_execute_action(
        "ext_check_1",
        idempotency_key="idem_ext_1",
        intent_digest="sha_intent_1",
        executor_fn=lambda: {"receipt_id": "rc_ok_1", "authorized": True, "ok": True},
        reward=0.2,
    )
    assert act_res["state"] == "settled"
    assert act_res["learning_receipt"].accepted is True

    replay_res = rt.propose_and_execute_action(
        "ext_check_1_dup",
        idempotency_key="idem_ext_1",
        intent_digest="sha_intent_1",
        executor_fn=lambda: {"receipt_id": "should_not_run"},
    )
    assert replay_res["idempotent_replay"] is True

    # Imagined feedback stays blocked by OutcomeRouter
    bad_ev = VerifiedOutcomeEvent(
        event_id="fake_1",
        action_ref="none",
        provenance="imagined",
        authorized=True,
        reward=1.0,
        prediction_error=0.0,
        cost=0.0,
    )
    bad_rec = rt.outcome_router.route_event(
        bad_ev, apply_fn=lambda _p: None, eval_old_fn=lambda: 1.0, eval_new_fn=lambda: 2.0
    )
    assert bad_rec.accepted is False

    # 4. Governed shadow growth + live policy reload + optimizer moment migration
    policy_path = tmp_path / "growth_policy.json"
    save_growth_policy(
        policy_path,
        NativeGrowthPolicy(
            auto_grow_enabled=True, delta_units=8, min_gain=0.0, max_retention_drop=1.0
        ),
    )
    opt = torch.optim.AdamW(rt.backbone.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (1, 16))
    y = torch.randint(0, 256, (1, 16))
    _, loss, _ = rt.backbone(x, labels=y)
    assert loss is not None
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    # Trigger pressure window
    rt.growth_controller.pressure.window = 2
    rt.growth_controller.pressure.error_threshold = 0.1
    assert (
        rt.governed_growth_step(
            0.5, 1, adapt_fn=lambda _m: None, eval_old_fn=lambda _m: 1.0, eval_new_fn=lambda _m: 1.0
        )["accepted"]
        is False
    )

    before_params = rt.backbone.count_parameters()
    rec_grow = rt.governed_growth_step(
        0.5,
        2,
        adapt_fn=lambda m: m.apply_selective_plasticity_grad_mask(),
        eval_old_fn=lambda _m: 1.0,
        eval_new_fn=lambda m: 1.1 if m.count_parameters() > before_params else 1.0,
        sample_input=x,
        optimizer=opt,
        policy_path=policy_path,
        seed=7,
    )
    assert rec_grow["accepted"] is True
    assert rec_grow["learning_receipt_accepted"] is True
    assert rt.backbone.count_parameters() > before_params
    assert rec_grow["optimizer_migration"]["migrated"] >= 6

    # 5. Checkpoint save & reload across restarts preserves weights, subject state & ledger
    ckpt = state_dir / "native_runtime.pt"
    rt.save(ckpt)
    rt2 = TaijiNativeRuntime.load(ckpt, state_dir=state_dir)
    assert rt2.backbone.count_parameters() == rt.backbone.count_parameters()
    assert torch.equal(rt2.subject_state.h_states[0], rt.subject_state.h_states[0])
    reopened_ledger = ActionLifecycleLedger.open(state_dir / "action_ledger.json")
    assert "ext_check_1" in reopened_ledger.records
