"""Small synthetic U2 graph regressions with zero optimizer steps."""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from taiji import u2_operators as ops
from taiji.u2_subject import (
    U2Config,
    U2Packet,
    U2Subject,
    run_reference_sequence,
    synthetic_fixture,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def model():
    return U2Subject(U2Config.small()).double()


def frozen_reference():
    path = ROOT / "scripts/training/verify_taiji_u2_math_22_1.py"
    spec = importlib.util.spec_from_file_location("u2_22_1_parity_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pending_fixture(model):
    fixture = synthetic_fixture(model.config, torch.float64)
    state = model.set_goal(model.initial("test-u2"), fixture["goal"])
    state = model.set_records(state, fixture["records"])
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    state, decision = model.choose_operation(state, mask)
    state, receipt_id = model.issue(state, decision)
    state["process_time"] = 0.05
    event = fixture["events"][0]
    packet = U2Packet(
        "p0",
        "sensor",
        "world",
        0.0,
        0.05,
        0.05,
        event["observations"],
        event["coordinates"],
        event["outcome"],
        receipt_id,
    )
    return state, packet


def test_promoted_fp1_matches_frozen_oracle_and_zero_chi_credit():
    reference = frozen_reference()
    memory = torch.tensor([[0.2]], dtype=torch.float64)
    key = torch.ones((1, 1), dtype=torch.float64)
    alpha = torch.tensor([0.8], dtype=torch.float64)
    gain = torch.ones(1, dtype=torch.float64)
    radius = torch.tensor(10.0, dtype=torch.float64)
    for point in (0.0, 0.25, 0.5, 1.0):
        chi = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        promoted = ops.fp1(memory, key, key, alpha, gain, chi, radius)
        oracle = reference.fp1(memory, key, key, alpha, gain, chi, radius)
        assert torch.allclose(promoted, oracle, atol=1e-12, rtol=1e-10)
        derivative = torch.autograd.grad(promoted.sum(), chi)[0]
        assert torch.allclose(derivative, derivative.new_tensor(0.76 / (1 + point) ** 2))
    assert ops.fp1(memory, key, key, alpha, gain, chi, radius, allowed=False) is memory


def test_empty_record_support_preserves_finite_backward_and_oracle_parity():
    reference = frozen_reference()
    dtype = torch.float64
    valid = torch.tensor([[True, True], [False, False]])
    ids = torch.tensor([[0, 1], [0, 0]])
    route = torch.tensor([[0.1, 0.2, -0.3], [-0.1, 0.2, 0.4]], dtype=dtype, requires_grad=True)
    position = torch.zeros((2, 2, 2), dtype=dtype, requires_grad=True)
    generator = torch.tensor([[0.2, 0.4, -0.3], [-0.2, 0.1, 0.3]], dtype=dtype, requires_grad=True)
    lam = torch.tensor([0.4, 0.6], dtype=dtype, requires_grad=True)
    gamma = torch.tensor([0.2, 0.3], dtype=dtype, requires_grad=True)
    state = ops.initial_cursor(2, 2, 2, dtype)
    alpha, beta = ops.record_inputs(route, position, valid)
    output = ops.emission(
        state,
        alpha,
        beta,
        lam,
        gamma,
        generator.log_softmax(-1),
        ids,
        valid,
        torch.ones(3, dtype=torch.bool),
    )
    oracle = reference.emission(
        state,
        alpha,
        beta,
        lam,
        gamma,
        generator.log_softmax(-1),
        ids,
        valid,
        torch.ones(3, dtype=torch.bool),
    )
    assert torch.allclose(output["log_output"], oracle["log_output"], atol=1e-12, rtol=1e-10)
    gradients = torch.autograd.grad(
        -output["log_output"][1], [route, position, generator, lam, gamma]
    )
    assert all(torch.isfinite(value).all() for value in gradients)
    assert torch.count_nonzero(gradients[0][:, 1]) == 0
    assert torch.count_nonzero(gradients[1][:, 1]) == 0


def test_rare_restart_survives_promoted_log_cursor():
    dtype = torch.float64
    state = ops.initial_cursor(1, 1, 2, dtype)
    state["log_m"][0, 0, 1] = 0
    state["log_zeta"][0] = -1000
    output = ops.emission(
        state,
        torch.tensor([[0.0, -torch.inf]], dtype=dtype),
        torch.tensor([[[0.0, -torch.inf]]], dtype=dtype),
        torch.ones(1, dtype=dtype),
        torch.zeros(1, dtype=dtype),
        torch.tensor([[-torch.inf, 0.0]], dtype=dtype),
        torch.tensor([[0, 1]]),
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )
    assert output["log_output"][0].item() == -1000
    assert (1 - state["log_m"].exp().sum()).item() == 0


def test_full_small_subject_has_finite_owned_gradients_and_zero_initial_read_output(model):
    fixture = synthetic_fixture(model.config, torch.float64)
    assert torch.count_nonzero(model.record_output.weight) == 0
    assert torch.count_nonzero(model.record_output.bias) == 0
    losses, state, trace, credit = run_reference_sequence(model, fixture)
    named = list(model.named_parameters())
    gradients = torch.autograd.grad(losses["total"], [p for _, p in named], allow_unused=True)
    assert len(named) == 191
    assert set(credit["owner_map"]) == {name for name, _ in named}
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert trace["events"] == 3
    assert trace["world_unchanged_by_expression"]
    assert not trace["teacher_forcing_committed"]
    assert set(state) == set(model.state_registry())
    assert credit["native_input_width"] == 8 * model.config.d


def test_teacher_forced_utterance_does_not_mutate_parent_world(model):
    fixture = synthetic_fixture(model.config, torch.float64)
    state = model.set_goal(model.initial(), fixture["goal"])
    state = model.set_records(state, fixture["records"])
    before = {name: state[name].detach().clone() for name in ("H", "F", "p", "v", "s", "l", "logw")}
    utterance = model.begin_utterance(state)
    goal = model.goal(state["goal_raw"])
    for symbol in fixture["target"]:
        _, _, output = model.output(utterance, goal, torch.zeros(15, dtype=torch.float64))
        utterance = model.accept_byte(utterance, symbol, output)
    assert all(torch.equal(state[name], value) for name, value in before.items())
    assert not state["utterance_active"]
    assert utterance["data_bytes"] == len(fixture["target"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "untrusted", "illegal event"),
        ("scope", "other", "illegal event"),
        ("t_arrival", 0.2, "not arrived"),
        ("receipt_id", "wrong", "receipt mismatch"),
    ],
)
def test_subject_rejects_illegal_or_future_event_before_state_change(model, field, value, message):
    state, packet = pending_fixture(model)
    pending_id = state["pending"]["id"]
    with pytest.raises(ValueError, match=message):
        model.receive(state, replace(packet, **{field: value}))
    assert state["pending"]["id"] == pending_id
    assert not state["seen_packets"]
    assert state["world_time"] == 0


def test_unknown_outcome_keeps_receipt_pending_then_settles_once_under_old_goal(model):
    state, packet = pending_fixture(model)
    old_goal = state["pending"]["goal_raw"]
    state = model.set_goal(state, b"new goal")
    unknown = packet.outcome.clone()
    unknown[4:] = 0
    waiting, result = model.receive(state, replace(packet, outcome=unknown))
    assert result == {"status": "awaiting_outcome"}
    assert waiting["pending"]["goal_raw"] == old_goal
    assert not waiting["settled"]
    assert not waiting["seen_packets"]
    repeated, repeated_result = model.receive(waiting, replace(packet, outcome=unknown))
    assert repeated is waiting
    assert repeated_result == result
    resumed, _ = model.receive(waiting, replace(packet, packet_id="completed"))
    assert resumed["pending"] is None
    assert resumed["goal_raw"] == b"new goal"
    assert resumed["evidence"][-1]["old_goal"] == old_goal
    assert all(torch.count_nonzero(value) == 0 for value in resumed["eligibility"].values())
    duplicate, result = model.receive(resumed, replace(packet, packet_id="completed"))
    assert duplicate is resumed and result == {}


def credit_fixture(model):
    credit = torch.zeros(
        (model.config.groups, model.config.d, 8 * model.config.d), dtype=torch.float64
    )
    credit[:, 0, 0] = 3
    credit[:, 1, 1] = 2
    return credit


def test_malformed_birth_is_atomic(model):
    state = model.initial()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    structure = copy.deepcopy(model.structure())
    with pytest.raises(ValueError, match="credit shape"):
        model.birth_units(torch.ones((1, 1), dtype=torch.float64), state=state)
    assert model.structure() == structure
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert state["structure"] == structure


def test_two_neutral_births_preserve_lineage_config_and_old_parameters(model):
    state = model.set_goal(model.initial(), b"birth fixture")
    goal = model.goal(state["goal_raw"])
    action = torch.zeros(15, dtype=torch.float64)
    event = torch.ones((model.config.branches, model.config.d), dtype=torch.float64) * 0.1
    old_output = model.native(state, goal, action, event).detach()
    old_r, old_p = model.native_r.detach().clone(), model.native_p.detach().clone()
    parent_units = model.config.units
    for generation, branch in ((1, 0), (2, 1)):
        model.birth_units(
            credit_fixture(model),
            count=1,
            branch=branch,
            state=state,
            birth_id=f"test-generation-{generation}",
        )
        assert model.config.units == parent_units + generation
        assert model.native_r.shape[1] == model.config.units
        assert model.native_p.shape[2] == model.config.units
        assert torch.equal(model.native_r[:, :parent_units], old_r)
        assert torch.equal(model.native_p[:, :, :parent_units], old_p)
        assert torch.allclose(
            model.native(state, goal, action, event), old_output, atol=1e-10, rtol=0
        )
    assert all(branches[-2:] == [0, 1] for branches in model.unit_branches)
    assert all(len(set(ids)) == model.config.units for ids in model.unit_ids)
    restored, restored_state = U2Subject.from_bundle(model.export_bundle(state))
    assert restored.config.units == parent_units + 2
    assert restored.structure() == model.structure() == restored_state["structure"]


def test_bundle_export_and_import_reject_unregistered_state(model):
    state = model.initial()
    with pytest.raises(ValueError, match="unregistered"):
        model.export_bundle({**state, "unregistered": 1})
    bundle = model.export_bundle(state)
    bundle["state"]["unregistered"] = 1
    with pytest.raises(ValueError, match="registry mismatch"):
        U2Subject.from_bundle(bundle)
