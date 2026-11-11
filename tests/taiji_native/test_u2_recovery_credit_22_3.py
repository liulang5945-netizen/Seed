"""U2 23.1 pending-result credit regressions, with zero optimizer steps."""

from __future__ import annotations

import io
from dataclasses import replace

import pytest
import torch

from taiji.u2_subject import U2Config, U2Packet, U2Subject, synthetic_fixture


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


def pending_fixture(model):
    fixture = synthetic_fixture(model.config, torch.float64)
    state = model.set_goal(model.initial("credit-recovery-fixture"), fixture["goal"])
    state = model.set_records(state, fixture["records"])
    legal = torch.zeros(9, dtype=torch.bool)
    legal[0] = True
    state, decision = model.choose_operation(state, legal)
    state, receipt_id = model.issue(state, decision)
    state["process_time"] = 0.05
    event = fixture["events"][0]
    packet = U2Packet(
        "credit-packet",
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


def roundtrip(bundle):
    buffer = io.BytesIO()
    torch.save(bundle, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def critic_gradients(model, loss):
    parameters = list(model.critic.parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    assert all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
    assert any(float(value.norm()) > 0 for value in gradients)
    return gradients


def test_pending_bundle_keeps_branchwise_critic_inputs_and_declares_body_boundary(model):
    state, _ = pending_fixture(model)
    decision = state["pending"]["decision"]
    assert decision["critic_features"].shape == (model.config.branches, 2 * model.config.d)
    assert decision["critic_weights"].shape == (model.config.branches,)
    assert torch.allclose(decision["critic_weights"].sum(), torch.tensor(1.0, dtype=torch.float64))
    assert decision["psi_version"] == model.parameter_revision
    bundle = model.export_bundle(state)
    assert bundle["schema"] == "taiji-u2-subject-v2"
    assert bundle["credit_contract"]["body_graph"] == "values_only_requires_window_replay"
    assert not bundle["state"]["pending"]["decision"]["value"].requires_grad


def test_pending_critic_credit_matches_after_value_only_serialization(model):
    state, packet = pending_fixture(model)
    restored, restored_state = U2Subject.from_bundle(roundtrip(model.export_bundle(state)))
    assert torch.equal(
        state["pending"]["decision"]["value"].detach(),
        restored_state["pending"]["decision"]["value"],
    )
    _, live = model.receive(state, packet)
    _, resumed = restored.receive(restored_state, packet)
    assert torch.allclose(live["critic"], resumed["critic"], atol=1e-12, rtol=1e-10)
    expected = critic_gradients(model, live["critic"])
    actual = critic_gradients(restored, resumed["critic"])
    for left, right in zip(expected, actual, strict=True):
        assert torch.allclose(left, right, atol=1e-12, rtol=1e-10)


def test_cached_value_only_mutant_has_no_critic_credit(model):
    state, packet = pending_fixture(model)
    restored, restored_state = U2Subject.from_bundle(roundtrip(model.export_bundle(state)))
    cached = restored_state["pending"]["decision"]["value"]
    mutant_loss = 0.5 * (cached.new_tensor(0.1) - cached).square()
    assert not mutant_loss.requires_grad
    _, received = restored.receive(restored_state, packet)
    assert received["critic"].requires_grad
    critic_gradients(restored, received["critic"])


def test_actor_eligibility_credit_matches_after_value_serialization(model):
    state, packet = pending_fixture(model)
    restored, restored_state = U2Subject.from_bundle(roundtrip(model.export_bundle(state)))
    _, live = model.receive(state, packet)
    _, resumed = restored.receive(restored_state, packet)
    before = torch.autograd.grad(live["actor"], [p for _, p in model.policy_named_parameters()])
    after = torch.autograd.grad(
        resumed["actor"], [p for _, p in restored.policy_named_parameters()]
    )
    assert any(float(value.norm()) > 0 for value in before)
    for left, right in zip(before, after, strict=True):
        assert bool(torch.isfinite(right).all())
        assert torch.allclose(left, right, atol=1e-12, rtol=1e-10)


def test_branchwise_value_cannot_be_replaced_by_value_of_flat_features(model):
    features = torch.stack(
        [
            torch.full((2 * model.config.d,), -2.0, dtype=torch.float64),
            torch.full((2 * model.config.d,), 4.0, dtype=torch.float64),
        ]
    )
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    expected = (weights * 2 * model.critic(features).tanh().squeeze(-1)).sum()
    decision = {
        "critic_features": features,
        "critic_weights": weights,
        "psi_version": model.parameter_revision,
        "value": expected.detach().clone(),
    }
    actual = model.pending_value(decision)
    flattened = (weights[:, None] * features).sum(0)
    incorrect = 2 * model.critic(flattened).tanh().squeeze()
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-10)
    assert float((actual - incorrect).detach().abs()) > 1e-3


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_features",
        "wrong_features",
        "wrong_weights",
        "negative_weights",
        "unnormalized_weights",
        "wrong_version",
    ],
)
def test_pending_critic_context_rejects_missing_or_mismatched_input(model, mutation):
    state, _ = pending_fixture(model)
    bundle = model.export_bundle(state)
    decision = bundle["state"]["pending"]["decision"]
    if mutation == "missing_features":
        del decision["critic_features"]
    elif mutation == "wrong_features":
        decision["critic_features"] = decision["critic_features"][:, :-1]
    elif mutation == "wrong_weights":
        decision["critic_weights"] = decision["critic_weights"][:-1]
    elif mutation == "negative_weights":
        decision["critic_weights"][0] = -1.0
    elif mutation == "unnormalized_weights":
        decision["critic_weights"] *= 2
    else:
        decision["psi_version"] += 1
    with pytest.raises(ValueError, match="(?i)(critic|pending|weight|version)"):
        U2Subject.from_bundle(bundle)


def test_goal_switch_and_unknown_result_preserve_old_credit_after_restore(model):
    state, packet = pending_fixture(model)
    old_goal = state["pending"]["goal_raw"]
    state = model.set_goal(state, b"a new goal while old action is pending")
    unknown = packet.outcome.clone()
    unknown[4:] = 0
    waiting, result = model.receive(state, replace(packet, outcome=unknown))
    assert result == {"status": "awaiting_outcome"}
    restored, restored_state = U2Subject.from_bundle(roundtrip(model.export_bundle(waiting)))
    assert restored_state["pending"]["goal_raw"] == old_goal
    assert restored_state["goal_raw"] == state["goal_raw"]
    complete = replace(packet, packet_id="credit-complete")
    _, expected = model.receive(waiting, complete)
    final, actual = restored.receive(restored_state, complete)
    assert torch.allclose(expected["critic"], actual["critic"], atol=1e-12, rtol=1e-10)
    critic_gradients(restored, actual["critic"])
    assert all(not bool(value.any()) for value in final["eligibility"].values())
    assert final["evidence"][-1]["old_goal"] == old_goal
    duplicate, receipt = restored.receive(final, complete)
    assert duplicate is final and receipt == {}


def test_terminal_result_has_no_bootstrap_in_reconstructed_critic(model):
    state, packet = pending_fixture(model)
    bundle = model.export_bundle(state)
    restored, restored_state = U2Subject.from_bundle(roundtrip(bundle))
    decision = restored_state["pending"]["decision"]
    old_value = (
        decision["critic_weights"]
        * 2
        * restored.critic(decision["critic_features"]).tanh().squeeze(-1)
    ).sum()
    outcome = packet.outcome.clone()
    outcome[3] = 1
    _, received = restored.receive(restored_state, replace(packet, outcome=outcome))
    discount = 0.99 ** max(1.0, restored_state["pending"]["dt"] / 0.05)
    utility = float(outcome.new_tensor([-1.0, 0.0, 1.0]) @ outcome[:3])
    expected_loss = 0.5 * ((1 - discount) * utility - old_value).square()
    assert torch.allclose(received["critic"], expected_loss, atol=1e-12, rtol=1e-10)


def test_legacy_v1_values_without_pending_can_migrate(model):
    state = model.initial("legacy-values")
    bundle = model.export_bundle(state)
    bundle["schema"] = "taiji-u2-subject-v1"
    bundle.pop("credit_contract", None)
    restored, restored_state = U2Subject.from_bundle(roundtrip(bundle))
    assert restored_state["subject_id"] == state["subject_id"]
    assert restored.export_bundle(restored_state)["schema"] == "taiji-u2-subject-v2"


def test_legacy_v1_pending_is_not_silently_upgraded(model):
    state, _ = pending_fixture(model)
    bundle = model.export_bundle(state)
    bundle["schema"] = "taiji-u2-subject-v1"
    bundle.pop("credit_contract", None)
    for name in ("critic_features", "critic_weights", "psi_version"):
        bundle["state"]["pending"]["decision"].pop(name, None)
    with pytest.raises(ValueError, match="(?is)(pending.*v1|v1.*pending)"):
        U2Subject.from_bundle(roundtrip(bundle))


def test_value_only_body_restore_does_not_claim_motion_credit_recovery(model):
    state, packet = pending_fixture(model)
    restored, restored_state = U2Subject.from_bundle(roundtrip(model.export_bundle(state)))
    _, live = model.receive(state, packet)
    _, resumed = restored.receive(restored_state, packet)
    original = torch.autograd.grad(
        live["predictive_logp"].sum(), model.motion.first.weight, allow_unused=True
    )[0]
    detached = torch.autograd.grad(
        resumed["predictive_logp"].sum(), restored.motion.first.weight, allow_unused=True
    )[0]
    assert original is not None and bool(torch.isfinite(original).all())
    assert float(original.norm()) > 0
    assert detached is None
