"""Regrouping preserves actual history, gradient ownership and inherited Adam state."""

import itertools

import pytest
import torch

from scripts.training.evidence_a_learner_22_5 import fresh_optimizer
from scripts.training.first_a_support_22_5 import make_model
from scripts.training.first_binding_objective_22_5 import add_gradients, quad_terms
from scripts.training.paired_compute_22_5 import shadow_optimizer
from scripts.training.paired_world_22_5 import build_partition
from scripts.training.probe_first_binding_22_5 import interaction, select_cells
from scripts.training.task_history_loss_22_5 import history_loss
from taiji.u2_evidence_subject import EvidenceSubject
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_task_goal import ActiveQuerySubject


def test_quad_ce_matches_original_actual_history_and_all_parameter_gradients():
    torch.set_num_threads(1)
    model = ActiveQuerySubject.from_parent(EvidenceSubject.from_parent(make_model("native")))
    with torch.no_grad():
        model.byte_head.bias[256] = 20
    partition = build_partition("train")
    cells, _ = select_cells(partition, 1, [0, 1])
    expected = torch.stack(
        [
            history_loss(model, partition, episode, "actual-self")["loss"]
            for row in cells
            for episode in row
        ]
    ).mean()
    expected_gradients = torch.autograd.grad(expected, list(model.parameters()), allow_unused=True)
    result = quad_terms(model, partition, 1, [0, 1])
    assert torch.equal(result["ce"].detach(), expected.detach())
    assert all(item == {"first_hex": "", "ended": True} for item in result["histories"])
    actual_gradients = torch.autograd.grad(
        result["ce"], list(model.parameters()), allow_unused=True
    )
    for actual, wanted in zip(actual_gradients, expected_gradients, strict=True):
        assert (actual is None) == (wanted is None)
        if actual is not None:
            torch.testing.assert_close(actual, wanted, atol=1e-6, rtol=1e-5)


def test_combining_objectives_preserves_none_and_detaches_storage_ownership():
    first = [None, torch.tensor([1.0]), None, torch.tensor([2.0])]
    second = [None, None, torch.tensor([3.0]), torch.tensor([4.0])]
    combined = add_gradients(first, second)
    assert combined[0] is None
    assert [float(item) for item in combined[1:]] == [1, 3, 6]
    assert combined[1].data_ptr() != first[1].data_ptr()
    assert combined[2].data_ptr() != second[2].data_ptr()


def test_full_quad_schedule_regroups_each_original_episode_exactly_once():
    partition = build_partition("train")
    grouped = []
    for position in range(3):
        for colors in itertools.combinations(range(6), 2):
            cells, _ = select_cells(partition, position, colors)
            grouped.extend(episode["id"] for row in cells for episode in row)
    assert len(grouped) == len(set(grouped)) == 180
    assert set(grouped) == {episode["id"] for episode in partition["episodes"]}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_softplus_credit_has_the_declared_sign_and_sequence_scale(dtype):
    coefficient = torch.tensor(0.25, dtype=dtype, requires_grad=True)
    x = torch.tensor([-1, 1], dtype=dtype)
    delta = interaction(coefficient * x[:, None] * x[None, :])
    loss = torch.nn.functional.softplus(-delta)
    actual = torch.autograd.grad(loss, coefficient)[0]
    torch.testing.assert_close(actual, -4 * torch.sigmoid(-delta.detach()))


@pytest.mark.parametrize("zero_first_moment", [False, True])
def test_real_shadow_adam_preserves_parent_and_unused_parameter_clock(zero_first_moment):
    model = torch.nn.Linear(1, 1)
    optimizer = fresh_optimizer(model)
    model(torch.ones(1, 1)).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before_parameters = value_digest(model.state_dict())
    before_optimizer = value_digest(optimizer.state_dict())
    temporary, shadow = shadow_optimizer(model, optimizer)
    before_bias = temporary[1].detach().clone()
    bias_state = {key: value.clone() for key, value in shadow.state[temporary[1]].items()}
    if zero_first_moment:
        for state in shadow.state.values():
            state["exp_avg"].zero_()
        bias_state["exp_avg"].zero_()
    temporary[0].grad = torch.ones_like(temporary[0])
    temporary[1].grad = None
    shadow.step()
    assert torch.equal(temporary[1], before_bias)
    assert_values_equal(shadow.state[temporary[1]], bias_state)
    assert int(shadow.state[temporary[0]]["step"]) == 2
    assert value_digest(model.state_dict()) == before_parameters
    assert value_digest(optimizer.state_dict()) == before_optimizer
