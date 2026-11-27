"""Aligned supervision preserves byte inference and exposes gradient ownership."""

import copy

import pytest
import torch

from scripts.training import calibrate_content_credit_22_5 as auxiliary
from scripts.training import calibrate_first_round_22_5 as original
from scripts.training.first_round_baseline_22_5 import (
    FirstRoundBaseline,
    generate,
    loss_and_logits,
    make_batch,
)
from scripts.training.paired_world_22_5 import build_partition
from taiji.u2_history import assert_values_equal


@pytest.fixture(scope="module")
def batch():
    torch.set_num_threads(1)
    return make_batch(build_partition("train"))


def test_labels_and_balancing_are_derived_only_from_raw_training_answers(batch):
    labels, classes, weights = auxiliary.labels_for(batch)
    assert len(labels) == 4
    for index in range(4):
        torch.testing.assert_close(weights[index] * (classes == index).sum(), torch.tensor(45.0))
    torch.testing.assert_close(weights[classes].mean(), torch.tensor(1.0))


def test_byte_loss_and_control_adam_are_exactly_the_original_recipe(batch):
    model = FirstRoundBaseline()
    control = copy.deepcopy(model)
    head = auxiliary.head_for(2026111310, 4)
    _, classes, weights = auxiliary.labels_for(batch)
    optimizer = original.optimizer_for(model)
    optimizer.add_param_group({"params": list(head.parameters())})
    old_optimizer = original.optimizer_for(control)
    current = auxiliary.terms(model, head, batch, classes, weights)
    loss, logp = loss_and_logits(control, batch)
    assert_values_equal(current["logp"], logp)
    assert_values_equal(current["byte"], loss)
    current["byte"].backward()
    loss.backward()
    assert all(parameter.grad is None for parameter in head.parameters())
    for first, second in zip(model.parameters(), control.parameters(), strict=True):
        assert_values_equal(first.grad, second.grad)
    optimizer.step()
    old_optimizer.step()
    assert_values_equal(model.state_dict(), control.state_dict())
    state = optimizer.state_dict()
    assert_values_equal(
        {"state": state["state"], "param_groups": state["param_groups"][:1]},
        old_optimizer.state_dict(),
    )


def test_content_credit_reaches_shared_context_and_keeps_decoder_ownership_explicit(batch):
    model = FirstRoundBaseline()
    head = auxiliary.head_for(2026111310, 4)
    _, classes, weights = auxiliary.labels_for(batch)
    result = auxiliary.terms(model, head, batch, classes, weights)
    parameters = list(model.named_parameters())
    gradients = torch.autograd.grad(result["content"], list(model.parameters()), allow_unused=True)
    for (name, _), value in zip(parameters, gradients, strict=True):
        if name.startswith(("gru.", "output.")):
            assert value is None
        else:
            assert (
                value is not None and bool(torch.isfinite(value).all()) and bool(value.norm() > 0)
            )


def test_compound_checkpoint_restores_both_moments_and_all_gradients(batch):
    model = FirstRoundBaseline()
    labels, classes, weights = auxiliary.labels_for(batch)
    head = auxiliary.head_for(2026111310, 4)
    optimizer = original.optimizer_for(model)
    optimizer.add_param_group({"params": list(head.parameters())})
    auxiliary.terms(model, head, batch, classes, weights)["total"].backward()
    optimizer.step()
    contract = {"head_seed": 2026111310, "data_digest": "fixture"}
    payload = auxiliary.snapshot(model, head, optimizer, 1, contract, batch, labels, weights, 0.1)
    restored, other_head, other_optimizer = auxiliary.restore(payload, contract, batch)
    assert_values_equal(optimizer.state_dict(), other_optimizer.state_dict())
    assert_values_equal(
        auxiliary.measure(model, head, batch, classes, weights),
        auxiliary.measure(restored, other_head, batch, classes, weights),
    )
    assert_values_equal(generate(model, batch), generate(restored, batch))
