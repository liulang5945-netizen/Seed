"""Accepted parameter values and moment-clock evolution have explicit ownership."""

import pytest
import torch

from scripts.training.credit_update_22_5 import accepted_adam_step
from taiji.u2_history import assert_values_equal, detached_tree


def test_backtracking_accepts_a_bounded_descent_and_records_actual_adam_call():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=10)
    calls = {"optimizer_steps": 0}
    result = accepted_adam_step([parameter], optimizer, lambda: parameter.square(), calls)
    assert result["accepted"] and result["scale"] < 1
    assert result["after"] < result["before"] and result["slope"] < 0
    assert calls["optimizer_steps"] == int(optimizer.state[parameter]["step"]) == 1


def test_finite_rejection_keeps_parameters_and_advances_real_moments():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=10)
    calls = {"optimizer_steps": 0}
    result = accepted_adam_step(
        [parameter], optimizer, lambda: parameter.square(), calls, scales=(1.0,)
    )
    assert not result["accepted"] and float(parameter.detach()) == 1
    assert calls["optimizer_steps"] == int(optimizer.state[parameter]["step"]) == 1
    assert optimizer.state[parameter]["exp_avg"].abs() > 0


def test_opposing_old_momentum_uses_norm_matched_negative_gradient():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    (-100 * parameter).backward()
    optimizer.step()
    calls = {"optimizer_steps": 0}
    result = accepted_adam_step([parameter], optimizer, lambda: 0.001 * parameter.square(), calls)
    assert result["fallback"] and result["accepted"] and result["slope"] < 0
    assert result["after"] < result["before"]


def test_exception_rolls_back_parameters_and_moments_but_keeps_attempt_count():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=10)
    before = detached_tree(optimizer.state_dict())
    calls = {"optimizer_steps": 0}
    objective_calls = []

    def objective():
        objective_calls.append(True)
        if len(objective_calls) > 1:
            raise ValueError("candidate_failed")
        return parameter.square()

    with pytest.raises(ValueError, match="candidate_failed"):
        accepted_adam_step([parameter], optimizer, objective, calls)
    assert float(parameter.detach()) == 1
    assert_values_equal(optimizer.state_dict(), before)
    assert calls["optimizer_steps"] == 1


def test_unused_parameters_keep_their_prior_values_and_clocks():
    used = torch.nn.Parameter(torch.tensor(1.0))
    unused = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.Adam([used, unused], lr=0.1)
    (used.square() + unused.square()).backward()
    optimizer.step()
    before = unused.detach().clone(), detached_tree(optimizer.state[unused])
    accepted_adam_step([used, unused], optimizer, lambda: used.square(), {"optimizer_steps": 0})
    assert_values_equal(unused, before[0])
    assert_values_equal(optimizer.state[unused], before[1])


def test_candidate_rechecks_use_one_noise_draw_and_zero_gradient_is_explicit():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=10)
    torch.manual_seed(99)
    torch.rand(())
    expected = torch.get_rng_state()
    torch.manual_seed(99)
    accepted_adam_step(
        [parameter], optimizer, lambda: parameter.square() + torch.rand(()), {"optimizer_steps": 0}
    )
    assert torch.equal(torch.get_rng_state(), expected)
    calls = {"optimizer_steps": 0}
    result = accepted_adam_step([parameter], optimizer, lambda: 0 * parameter, calls)
    assert result["reason"] == "zero_gradient" and calls["optimizer_steps"] == 0


def test_zero_rate_proposal_advances_moments_without_claiming_a_parameter_update():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0)
    calls = {"optimizer_steps": 0}
    result = accepted_adam_step([parameter], optimizer, lambda: parameter.square(), calls)
    assert not result["accepted"] and result["reason"] == "zero_parameter_proposal"
    assert calls["optimizer_steps"] == int(optimizer.state[parameter]["step"]) == 1
    assert float(parameter.detach()) == 1
