"""Train-objective-accepted Adam proposals for an isolated calibration learner."""

from __future__ import annotations

import math

import torch

from taiji.u2_history import detached_tree


def accepted_adam_step(
    parameters,
    optimizer,
    objective,
    counters,
    *,
    scales=(1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125),
    armijo=1e-4,
    tolerance=1e-6,
):
    """Keep finite proposal moments, selecting a sufficient-decrease parameter step.

    Call on a single-writer, isolated learner. Evaluation uses identical CPU RNG.
    Rejected finite proposals advance moments. Exceptions restore the prior state.
    """
    scales = tuple(scales)
    if (
        not 0 < armijo < 1
        or not math.isfinite(tolerance)
        or tolerance < 0
        or not scales
        or any(not math.isfinite(scale) or not 0 < scale <= 1 for scale in scales)
    ):
        raise ValueError("invalid_credit_update_search")
    parameters = list(parameters)
    tracked = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert len(parameters) == len(tracked) and {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in tracked
    }
    original = [parameter.detach().clone() for parameter in parameters]
    old_optimizer = detached_tree(optimizer.state_dict())
    before_rng = torch.get_rng_state()
    after_rng = before_rng
    try:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        assert bool(torch.isfinite(loss))
        loss.backward()
        after_rng = torch.get_rng_state()
        gradients = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        ]
        assert all(value is None or bool(torch.isfinite(value).all()) for value in gradients)
        gradient_norm = math.sqrt(
            sum(float(value.double().square().sum()) for value in gradients if value is not None)
        )
        before = float(loss.detach())
        if gradient_norm == 0:
            return {
                "accepted": False,
                "scale": 0.0,
                "before": before,
                "after": before,
                "fallback": False,
                "evaluations": 0,
                "reason": "zero_gradient",
            }
        counters["optimizer_steps"] += 1
        optimizer.step()
        assert all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
        assert all(
            bool(torch.isfinite(value).all())
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
        direction = [
            parameter.detach().double() - old.double()
            for parameter, old in zip(parameters, original, strict=True)
        ]
        step_norm = math.sqrt(sum(float(value.double().square().sum()) for value in direction))
        slope = sum(
            float((value.double() * move.double()).sum())
            for value, move in zip(gradients, direction, strict=True)
            if value is not None
        )
        assert math.isfinite(step_norm) and math.isfinite(slope)
        if step_norm == 0:
            return {
                "accepted": False,
                "scale": 0.0,
                "before": before,
                "after": before,
                "fallback": False,
                "evaluations": 0,
                "reason": "zero_parameter_proposal",
            }
        fallback = slope >= 0
        if fallback:
            direction = [
                (
                    torch.zeros_like(parameter, dtype=torch.float64)
                    if value is None
                    else -value.double() * (step_norm / gradient_norm)
                )
                for parameter, value in zip(parameters, gradients, strict=True)
            ]
            slope = sum(
                float((value.double() * move.double()).sum())
                for value, move in zip(gradients, direction, strict=True)
                if value is not None
            )
        accepted, selected_scale, after, evaluations = False, 0.0, before, 0
        with torch.no_grad():
            for scale in scales:
                for parameter, old, move in zip(parameters, original, direction, strict=True):
                    parameter.copy_(old.double() + scale * move)
                if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
                    continue
                torch.set_rng_state(before_rng)
                candidate = objective()
                evaluations += 1
                value = float(candidate)
                moved = any(
                    bool((parameter.detach() != old).any())
                    for parameter, old in zip(parameters, original, strict=True)
                )
                if (
                    moved
                    and math.isfinite(value)
                    and value <= before + armijo * scale * slope + tolerance
                ):
                    accepted, selected_scale, after = True, scale, value
                    break
            if not accepted:
                for parameter, old in zip(parameters, original, strict=True):
                    parameter.copy_(old)
        return {
            "accepted": accepted,
            "scale": selected_scale,
            "before": before,
            "after": after,
            "fallback": fallback,
            "evaluations": evaluations,
            "slope": slope,
            "gradient_norm": gradient_norm,
            "proposal_norm": step_norm,
            "reason": "accepted" if accepted else "finite_search_rejected",
        }
    except Exception:
        with torch.no_grad():
            for parameter, old in zip(parameters, original, strict=True):
                parameter.copy_(old)
        optimizer.load_state_dict(old_optimizer)
        raise
    finally:
        torch.set_rng_state(after_rng)
