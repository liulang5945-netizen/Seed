"""PLAN67: project the actual Adam proposal, retaining all gradients and moments."""

from __future__ import annotations

import math

import torch

from scripts.training import run_taiji_n1_relation_paired as runner

POLICY = dict(
    schema="n1-five-group-actual-adam-relative-v1",
    relative_limit=0.001,
    norm_floor=1.0,
    groups=["reader", "writer", "local", "context", "relation"],
    moments="unaltered-original-adam",
)


def grouped_parameters(model):
    groups = {key: [] for key in POLICY["groups"]}
    for name, parameter in model.named_parameters():
        label = "writer" if name.startswith("writer.") else runner.core.parameter_group(name)
        groups[label].append((name, parameter))
    if (
        sum(len(value) for value in groups.values()) != 41
        or len(groups["writer"]) != 4
        or not all(groups.values())
        or sum(p.numel() for values in groups.values() for _, p in values) != 62056
    ):
        raise ValueError("five disjoint groups must cover all41 native relation parameters")
    return groups


@torch.no_grad()
def bounded_step(model, optimizer, original_step, *, relative_limit=0.001, audit=False):
    if not math.isfinite(relative_limit) or relative_limit <= 0:
        raise ValueError("positive finite relative limit required")
    groups = grouped_parameters(model)
    before = {name: p.detach().clone() for values in groups.values() for name, p in values}
    gradients = {
        name: None if p.grad is None else p.grad.detach()
        for values in groups.values()
        for name, p in values
    }
    gradient_norm = {
        label: math.sqrt(
            sum(
                float(torch.linalg.vector_norm(p.grad.detach())) ** 2
                for _, p in values
                if p.grad is not None
            )
        )
        for label, values in groups.items()
    }
    returned = original_step()
    moment_before = runner.paired.fingerprint(optimizer.state_dict()) if audit else None
    gradient_before = runner.paired.fingerprint(gradients) if audit else None
    measurements = {}
    for label, values in groups.items():
        parameter_norm = math.sqrt(
            sum(float(torch.linalg.vector_norm(before[name])) ** 2 for name, _ in values)
        )
        deltas = {name: p.detach() - before[name] for name, p in values}
        proposed_norm = math.sqrt(
            sum(float(torch.linalg.vector_norm(delta)) ** 2 for delta in deltas.values())
        )
        if not math.isfinite(proposed_norm):
            raise ValueError("nonfinite actual Adam proposal")
        cap = relative_limit * max(parameter_norm, POLICY["norm_floor"])
        scale = min(1.0, cap / proposed_norm) if proposed_norm else 1.0
        if scale < 1.0:
            for name, parameter in values:
                parameter.copy_(before[name] + deltas[name] * scale)
        actual_norm = math.sqrt(
            sum(
                float(torch.linalg.vector_norm(p.detach() - before[name])) ** 2
                for name, p in values
            )
        )
        if actual_norm > cap * 1.0001 + 1e-6:
            raise ValueError("actual group displacement exceeded cap beyond float tolerance")
        measurements[label] = dict(
            parameter_l2=parameter_norm,
            gradient_l2=gradient_norm[label],
            proposed_l2=proposed_norm,
            cap=cap,
            scale=scale,
            clipped=scale < 1.0,
            actual_l2=actual_norm,
        )
    if audit and (
        runner.paired.fingerprint(optimizer.state_dict()) != moment_before
        or runner.paired.fingerprint(gradients) != gradient_before
    ):
        raise ValueError("proposal projection changed Adam moments or gradients")
    return returned, measurements


def train_step(model, optimizer, item, session, second, *, relative_limit=0.001, audit=False):
    original_step = optimizer.step
    observed = []

    def proposed_step():
        returned, measurement = bounded_step(
            model, optimizer, original_step, relative_limit=relative_limit, audit=audit
        )
        observed.append(measurement)
        return returned

    optimizer.step = proposed_step
    try:
        loss = runner.train_step(model, optimizer, item, session, "position", second)
    finally:
        optimizer.step = original_step
    if len(observed) != 1:
        raise ValueError("exactly one Adam proposal required")
    return loss, observed[0]
