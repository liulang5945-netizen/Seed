"""Shared relative trust region for the reader/writer/local Adam proposal block."""

from __future__ import annotations

import math

import torch

from scripts.training import _taiji_n1_group_relative_adam as bounded
from scripts.training import run_taiji_n1_relation_paired as runner

POLICY = {
    "schema": "n1-coupled-reader-writer-local-relative-adam-v1",
    "relative_limit": 0.001,
    "norm_floor": 1.0,
    "groups": ["reader", "writer", "local", "context", "relation"],
    "shared_block": ["reader", "writer", "local"],
    "shared_block_rule": "one_scale_for_combined_actual_adam_proposal",
    "moments": "unaltered-original-adam",
}


def project_deltas(
    groups: dict[str, list[tuple[str, torch.Tensor]]],
    before: dict[str, torch.Tensor],
    proposed: dict[str, torch.Tensor],
    *,
    relative_limit: float = POLICY["relative_limit"],
) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    if not math.isfinite(relative_limit) or relative_limit <= 0:
        raise ValueError("positive finite relative limit required")
    if tuple(groups) != tuple(POLICY["groups"]):
        raise ValueError("five frozen parameter groups required in order")
    names = [name for values in groups.values() for name, _ in values]
    if len(names) != len(set(names)) or set(names) != set(before) or set(names) != set(proposed):
        raise ValueError("parameter groups must form a complete disjoint partition")
    for name in names:
        if before[name].shape != proposed[name].shape or before[name].dtype != proposed[name].dtype:
            raise ValueError(f"parameter proposal shape/dtype mismatch: {name}")
        if not torch.isfinite(before[name]).all() or not torch.isfinite(proposed[name]).all():
            raise ValueError(f"nonfinite parameter/proposal: {name}")

    block_names = {name for label in POLICY["shared_block"] for name, _ in groups[label]}
    block_parameter_norm = math.sqrt(
        sum(float(torch.linalg.vector_norm(before[name])) ** 2 for name in block_names)
    )
    block_proposed_norm = math.sqrt(
        sum(float(torch.linalg.vector_norm(proposed[name])) ** 2 for name in block_names)
    )
    block_cap = relative_limit * max(block_parameter_norm, POLICY["norm_floor"])
    block_scale = min(1.0, block_cap / block_proposed_norm) if block_proposed_norm else 1.0

    projected: dict[str, torch.Tensor] = {}
    measurements: dict[str, dict] = {}
    for label, values in groups.items():
        parameter_norm = math.sqrt(
            sum(float(torch.linalg.vector_norm(before[name])) ** 2 for name, _ in values)
        )
        proposed_norm = math.sqrt(
            sum(float(torch.linalg.vector_norm(proposed[name])) ** 2 for name, _ in values)
        )
        shared = label in POLICY["shared_block"]
        if shared:
            cap, scale, scope = block_cap, block_scale, "shared_rwl_block"
        else:
            cap = relative_limit * max(parameter_norm, POLICY["norm_floor"])
            scale = min(1.0, cap / proposed_norm) if proposed_norm else 1.0
            scope = "per_group"
        actual_norm = 0.0
        for name, _ in values:
            delta = proposed[name] * scale
            projected[name] = delta
            actual_norm += float(torch.linalg.vector_norm(delta)) ** 2
        actual_norm = math.sqrt(actual_norm)
        if actual_norm > cap * 1.0001 + 1e-6:
            raise ValueError(f"projected displacement exceeded {scope} cap: {label}")
        measurements[label] = {
            "parameter_l2": parameter_norm,
            "proposed_l2": proposed_norm,
            "cap": cap,
            "cap_scope": scope,
            "scale": scale,
            "clipped": scale < 1.0,
            "actual_l2": actual_norm,
            **(
                {
                    "shared_block_parameter_l2": block_parameter_norm,
                    "shared_block_proposed_l2": block_proposed_norm,
                    "shared_block_cap": block_cap,
                    "shared_block_scale": block_scale,
                }
                if shared
                else {}
            ),
        }
    return projected, measurements


@torch.no_grad()
def bounded_step(model, optimizer, original_step, *, relative_limit=0.001, audit=False):
    groups = bounded.grouped_parameters(model)
    before = {
        name: parameter.detach().clone() for values in groups.values() for name, parameter in values
    }
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for values in groups.values()
        for name, parameter in values
    }
    gradient_norm = {
        label: math.sqrt(
            sum(
                float(torch.linalg.vector_norm(parameter.grad.detach())) ** 2
                for _, parameter in values
                if parameter.grad is not None
            )
        )
        for label, values in groups.items()
    }
    returned = original_step()
    moments_after_step = runner.paired.fingerprint(optimizer.state_dict()) if audit else None
    gradients_after_step = runner.paired.fingerprint(gradients) if audit else None
    proposed = {
        name: parameter.detach() - before[name]
        for values in groups.values()
        for name, parameter in values
    }
    projected, measurements = project_deltas(
        groups, before, proposed, relative_limit=relative_limit
    )
    for label, values in groups.items():
        for name, parameter in values:
            parameter.copy_(before[name] + projected[name])
        measurements[label]["gradient_l2"] = gradient_norm[label]
    if audit and (
        runner.paired.fingerprint(optimizer.state_dict()) != moments_after_step
        or runner.paired.fingerprint(gradients) != gradients_after_step
    ):
        raise ValueError("shared proposal projection changed Adam moments or gradients")
    return returned, measurements
