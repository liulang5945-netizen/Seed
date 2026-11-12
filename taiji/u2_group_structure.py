"""Actual group-local residual units and inherited optimizer ownership for U2."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import torch
from torch import nn

from .u2_history import detached_tree
from .u2_operators import choose_left_directions


class GroupResidual(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.incoming = nn.Parameter(incoming)
        self.outgoing = nn.Parameter(outgoing)


def group_widths(config):
    return (
        tuple(config.group_units)
        if config.group_units is not None
        else (config.units,) * config.groups
    )


def validate_structure(model):
    widths = group_widths(model.c)
    base = model.c.base_units or model.c.units
    if model.native_r.shape != (model.c.groups, base, 8 * model.c.d) or model.native_p.shape != (
        model.c.groups,
        model.c.d,
        base,
    ):
        raise ValueError("Base parameters differ from the structural registry")
    if len(model.unit_ids) != model.c.groups or len(model.unit_branches) != model.c.groups:
        raise ValueError("Invalid group lineage count")
    if any(
        len(ids) != width
        or len(branches) != width
        or any(branch not in {-1, 0, 1} for branch in branches)
        for ids, branches, width in zip(model.unit_ids, model.unit_branches, widths, strict=True)
    ):
        raise ValueError("Unit lineage differs from actual group widths")
    ids = [value for group in model.unit_ids for value in group]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Unit identities must be unique")
    expected = {str(group) for group, width in enumerate(widths) if width > base}
    if set(model.group_extensions) != expected:
        raise ValueError("Group extension registry mismatch")
    for key, extension in model.group_extensions.items():
        extra = widths[int(key)] - base
        if extension.incoming.shape != (extra, 8 * model.c.d) or extension.outgoing.shape != (
            model.c.d,
            extra,
        ):
            raise ValueError("Extension parameters differ from actual widths")


def install_extensions(model):
    model.group_extensions = nn.ModuleDict()
    base = model.c.base_units or model.c.units
    for group, width in enumerate(group_widths(model.c)):
        extra = width - base
        if extra:
            outgoing = model.native_w.new_empty(model.c.d, extra)
            nn.init.normal_(outgoing, std=0.1 / math.sqrt(model.c.d))
            model.group_extensions[str(group)] = GroupResidual(
                model.native_w.new_zeros(extra, 8 * model.c.d),
                outgoing,
            )


def add_group_residuals(model, pre, inputs, content_gates):
    result = pre.clone()
    base = model.native_r.shape[1]
    for key, extension in model.group_extensions.items():
        group = int(key)
        gates = torch.stack(
            [
                (
                    torch.ones_like(content_gates[:, group, 0])
                    if branch < 0
                    else content_gates[:, group, branch]
                )
                for branch in model.unit_branches[group][base:]
            ],
            -1,
        )
        hidden = (inputs[:, group] @ extension.incoming.T).tanh() * gates
        result[:, group] = result[:, group] + hidden @ extension.outgoing.T
    return result


def ancestry_compatible(root, config):
    current = asdict(config)
    # Old value-only bundles predate these optional graph fields.
    original = dict(root)
    for key in ("record_metadata", "separate_requests", "base_units", "group_units"):
        original.setdefault(key, False if key in {"record_metadata", "separate_requests"} else None)
    ignored = {"units", "base_units", "group_units"}
    if set(original) != set(current) or any(
        value != current[name] for name, value in original.items() if name not in ignored
    ):
        return False
    before = (
        tuple(original["group_units"])
        if original["group_units"] is not None
        else (original["units"],) * config.groups
    )
    after = group_widths(config)
    if original["group_units"] is not None and config.group_units is None:
        return False
    return (
        len(before) == len(after)
        and all(old <= new for old, new in zip(before, after, strict=True))
        and (original["base_units"] or original["units"]) <= (config.base_units or config.units)
    )


def _owner(optimizer, parameter):
    matches = [
        group
        for group in optimizer.param_groups
        if any(value is parameter for value in group["params"])
    ]
    if len(matches) != 1:
        raise ValueError("Group birth needs a unique inherited optimizer owner")
    return matches[0]


def _moments(optimizer, old, new, axis):
    result = {}
    for key, value in optimizer.state.get(old, {}).items():
        if isinstance(value, torch.Tensor) and value.shape == old.shape:
            target = torch.zeros_like(new)
            slices = [slice(None)] * new.ndim
            slices[axis] = slice(0, old.shape[axis])
            target[tuple(slices)] = value
            result[key] = target
        else:
            result[key] = detached_tree(value)
    return result


def birth_groups(model, credits, *, groups, count, branch, state, optimizer, birth_id):
    selected = tuple(groups)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(group) is not int or not 0 <= group < model.c.groups for group in selected)
    ):
        raise ValueError("Invalid growth groups")
    if credits.shape != (model.c.groups, model.c.d, 8 * model.c.d):
        raise ValueError("Group birth credit shape mismatch")
    identity = birth_id or f"birth-{model.generation+1}"
    if any(value.startswith(identity + ":g") for row in model.unit_ids for value in row):
        raise ValueError("Birth identity has already been committed")
    if state is not None and (
        state["pending"] is not None
        or state["utterance_active"]
        or state["revision"] != model.parameter_revision
        or state["structure"] != model.structure()
    ):
        raise ValueError("Group birth requires the current committed boundary")
    proposals = []
    for group in selected:
        directions = choose_left_directions(credits[group], count)
        if directions["effective_budget"] != count:
            raise ValueError("insufficient independent group credit")
        old = dict(model.group_extensions.items()).get(str(group))
        incoming = model.native_w.new_zeros(count, 8 * model.c.d)
        outgoing = directions["directions"] * 0.1
        extension = GroupResidual(
            incoming if old is None else torch.cat([old.incoming.detach(), incoming]),
            outgoing if old is None else torch.cat([old.outgoing.detach(), outgoing], -1),
        )
        owners = []
        if optimizer is not None:
            for old_parameter, new_parameter, base_parameter, axis in [
                (None if old is None else old.incoming, extension.incoming, model.native_r, 0),
                (None if old is None else old.outgoing, extension.outgoing, model.native_p, 1),
            ]:
                owner = _owner(
                    optimizer, base_parameter if old_parameter is None else old_parameter
                )
                values = (
                    {}
                    if old_parameter is None
                    else _moments(optimizer, old_parameter, new_parameter, axis)
                )
                owners.append((owner, old_parameter, new_parameter, values))
        proposals.append((group, extension, owners))
    widths = list(group_widths(model.c))
    base = model.c.base_units or model.c.units
    for group, extension, owners in proposals:
        model.group_extensions[str(group)] = extension
        widths[group] += count
        model.unit_ids[group].extend(f"{identity}:g{group}:u{unit}" for unit in range(count))
        model.unit_branches[group].extend([branch] * count)
        for owner, old_parameter, new_parameter, values in owners:
            if old_parameter is None:
                owner["params"].append(new_parameter)
            else:
                owner["params"] = [
                    new_parameter if value is old_parameter else value for value in owner["params"]
                ]
                optimizer.state.pop(old_parameter, None)
            optimizer.state[new_parameter] = values
    model.group_extensions = nn.ModuleDict(
        dict(sorted(model.group_extensions.items(), key=lambda item: int(item[0])))
    )
    model.c = replace(model.c, units=max(widths), base_units=base, group_units=tuple(widths))
    model.generation += 1
    model.parameter_revision += 1
    if state is not None:
        state["revision"] = model.parameter_revision
        state["structure"] = model.structure()
        state["history"] = [
            {
                **row,
                "parameter_revision": model.parameter_revision,
                "neutral_birth_rebase": identity,
            }
            for row in state["history"]
        ]
    return {
        "birth_id": identity,
        "generation": model.generation,
        "parent_units": max(
            width - count if group in selected else width for group, width in enumerate(widths)
        ),
        "units": model.c.units,
        "group_units": tuple(widths),
        "groups": selected,
        "count": count,
    }
