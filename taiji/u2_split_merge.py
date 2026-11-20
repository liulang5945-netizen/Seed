"""Shared-unit split and merge on top of the core NeuralUnits (new module; core untouched)."""

from __future__ import annotations

import copy
from typing import Any

import torch

from taiji.u2_neural_birth import live_entry_birth
from taiji.u2_neural_units import LEGAL_INPUT_CHANNELS, NeuralUnits


def split_unit(
    units: NeuralUnits, parent_index: int, group_id: int = 0
) -> tuple[NeuralUnits, dict[str, Any]]:
    parent_id = units.identities["units"][parent_index]["unit_id"]
    grown, detail = live_entry_birth(units, group_id, [parent_id])
    child_index = grown.n_units - 1
    child_id = grown.identities["units"][child_index]["unit_id"]
    grown.set_lifecycle(child_id, "mature")
    with torch.no_grad():
        column = units.p[:, parent_index] / 2.0
        grown.p[:, parent_index] = column
        grown.p[:, child_index] = column
        grown.c[child_index] = grown.c[parent_index]
        grown.eta[child_index] = grown.eta[parent_index]
        grown.b[child_index] = grown.b[parent_index]
    return grown, {"parent": parent_index, "child": child_index, "detail": detail}


def merge_units(units: NeuralUnits, keep: int, drop: int) -> tuple[NeuralUnits, dict[str, Any]]:
    identities = copy.deepcopy(units.identities)
    kept_id = identities["units"][keep]["unit_id"]
    dropped_id = identities["units"][drop]["unit_id"]
    remaining = [unit for position, unit in enumerate(identities["units"]) if position != drop]
    kept_position = keep if keep < drop else keep - 1
    identities["units"] = remaining
    identities["edges"] = [
        edge
        for edge in identities["edges"]
        if edge["target"] != dropped_id and edge["source"] != dropped_id
    ]
    merged = NeuralUnits(units.spec, identities)
    sources = [position if position < drop else position + 1 for position in range(merged.n_units)]
    with torch.no_grad():
        for position, source in enumerate(sources):
            if position == kept_position:
                continue
            merged.p[:, position] = units.p[:, source]
            merged.r[position] = units.r[source]
            for name in ("c", "eta", "b"):
                getattr(merged, name)[position] = getattr(units, name)[source]
            for other, other_source in enumerate(sources):
                merged.w[position, other] = units.w[source, other_source]
        merged.p[:, kept_position] = units.p[:, keep] + units.p[:, drop]
        merged.r[kept_position] = (units.r[keep] + units.r[drop]) / 2.0
        for name in ("c", "eta", "b"):
            source = getattr(units, name)
            getattr(merged, name)[kept_position] = (source[keep] + source[drop]) / 2.0
        for position, source in enumerate(sources):
            if position == kept_position:
                continue
            merged.w[position, kept_position] = units.w[source, keep] + units.w[source, drop]
            merged.w[kept_position, position] = (
                units.w[keep, source] + units.w[drop, source]
            ) / 2.0
    merged.project_parameters_()
    return merged, {"kept": keep, "dropped": drop, "kept_id": kept_id, "dropped_id": dropped_id}


def train_pair(graph, features, labels, pair: list[int], steps: int, lr: float, generator) -> None:
    units = graph.units
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        batch = torch.randint(0, features.shape[0], (64,), generator=generator)
        gates = torch.ones(batch.numel(), units.n_units)
        units.zero_grad(set_to_none=True)
        batch_channels = {
            name: features[batch, index * 8 : (index + 1) * 8]
            for index, name in enumerate(LEGAL_INPUT_CHANNELS)
        }
        logits = graph(batch_channels, gates, graph.initial_state())["readout"]
        loss = loss_fn(logits, labels[batch])
        loss.backward()
        with torch.no_grad():
            for position in pair:
                if units.p.grad is not None:
                    units.p[:, position] -= lr * units.p.grad[:, position]
                if units.r.grad is not None:
                    units.r[position, :] -= lr * units.r.grad[position, :]
                for name in ("c", "eta", "b"):
                    parameter = getattr(units, name)
                    if parameter.grad is not None:
                        parameter[position] -= lr * parameter.grad[position]
        units.zero_grad(set_to_none=True)
        units.project_parameters_()
