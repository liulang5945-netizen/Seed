"""The subject's own descriptor-read probe, lifted out of the 10.21.1 runner.

14.3.0: one implementation, two users. ``verify_taiji_10_21_1.py`` imported to keep its
own copies; this module is now the single source and the runner imports it. The anchor
test in ``tests/taiji_native/test_composition_read_probe_anchor.py`` proves the lifted
code reproduces the fifteen published rows of 10.21.1 bit-for-bit, so nothing about that
card's readings changed when the code moved.

The read is ``EpisodicRead(state.h, goal_vector(descriptor), view(records))`` -- the
model's own retrieval path, with the per-group attention split into record masses so a
"did the query pick a record" question can be answered without touching any parameter.
"""

from __future__ import annotations

import numpy as np
import torch

from taiji.living_core import RawSource
from taiji.slot_composition_head import read_slot

TIE = 1e-12


def record_spans(records: tuple[RawSource, ...]) -> list[tuple[int, int]]:
    spans, start = [], 0
    for record in records:
        end = start + len(record.raw)
        spans.append((start, end))
        start = end
    return spans


def descriptor_read(model, state, descriptor: str, records: tuple[RawSource, ...]) -> dict:
    slot, attention = read_slot(model, state, descriptor, records)
    groups = model.config.groups
    weights = attention.detach().reshape(groups, -1)
    spans = record_spans(records)
    record_mass = torch.stack(
        [weights[:, start:end].sum(dim=1) for start, end in spans]
    )  # (records, groups)
    null_mass = weights[:, -1]
    mean_mass = record_mass.mean(dim=1)
    mean_null = null_mass.mean()
    targets = [
        index for index, record in enumerate(records) if record.raw.startswith(descriptor.encode())
    ]
    if len(targets) != 1:
        raise ValueError(f"descriptor {descriptor!r} does not own exactly one record")
    target = targets[0]

    def top_one(values: list[float]):
        best = max(values)
        close = sum(1 for value in values if abs(value - best) <= TIE)
        if close > 1:
            return None, True
        return int(np.argmax(values)), False

    mean_values = [float(value) for value in mean_mass] + [float(mean_null)]
    null_index = len(mean_mass)
    top1, ambiguous = top_one(mean_values)
    groups_rows = []
    for group in range(groups):
        group_values = [float(record_mass[index, group]) for index in range(len(records))]
        group_values.append(float(null_mass[group]))
        group_top1, group_ambiguous = top_one(group_values)
        groups_rows.append(
            {
                "target_mass": float(record_mass[target, group]),
                "null_mass": float(null_mass[group]),
                "top1": (
                    "ambiguous"
                    if group_ambiguous
                    else ("null" if group_top1 == null_index else group_top1)
                ),
                "top1_ambiguous": group_ambiguous,
                "top1_is_target": bool(group_top1 == target and not group_ambiguous),
            }
        )
    selected = None if top1 in (None, null_index) else records[top1]
    selected_entity = None
    if selected is not None:
        selected_entity = selected.raw.decode("utf-8").split("是")[1].rstrip("。")
    return {
        "target_index": target,
        "target_source_id": records[target].source_id,
        "target_mass": float(mean_mass[target]),
        "target_density": float(mean_mass[target]) / len(records[target].raw),
        "null_mass": float(mean_null),
        "record_masses": [float(value) for value in mean_mass],
        "top1": "ambiguous" if ambiguous else ("null" if top1 == null_index else top1),
        "top1_ambiguous": ambiguous,
        "top1_source_id": None if selected is None else selected.source_id,
        "top1_is_target": bool(top1 == target and not ambiguous),
        "record_only_top1_is_target": bool(
            int(np.argmax([float(value) for value in mean_mass])) == target
        ),
        "selected_entity": selected_entity,
        "recalled_norm": float(slot.norm()),
        "groups": groups_rows,
    }
