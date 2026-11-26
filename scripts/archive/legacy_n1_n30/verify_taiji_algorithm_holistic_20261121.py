"""Holistic algorithm and architecture verification suite (5 suites, 0 check / 0 final).

Audits root causes of earlier experimental stalls and verifies additive fixes in
``taiji/u2_selective_units.py`` across 3 fixed seeds without modifying ``taiji/u2_neural_units.py``:
1. ``arch_bug_audit``: batch-as-time bleed, cross-group edges, collision checks, bit parity.
2. ``M1_selective_state_hold``: horizons T=(5, 10, 20), random target ticks, state_final only,
   old-skill damage=0.0, capsule_v2 atomic roundtrip <= 1e-10.
3. ``M2_midstream_overwrite``: T=12 old->new overwrite controllability + old_error_rate.
4. ``M3_multi_item_capacity_scaling``: T=12, 3 keyed items, delayed query at T-1, N=(2, 8, 32).
5. ``B1_compositional_action_binding``: 5 train / 2 held-out seen-char templates,
   flat MLP vs compositional slot binding vs selective unit binding.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import (  # noqa: E402
    SEEDS,
    build_graph,
    channels,
    lr_at,
)
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402
from scripts.training.run_s2_action_prior_20261121 import BODY_SLOTS  # noqa: E402
from scripts.training.run_s2_action_prior_20261121 import CLASSES as ACTION_CLASSES  # noqa: E402
from scripts.training.run_s2_action_prior_20261121 import COLOURS as ACTION_COLOURS  # noqa: E402
from scripts.training.run_s2_action_prior_20261121 import (  # noqa: E402
    DIRECTIONS as ACTION_DIRECTIONS,
)
from taiji.u2_neural_units import LEGAL_INPUT_CHANNELS  # noqa: E402
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    SelectiveUnitSpec,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_holistic_algorithm_verify_20261121.json"
CAP_CPU = 1200.0
FIT_OFFSET = 10_000_000
TRAIN = 512
FIT = 256
BATCH = 64

ACTION_TRAIN_TEMPLATES = (
    "把{c}球放到{d}边",
    "{c}球放{d}边",
    "让{c}球去{d}边",
    "将{c}球移到{d}边",
    "{c}球去{d}边放",
)
ACTION_FIT_TEMPLATES = (
    "把{c}球放去{d}边",
    "将{c}球放到{d}边",
)
ACTION_VOCAB = tuple(
    sorted(
        {
            ch
            for tpl in (*ACTION_TRAIN_TEMPLATES, *ACTION_FIT_TEMPLATES)
            for ch in tpl.replace("{c}", "").replace("{d}", "")
        }
        | set(ACTION_COLOURS)
        | set(ACTION_DIRECTIONS)
    )
)
ACTION_CHAR_INDEX = {ch: idx for idx, ch in enumerate(ACTION_VOCAB)}
_PROJ_GEN = torch.Generator().manual_seed(2026112101)
_PROJ_RAW = torch.randn(len(ACTION_VOCAB), 16, generator=_PROJ_GEN)
ACTION_PROJECTION = _PROJ_RAW / _PROJ_RAW.norm(dim=0, keepdim=True).clamp(min=1e-6)


def _channels_seq(ticks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert ``[B, T, 48]`` sequence tensor into 6 named ``[B, T, 8]`` legal channels."""
    parts = ticks.split(8, dim=-1)
    return {name: parts[idx] for idx, name in enumerate(LEGAL_INPUT_CHANNELS)}


def _clip_and_step(
    optimizer: torch.optim.Optimizer,
    graph: SelectivePersistentGraph,
    loss: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        [p for p in graph.parameters() if p.requires_grad],
        max_norm=1.0,
    )
    optimizer.step()
    graph.selective_units.project_parameters_()


def _trainable_params(graph: SelectivePersistentGraph) -> list[nn.Parameter]:
    return [p for p in graph.parameters() if p.requires_grad]


def _make_m1_data(count: int, horizon: int, seed: int) -> dict[str, torch.Tensor]:
    """Random-position write at ``t_target in [0, T-2]``, distractors everywhere else including ``T-1``."""
    generator = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(count, horizon, 48, dtype=torch.float32)
    labels = torch.randint(0, 8, (count,), generator=generator)
    target_ticks = torch.randint(0, horizon - 1, (count,), generator=generator)

    # All ticks get distractor colour codes in rgb (0..7) + distractor audio noise in audio (8..15).
    distractors = torch.randint(0, 8, (count, horizon), generator=generator)
    ticks.scatter_(2, distractors.unsqueeze(-1), 1.0)
    ticks[:, :, 8:16] = torch.randn(count, horizon, 8, generator=generator) * 0.25

    row_idx = torch.arange(count)
    # Overwrite target tick rgb with true colour label and assert write flag goal[0]=1.0 (ch index 32).
    ticks[row_idx, target_ticks, :8] = 0.0
    ticks[row_idx, target_ticks, labels] = 1.0
    ticks[row_idx, target_ticks, 32] = 1.0
    # Mark final tick as query step goal[1]=1.0 (ch index 33).
    ticks[:, horizon - 1, 33] = 1.0
    return {"ticks": ticks, "labels": labels, "target_ticks": target_ticks}


def _make_m2_data(count: int, seed: int, horizon: int = 12) -> dict[str, torch.Tensor]:
    """T=12 midstream overwrite: write c_old in [0..4], overwrite c_new in [6..10], query at T-1."""
    generator = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(count, horizon, 48, dtype=torch.float32)
    old_labels = torch.randint(0, 8, (count,), generator=generator)
    offsets = torch.randint(1, 8, (count,), generator=generator)
    new_labels = (old_labels + offsets) % 8
    t1 = torch.randint(0, 5, (count,), generator=generator)
    t2 = torch.randint(6, 11, (count,), generator=generator)

    distractors = torch.randint(0, 8, (count, horizon), generator=generator)
    ticks.scatter_(2, distractors.unsqueeze(-1), 1.0)
    ticks[:, :, 8:16] = torch.randn(count, horizon, 8, generator=generator) * 0.25

    row_idx = torch.arange(count)
    ticks[row_idx, t1, :8] = 0.0
    ticks[row_idx, t1, old_labels] = 1.0
    ticks[row_idx, t1, 32] = 1.0

    ticks[row_idx, t2, :8] = 0.0
    ticks[row_idx, t2, new_labels] = 1.0
    ticks[row_idx, t2, 32] = 1.0

    ticks[:, horizon - 1, 33] = 1.0
    return {"ticks": ticks, "old_labels": old_labels, "new_labels": new_labels}


def _make_m3_data(count: int, seed: int, horizon: int = 12) -> dict[str, torch.Tensor]:
    """T=12 multi-item binding: store 3 keyed 4-class values in [0..8], query slot revealed at T-1."""
    generator = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(count, horizon, 48, dtype=torch.float32)
    # Distractor noise in rgb (0..3) and key question (24..26).
    d_val = torch.randint(0, 4, (count, horizon), generator=generator)
    d_key = torch.randint(0, 3, (count, horizon), generator=generator)
    ticks.scatter_(2, d_val.unsqueeze(-1), 1.0)
    ticks.scatter_(2, (24 + d_key).unsqueeze(-1), 1.0)
    ticks[:, :, 8:16] = torch.randn(count, horizon, 8, generator=generator) * 0.2

    slot_vals = torch.randint(0, 4, (count, 3), generator=generator)
    write_windows = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    row_idx = torch.arange(count)

    for b in range(count):
        perm = torch.randperm(3, generator=generator).tolist()
        for w_idx, slot in enumerate(perm):
            win = write_windows[w_idx]
            t_pick = win[int(torch.randint(0, len(win), (1,), generator=generator))]
            ticks[b, t_pick, :8] = 0.0
            ticks[b, t_pick, int(slot_vals[b, slot])] = 1.0
            ticks[b, t_pick, 24:32] = 0.0
            ticks[b, t_pick, 24 + slot] = 1.0
            ticks[b, t_pick, 32] = 1.0  # WRITE

    query_slots = torch.randint(0, 3, (count,), generator=generator)
    labels = slot_vals[row_idx, query_slots]
    ticks[:, horizon - 1, :8] = 0.0
    ticks[row_idx, horizon - 1, 24:32] = 0.0
    ticks[row_idx, horizon - 1, 24 + query_slots] = 1.0
    ticks[:, horizon - 1, 33] = 1.0  # QUERY at final tick ONLY
    return {"ticks": ticks, "labels": labels}


def _train_selective_classifier(
    graph: SelectivePersistentGraph,
    train_data: dict[str, torch.Tensor],
    label_key: str,
    steps: int,
    lr: float,
    seed: int,
) -> None:
    params = _trainable_params(graph)
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 19)
    n_train = train_data["ticks"].shape[0]
    n_units = graph.selective_spec.n_units
    for step in range(steps):
        cur_lr = max(lr * 0.05, lr_at(step, steps) * (lr / 3e-3))
        for group in optimizer.param_groups:
            group["lr"] = min(lr, cur_lr)
        idx = torch.randint(0, n_train, (BATCH,), generator=generator)
        seq = train_data["ticks"][idx]
        gates = torch.ones(BATCH, seq.shape[1], n_units)
        out = graph(_channels_seq(seq), gates, graph.initial_state(batch_size=BATCH))
        logits = out["readout"]
        loss = loss_fn(logits, train_data[label_key][idx])
        _clip_and_step(optimizer, graph, loss)


def _eval_selective_preds(
    graph: SelectivePersistentGraph,
    ticks: torch.Tensor,
) -> torch.Tensor:
    n_units = graph.selective_spec.n_units
    gates = torch.ones(ticks.shape[0], ticks.shape[1], n_units)
    with torch.no_grad():
        out = graph(_channels_seq(ticks), gates, graph.initial_state(batch_size=ticks.shape[0]))
        return cast(torch.Tensor, out["readout"].argmax(dim=-1))


def _skill_scores(
    graph: SelectivePersistentGraph, skill_data: dict[str, torch.Tensor]
) -> dict[str, float]:
    was_per_sample = graph.per_sample_state
    graph.set_per_sample_state(False)
    scores: dict[str, float] = {}
    with torch.no_grad():
        for name, x_key, y_key, head in (
            ("s1", "x1f", "y1f", 0),
            ("s2", "x2f", "y2f", 1),
            ("s3", "x3f", "y3f", 2),
        ):
            features = skill_data[x_key]
            labels = skill_data[y_key]
            gates = torch.ones(features.shape[0], graph.base_graph.units.n_units)
            logits = graph(channels(features), gates, graph.initial_state(), head=head)["readout"]
            k = int(labels.max()) + 1
            scores[name] = float((logits[:, :k].argmax(dim=-1) == labels).float().mean())
    graph.set_per_sample_state(was_per_sample)
    return scores


def run_suite_arch_audit(seed: int, base_graph: Any) -> dict[str, Any]:
    """Suite 1: empirically verify the root architectural bugs and their elimination."""
    gen = torch.Generator().manual_seed(seed + 100)
    flat_a = torch.randn(16, 48, generator=gen)
    flat_b = flat_a.clone()
    flat_b[:-1] += 3.0
    gates_old = torch.ones(16, base_graph.units.n_units)

    fresh_graph = build_graph(seed)
    with torch.no_grad():
        fresh_a = fresh_graph(channels(flat_a), gates_old, fresh_graph.initial_state(), head=0)
        fresh_b = fresh_graph(channels(flat_b), gates_old, fresh_graph.initial_state(), head=0)
        out_old_a = base_graph(channels(flat_a), gates_old, base_graph.initial_state(), head=0)
        out_old_b = base_graph(channels(flat_b), gates_old, base_graph.initial_state(), head=0)

    fresh_old_batch_as_time_bleed = float(
        (fresh_a["readout"][-1] - fresh_b["readout"][-1]).abs().max()
    )
    trained_parent_batch_bleed = float(
        (out_old_a["readout"][-1] - out_old_b["readout"][-1]).abs().max()
    )
    trained_parent_mean_c = float(base_graph.units.c.detach().mean())
    trained_parent_max_c = float(base_graph.units.c.detach().max())

    edge_mask = base_graph.units.edge_mask
    g_idx = base_graph.units.group_index
    cross_group_mask = (g_idx.unsqueeze(0) != g_idx.unsqueeze(1)).to(edge_mask.dtype)
    old_cross_edges = int((edge_mask * cross_group_mask).sum().item())

    sel_default = SelectivePersistentGraph(base_graph)
    with torch.no_grad():
        out_sel_off = sel_default(channels(flat_a), gates_old, sel_default.initial_state(), head=0)
    default_bit_identical_diff = float((out_sel_off["readout"] - out_old_a["readout"]).abs().max())

    sel_spec = SelectiveUnitSpec(
        groups=6,
        units_per_group=5,
        output_dim=8,
        per_sample_state=True,
        selective_gate=True,
        seed=seed,
    )
    sel_on = SelectivePersistentGraph(base_graph, sel_spec)
    new_cross_edges = int(
        (
            sel_on.selective_units.edge_mask
            * (
                sel_on.selective_units.group_index.unsqueeze(0)
                != sel_on.selective_units.group_index.unsqueeze(1)
            ).to(sel_on.selective_units.edge_mask.dtype)
        )
        .sum()
        .item()
    )
    seq_a = torch.randn(8, 10, 48, generator=gen)
    seq_b = seq_a.clone()
    seq_b[:-1] += 3.0
    gates_seq = torch.ones(8, 10, sel_spec.n_units)
    with torch.no_grad():
        s_out_a = sel_on(_channels_seq(seq_a), gates_seq, sel_on.initial_state(batch_size=8))
        s_out_b = sel_on(_channels_seq(seq_b), gates_seq, sel_on.initial_state(batch_size=8))
    per_sample_bleed = float((s_out_a["readout"][-1] - s_out_b["readout"][-1]).abs().max())

    # Also verify training templates vs fit templates character coverage for A3c/B1 action.
    train_chars = {
        ch for tpl in ACTION_TRAIN_TEMPLATES for ch in tpl.replace("{c}", "").replace("{d}", "")
    }
    fit_chars = {
        ch for tpl in ACTION_FIT_TEMPLATES for ch in tpl.replace("{c}", "").replace("{d}", "")
    }
    return {
        "fresh_old_batch_as_time_bleed": fresh_old_batch_as_time_bleed,
        "trained_parent_batch_bleed": trained_parent_batch_bleed,
        "trained_parent_mean_c": trained_parent_mean_c,
        "trained_parent_max_c": trained_parent_max_c,
        "old_cross_group_edges": old_cross_edges,
        "default_mode_max_diff": default_bit_identical_diff,
        "new_per_sample_bleed": per_sample_bleed,
        "new_cross_group_edges": new_cross_edges,
        "b1_unseen_fit_chars": sorted(fit_chars - train_chars),
    }


def run_suite_m1(
    seed: int,
    base_graph: Any,
    skill_data: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Suite 2: M1 selective state hold across horizons T=(5, 10, 20)."""
    horizon_rows: dict[str, Any] = {}
    base_before = _skill_scores(SelectivePersistentGraph(base_graph), skill_data)
    max_capsule_diff = 0.0
    max_skill_damage = 0.0

    for horizon in (5, 10, 20):
        train_data = _make_m1_data(TRAIN, horizon, seed + horizon * 100)
        fit_data = _make_m1_data(FIT, horizon, seed + horizon * 100 + FIT_OFFSET)

        arm_accs: dict[str, float] = {}
        for arm_name, use_selective in (
            ("static_c_state_final", False),
            ("selective_state_final", True),
        ):
            cloned_base = copy.deepcopy(base_graph)
            spec = SelectiveUnitSpec(
                groups=6,
                units_per_group=5,
                output_dim=8,
                per_sample_state=True,
                selective_gate=use_selective,
                init_write_bias=-2.0,
                seed=seed + horizon,
            )
            graph = SelectivePersistentGraph(cloned_base, spec)
            graph.freeze_base_()
            _train_selective_classifier(
                graph, train_data, "labels", steps=500, lr=6e-3, seed=seed + horizon
            )
            preds = _eval_selective_preds(graph, fit_data["ticks"])
            arm_accs[arm_name] = float((preds == fit_data["labels"]).float().mean())

            if use_selective:
                after = _skill_scores(graph, skill_data)
                damage = max(max(0.0, base_before[k] - after[k]) for k in base_before)
                max_skill_damage = max(max_skill_damage, damage)

                if horizon == 20:
                    cap = build_capsule_v2(graph, clocks=(5, 2))
                    with tempfile.TemporaryDirectory() as tmpdir:
                        path = Path(tmpdir) / f"cap_{seed}.pt"
                        save_capsule_v2(cap, path)
                        loaded = load_capsule_v2(path)
                    rebuilt = SelectivePersistentGraph(copy.deepcopy(base_graph))
                    apply_capsule_v2(rebuilt, loaded)
                    gates = torch.ones(fit_data["ticks"].shape[0], horizon, spec.n_units)
                    with torch.no_grad():
                        o1 = graph(
                            _channels_seq(fit_data["ticks"]),
                            gates,
                            graph.initial_state(batch_size=FIT),
                        )["readout"]
                        o2 = rebuilt(
                            _channels_seq(fit_data["ticks"]),
                            gates,
                            rebuilt.initial_state(batch_size=FIT),
                        )["readout"]
                    max_capsule_diff = max(max_capsule_diff, float((o1 - o2).abs().max()))

        horizon_rows[str(horizon)] = arm_accs

    return {
        "horizons": horizon_rows,
        "old_skill_damage": max_skill_damage,
        "capsule_v2_max_abs_diff": max_capsule_diff,
    }


def run_suite_m2(seed: int, base_graph: Any) -> dict[str, Any]:
    """Suite 3: M2 midstream overwrite at T=12."""
    train_data = _make_m2_data(TRAIN, seed + 5000)
    fit_data = _make_m2_data(FIT, seed + 5000 + FIT_OFFSET)
    out: dict[str, Any] = {}
    for arm_name, use_selective in (
        ("static_c_state_final", False),
        ("selective_state_final", True),
    ):
        cloned_base = copy.deepcopy(base_graph)
        spec = SelectiveUnitSpec(
            groups=6,
            units_per_group=5,
            output_dim=8,
            per_sample_state=True,
            selective_gate=use_selective,
            init_write_bias=-2.0,
            seed=seed + 5000,
        )
        graph = SelectivePersistentGraph(cloned_base, spec)
        graph.freeze_base_()
        _train_selective_classifier(
            graph, train_data, "new_labels", steps=550, lr=6e-3, seed=seed + 5000
        )
        preds = _eval_selective_preds(graph, fit_data["ticks"])
        acc = float((preds == fit_data["new_labels"]).float().mean())
        old_err = float((preds == fit_data["old_labels"]).float().mean())
        out[arm_name] = {"acc": acc, "old_error_rate": old_err}
    return out


def run_suite_m3(seed: int, base_graph: Any) -> dict[str, Any]:
    """Suite 4: M3 multi-item delayed query at T=12 across n_mem_units=(2, 8, 32)."""
    fit_data = _make_m3_data(FIT, seed + 9000 + FIT_OFFSET)
    unit_configs = (
        ("2", 2, 1),
        ("8", 4, 2),
        ("32", 8, 4),
    )
    out: dict[str, float] = {}
    loss_fn = nn.CrossEntropyLoss()
    steps = 1000
    lr = 7e-3
    for label, groups, upg in unit_configs:
        cloned_base = copy.deepcopy(base_graph)
        spec = SelectiveUnitSpec(
            groups=groups,
            units_per_group=upg,
            output_dim=4,
            per_sample_state=True,
            selective_gate=True,
            init_write_bias=-2.0,
            seed=seed + groups * 100 + upg,
        )
        graph = SelectivePersistentGraph(cloned_base, spec)
        graph.freeze_base_()
        params = _trainable_params(graph)
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        for step in range(steps):
            batch = _make_m3_data(BATCH, seed * 10000 + step)
            gates = torch.ones(BATCH, batch["ticks"].shape[1], spec.n_units)
            out_step = graph(
                _channels_seq(batch["ticks"]), gates, graph.initial_state(batch_size=BATCH)
            )
            loss = loss_fn(out_step["readout"], batch["labels"])
            _clip_and_step(optimizer, graph, loss)
        preds = _eval_selective_preds(graph, fit_data["ticks"])
        out[label] = float((preds == fit_data["labels"]).float().mean())
    return out


def _action_episode(seed: int, template_str: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    colours = torch.randperm(4, generator=generator)[:3].tolist()
    positions = torch.randint(0, 4, (3,), generator=generator).tolist()
    target_slot = int(torch.randint(0, 3, (1,), generator=generator))
    direction = ACTION_DIRECTIONS[int(torch.randint(0, 4, (1,), generator=generator))]
    features = torch.zeros(3 * BODY_SLOTS + 16, dtype=torch.float32)
    for slot in range(3):
        base = slot * BODY_SLOTS
        features[base + colours[slot]] = 1.0
        features[base + 6 + positions[slot]] = 1.0
    text = template_str.format(c=ACTION_COLOURS[colours[target_slot]], d=direction)
    bag = torch.zeros(len(ACTION_VOCAB), dtype=torch.float32)
    for ch in text:
        if ch in ACTION_CHAR_INDEX:
            bag[ACTION_CHAR_INDEX[ch]] += 1.0
    features[30:] = bag @ ACTION_PROJECTION
    return {
        "features": features,
        "label": torch.tensor(target_slot * 4 + ACTION_DIRECTIONS.index(direction)),
    }


def _build_action_split(
    count: int, seed: int, templates: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    rows = [_action_episode(seed + idx, templates[idx % len(templates)]) for idx in range(count)]
    return {
        "features": torch.stack([r["features"] for r in rows]).float(),
        "labels": torch.stack([r["label"] for r in rows]).long(),
    }


class _FlatActionMLP(nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(46, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, ACTION_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(features))


class _CompositionalSlotBinding(nn.Module):
    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.slot_enc = nn.Sequential(
            nn.Linear(BODY_SLOTS, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim)
        )
        self.target_q = nn.Sequential(
            nn.Linear(16, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim)
        )
        self.dir_head = nn.Sequential(
            nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, len(ACTION_DIRECTIONS))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        slots = features[:, : 3 * BODY_SLOTS].reshape(features.shape[0], 3, BODY_SLOTS)
        instr = features[:, 3 * BODY_SLOTS :]
        keys = self.slot_enc(slots)
        query = self.target_q(instr)
        slot_logits = torch.einsum("bsd,bd->bs", keys, query)
        dir_logits = self.dir_head(instr)
        log_joint = F.log_softmax(slot_logits, dim=-1).unsqueeze(-1) + F.log_softmax(
            dir_logits, dim=-1
        ).unsqueeze(1)
        return log_joint.reshape(features.shape[0], ACTION_CLASSES)


class _SelectiveUnitActionBinding(nn.Module):
    """Presents 3 slots + instruction over 4 unit ticks and binds per-slot states to query."""

    def __init__(self, base_graph: Any, seed: int) -> None:
        super().__init__()
        spec = SelectiveUnitSpec(
            groups=6,
            units_per_group=5,
            output_dim=ACTION_CLASSES,
            per_sample_state=True,
            selective_gate=True,
            init_write_bias=-1.0,
            seed=seed,
        )
        self.graph = SelectivePersistentGraph(copy.deepcopy(base_graph), spec)
        self.graph.freeze_base_()
        n_units = spec.n_units
        self.slot_proj = nn.Linear(n_units, 32)
        self.query_proj = nn.Linear(2 * n_units, 32)
        self.dir_head = nn.Linear(2 * n_units, len(ACTION_DIRECTIONS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        seq = torch.zeros(batch_size, 4, 48, dtype=features.dtype)
        slots = features[:, : 3 * BODY_SLOTS].reshape(batch_size, 3, BODY_SLOTS)
        seq[:, :3, :BODY_SLOTS] = slots
        seq[:, 3, 24:40] = features[:, 3 * BODY_SLOTS :]
        gates = torch.ones(batch_size, 4, self.graph.selective_spec.n_units)
        out = self.graph(
            _channels_seq(seq),
            gates,
            self.graph.initial_state(batch_size=batch_size),
        )
        # Activity at ticks 0..2 corresponds to the 3 object slots; state_final captures tick-3 query.
        slot_states = self.slot_proj(out["unit_result"]["activity"][:, :3, :])
        query_vec = self.query_proj(out["state_final"])
        slot_logits = torch.einsum("bsd,bd->bs", slot_states, query_vec)
        dir_logits = self.dir_head(out["state_final"])
        log_joint = F.log_softmax(slot_logits, dim=-1).unsqueeze(-1) + F.log_softmax(
            dir_logits, dim=-1
        ).unsqueeze(1)
        return log_joint.reshape(batch_size, ACTION_CLASSES)


def _train_and_eval_action(
    model: nn.Module,
    train_data: dict[str, torch.Tensor],
    fit_data: dict[str, torch.Tensor],
    seed: int,
    steps: int = 1200,
    lr: float = 4e-3,
    project_graph: SelectivePersistentGraph | None = None,
) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 41)
    n_train = train_data["features"].shape[0]
    for step in range(steps):
        cur_lr = max(lr * 0.05, lr_at(step, steps) * (lr / 3e-3))
        for group in optimizer.param_groups:
            group["lr"] = min(lr, cur_lr)
        idx = torch.randint(0, n_train, (BATCH,), generator=generator)
        logits = model(train_data["features"][idx])
        loss = loss_fn(logits, train_data["labels"][idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if project_graph is not None:
            project_graph.selective_units.project_parameters_()
    with torch.no_grad():
        preds = model(fit_data["features"]).argmax(dim=-1)
    return float((preds == fit_data["labels"]).float().mean())


def run_suite_b1(seed: int, base_graph: Any) -> dict[str, float]:
    """Suite 5: compositional action binding on 5 train / 2 held-out seen-char templates."""
    train_data = _build_action_split(1024, seed + 12000, ACTION_TRAIN_TEMPLATES)
    fit_data = _build_action_split(FIT, seed + 12000 + FIT_OFFSET, ACTION_FIT_TEMPLATES)

    torch.manual_seed(seed + 1)
    flat = _FlatActionMLP()
    flat_acc = _train_and_eval_action(flat, train_data, fit_data, seed + 1)

    torch.manual_seed(seed + 2)
    comp = _CompositionalSlotBinding()
    comp_acc = _train_and_eval_action(comp, train_data, fit_data, seed + 2)

    torch.manual_seed(seed + 3)
    unit_bind = _SelectiveUnitActionBinding(base_graph, seed + 3)
    unit_acc = _train_and_eval_action(
        unit_bind,
        train_data,
        fit_data,
        seed + 3,
        steps=1400,
        lr=5e-3,
        project_graph=unit_bind.graph,
    )
    return {
        "flat_mlp": flat_acc,
        "compositional_slot_binding": comp_acc,
        "selective_unit_binding": unit_acc,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    started = time.process_time()
    torch.set_num_threads(1)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        if time.process_time() - started > CAP_CPU:
            raise RuntimeError("holistic_verify_budget_exceeded")
        base_graph, skill_data = train_joint_parent(seed)
        row = {
            "seed": seed,
            "arch_bug_audit": run_suite_arch_audit(seed, base_graph),
            "M1_selective_state_hold": run_suite_m1(seed, base_graph, skill_data),
            "M2_midstream_overwrite": run_suite_m2(seed, base_graph),
            "M3_multi_item_capacity_scaling": run_suite_m3(seed, base_graph),
            "B1_compositional_action_binding": run_suite_b1(seed, base_graph),
        }
        rows.append(row)

    summary: dict[str, Any] = {
        "arch_bug_audit": {
            "fresh_old_batch_as_time_bleed_max": max(
                r["arch_bug_audit"]["fresh_old_batch_as_time_bleed"] for r in rows
            ),
            "trained_parent_batch_bleed_max": max(
                r["arch_bug_audit"]["trained_parent_batch_bleed"] for r in rows
            ),
            "trained_parent_mean_c": _mean(
                [r["arch_bug_audit"]["trained_parent_mean_c"] for r in rows]
            ),
            "trained_parent_max_c": max(r["arch_bug_audit"]["trained_parent_max_c"] for r in rows),
            "old_cross_group_edges": rows[0]["arch_bug_audit"]["old_cross_group_edges"],
            "default_mode_max_diff": max(
                r["arch_bug_audit"]["default_mode_max_diff"] for r in rows
            ),
            "new_per_sample_bleed_max": max(
                r["arch_bug_audit"]["new_per_sample_bleed"] for r in rows
            ),
            "new_cross_group_edges": rows[0]["arch_bug_audit"]["new_cross_group_edges"],
            "b1_unseen_fit_chars": rows[0]["arch_bug_audit"]["b1_unseen_fit_chars"],
        },
        "M1_selective_state_hold": {
            str(h): {
                arm: _mean([r["M1_selective_state_hold"]["horizons"][str(h)][arm] for r in rows])
                for arm in ("static_c_state_final", "selective_state_final")
            }
            for h in (5, 10, 20)
        }
        | {
            "old_skill_damage_max": max(
                r["M1_selective_state_hold"]["old_skill_damage"] for r in rows
            ),
            "capsule_v2_max_abs_diff": max(
                r["M1_selective_state_hold"]["capsule_v2_max_abs_diff"] for r in rows
            ),
        },
        "M2_midstream_overwrite": {
            arm: {
                "acc": _mean([r["M2_midstream_overwrite"][arm]["acc"] for r in rows]),
                "old_error_rate": _mean(
                    [r["M2_midstream_overwrite"][arm]["old_error_rate"] for r in rows]
                ),
            }
            for arm in ("static_c_state_final", "selective_state_final")
        },
        "M3_multi_item_capacity_scaling": {
            k: _mean([r["M3_multi_item_capacity_scaling"][k] for r in rows])
            for k in ("2", "8", "32")
        },
        "B1_compositional_action_binding": {
            arm: _mean([r["B1_compositional_action_binding"][arm] for r in rows])
            for arm in ("flat_mlp", "compositional_slot_binding", "selective_unit_binding")
        },
    }

    m1 = summary["M1_selective_state_hold"]
    m2 = summary["M2_midstream_overwrite"]
    m3 = summary["M3_multi_item_capacity_scaling"]
    b1 = summary["B1_compositional_action_binding"]
    arch = summary["arch_bug_audit"]

    gates = {
        "arch_default_bit_identical": arch["default_mode_max_diff"] == 0.0,
        "arch_batch_isolation_fixed": (
            arch["fresh_old_batch_as_time_bleed_max"] > 0.0
            and arch["new_per_sample_bleed_max"] == 0.0
        ),
        "arch_cross_group_restored": (
            arch["old_cross_group_edges"] == 0 and arch["new_cross_group_edges"] > 0
        ),
        "m1_hold_t5": m1["5"]["selective_state_final"] >= 0.85,
        "m1_hold_t10": m1["10"]["selective_state_final"] >= 0.75,
        "m1_hold_t20": m1["20"]["selective_state_final"] >= 0.70,
        "m1_beats_static_c": (
            m1["20"]["selective_state_final"] > m1["20"]["static_c_state_final"] + 0.15
        ),
        "old_skill_damage_zero": m1["old_skill_damage_max"] == 0.0,
        "capsule_v2_roundtrip_exact": m1["capsule_v2_max_abs_diff"] <= 1e-10,
        "m2_overwrite_pass": m2["selective_state_final"]["acc"] >= 0.70,
        "m3_capacity_scales": m3["32"] > m3["2"] + 0.15,
        "b1_compositional_ge_080": b1["compositional_slot_binding"] >= 0.80,
        "b1_unit_binding_ge_080": b1["selective_unit_binding"] >= 0.80,
    }
    verdict = "holistic_algorithm_verified" if all(gates.values()) else "holistic_algorithm_partial"

    payload: dict[str, Any] = {
        "schema": "taiji-u2-holistic-algorithm-verify-v1",
        "card": "Holistic 5-suite verification of Taiji selective persistent state & compositional binding",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "cap_cpu": CAP_CPU,
            "train": TRAIN,
            "fit": FIT,
            "fit_offset": FIT_OFFSET,
            "m3_training": "online_disjoint_batches_1000_steps_wd_1e-4",
            "state_readout": "state_final_and_read_gate_at_t_minus_1_only_no_external_kv_history",
        },
        "rows": rows,
        "summary": summary,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "summary": summary,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if verdict == "holistic_algorithm_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
