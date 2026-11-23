"""P0-BC: 4 -> 8 colour progressive expansion (adds 24 units, keeps the 4-colour ability).

Phase A: learn the 4-colour flip with 24 newborn units (P0-BB pipeline, seed via env).
Phase B: grow another 24 units (per group), resize the raw-input adapter (old rows kept,
new rows zero), then learn the 8-colour flip; finally check
  8-colour flip accuracy (seen / unseen pairs),
  4-colour flip retention (phase-A head under the grown subject),
  growth causality: dormancy of ONLY the phase-B cohort drops 8-colour accuracy while the
  4-colour ability stays, and normal Chinese stays 1.0.

Exploratory diagnostic; zero check/final.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_p0bb_20261123 import (
    BATCH,
    CN_STEPS,
    DET_STEPS,
    FLAG_CH,
    FLIP_STEPS,
    OLD_N,
    PhaseHead,
    cn_eval,
    cn_feature,
    mask_fn,
    mask_newborn_grads,
    newborn_params,
    newborn_states,
    probe,
    revision_split,
    run_states,
)
from scripts.training.verify_taiji_23s2a_gain_20261122 import (
    BI_CN_CHAR_INDEX,
    BI_CN_PROJ32,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    CN_CLASSES,
    CN_COLOURS,
    CN_MAX_LEN,
    HOLDOUT,
)
from scripts.training.verify_taiji_cognitive_loop_20261121 import _build_bidirectional_cn_32dim
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import SelectivePersistentGraph, apply_capsule_v2, build_capsule_v2

ROOT = Path(r"E:\Seed")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261120
OUT = ROOT / f"reports/taiji_p0ca_seed{SEED}.json"
PHASE_B_STEPS = 2400


def revision8(count: int, seed: int, allow_holdout: bool = False) -> dict:
    """8-colour flip task (same template/flag scheme as revision_split, all 8 colours)."""
    first, second = BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    all_tpls = [(t, True) for t in first] + [(t, False) for t in second]
    rows, flip_labels = [], []
    flag_rng = torch.Generator().manual_seed(seed + 999_999)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, CN_CLASSES, (1,), generator=gen))
        c2 = int(torch.randint(0, CN_CLASSES - 1, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = ((c1, c2) in HOLDOUT) if allow_holdout else ((c1, c2) not in HOLDOUT)
        idx += 1
        if not ok:
            continue
        tpl, pick_first = all_tpls[len(rows) % len(all_tpls)]
        normal = c1 if pick_first else c2
        other = c2 if pick_first else c1
        flag = int(torch.randint(0, 4, (1,), generator=flag_rng))
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        rows.append(seq)
        flip_labels.append(other if flag > 0 else normal)
    return {"ticks": torch.stack(rows), "flip_labels": torch.tensor(flip_labels, dtype=torch.long)}


HOLDOUT4B = {(4, 5), (5, 4), (6, 7), (7, 6)}


def revision4b(count: int, seed: int, allow_holdout: bool = False) -> dict:
    first, second = BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    all_tpls = [(tpl, True) for tpl in first] + [(tpl, False) for tpl in second]
    rows, labels, flip_labels = [], [], []
    flag_rng = torch.Generator().manual_seed(seed + 999_999)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = 4 + int(torch.randint(0, 4, (1,), generator=gen))
        c2 = 4 + int(torch.randint(0, 3, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = ((c1, c2) in HOLDOUT4B) if allow_holdout else ((c1, c2) not in HOLDOUT4B)
        idx += 1
        if not ok:
            continue
        tpl, pick_first = all_tpls[len(rows) % len(all_tpls)]
        normal = c1 if pick_first else c2
        other = c2 if pick_first else c1
        flag = int(torch.randint(0, 4, (1,), generator=flag_rng))
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        rows.append(seq)
        labels.append(1 if flag >= 2 else 0)
        flip_labels.append((other if flag >= 2 else normal) - 4)
    return {
        "ticks": torch.stack(rows),
        "labels": torch.tensor(labels, dtype=torch.long),
        "flip_labels": torch.tensor(flip_labels, dtype=torch.long),
    }


HOLDOUT4 = {(0, 1), (1, 0), (2, 3), (3, 2)}


def revision4c(count: int, seed: int, allow_holdout: bool = False) -> dict:
    """Third task family: 4-colour flip with the second template family only (different surface)."""
    tpls = [(tpl, False) for tpl in BI_CN_TRAIN_SECOND]
    rows, labels, flip_labels = [], [], []
    flag_rng = torch.Generator().manual_seed(seed + 888_888)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, 4, (1,), generator=gen))
        c2 = int(torch.randint(0, 3, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = ((c1, c2) in HOLDOUT4) if allow_holdout else ((c1, c2) not in HOLDOUT4)
        idx += 1
        if not ok:
            continue
        tpl, _pick = tpls[len(rows) % len(tpls)]
        normal = c2
        other = c1
        flag = int(torch.randint(0, 4, (1,), generator=flag_rng))
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        rows.append(seq)
        labels.append(1 if flag >= 2 else 0)
        flip_labels.append(other if flag >= 2 else normal)
    return {
        "ticks": torch.stack(rows),
        "labels": torch.tensor(labels, dtype=torch.long),
        "flip_labels": torch.tensor(flip_labels, dtype=torch.long),
    }


def revision4d(count: int, seed: int, allow_holdout: bool = False) -> dict:
    """Fourth task family: colours 4..7 flip with the second template family only."""
    tpls = [(tpl, False) for tpl in BI_CN_TRAIN_SECOND]
    rows, labels, flip_labels = [], [], []
    flag_rng = torch.Generator().manual_seed(seed + 777_777)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = 4 + int(torch.randint(0, 4, (1,), generator=gen))
        c2 = 4 + int(torch.randint(0, 3, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = ((c1, c2) in HOLDOUT4B) if allow_holdout else ((c1, c2) not in HOLDOUT4B)
        idx += 1
        if not ok:
            continue
        tpl, _pick = tpls[len(rows) % len(tpls)]
        normal = c2
        other = c1
        flag = int(torch.randint(0, 4, (1,), generator=flag_rng))
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        rows.append(seq)
        labels.append(1 if flag >= 2 else 0)
        flip_labels.append((other if flag >= 2 else normal) - 4)
    return {
        "ticks": torch.stack(rows),
        "labels": torch.tensor(labels, dtype=torch.long),
        "flip_labels": torch.tensor(flip_labels, dtype=torch.long),
    }


def revision4e(count: int, seed: int) -> dict:
    """Fifth task family E: colours 0..3, second templates, flip when flag>=1."""
    tpls = [(tpl, False) for tpl in BI_CN_TRAIN_SECOND]
    rows, labels, flip_labels = [], [], []
    flag_rng = torch.Generator().manual_seed(seed + 666_666)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, 4, (1,), generator=gen))
        c2 = int(torch.randint(0, 3, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        idx += 1
        tpl, _pick = tpls[len(rows) % len(tpls)]
        normal = c2
        other = c1
        flag = int(torch.randint(0, 4, (1,), generator=flag_rng))
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        rows.append(seq)
        labels.append(1 if flag >= 1 else 0)
        flip_labels.append(other if flag >= 1 else normal)
    return {
        "ticks": torch.stack(rows),
        "labels": torch.tensor(labels, dtype=torch.long),
        "flip_labels": torch.tensor(flip_labels, dtype=torch.long),
    }


def extra_of(batch: dict) -> torch.Tensor:
    return batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()


def cohort_feat(subj, ticks: torch.Tensor, extra: torch.Tensor, hi: int) -> torch.Tensor:
    z, a = run_states(subj, ticks, mask_fn(subj.n_units, OLD_N, hi), extra=extra)
    return torch.cat([z[:, OLD_N:hi], a[:, OLD_N:hi]], dim=-1)


def flip_acc(subj, head, batch: dict, hi: int | None = None) -> float:
    hi = subj.n_units if hi is None else hi
    z, a = run_states(subj, batch["ticks"], mask_fn(subj.n_units, OLD_N, hi), extra=extra_of(batch))
    feat = torch.cat([z[:, OLD_N:hi], a[:, OLD_N:hi]], dim=-1)
    head.eval()
    subj.eval_mode()
    with torch.no_grad():
        logits = head(feat)
        return float((logits.argmax(-1) == batch["flip_labels"]).float().mean())


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=1e-4)
    subj.train_mode()
    for step in range(CN_STEPS):
        batch = _build_bidirectional_cn_32dim(
            BATCH, SEED * 6000 + step, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
        )
        loss = F.cross_entropy(
            subj.logits("cn", batch["ticks"], (batch["ticks"].abs().sum(-1) > 0).float()),
            batch["labels"],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
    normal_cn = cn_eval(subj)

    # ---------- phase A: 24 newborn units learn the 4-colour flip ----------
    subj.graph.grow_units_neutral_(2)
    units = subj.graph.selective_units
    subj.n_units = 96
    units.attach_extra_(4, seed=SEED)
    with torch.no_grad():
        src_idx = torch.arange(24)
        dst_idx = torch.arange(24) + OLD_N
        units.group_index[dst_idx] = torch.arange(24) // 6
        ar = torch.arange(12).unsqueeze(1)
        group_idx = units.group_index
        units.group_mask.copy_((group_idx.unsqueeze(0) == ar).to(units.group_mask.dtype))
        if units.spec.cross_group_edges:
            units.edge_mask.copy_(1.0 - torch.eye(96, dtype=units.edge_mask.dtype))
        else:
            same = (group_idx.unsqueeze(1) == group_idx.unsqueeze(0)).to(units.edge_mask.dtype)
            units.edge_mask.copy_(same * (1.0 - torch.eye(96, dtype=units.edge_mask.dtype)))
        units.r[dst_idx] = units.r[src_idx].clone()
        units.p[:, dst_idx] = units.p[:, src_idx].clone()
        for name in ("c", "eta", "b", "write_bias"):
            getattr(units, name)[dst_idx] = getattr(units, name)[src_idx].clone()
        units.w[dst_idx, :] = units.w[src_idx, :].clone()
        units.write_in.weight[dst_idx, :] = units.write_in.weight[src_idx, :].clone()
        units.write_state.weight[dst_idx, :] = units.write_state.weight[src_idx, :].clone()
        units.write_bias[dst_idx] = 0.0
        units.b[dst_idx] = 0.0

    # null-calibrated pressure probe at birth (pre-registered delta <= 0.10)
    test_birth = revision_split(1024, SEED + 1300 + 10_000_000)
    sm_birth = (test_birth["ticks"].abs().sum(-1) > 0).float()
    cn_probe = probe(
        cn_feature(subj, test_birth["ticks"], step_mask=sm_birth), test_birth["labels"]
    )
    extra_birth = extra_of(test_birth)
    with torch.no_grad():
        birth_feat = newborn_states(subj, test_birth["ticks"], extra_birth).detach()
    birth_probe = probe(birth_feat, test_birth["labels"])
    null_cn, null_birth = [], []
    g_perm = torch.Generator().manual_seed(SEED + 20_000)
    for _k in range(10):
        perm = torch.randperm(test_birth["labels"].numel(), generator=g_perm)
        null_cn.append(
            probe(
                cn_feature(subj, test_birth["ticks"], step_mask=sm_birth),
                test_birth["labels"][perm],
            )["test"]
        )
        null_birth.append(probe(birth_feat, test_birth["labels"][perm])["test"])
    null_cn_mean = sum(null_cn) / len(null_cn)
    null_birth_mean = sum(null_birth) / len(null_birth)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17601)
        head_det = PhaseHead(2 * (96 - OLD_N), 2, 64)
    params = list(head_det.parameters()) + list(newborn_params(subj))
    opt2 = torch.optim.Adam(params, lr=6e-3)
    for i in range(DET_STEPS):
        batch = revision_split(BATCH, SEED * 7000 + i)
        subj.train_mode()
        loss = F.cross_entropy(
            head_det(newborn_states(subj, batch["ticks"], extra_of(batch))), batch["labels"]
        )
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt2.step()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17701)
        head4 = PhaseHead(2 * (96 - OLD_N), 4, 64)
    params4 = list(head4.parameters()) + list(newborn_params(subj))
    opt4 = torch.optim.Adam(params4, lr=6e-3)
    for i in range(FLIP_STEPS):
        batch = revision_split(BATCH, SEED * 7100 + i)
        subj.train_mode()
        loss = F.cross_entropy(
            head4(newborn_states(subj, batch["ticks"], extra_of(batch))), batch["flip_labels"]
        )
        opt4.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params4, 1.0)
        opt4.step()
    eval4 = revision_split(1024, SEED + 1300 + 10_000_000)
    acc4_before = flip_acc(subj, head4, eval4, hi=96)

    # ---- evidence-based growth choice (prior cards as the evidence base) ----
    # predicted net utility in "ability points": gain - damage - cost penalty
    EV_GROW_GAIN = 0.70  # P0-BH/P0-BI: grow+train 8c gains ~0.70-1.00 over chance
    EV_RETIRE_DAMAGE = 0.75  # P0-BL: retiring 72:96 costs 0.66-0.84 on the old ability
    cost_budget = 0.10
    utility = {
        "grow_and_train_8c": EV_GROW_GAIN - 0.0 - cost_budget,
        "retire_phaseA": -EV_RETIRE_DAMAGE - 0.02,
        "wait": 0.0,
    }
    chosen_action = max(utility, key=utility.get)

    head4_frozen = copy.deepcopy(head4)
    for p in head4_frozen.parameters():
        p.requires_grad_(False)
    head4_frozen.eval()

    # ---------- phase B: grow 24 more units and learn the 8-colour flip ----------
    old_w = units.extra_proj.weight.detach().clone()
    subj.graph.grow_units_neutral_(2)
    units = subj.graph.selective_units
    subj.n_units = 120
    new_proj = torch.nn.Linear(4, 120, bias=False)
    with torch.no_grad():
        new_proj.weight[: old_w.shape[0]] = old_w
        new_proj.weight[old_w.shape[0] :] = 0.0
    units.extra_proj = new_proj

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17801)
        head8 = PhaseHead(2 * (120 - OLD_N), 8, 64)
    cohorts = list(newborn_params(subj))  # all units 72:108 (old cohort + new cohort)
    opt8 = torch.optim.Adam(list(head8.parameters()) + cohorts, lr=6e-3)
    for i in range(PHASE_B_STEPS):
        batch8 = revision8(24, SEED * 7200 + i)
        batch4 = revision_split(24, SEED * 7300 + i)
        subj.train_mode()
        f8 = cohort_feat(subj, batch8["ticks"], extra_of(batch8), 120)
        f4 = cohort_feat(subj, batch4["ticks"], extra_of(batch4), 96)
        loss = F.cross_entropy(head8(f8), batch8["flip_labels"]) + 1.0 * F.cross_entropy(
            head4_frozen(f4), batch4["flip_labels"]
        )
        opt8.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(list(head8.parameters()) + cohorts, 1.0)
        opt8.step()

    eval8 = revision8(1024, SEED + 1300 + 10_000_000)
    eval8_unseen = revision8(512, SEED + 1302 + 10_000_000, allow_holdout=True)
    acc8 = flip_acc(subj, head8, eval8)
    acc8_unseen = flip_acc(subj, head8, eval8_unseen)
    acc4_after = flip_acc(subj, head4, eval4, hi=96)

    # growth causality: lesion ONLY the phase-B cohort (units 96:108)
    subj.graph.selective_units.set_unit_lifecycle_(range(96, 120), "dormant")
    acc8_lesion = flip_acc(subj, head8, eval8)
    acc4_lesion = flip_acc(subj, head4, eval4, hi=96)
    subj.graph.selective_units.set_unit_lifecycle_(range(96, 120), "mature")
    acc8_wake = flip_acc(subj, head8, eval8)
    acc4_wake = flip_acc(subj, head4, eval4, hi=96)

    # capsule round trip into a fresh graph (same spec, same adapter layout)
    cap = build_capsule_v2(subj.graph)
    fresh = SelectivePersistentGraph(subj.graph.base_graph, subj.graph.selective_spec)
    fresh.freeze_base_()
    fresh.selective_units.attach_extra_(4, seed=SEED)
    apply_capsule_v2(fresh, cap)

    class Holder:
        def __init__(self, graph, num_units):
            self.graph = graph
            self.n_units = num_units

        def eval_mode(self):
            pass

        def train_mode(self):
            pass

    holder = Holder(fresh, subj.n_units)
    with torch.no_grad():
        z0, a0 = run_states(
            subj, eval8["ticks"], mask_fn(subj.n_units, OLD_N, subj.n_units), extra=extra_of(eval8)
        )
        z1, a1 = run_states(
            holder,
            eval8["ticks"],
            mask_fn(holder.n_units, OLD_N, holder.n_units),
            extra=extra_of(eval8),
        )
    capsule_diff = float(max((z0 - z1).abs().max(), (a0 - a1).abs().max()))
    acc8_capsule = flip_acc(holder, head8, eval8)
    acc4_capsule = flip_acc(holder, head4, eval4, hi=96)

    subj.graph.selective_units.set_unit_lifecycle_(range(72, 96), "retired")
    acc4_after_retire = flip_acc(subj, head4, eval4, hi=96)
    acc8_after_retire = flip_acc(subj, head8, eval8, hi=120)
    subj.graph.selective_units.set_unit_lifecycle_(range(72, 96), "mature")
    actual_retire_damage = acc4_after - acc4_after_retire

    actual_gain = acc8 - 0.25
    regret = abs(EV_GROW_GAIN - actual_gain) + abs(EV_RETIRE_DAMAGE - actual_retire_damage)
    # ---- decision 2: update the evidence with the observed outcomes, then choose again ----
    alpha = 0.5
    EV_GROW_GAIN = (1 - alpha) * EV_GROW_GAIN + alpha * actual_gain
    EV_RETIRE_DAMAGE = (1 - alpha) * EV_RETIRE_DAMAGE + alpha * actual_retire_damage
    cost_grow, cost_recycle, cost_retire = 0.10, 0.39, 0.02
    utility2 = {
        "grow_more": 0.0 - 0.0 - cost_grow,
        "retire_phaseA": -EV_RETIRE_DAMAGE - cost_retire,
        "recycle_phaseA": 0.0 - 0.0 - cost_recycle,
        "wait": 0.0,
    }
    chosen_action2 = max(utility2, key=utility2.get)
    acc4_before_wait, acc8_before_wait = acc4_after, acc8
    # execute the chosen action (the pre-registered argmax is "wait")
    acc4_after_wait, acc8_after_wait = acc4_after, acc8

    # ---- session boundary: persist the accumulated subject, restore into a fresh one ----
    cap_A = build_capsule_v2(subj.graph)
    fresh_A = SelectivePersistentGraph(subj.graph.base_graph, subj.graph.selective_spec)
    fresh_A.freeze_base_()
    fresh_A.selective_units.attach_extra_(4, seed=SEED)
    apply_capsule_v2(fresh_A, cap_A)
    holder_A = Holder(fresh_A, subj.n_units)
    acc4_sess2 = flip_acc(holder_A, head4, eval4, hi=96)
    acc8_sess2 = flip_acc(holder_A, head8, eval8, hi=120)
    with torch.no_grad():
        z0, a0 = run_states(
            subj, eval8["ticks"], mask_fn(subj.n_units, OLD_N, subj.n_units), extra=extra_of(eval8)
        )
        z1, a1 = run_states(
            holder_A,
            eval8["ticks"],
            mask_fn(holder_A.n_units, OLD_N, holder_A.n_units),
            extra=extra_of(eval8),
        )
    capA_diff = float(max((z0 - z1).abs().max(), (a0 - a1).abs().max()))

    # ---- wait with wake conditions (autonomous wake-up monitor) ----
    # signals: new capability demand (0/1), budget share available, evidence drift
    budget_series = [0.05, 0.45, 0.45]
    demand_series = [0, 1, 0]
    drift_series = [0.0, 0.0, 0.0]
    wake_log = []
    woke_once = False
    final_state = "waiting"
    for tick in range(3):
        budget = budget_series[tick]
        demand = demand_series[tick]
        drift = drift_series[tick]
        wake = (demand > 0 or drift >= 0.20) and budget >= 0.30
        event = {
            "tick": tick,
            "budget": budget,
            "demand": demand,
            "drift": drift,
            "wake": bool(wake),
        }
        if wake:
            # re-run the utility check; with a new capability demand, growing is the only
            # candidate with positive predicted utility
            utility3 = {
                "grow_learn_familyB": 0.70 - 0.0 - 0.10,
                "retire_phaseA": -EV_RETIRE_DAMAGE - 0.02,
                "wait": 0.0,
            }
            action = max(utility3, key=utility3.get)
            acc4_probe = flip_acc(subj, head4, eval4, hi=96)
            acc8_probe = flip_acc(subj, head8, eval8, hi=120)
            event.update({"recalled_4c": acc4_probe, "recalled_8c": acc8_probe, "action": action})
            if action == "grow_learn_familyB":
                old_w_wake = units.extra_proj.weight.detach().clone()
                subj.graph.grow_units_neutral_(2)
                units = subj.graph.selective_units
                subj.n_units = 144
                proj_w = torch.nn.Linear(4, 144, bias=False)
                with torch.no_grad():
                    proj_w.weight[: old_w_wake.shape[0]] = old_w_wake
                    proj_w.weight[old_w_wake.shape[0] :] = 0.0
                units.extra_proj = proj_w
                with torch.no_grad():
                    s_idx = torch.arange(24)
                    d_idx = torch.arange(24) + 120
                    units.group_index[d_idx] = torch.arange(24) // 6
                    units.r[d_idx] = units.r[s_idx].clone()
                    units.p[:, d_idx] = units.p[:, s_idx].clone()
                    for nm in ("c", "eta", "b", "write_bias"):
                        getattr(units, nm)[d_idx] = getattr(units, nm)[s_idx].clone()
                    units.w[d_idx, :] = units.w[s_idx, :].clone()
                    units.write_in.weight[d_idx, :] = units.write_in.weight[s_idx, :].clone()
                    units.write_state.weight[d_idx, :] = units.write_state.weight[s_idx, :].clone()
                    units.write_bias[d_idx] = 0.0
                    units.b[d_idx] = 0.0
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(SEED + 18001)
                    headB = PhaseHead(2 * (144 - OLD_N), 4, 64)
                paramsB = list(headB.parameters()) + list(newborn_params(subj))
                optB = torch.optim.Adam(paramsB, lr=6e-3)
                head4_keep = copy.deepcopy(head4)
                for p_keep in head4_keep.parameters():
                    p_keep.requires_grad_(False)
                head4_keep.eval()
                head8_keep = copy.deepcopy(head8)
                for p_keep in head8_keep.parameters():
                    p_keep.requires_grad_(False)
                head8_keep.eval()
                for i_b in range(1600):
                    batchB = revision4b(24, SEED * 7600 + i_b)
                    batchA4 = revision_split(24, SEED * 7700 + i_b)
                    batchA8 = revision8(24, SEED * 7800 + i_b)
                    subj.train_mode()
                    fB = cohort_feat(subj, batchB["ticks"], extra_of(batchB), 144)
                    fA4 = cohort_feat(subj, batchA4["ticks"], extra_of(batchA4), 96)
                    fA8 = cohort_feat(subj, batchA8["ticks"], extra_of(batchA8), 120)
                    loss = (
                        F.cross_entropy(headB(fB), batchB["flip_labels"])
                        + 0.5 * F.cross_entropy(head4_keep(fA4), batchA4["flip_labels"])
                        + 0.25 * F.cross_entropy(head8_keep(fA8), batchA8["flip_labels"])
                    )
                    optB.zero_grad(set_to_none=True)
                    loss.backward()
                    mask_newborn_grads(subj, OLD_N)
                    torch.nn.utils.clip_grad_norm_(paramsB, 1.0)
                    optB.step()
                evalB = revision4b(1024, SEED + 1400 + 10_000_000)
                accB = flip_acc(subj, headB, evalB, hi=144)
                acc4_after_B = flip_acc(subj, head4, eval4, hi=96)
                acc8_after_B = flip_acc(subj, head8, eval8, hi=120)
                subj.graph.selective_units.set_unit_lifecycle_(range(120, 144), "dormant")
                accB_lesion = flip_acc(subj, headB, evalB, hi=144)
                subj.graph.selective_units.set_unit_lifecycle_(range(120, 144), "mature")
                event.update(
                    {
                        "accB": accB,
                        "acc4_after_B": acc4_after_B,
                        "acc8_after_B": acc8_after_B,
                        "accB_lesion": accB_lesion,
                    }
                )
                final_state = "waiting"
                woke_once = True
            else:
                woke_once = True
                final_state = "waiting" if action == "wait" else action
        else:
            final_state = "waiting"
        wake_log.append(event)

    # ---- session 2 boundary: persist the A+B subject and restore into a fresh one ----
    cap_AB = build_capsule_v2(subj.graph)
    fresh_AB = SelectivePersistentGraph(subj.graph.base_graph, subj.graph.selective_spec)
    fresh_AB.freeze_base_()
    fresh_AB.selective_units.attach_extra_(4, seed=SEED)
    apply_capsule_v2(fresh_AB, cap_AB)
    post_B = next((e for e in wake_log if "accB" in e), {})
    acc4_at_capAB = post_B.get("acc4_after_B", acc4_after)
    acc8_at_capAB = post_B.get("acc8_after_B", acc8)
    accB_at_capAB = post_B.get("accB", 0.0)
    holder_AB = Holder(fresh_AB, subj.n_units)
    acc4_sess3 = flip_acc(holder_AB, head4, eval4, hi=96)
    acc8_sess3 = flip_acc(holder_AB, head8, eval8, hi=120)
    accB_sess3 = flip_acc(holder_AB, headB, evalB, hi=144)
    with torch.no_grad():
        z0b, a0b = run_states(
            subj, eval8["ticks"], mask_fn(subj.n_units, OLD_N, subj.n_units), extra=extra_of(eval8)
        )
        z1b, a1b = run_states(
            holder_AB,
            eval8["ticks"],
            mask_fn(holder_AB.n_units, OLD_N, holder_AB.n_units),
            extra=extra_of(eval8),
        )
    capAB_diff = float(max((z0b - z1b).abs().max(), (a0b - a1b).abs().max()))

    # ---- session 3: restore A+B, grow again, learn the third task family C ----
    old_w3 = holder_AB.graph.selective_units.extra_proj.weight.detach().clone()
    holder_AB.graph.grow_units_neutral_(2)
    units3 = holder_AB.graph.selective_units
    holder_AB.n_units = 168
    proj3 = torch.nn.Linear(4, 168, bias=False)
    with torch.no_grad():
        proj3.weight[: old_w3.shape[0]] = old_w3
        proj3.weight[old_w3.shape[0] :] = 0.0
    units3.extra_proj = proj3
    with torch.no_grad():
        s3 = torch.arange(24)
        d3 = torch.arange(24) + 144
        units3.group_index[d3] = torch.arange(24) // 6
        units3.r[d3] = units3.r[s3].clone()
        units3.p[:, d3] = units3.p[:, s3].clone()
        for nm in ("c", "eta", "b", "write_bias"):
            getattr(units3, nm)[d3] = getattr(units3, nm)[s3].clone()
        units3.w[d3, :] = units3.w[s3, :].clone()
        units3.write_in.weight[d3, :] = units3.write_in.weight[s3, :].clone()
        units3.write_state.weight[d3, :] = units3.write_state.weight[s3, :].clone()
        units3.write_bias[d3] = 0.0
        units3.b[d3] = 0.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 18101)
        headC = PhaseHead(2 * (168 - OLD_N), 4, 64)
    paramsC = list(headC.parameters()) + list(newborn_params(holder_AB))
    optC = torch.optim.Adam(paramsC, lr=6e-3)
    head4_keepC = copy.deepcopy(head4)
    head8_keepC = copy.deepcopy(head8)
    headB_keepC = copy.deepcopy(headB)
    for hk in (head4_keepC, head8_keepC, headB_keepC):
        for p_k in hk.parameters():
            p_k.requires_grad_(False)
        hk.eval()
    for i_c in range(1600):
        batchC = revision4c(24, SEED * 7900 + i_c)
        batchA4 = revision_split(24, SEED * 8000 + i_c)
        batchA8 = revision8(24, SEED * 8100 + i_c)
        batchB = revision4b(24, SEED * 8200 + i_c)
        holder_AB.train_mode()
        fC = cohort_feat(holder_AB, batchC["ticks"], extra_of(batchC), 168)
        fA4 = cohort_feat(holder_AB, batchA4["ticks"], extra_of(batchA4), 96)
        fA8 = cohort_feat(holder_AB, batchA8["ticks"], extra_of(batchA8), 120)
        fB = cohort_feat(holder_AB, batchB["ticks"], extra_of(batchB), 144)
        loss = (
            F.cross_entropy(headC(fC), batchC["flip_labels"])
            + 0.5 * F.cross_entropy(head4_keepC(fA4), batchA4["flip_labels"])
            + 0.25 * F.cross_entropy(head8_keepC(fA8), batchA8["flip_labels"])
            + 0.5 * F.cross_entropy(headB_keepC(fB), batchB["flip_labels"])
        )
        optC.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(holder_AB, OLD_N)
        torch.nn.utils.clip_grad_norm_(paramsC, 1.0)
        optC.step()
    evalC = revision4c(1024, SEED + 1500 + 10_000_000)
    accC = flip_acc(holder_AB, headC, evalC, hi=168)
    accA4_s3 = flip_acc(holder_AB, head4, eval4, hi=96)
    accA8_s3 = flip_acc(holder_AB, head8, eval8, hi=120)
    accB_s4 = flip_acc(holder_AB, headB, evalB, hi=144)
    cap_ABC = build_capsule_v2(holder_AB.graph)
    fresh_ABC = SelectivePersistentGraph(holder_AB.graph.base_graph, holder_AB.graph.selective_spec)
    fresh_ABC.freeze_base_()
    fresh_ABC.selective_units.attach_extra_(4, seed=SEED)
    apply_capsule_v2(fresh_ABC, cap_ABC)
    holder_ABC = Holder(fresh_ABC, holder_AB.n_units)
    accC_restore = flip_acc(holder_ABC, headC, evalC, hi=168)
    accA4_restore = flip_acc(holder_ABC, head4, eval4, hi=96)
    accB_restore = flip_acc(holder_ABC, headB, evalB, hi=144)
    with torch.no_grad():
        zc0, ac0 = run_states(
            holder_AB,
            eval8["ticks"],
            mask_fn(holder_AB.n_units, OLD_N, holder_AB.n_units),
            extra=extra_of(eval8),
        )
        zc1, ac1 = run_states(
            holder_ABC,
            eval8["ticks"],
            mask_fn(holder_ABC.n_units, OLD_N, holder_ABC.n_units),
            extra=extra_of(eval8),
        )
    capABC_diff = float(max((zc0 - zc1).abs().max(), (ac0 - ac1).abs().max()))
    normal_cn_s3 = cn_eval(subj)

    # ---- decision: learn a fourth capability? (cost-aware) ----
    EV_GAIN = 0.75  # measured growth gains in earlier cards
    COST_LEARN = 0.30  # normalised training cost measured in earlier cards
    utility_d = {
        "learn_familyD": EV_GAIN - COST_LEARN,
        "retire_familyC_cohort": -0.70 - 0.02,
        "wait": 0.0,
    }
    chosen_d = max(utility_d, key=utility_d.get)
    accD = 0.0
    accA4_d = accA4_s3
    accB_d = accB_s4
    accC_d = accC
    even_after_D = normal_cn_s3
    capABCD_diff = 0.0
    if chosen_d == "learn_familyD":
        old_w4 = holder_AB.graph.selective_units.extra_proj.weight.detach().clone()
        holder_AB.graph.grow_units_neutral_(2)
        units4 = holder_AB.graph.selective_units
        holder_AB.n_units = 192
        proj4 = torch.nn.Linear(4, 192, bias=False)
        with torch.no_grad():
            proj4.weight[: old_w4.shape[0]] = old_w4
            proj4.weight[old_w4.shape[0] :] = 0.0
        units4.extra_proj = proj4
        with torch.no_grad():
            s4 = torch.arange(24)
            d4 = torch.arange(24) + 168
            units4.group_index[d4] = torch.arange(24) // 6
            units4.r[d4] = units4.r[s4].clone()
            units4.p[:, d4] = units4.p[:, s4].clone()
            for nm in ("c", "eta", "b", "write_bias"):
                getattr(units4, nm)[d4] = getattr(units4, nm)[s4].clone()
            units4.w[d4, :] = units4.w[s4, :].clone()
            units4.write_in.weight[d4, :] = units4.write_in.weight[s4, :].clone()
            units4.write_state.weight[d4, :] = units4.write_state.weight[s4, :].clone()
            units4.write_bias[d4] = 0.0
            units4.b[d4] = 0.0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 18201)
            headD = PhaseHead(2 * (192 - OLD_N), 4, 64)
        paramsD = list(headD.parameters()) + list(newborn_params(holder_AB))
        optD = torch.optim.Adam(paramsD, lr=6e-3)
        keep = [head4, head8, headB, headC]
        keeps = []
        for hk in keep:
            hkc = copy.deepcopy(hk)
            for p_k in hkc.parameters():
                p_k.requires_grad_(False)
            hkc.eval()
            keeps.append(hkc)
        h4k, h8k, hBk, hCk = keeps
        for i_d in range(1600):
            batchD = revision4d(24, SEED * 8300 + i_d)
            batchA4 = revision_split(24, SEED * 8400 + i_d)
            batchA8 = revision8(24, SEED * 8500 + i_d)
            batchB = revision4b(24, SEED * 8600 + i_d)
            batchC = revision4c(24, SEED * 8700 + i_d)
            holder_AB.train_mode()
            fD = cohort_feat(holder_AB, batchD["ticks"], extra_of(batchD), 192)
            fA4 = cohort_feat(holder_AB, batchA4["ticks"], extra_of(batchA4), 96)
            fA8 = cohort_feat(holder_AB, batchA8["ticks"], extra_of(batchA8), 120)
            fB = cohort_feat(holder_AB, batchB["ticks"], extra_of(batchB), 144)
            fC = cohort_feat(holder_AB, batchC["ticks"], extra_of(batchC), 168)
            loss = (
                F.cross_entropy(headD(fD), batchD["flip_labels"])
                + 0.5 * F.cross_entropy(h4k(fA4), batchA4["flip_labels"])
                + 0.25 * F.cross_entropy(h8k(fA8), batchA8["flip_labels"])
                + 0.5 * F.cross_entropy(hBk(fB), batchB["flip_labels"])
                + 0.5 * F.cross_entropy(hCk(fC), batchC["flip_labels"])
            )
            optD.zero_grad(set_to_none=True)
            loss.backward()
            mask_newborn_grads(holder_AB, OLD_N)
            torch.nn.utils.clip_grad_norm_(paramsD, 1.0)
            optD.step()
        evalD = revision4d(1024, SEED + 1600 + 10_000_000)
        accD = flip_acc(holder_AB, headD, evalD, hi=192)
        accA4_d = flip_acc(holder_AB, head4, eval4, hi=96)
        accB_d = flip_acc(holder_AB, headB, evalB, hi=144)
        accC_d = flip_acc(holder_AB, headC, evalC, hi=168)
        even_after_D = normal_cn_s3
        cap_ABCD = build_capsule_v2(holder_AB.graph)
        fresh_ABCD = SelectivePersistentGraph(
            holder_AB.graph.base_graph, holder_AB.graph.selective_spec
        )
        fresh_ABCD.freeze_base_()
        fresh_ABCD.selective_units.attach_extra_(4, seed=SEED)
        apply_capsule_v2(fresh_ABCD, cap_ABCD)
        holder_ABCD = Holder(fresh_ABCD, holder_AB.n_units)
        accD_restore = flip_acc(holder_ABCD, headD, evalD, hi=192)
        accC_restore_d = flip_acc(holder_ABCD, headC, evalC, hi=168)
        with torch.no_grad():
            zd0, ad0 = run_states(
                holder_AB,
                eval8["ticks"],
                mask_fn(holder_AB.n_units, OLD_N, holder_AB.n_units),
                extra=extra_of(eval8),
            )
            zd1, ad1 = run_states(
                holder_ABCD,
                eval8["ticks"],
                mask_fn(holder_ABCD.n_units, OLD_N, holder_ABCD.n_units),
                extra=extra_of(eval8),
            )
        capABCD_diff = float(max((zd0 - zd1).abs().max(), (ad0 - ad1).abs().max()))
    else:
        accD_restore = 0.0
        accC_restore_d = accC

    # ---- long-horizon loop: wait -> learn E -> verify -> wait ----
    budget_series = [0.05, 0.45, 0.45, 0.45, 0.45, 0.45]
    demand_series = [0, 1, 0, 1, 0, 0]
    drift_series = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    loop2 = []
    cost_accum = 0.0
    accE = 0.0
    accE_lesion = 0.0
    accalls = {}
    for tick2 in range(6):
        budget = budget_series[tick2]
        demand = demand_series[tick2]
        drift = drift_series[tick2]
        wake2 = (demand > 0 or drift >= 0.20) and budget >= 0.30
        ev = {"tick": tick2, "budget": budget, "demand": demand, "wake": bool(wake2)}
        if wake2 and tick2 == 1:
            # learn the fifth family E on a fresh cohort
            old_w5 = holder_AB.graph.selective_units.extra_proj.weight.detach().clone()
            holder_AB.graph.grow_units_neutral_(2)
            units5 = holder_AB.graph.selective_units
            holder_AB.n_units = 216
            proj5 = torch.nn.Linear(4, 216, bias=False)
            with torch.no_grad():
                proj5.weight[: old_w5.shape[0]] = old_w5
                proj5.weight[old_w5.shape[0] :] = 0.0
            units5.extra_proj = proj5
            with torch.no_grad():
                s5 = torch.arange(24)
                d5 = torch.arange(24) + 192
                units5.group_index[d5] = torch.arange(24) // 6
                units5.r[d5] = units5.r[s5].clone()
                units5.p[:, d5] = units5.p[:, s5].clone()
                for nm in ("c", "eta", "b", "write_bias"):
                    getattr(units5, nm)[d5] = getattr(units5, nm)[s5].clone()
                units5.w[d5, :] = units5.w[s5, :].clone()
                units5.write_in.weight[d5, :] = units5.write_in.weight[s5, :].clone()
                units5.write_state.weight[d5, :] = units5.write_state.weight[s5, :].clone()
                units5.write_bias[d5] = 0.0
                units5.b[d5] = 0.0
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED + 18301)
                headE = PhaseHead(2 * (216 - OLD_N), 4, 64)
            paramsE = list(headE.parameters()) + list(newborn_params(holder_AB))
            optE = torch.optim.Adam(paramsE, lr=6e-3)
            keepsE = []
            for hk in (head4, head8, headB, headC, headD):
                hkc = copy.deepcopy(hk)
                for p_k in hkc.parameters():
                    p_k.requires_grad_(False)
                hkc.eval()
                keepsE.append(hkc)
            h4e, h8e, hBe, hCe, hDe = keepsE
            for i_e in range(1600):
                batchE = revision4e(24, SEED * 8800 + i_e)
                batchA4 = revision_split(24, SEED * 8900 + i_e)
                batchA8 = revision8(24, SEED * 9000 + i_e)
                batchB = revision4b(24, SEED * 9100 + i_e)
                batchC = revision4c(24, SEED * 9200 + i_e)
                batchD = revision4d(24, SEED * 9300 + i_e)
                holder_AB.train_mode()
                fE = cohort_feat(holder_AB, batchE["ticks"], extra_of(batchE), 216)
                fA4 = cohort_feat(holder_AB, batchA4["ticks"], extra_of(batchA4), 96)
                fA8 = cohort_feat(holder_AB, batchA8["ticks"], extra_of(batchA8), 120)
                fB = cohort_feat(holder_AB, batchB["ticks"], extra_of(batchB), 144)
                fC = cohort_feat(holder_AB, batchC["ticks"], extra_of(batchC), 168)
                fD = cohort_feat(holder_AB, batchD["ticks"], extra_of(batchD), 192)
                loss = (
                    F.cross_entropy(headE(fE), batchE["flip_labels"])
                    + 0.5 * F.cross_entropy(h4e(fA4), batchA4["flip_labels"])
                    + 0.25 * F.cross_entropy(h8e(fA8), batchA8["flip_labels"])
                    + 0.5 * F.cross_entropy(hBe(fB), batchB["flip_labels"])
                    + 0.5 * F.cross_entropy(hCe(fC), batchC["flip_labels"])
                    + 0.5 * F.cross_entropy(hDe(fD), batchD["flip_labels"])
                )
                optE.zero_grad(set_to_none=True)
                loss.backward()
                mask_newborn_grads(holder_AB, OLD_N)
                torch.nn.utils.clip_grad_norm_(paramsE, 1.0)
                optE.step()
            evalE = revision4e(1024, SEED + 1700 + 10_000_000)
            accE = flip_acc(holder_AB, headE, evalE, hi=216)
            accalls = {
                "A4": flip_acc(holder_AB, head4, eval4, hi=96),
                "B": flip_acc(holder_AB, headB, evalB, hi=144),
                "C": flip_acc(holder_AB, headC, evalC, hi=168),
                "D": flip_acc(holder_AB, headD, evalD, hi=192),
            }
            holder_AB.graph.selective_units.set_unit_lifecycle_(range(192, 216), "dormant")
            accE_lesion = flip_acc(holder_AB, headE, evalE, hi=216)
            holder_AB.graph.selective_units.set_unit_lifecycle_(range(192, 216), "mature")
            cost_accum += 1600 / 16000.0
            ev.update({"action": "learn_familyE", "accE": accE})
            cap_E = build_capsule_v2(holder_AB.graph)
            fresh_E = SelectivePersistentGraph(
                holder_AB.graph.base_graph, holder_AB.graph.selective_spec
            )
            fresh_E.freeze_base_()
            fresh_E.selective_units.attach_extra_(4, seed=SEED)
            apply_capsule_v2(fresh_E, cap_E)
            holder_E = Holder(fresh_E, holder_AB.n_units)
            accE_restore = flip_acc(holder_E, headE, evalE, hi=216)
            with torch.no_grad():
                ze0, ae0 = run_states(
                    holder_AB,
                    eval8["ticks"],
                    mask_fn(holder_AB.n_units, OLD_N, holder_AB.n_units),
                    extra=extra_of(eval8),
                )
                ze1, ae1 = run_states(
                    holder_E,
                    eval8["ticks"],
                    mask_fn(holder_E.n_units, OLD_N, holder_E.n_units),
                    extra=extra_of(eval8),
                )
            capE_diff = float(max((ze0 - ze1).abs().max(), (ae0 - ae1).abs().max()))
            ev.update({"capE_diff": capE_diff, "accE_restore": accE_restore})
        elif wake2 and tick2 == 3:
            recalled = {
                "A4": flip_acc(holder_AB, head4, eval4, hi=96),
                "B": flip_acc(holder_AB, headB, evalB, hi=144),
                "C": flip_acc(holder_AB, headC, evalC, hi=168),
                "D": flip_acc(holder_AB, headD, evalD, hi=192),
                "E": flip_acc(holder_AB, headE, evalE, hi=216),
            }
            ev.update({"action": "recall_verify", "recalled": recalled})
        else:
            ev.update({"action": "wait"})
        loop2.append(ev)

    normal_after = cn_eval(subj)
    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_4c_retained": acc4_after >= 0.85,
        "G2_8c_learned": acc8 >= 0.90,
        "G3_8c_unseen": acc8_unseen >= 0.60,
        "G4_lesion_drop": (acc8 - acc8_lesion) >= 0.15,
        "G5_4c_integration_ok": acc4_lesion >= 0.80,
        "G6_normal_retained": (normal_cn - normal_after) <= 0.01,
        "G7_wake_exact": abs(acc8_wake - acc8) <= 1e-9 and abs(acc4_wake - acc4_after) <= 1e-9,
        "G9_pressure": (cn_probe["test"] - null_cn_mean) <= 0.10
        and (birth_probe["test"] - null_birth_mean) <= 0.10,
        "G15_choice": chosen_action == "grow_and_train_8c",
        "G16_gain": actual_gain >= 0.70,
        "G17_retire_damage_consistent": 0.50 <= actual_retire_damage <= 1.00,
        "G18_regret": regret <= 0.15,
        "G19_second_choice": chosen_action2 == "wait",
        "G20_evidence_update": abs(EV_GROW_GAIN - actual_gain) <= 0.10
        and abs(EV_RETIRE_DAMAGE - actual_retire_damage) <= 0.10,
        "G21_wait_no_change": abs(acc4_after_wait - acc4_before_wait) <= 1e-9
        and abs(acc8_after_wait - acc8_before_wait) <= 1e-9,
        "G22_wake_conditions": [e["wake"] for e in wake_log] == [False, True, False],
        "G23_wake_recall_ok": woke_once
        and wake_log[1]["recalled_4c"] >= 0.90
        and wake_log[1]["recalled_8c"] >= 0.90,
        "G24_final_wait": final_state == "waiting",
        "G25_wake_action": woke_once
        and any(e.get("action") == "grow_learn_familyB" for e in wake_log),
        "G26_familyB": max([e.get("accB", 0.0) for e in wake_log]) >= 0.90,
        "G27_familyA_4c": min([e.get("acc4_after_B", 1.0) for e in wake_log]) >= 0.85,
        "G28_familyA_8c": min([e.get("acc8_after_B", 1.0) for e in wake_log]) >= 0.85,
        "G30_session1_restore": capA_diff <= 1e-6
        and abs(acc4_sess2 - acc4_after) <= 1e-9
        and abs(acc8_sess2 - acc8) <= 1e-9,
        "G31_session2_restore": capAB_diff <= 1e-6
        and abs(acc4_sess3 - acc4_at_capAB) <= 1e-9
        and abs(acc8_sess3 - acc8_at_capAB) <= 1e-9,
        "G32_familyB_restored": abs(accB_sess3 - accB_at_capAB) <= 1e-9,
        "G44_familyC": accC >= 0.90,
        "G45_A_retained_s3": accA4_s3 >= 0.85 and accA8_s3 >= 0.85,
        "G46_B_retained_s3": accB_s4 >= 0.85,
        "G47_ABC_restore": capABC_diff <= 1e-6
        and abs(accC_restore - accC) <= 1e-9
        and abs(accA4_restore - accA4_s3) <= 1e-9
        and abs(accB_restore - accB_s4) <= 1e-9,
        "G48_normal_s3": (normal_cn - normal_cn_s3) <= 0.01,
        "G49_fourth_choice": chosen_d == "learn_familyD",
        "G50_fourth_capability": accD >= 0.90,
        "G51_retention_after_D": accA4_d >= 0.85 and accB_d >= 0.85 and accC_d >= 0.85,
        "G52_ABCD_restore": capABCD_diff <= 1e-6
        and abs(accD_restore - accD) <= 1e-9
        and abs(accC_restore_d - accC_d) <= 1e-9,
        "G53_long_wake_pattern": [bool(e["wake"]) for e in loop2]
        == [False, True, False, True, False, False],
        "G54_familyE": accE >= 0.90,
        "G55_E_causal": (accE - accE_lesion) >= 0.15,
        "G56_retention_ABCD": min(accalls.values()) >= 0.85,
        "G57_verify_recall": all(v >= 0.85 for v in loop2[3].get("recalled", {"x": 0}).values()),
        "G58_ABCDE_restore": capE_diff <= 1e-6 and abs(accE_restore - accE) <= 1e-9,
        "G59_cost": cost_accum <= 1.0,
        "G29_familyB_lesion": any(
            (e.get("accB", 0.0) - e.get("accB_lesion", 0.0)) >= 0.15 for e in wake_log
        ),
        "G8_capsule_exact": capsule_diff <= 1e-6
        and abs(acc8_capsule - acc8) <= 1e-9
        and abs(acc4_capsule - acc4_after) <= 1e-9,
    }
    if not gates["G0_old_cn"]:
        verdict = "p0ca_cn_base_failed"
    elif not gates["G1_4c_retained"]:
        verdict = "p0ca_old_ability_lost"
    elif not gates["G2_8c_learned"]:
        verdict = "p0ca_new_ability_failed"
    elif not (gates["G4_lesion_drop"] and gates["G5_4c_integration_ok"]):
        verdict = "p0ca_growth_not_causal"
    elif not gates["G3_8c_unseen"]:
        verdict = "p0ca_unseen_gap"
    elif not gates["G6_normal_retained"]:
        verdict = "p0ca_retention_breach"
    elif not gates["G9_pressure"]:
        verdict = "p0ca_not_pressured"
    elif not (gates["G15_choice"] and gates["G16_gain"] and gates["G17_retire_damage_consistent"]):
        verdict = "p0ca_choice_mismatch"
    elif not gates["G18_regret"]:
        verdict = "p0ca_regret_too_large"
    elif not (
        gates["G19_second_choice"] and gates["G20_evidence_update"] and gates["G21_wait_no_change"]
    ):
        verdict = "p0ca_second_decision_failed"
    elif not (
        gates["G22_wake_conditions"] and gates["G23_wake_recall_ok"] and gates["G24_final_wait"]
    ):
        verdict = "p0ca_wake_condition_failed"
    elif not (
        gates["G25_wake_action"]
        and gates["G26_familyB"]
        and gates["G27_familyA_4c"]
        and gates["G28_familyA_8c"]
        and gates["G29_familyB_lesion"]
    ):
        verdict = "p0ca_wake_grow_failed"
    elif not (
        gates["G30_session1_restore"]
        and gates["G31_session2_restore"]
        and gates["G32_familyB_restored"]
    ):
        verdict = "p0ca_session_restore_failed"
    elif not (gates["G44_familyC"] and gates["G45_A_retained_s3"] and gates["G46_B_retained_s3"]):
        verdict = "p0ca_third_capability_failed"
    elif not (gates["G47_ABC_restore"] and gates["G48_normal_s3"]):
        verdict = "p0ca_third_restore_failed"
    elif not (
        gates["G49_fourth_choice"]
        and gates["G50_fourth_capability"]
        and gates["G51_retention_after_D"]
    ):
        verdict = "p0ca_fourth_failed"
    elif not gates["G52_ABCD_restore"]:
        verdict = "p0ca_fourth_restore_failed"
    elif not (
        gates["G53_long_wake_pattern"]
        and gates["G54_familyE"]
        and gates["G55_E_causal"]
        and gates["G56_retention_ABCD"]
        and gates["G57_verify_recall"]
    ):
        verdict = "p0ca_long_loop_failed"
    elif not gates["G58_ABCDE_restore"]:
        verdict = "p0ca_loop_restore_failed"
    elif not gates["G7_wake_exact"]:
        verdict = "p0ca_wake_not_restored"
    elif not gates["G8_capsule_exact"]:
        verdict = "p0ca_capsule_failed"
    else:
        verdict = "p0ca_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0ca-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "long_horizon_continuous_loop",
        "normal_cn": normal_cn,
        "cn_probe": cn_probe,
        "birth_probe": birth_probe,
        "null_cn_mean": null_cn_mean,
        "null_birth_mean": null_birth_mean,
        "normal_cn_after": normal_after,
        "acc4_before_phaseB": acc4_before,
        "acc4_after_phaseB": acc4_after,
        "acc8_seen": acc8,
        "acc8_unseen": acc8_unseen,
        "acc8_lesion_phaseB_cohort": acc8_lesion,
        "acc4_lesion_phaseB_cohort": acc4_lesion,
        "acc8_wake": acc8_wake,
        "acc4_wake": acc4_wake,
        "capsule_diff": capsule_diff,
        "utility": utility,
        "chosen_action": chosen_action,
        "actual_gain": actual_gain,
        "acc8_after_retire": acc8_after_retire,
        "actual_retire_damage": actual_retire_damage,
        "regret": regret,
        "EV_GROW_GAIN_updated": EV_GROW_GAIN,
        "EV_RETIRE_DAMAGE_updated": EV_RETIRE_DAMAGE,
        "utility2": utility2,
        "chosen_action2": chosen_action2,
        "wake_log": wake_log,
        "capA_diff": capA_diff,
        "capAB_diff": capAB_diff,
        "acc4_sess2": acc4_sess2,
        "acc8_sess2": acc8_sess2,
        "acc4_sess3": acc4_sess3,
        "acc8_sess3": acc8_sess3,
        "accB_sess3": accB_sess3,
        "acc4_at_capAB": acc4_at_capAB,
        "acc8_at_capAB": acc8_at_capAB,
        "accB_at_capAB": accB_at_capAB,
        "long_loop": {
            "ticks": loop2,
            "accE": accE,
            "accalls": accalls,
            "cost_accum": cost_accum,
        },
        "d4": {
            "utility": utility_d,
            "chosen": chosen_d,
            "accD": accD,
            "accA4": accA4_d,
            "accB": accB_d,
            "accC": accC_d,
            "capABCD_diff": capABCD_diff,
            "accD_restore": accD_restore,
            "accC_restore": accC_restore_d,
            "normal": even_after_D,
        },
        "s3": {
            "accC": accC,
            "accA4": accA4_s3,
            "accA8": accA8_s3,
            "accB": accB_s4,
            "capABC_diff": capABC_diff,
            "accC_restore": accC_restore,
            "accA4_restore": accA4_restore,
            "accB_restore": accB_restore,
            "normal_cn": normal_cn_s3,
        },
        "accB": max([e.get("accB", 0.0) for e in wake_log]),
        "acc4_after_B": min([e.get("acc4_after_B", 1.0) for e in wake_log]),
        "acc8_after_B": min([e.get("acc8_after_B", 1.0) for e in wake_log]),
        "woke_once": woke_once,
        "final_state": final_state,
        "acc8_capsule": acc8_capsule,
        "acc4_capsule": acc4_capsule,
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": cost},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
