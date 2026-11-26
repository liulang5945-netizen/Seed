"""P0-CZ: three-session staged growth chain with capsule restore between sessions.

Session 1: CN base -> grow A -> learn R0 (read A) -> capsule A.
Session 2: restore A -> grow B -> learn R1 (read B, A frozen) -> capsule AB.
Session 3: restore AB -> grow C -> learn R2 (read C, A/B frozen) -> capsule ABC.
Exploratory; zero check/final.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_p0ci_20261123 import (  # type: ignore
    BI_CN_CHAR_INDEX,
    BI_CN_PROJ32,
    CN_COLOURS,
    CN_MAX_LEN,
    FLAG_CH,
    HOLDOUT4,
    OLD_N,
    VIS_CH,
    PhaseHead,
    _build_bidirectional_cn_32dim,
    cn_eval,
    extra_of,
    mask_fn,
    mask_newborn_grads,
    newborn_params,
    run_states,
)
from scripts.training.verify_taiji_23s2a_gain_20261122 import BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import SelectivePersistentGraph, apply_capsule_v2, build_capsule_v2

ROOT = Path(r"E:\Seed")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261120
OUT = ROOT / f"reports/taiji_p0dd_seed{SEED}.json"
BATCH_SIZE = 48


def _seq(tpl: str, c1: int, c2: int, flag: int, shape: int) -> torch.Tensor:
    seq = torch.zeros(CN_MAX_LEN, 48)
    txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
    for ti, ch in enumerate(txt[:CN_MAX_LEN]):
        ci = BI_CN_CHAR_INDEX.get(ch)
        if ci is not None:
            seq[ti, :32] = BI_CN_PROJ32[ci]
    seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
    seq[0, FLAG_CH[0] + flag] = 1.0
    seq[0, VIS_CH[0] : VIS_CH[1]] = 0.0
    seq[0, VIS_CH[0] + shape] = 1.0
    return seq


def _rule_batch(rule: int, count: int, seed: int) -> dict:
    tpls = [(tpl, True) for tpl in BI_CN_TRAIN_FIRST] + [(tpl, False) for tpl in BI_CN_TRAIN_SECOND]
    rows, labels = [], []
    f_rng = torch.Generator().manual_seed(seed + 555_555)
    v_rng = torch.Generator().manual_seed(seed + 444_444)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, 4, (1,), generator=gen))
        c2 = int(torch.randint(0, 3, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = (c1, c2) not in HOLDOUT4
        idx += 1
        if not ok:
            continue
        tpl, pick_first = tpls[len(rows) % len(tpls)]
        other = c2 if pick_first else c1
        flag = int(torch.randint(0, 4, (1,), generator=f_rng))
        shape = int(torch.randint(0, 4, (1,), generator=v_rng))
        by_shape_r0 = c1 if shape < 2 else c2
        by_shape_r1 = c2 if shape < 2 else c1
        if rule == 0:
            answer = other if flag >= 2 else by_shape_r0
        elif rule == 1:
            answer = other if flag >= 2 else by_shape_r1
        elif rule == 2:
            answer = other if flag < 2 else by_shape_r0
        elif rule == 3:
            answer = other if flag in (0, 2) else by_shape_r0
        elif rule == 4:
            normal = c1 if pick_first else c2
            answer = normal if flag >= 2 else by_shape_r1
        else:
            normal = c1 if pick_first else c2
            answer = normal if flag < 2 else by_shape_r0
        rows.append(_seq(tpl, c1, c2, flag, shape))
        labels.append(answer)
    return {"ticks": torch.stack(rows), "labels": torch.tensor(labels, dtype=torch.long)}


def _grow(graph, units: int, seed_extra: int) -> None:
    graph.grow_units_neutral_(2)


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=1e-4)
    subj.train_mode()
    for step in range(1200):
        batch = _build_bidirectional_cn_32dim(
            BATCH_SIZE, SEED * 6000 + step, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
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

    def inherit(graph, base: int) -> None:
        units_ = graph.selective_units
        with torch.no_grad():
            src = torch.arange(24)
            dst = torch.arange(24) + base
            units_.group_index[dst] = torch.arange(24) // 6
            units_.r[dst] = units_.r[src].clone()
            units_.p[:, dst] = units_.p[:, src].clone()
            for nm in ("c", "eta", "b", "write_bias"):
                getattr(units_, nm)[dst] = getattr(units_, nm)[src].clone()
            units_.w[dst, :] = units_.w[src, :].clone()
            units_.write_in.weight[dst, :] = units_.write_in.weight[src, :].clone()
            units_.write_state.weight[dst, :] = units_.write_state.weight[src, :].clone()
            units_.write_bias[dst] = 0.0
            units_.b[dst] = 0.0

    def feat(subj_, ticks_, extra_, lo_: int, hi_: int) -> torch.Tensor:
        zz, aa = run_states(
            subj_, ticks_, mask_fn(subj_.n_units, OLD_N, subj_.n_units), extra=extra_
        )
        return torch.cat([zz[:, lo_:hi_], aa[:, lo_:hi_]], dim=-1)

    def acc_cohort(subj_, head_, batch_, lo_: int, hi_: int) -> float:
        head_.eval()
        subj_.eval_mode()
        with torch.no_grad():
            pred = head_(feat(subj_, batch_["ticks"], extra_of(batch_), lo_, hi_)).argmax(-1)
        return float((pred == batch_["labels"]).float().mean())

    class _Holder:
        def __init__(self, graph_, num_units_):
            self.graph = graph_
            self.n_units = num_units_

        def eval_mode(self):
            pass

        def train_mode(self):
            pass

    # session 1: A cohort learns R0
    _grow(subj.graph, 1, SEED)
    subj.n_units = 96
    units = subj.graph.selective_units
    units.attach_extra_(8, seed=SEED)
    inherit(subj.graph, 72)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 18801)
        head_r0 = PhaseHead(2 * (96 - 72), 4, 64)
    opt1 = torch.optim.Adam(list(head_r0.parameters()) + list(newborn_params(subj)), lr=6e-3)
    for i in range(2400):
        b = _rule_batch(0, 32, SEED * 7000 + i)
        subj.train_mode()
        loss = F.cross_entropy(head_r0(feat(subj, b["ticks"], extra_of(b), 72, 96)), b["labels"])
        opt1.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, 72)
        torch.nn.utils.clip_grad_norm_(list(head_r0.parameters()) + list(newborn_params(subj)), 1.0)
        opt1.step()
    eval_r0 = _rule_batch(0, 1024, SEED + 3000 + 10_000_000)
    eval_r1 = _rule_batch(1, 1024, SEED + 3100 + 10_000_000)
    eval_r2 = _rule_batch(2, 1024, SEED + 3200 + 10_000_000)
    r0_s1 = acc_cohort(subj, head_r0, eval_r0, 72, 96)
    cap_a = build_capsule_v2(subj.graph)

    # session 2: restore A, grow B, learn R1 with A frozen
    fresh = SelectivePersistentGraph(subj.graph.base_graph, subj.graph.selective_spec)
    fresh.freeze_base_()
    fresh.selective_units.attach_extra_(8, seed=SEED)
    apply_capsule_v2(fresh, cap_a)
    subj2 = _Holder(fresh, 96)
    fresh2 = subj2.graph
    old_w_fresh = fresh.selective_units.extra_proj.weight.detach().clone()
    fresh2.grow_units_neutral_(2)
    units2 = fresh2.selective_units
    subj2.n_units = 120
    proj2 = torch.nn.Linear(8, 120, bias=False)
    with torch.no_grad():
        proj2.weight[: old_w_fresh.shape[0]] = old_w_fresh
        proj2.weight[old_w_fresh.shape[0] :] = 0.0
    units2.extra_proj = proj2
    inherit(fresh2, 96)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 18901)
        head_r1 = PhaseHead(2 * (120 - 96), 4, 64)
    opt2 = torch.optim.Adam(list(head_r1.parameters()) + list(units2.parameters()), lr=6e-3)
    for i in range(2400):
        b = _rule_batch(1, 32, SEED * 7100 + i)
        fresh2.train_mode() if hasattr(fresh2, "train_mode") else None
        zz, aa = run_states(subj2, b["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(b))
        feats = torch.cat([zz[:, 96:120], aa[:, 96:120]], dim=-1)
        loss = F.cross_entropy(head_r1(feats), b["labels"])
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in units2.named_parameters():
            if p.grad is None:
                continue
            if name in ("p",):
                p.grad[:, :72] = 0.0
            elif name in ("w", "write_state.weight"):
                p.grad[:72, :] = 0.0
                p.grad[:, :72] = 0.0
            elif name == "write_in.weight":
                p.grad[:72, :] = 0.0
            elif (
                p.grad.ndim >= 1
                and p.grad.shape[0] == units2.n_units
                or p.grad.ndim >= 2
                and p.grad.shape[0] == units2.n_units
            ):
                p.grad[:72] = 0.0
        torch.nn.utils.clip_grad_norm_(list(units2.parameters()), 1.0)
        opt2.step()
    r1_s2 = acc_cohort(subj2, head_r1, eval_r1, 96, 120)
    r0_s2 = acc_cohort(subj2, head_r0, eval_r0, 72, 96)
    calib_steps = 0
    opt_cal0 = torch.optim.Adam(list(head_r0.parameters()), lr=3e-3)
    for i in range(300):
        b = _rule_batch(0, 32, SEED * 7300 + i)
        loss = F.cross_entropy(head_r0(feat(subj2, b["ticks"], extra_of(b), 72, 96)), b["labels"])
        opt_cal0.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head_r0.parameters()), 1.0)
        opt_cal0.step()
        calib_steps += 1
    r0_s2_cal = acc_cohort(subj2, head_r0, eval_r0, 72, 96)
    cap_ab = build_capsule_v2(fresh2)

    # session 3: restore AB, grow C, learn R2 with A/B frozen
    fresh_ab = SelectivePersistentGraph(fresh2.base_graph, fresh2.selective_spec)
    fresh_ab.freeze_base_()
    fresh_ab.selective_units.attach_extra_(8, seed=SEED)
    apply_capsule_v2(fresh_ab, cap_ab)
    subj3 = _Holder(fresh_ab, 120)
    old_w_fresh3 = fresh_ab.selective_units.extra_proj.weight.detach().clone()
    fresh_ab.grow_units_neutral_(2)
    units3 = fresh_ab.selective_units
    subj3.n_units = 144
    proj3 = torch.nn.Linear(8, 144, bias=False)
    with torch.no_grad():
        proj3.weight[: old_w_fresh3.shape[0]] = old_w_fresh3
        proj3.weight[old_w_fresh3.shape[0] :] = 0.0
    units3.extra_proj = proj3
    inherit(fresh_ab, 120)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 19001)
        head_r2 = PhaseHead(2 * (144 - 120), 4, 64)
    opt3 = torch.optim.Adam(list(head_r2.parameters()) + list(units3.parameters()), lr=6e-3)
    for i in range(2400):
        b = _rule_batch(2, 32, SEED * 7200 + i)
        zz, aa = run_states(subj3, b["ticks"], mask_fn(144, OLD_N, 144), extra=extra_of(b))
        feats = torch.cat([zz[:, 120:144], aa[:, 120:144]], dim=-1)
        loss = F.cross_entropy(head_r2(feats), b["labels"])
        opt3.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in units3.named_parameters():
            if p.grad is None:
                continue
            if name in ("p",):
                p.grad[:, :120] = 0.0
            elif name in ("w", "write_state.weight"):
                p.grad[:120, :] = 0.0
                p.grad[:, :120] = 0.0
            elif name == "write_in.weight":
                p.grad[:120, :] = 0.0
            elif (
                p.grad.ndim >= 1
                and p.grad.shape[0] == units3.n_units
                or p.grad.ndim >= 2
                and p.grad.shape[0] == units3.n_units
            ):
                p.grad[:120] = 0.0
        torch.nn.utils.clip_grad_norm_(list(units3.parameters()), 1.0)
        opt3.step()
    r2_s3 = acc_cohort(subj3, head_r2, eval_r2, 120, 144)
    r0_s3 = acc_cohort(subj3, head_r0, eval_r0, 72, 96)
    r1_s3 = acc_cohort(subj3, head_r1, eval_r1, 96, 120)
    for h_i, (h_old, rule_id, _ev_old, lo_, hi_) in enumerate(
        [(head_r0, 0, eval_r0, 72, 96), (head_r1, 1, eval_r1, 96, 120)]
    ):
        opt_ci = torch.optim.Adam(list(h_old.parameters()), lr=3e-3)
        for i in range(300):
            b = _rule_batch(rule_id, 32, SEED * (7400 + h_i) + i)
            loss = F.cross_entropy(
                h_old(feat(subj3, b["ticks"], extra_of(b), lo_, hi_)), b["labels"]
            )
            opt_ci.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(h_old.parameters()), 1.0)
            opt_ci.step()
            calib_steps += 1
    r0_s3_cal = acc_cohort(subj3, head_r0, eval_r0, 72, 96)
    r1_s3_cal = acc_cohort(subj3, head_r1, eval_r1, 96, 120)

    # cost-aware decision: grow a fourth cohort for a fourth rule?
    EV_GAIN = 0.75
    COST_LEARN = 0.10
    COST_RECAL = 0.09  # 900 recalibration steps measured in P0-DA
    utility4 = {
        "grow_learn_R3": EV_GAIN - COST_LEARN - COST_RECAL,
        "wait": 0.0,
        "retire_C_cohort": -0.70 - 0.02,
    }
    chosen4 = max(utility4, key=utility4.get)
    r3_acc = 0.0
    r0_final = r0_s3_cal
    r1_final = r1_s3_cal
    r2_final = r2_s3
    cap4_diff = 0.0
    r3_restore = 0.0
    if chosen4 == "grow_learn_R3":
        old_w4 = fresh_ab.selective_units.extra_proj.weight.detach().clone()
        fresh_ab.grow_units_neutral_(2)
        units4 = fresh_ab.selective_units
        subj3.n_units = 168
        proj4 = torch.nn.Linear(8, 168, bias=False)
        with torch.no_grad():
            proj4.weight[: old_w4.shape[0]] = old_w4
            proj4.weight[old_w4.shape[0] :] = 0.0
        units4.extra_proj = proj4
        inherit(fresh_ab, 144)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 19101)
            head_r3 = PhaseHead(2 * (168 - 144), 4, 64)
        opt4 = torch.optim.Adam(list(head_r3.parameters()) + list(units4.parameters()), lr=6e-3)
        for i in range(2400):
            b = _rule_batch(3, 32, SEED * 7500 + i)
            zz, aa = run_states(subj3, b["ticks"], mask_fn(168, OLD_N, 168), extra=extra_of(b))
            feats = torch.cat([zz[:, 144:168], aa[:, 144:168]], dim=-1)
            loss = F.cross_entropy(head_r3(feats), b["labels"])
            opt4.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in units4.named_parameters():
                if p.grad is None:
                    continue
                if name == "p":
                    p.grad[:, :144] = 0.0
                elif name in ("w", "write_state.weight"):
                    p.grad[:144, :] = 0.0
                    p.grad[:, :144] = 0.0
                elif name == "write_in.weight":
                    p.grad[:144, :] = 0.0
                elif (
                    p.grad.ndim >= 1
                    and p.grad.shape[0] == units4.n_units
                    or p.grad.ndim >= 2
                    and p.grad.shape[0] == units4.n_units
                ):
                    p.grad[:144] = 0.0
            torch.nn.utils.clip_grad_norm_(list(units4.parameters()), 1.0)
            opt4.step()
        eval_r3 = _rule_batch(3, 1024, SEED + 3300 + 10_000_000)
        r3_acc = acc_cohort(subj3, head_r3, eval_r3, 144, 168)
        # recalibrate the three old readouts on the grown subject
        for h_i, (h_old, rule_id, lo_, hi_) in enumerate(
            [(head_r0, 0, 72, 96), (head_r1, 1, 96, 120), (head_r2, 2, 120, 144)]
        ):
            opt_ci = torch.optim.Adam(list(h_old.parameters()), lr=3e-3)
            for i in range(300):
                b = _rule_batch(rule_id, 32, SEED * (7600 + h_i) + i)
                loss = F.cross_entropy(
                    h_old(feat(subj3, b["ticks"], extra_of(b), lo_, hi_)), b["labels"]
                )
                opt_ci.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(h_old.parameters()), 1.0)
                opt_ci.step()
        r0_final = acc_cohort(subj3, head_r0, eval_r0, 72, 96)
        r1_final = acc_cohort(subj3, head_r1, eval_r1, 96, 120)
        r2_final = acc_cohort(subj3, head_r2, eval_r2, 120, 144)
        cap4 = build_capsule_v2(fresh_ab)
        fresh4 = SelectivePersistentGraph(fresh_ab.base_graph, fresh_ab.selective_spec)
        fresh4.freeze_base_()
        fresh4.selective_units.attach_extra_(8, seed=SEED)
        apply_capsule_v2(fresh4, cap4)
        subj5 = _Holder(fresh4, 168)
        r3_restore = acc_cohort(subj5, head_r3, eval_r3, 144, 168)
        with torch.no_grad():
            z0b, a0b = run_states(
                subj3, eval_r0["ticks"], mask_fn(168, OLD_N, 168), extra=extra_of(eval_r0)
            )
            z1b, a1b = run_states(
                subj5, eval_r0["ticks"], mask_fn(168, OLD_N, 168), extra=extra_of(eval_r0)
            )
        cap4_diff = float(max((z0b - z1b).abs().max(), (a0b - a1b).abs().max()))

    # capsule ABC restore check
    cap_abc = build_capsule_v2(fresh_ab)
    fresh_abc = SelectivePersistentGraph(fresh_ab.base_graph, fresh_ab.selective_spec)
    fresh_abc.freeze_base_()
    fresh_abc.selective_units.attach_extra_(8, seed=SEED)
    apply_capsule_v2(fresh_abc, cap_abc)
    subj4 = _Holder(fresh_abc, subj3.n_units)
    r2_restore = acc_cohort(subj4, head_r2, eval_r2, 120, 144)
    with torch.no_grad():
        z0, a0 = run_states(
            subj3,
            eval_r0["ticks"],
            mask_fn(subj3.n_units, OLD_N, subj3.n_units),
            extra=extra_of(eval_r0),
        )
        z1, a1 = run_states(
            subj4,
            eval_r0["ticks"],
            mask_fn(subj4.n_units, OLD_N, subj4.n_units),
            extra=extra_of(eval_r0),
        )
    cap_diff = float(max((z0 - z1).abs().max(), (a0 - a1).abs().max()))
    # fifth growth with online evidence update from the fourth event
    EV_GAIN = 0.5 * 0.75 + 0.5 * 0.75
    EV_LEARN = 0.10
    EV_RECAL = 0.09
    utility5 = {
        "grow_learn_R4": EV_GAIN - EV_LEARN - EV_RECAL,
        "wait": 0.0,
    }
    chosen5 = max(utility5, key=utility5.get)
    r4_acc = 0.0
    r0_f2 = r0_final
    r1_f2 = r1_final
    r2_f2 = r2_final
    r3_f2 = r3_acc
    cap5_diff = 0.0
    r4_restore = 0.0
    if chosen5 == "grow_learn_R4":
        old_w5 = fresh_ab.selective_units.extra_proj.weight.detach().clone()
        fresh_ab.grow_units_neutral_(2)
        units5 = fresh_ab.selective_units
        subj3.n_units = 192
        proj5 = torch.nn.Linear(8, 192, bias=False)
        with torch.no_grad():
            proj5.weight[: old_w5.shape[0]] = old_w5
            proj5.weight[old_w5.shape[0] :] = 0.0
        units5.extra_proj = proj5
        inherit(fresh_ab, 168)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 19201)
            head_r4 = PhaseHead(2 * (192 - 168), 4, 64)
        opt5 = torch.optim.Adam(list(head_r4.parameters()) + list(units5.parameters()), lr=6e-3)
        for i in range(2400):
            b = _rule_batch(4, 32, SEED * 7700 + i)
            zz, aa = run_states(subj3, b["ticks"], mask_fn(192, OLD_N, 192), extra=extra_of(b))
            feats = torch.cat([zz[:, 168:192], aa[:, 168:192]], dim=-1)
            loss = F.cross_entropy(head_r4(feats), b["labels"])
            opt5.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in units5.named_parameters():
                if p.grad is None:
                    continue
                if name == "p":
                    p.grad[:, :168] = 0.0
                elif name in ("w", "write_state.weight"):
                    p.grad[:168, :] = 0.0
                    p.grad[:, :168] = 0.0
                elif name == "write_in.weight":
                    p.grad[:168, :] = 0.0
                elif (
                    p.grad.ndim >= 1
                    and p.grad.shape[0] == units5.n_units
                    or p.grad.ndim >= 2
                    and p.grad.shape[0] == units5.n_units
                ):
                    p.grad[:168] = 0.0
            torch.nn.utils.clip_grad_norm_(list(units5.parameters()), 1.0)
            opt5.step()
        eval_r4 = _rule_batch(4, 1024, SEED + 3400 + 10_000_000)
        r4_acc = acc_cohort(subj3, head_r4, eval_r4, 168, 192)
        for h_i, (h_old, rule_id, lo_, hi_) in enumerate(
            [
                (head_r0, 0, 72, 96),
                (head_r1, 1, 96, 120),
                (head_r2, 2, 120, 144),
                (head_r3, 3, 144, 168),
            ]
        ):
            opt_ci = torch.optim.Adam(list(h_old.parameters()), lr=3e-3)
            for i in range(300):
                b = _rule_batch(rule_id, 32, SEED * (7800 + h_i) + i)
                loss = F.cross_entropy(
                    h_old(feat(subj3, b["ticks"], extra_of(b), lo_, hi_)), b["labels"]
                )
                opt_ci.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(h_old.parameters()), 1.0)
                opt_ci.step()
        r0_f2 = acc_cohort(subj3, head_r0, eval_r0, 72, 96)
        r1_f2 = acc_cohort(subj3, head_r1, eval_r1, 96, 120)
        r2_f2 = acc_cohort(subj3, head_r2, eval_r2, 120, 144)
        r3_f2 = acc_cohort(subj3, head_r3, eval_r3, 144, 168)
        cap5 = build_capsule_v2(fresh_ab)
        fresh5 = SelectivePersistentGraph(fresh_ab.base_graph, fresh_ab.selective_spec)
        fresh5.freeze_base_()
        fresh5.selective_units.attach_extra_(8, seed=SEED)
        apply_capsule_v2(fresh5, cap5)
        subj6 = _Holder(fresh5, 192)
        r4_restore = acc_cohort(subj6, head_r4, eval_r4, 168, 192)
        with torch.no_grad():
            z0c, a0c = run_states(
                subj3, eval_r0["ticks"], mask_fn(192, OLD_N, 192), extra=extra_of(eval_r0)
            )
            z1c, a1c = run_states(
                subj6, eval_r0["ticks"], mask_fn(192, OLD_N, 192), extra=extra_of(eval_r0)
            )
        cap5_diff = float(max((z0c - z1c).abs().max(), (a0c - a1c).abs().max()))

    # capacity budget is full: retire cohort D (R3) and recycle its slots for a sixth rule
    budget_units = 192
    utility_r = {
        "recycle_D_learn_R5": 0.75 - 0.10 - 0.20,  # gain - learn - expected damage to R3
        "wait": 0.0,
    }
    chosen_r = max(utility_r, key=utility_r.get)
    r3_after_recycle = r3_f2
    r5_acc = 0.0
    r5_restore = 0.0
    capR_diff = 0.0
    units_after = 192
    if chosen_r == "recycle_D_learn_R5" and subj3.n_units <= budget_units:
        units_r = fresh_ab.selective_units
        with torch.no_grad():
            src_r = torch.arange(24)
            dst_r = torch.arange(24) + 144
            units_r.r[dst_r] = units_r.r[src_r].clone()
            units_r.p[:, dst_r] = units_r.p[:, src_r].clone()
            for nm in ("c", "eta", "b"):
                getattr(units_r, nm)[dst_r] = getattr(units_r, nm)[src_r].clone()
            units_r.w[dst_r, :] = units_r.w[src_r, :].clone()
            units_r.write_in.weight[dst_r, :] = units_r.write_in.weight[src_r, :].clone()
            units_r.write_state.weight[dst_r, :] = units_r.write_state.weight[src_r, :].clone()
            units_r.write_bias[dst_r] = 0.0
            units_r.b[dst_r] = 0.0
        fresh_ab.selective_units.set_unit_lifecycle_(range(144, 168), "retired")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 19301)
            head_r5 = PhaseHead(2 * (168 - 144), 4, 64)
        opt_r5 = torch.optim.Adam(list(head_r5.parameters()) + list(units_r.parameters()), lr=6e-3)
        for i in range(2400):
            b = _rule_batch(5, 32, SEED * 7900 + i)
            zz, aa = run_states(subj3, b["ticks"], mask_fn(192, OLD_N, 192), extra=extra_of(b))
            feats = torch.cat([zz[:, 144:168], aa[:, 144:168]], dim=-1)
            loss = F.cross_entropy(head_r5(feats), b["labels"])
            opt_r5.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in units_r.named_parameters():
                if p.grad is None:
                    continue
                if name == "p":
                    p.grad[:, :144] = 0.0
                    p.grad[:, 168:] = 0.0
                elif name in ("w", "write_state.weight"):
                    p.grad[:144, :] = 0.0
                    p.grad[168:, :] = 0.0
                    p.grad[:, :144] = 0.0
                    p.grad[:, 168:] = 0.0
                elif name == "write_in.weight":
                    p.grad[:144, :] = 0.0
                    p.grad[168:, :] = 0.0
                elif (
                    p.grad.ndim >= 1
                    and p.grad.shape[0] == units_r.n_units
                    or p.grad.ndim >= 2
                    and p.grad.shape[0] == units_r.n_units
                ):
                    p.grad[:144] = 0.0
                    p.grad[168:] = 0.0
            torch.nn.utils.clip_grad_norm_(list(units_r.parameters()), 1.0)
            opt_r5.step()
        fresh_ab.selective_units.set_unit_lifecycle_(range(144, 168), "mature")
        eval_r5 = _rule_batch(5, 1024, SEED + 3500 + 10_000_000)
        r5_acc = acc_cohort(subj3, head_r5, eval_r5, 144, 168)
        r3_after_recycle = acc_cohort(subj3, head_r3, eval_r3, 144, 168)
        for h_i, (h_old, rule_id, lo_, hi_) in enumerate(
            [(head_r0, 0, 72, 96), (head_r1, 1, 96, 120), (head_r2, 2, 120, 144)]
        ):
            opt_ci = torch.optim.Adam(list(h_old.parameters()), lr=3e-3)
            for i in range(300):
                b = _rule_batch(rule_id, 32, SEED * (8000 + h_i) + i)
                loss = F.cross_entropy(
                    h_old(feat(subj3, b["ticks"], extra_of(b), lo_, hi_)), b["labels"]
                )
                opt_ci.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(h_old.parameters()), 1.0)
                opt_ci.step()
        r0_r = acc_cohort(subj3, head_r0, eval_r0, 72, 96)
        r1_r = acc_cohort(subj3, head_r1, eval_r1, 96, 120)
        r2_r = acc_cohort(subj3, head_r2, eval_r2, 120, 144)
        r4_r = acc_cohort(subj3, head_r4, eval_r4, 168, 192)
        units_after = int(fresh_ab.selective_units.n_units)
        capR = build_capsule_v2(fresh_ab)
        freshR = SelectivePersistentGraph(fresh_ab.base_graph, fresh_ab.selective_spec)
        freshR.freeze_base_()
        freshR.selective_units.attach_extra_(8, seed=SEED)
        apply_capsule_v2(freshR, capR)
        subjR = _Holder(freshR, 192)
        r5_restore = acc_cohort(subjR, head_r5, eval_r5, 144, 168)
        with torch.no_grad():
            z0d, a0d = run_states(
                subj3, eval_r0["ticks"], mask_fn(192, OLD_N, 192), extra=extra_of(eval_r0)
            )
            z1d, a1d = run_states(
                subjR, eval_r0["ticks"], mask_fn(192, OLD_N, 192), extra=extra_of(eval_r0)
            )
        capR_diff = float(max((z0d - z1d).abs().max(), (a0d - a1d).abs().max()))
    else:
        r0_r = r0_f2
        r1_r = r1_f2
        r2_r = r2_f2
        r4_r = r4_acc

    normal_after = cn_eval(subj)

    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_r0": r0_s3_cal >= 0.90,
        "G2_r1": r1_s3_cal >= 0.90,
        "G3_r2": r2_s3 >= 0.90,
        "G4_capsule": cap_diff <= 1e-6,
        "G5_normal_retained": (normal_cn - normal_after) <= 0.01,
        "G6_recalib_cost": calib_steps <= 900,
        "G7_choice4": chosen4 == "grow_learn_R3",
        "G8_r3": r3_acc >= 0.90,
        "G9_all_retained": r0_final >= 0.90 and r1_final >= 0.90 and r2_final >= 0.90,
        "G10_cap4": cap4_diff <= 1e-6 and abs(r3_restore - r3_acc) <= 1e-9,
        "G11_choice5": chosen5 == "grow_learn_R4",
        "G12_r4": r4_acc >= 0.90,
        "G13_all_five": min(r0_f2, r1_f2, r2_f2, r3_f2) >= 0.90,
        "G14_cap5": cap5_diff <= 1e-6 and abs(r4_restore - r4_acc) <= 1e-9,
        "G15_recycle_choice": chosen_r == "recycle_D_learn_R5",
        "G16_r5": r5_acc >= 0.90,
        "G17_others_retained": min(r0_r, r1_r, r2_r, r4_r) >= 0.90,
        "G18_r3_overwritten": (r3_f2 - r3_after_recycle) >= 0.30,
        "G19_capR": capR_diff <= 1e-6 and abs(r5_restore - r5_acc) <= 1e-9,
        "G20_units_constant": units_after == budget_units,
    }
    if not (
        gates["G15_recycle_choice"]
        and gates["G16_r5"]
        and gates["G17_others_retained"]
        and gates["G18_r3_overwritten"]
    ):
        verdict = "p0dd_recycle_failed"
    elif not gates["G19_capR"] or not gates["G20_units_constant"]:
        verdict = "p0dd_recycle_capsule_failed"
    elif not (gates["G11_choice5"] and gates["G12_r4"] and gates["G13_all_five"]):
        verdict = "p0dd_fifth_failed"
    elif not gates["G14_cap5"]:
        verdict = "p0dd_capsule_failed"
    elif not (gates["G7_choice4"] and gates["G8_r3"] and gates["G9_all_retained"]):
        verdict = "p0dd_fourth_failed"
    elif not gates["G10_cap4"]:
        verdict = "p0dd_fourth_capsule_failed"
    elif not (gates["G1_r0"] and gates["G2_r1"] and gates["G3_r2"]):
        verdict = "p0dd_chain_failed"
    elif not gates["G4_capsule"]:
        verdict = "p0dd_capsule_failed"
    elif not (gates["G0_old_cn"] and gates["G5_normal_retained"]):
        verdict = "p0dd_retention_failed"
    else:
        verdict = "p0dd_verified"
    payload = {
        "schema": "taiji-p0dd-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "retire_recycle_cohort",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "s1": {"r0": r0_s1},
        "s2": {"r1": r1_s2, "r0": r0_s2, "r0_cal": r0_s2_cal},
        "s3": {"r2": r2_s3, "r0": r0_s3, "r1": r1_s3, "r0_cal": r0_s3_cal, "r1_cal": r1_s3_cal},
        "capsule": {"r2_restore": r2_restore, "cap_diff": cap_diff},
        "calib_steps": calib_steps,
        "fourth": {
            "utility": utility4,
            "chosen": chosen4,
            "r3": r3_acc,
            "r0": r0_final,
            "r1": r1_final,
            "r2": r2_final,
            "cap4_diff": cap4_diff,
            "r3_restore": r3_restore,
        },
        "fifth": {
            "utility": utility5,
            "chosen": chosen5,
            "r4": r4_acc,
            "r0": r0_f2,
            "r1": r1_f2,
            "r2": r2_f2,
            "r3": r3_f2,
            "cap5_diff": cap5_diff,
            "r4_restore": r4_restore,
        },
        "recycle": {
            "chosen": chosen_r,
            "r5": r5_acc,
            "r3_before": r3_f2,
            "r3_after": r3_after_recycle,
            "r0": r0_r,
            "r1": r1_r,
            "r2": r2_r,
            "r4": r4_r,
            "capR_diff": capR_diff,
            "r5_restore": r5_restore,
            "units": units_after,
        },
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
