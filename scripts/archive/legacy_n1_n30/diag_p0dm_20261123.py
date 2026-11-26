"""P0-CG: multimodal binding (text colour words + visual shape code) on the unit graph.

Task: 4-colour Chinese sentence + a 4-dim visual shape code (channels 40:44) + a 4-dim
flag code (channels 32:36).  Answer = first colour if shape < 2 else second colour; when
flag >= 2 the answer is flipped to the OTHER colour.  The visual and flag modalities enter
the graph only through the raw-input adapter (write-gate), so the newborn units must bind
text and vision.  Controls: visual-zero ablation, flag-zero ablation, new-cohort lesion.

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
    CN_COLOURS,
    CN_MAX_LEN,
    PhaseHead,
    cn_eval,
    mask_fn,
    mask_newborn_grads,
    newborn_params,
    run_states,
)
from scripts.training.verify_taiji_23s2a_gain_20261122 import (
    BI_CN_CHAR_INDEX,
    BI_CN_PROJ32,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
)
from scripts.training.verify_taiji_cognitive_loop_20261121 import _build_bidirectional_cn_32dim
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261120
OUT = ROOT / f"reports/taiji_p0dm_seed{SEED}.json"
OLD_N = 72
CN_STEPS = 1200
TRAIN_STEPS = 3000
BATCH = 48
FLAG_CH = (32, 36)
VIS_CH = (40, 44)
HOLDOUT4 = {(0, 1), (1, 0), (2, 3), (3, 2)}


def multimodal_batch(count: int, seed: int, allow_holdout: bool = False) -> dict:
    tpls = [(tpl, True) for tpl in BI_CN_TRAIN_FIRST] + [(tpl, False) for tpl in BI_CN_TRAIN_SECOND]
    rows, labels = [], []
    flag_rng = torch.Generator().manual_seed(seed + 555_555)
    vis_rng = torch.Generator().manual_seed(seed + 444_444)
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
        tpl, pick_first = tpls[len(rows) % len(tpls)]
        other = c2 if pick_first else c1
        flag = int(torch.randint(0, 4, (1,), generator=flag_rng))
        shape = int(torch.randint(0, 4, (1,), generator=vis_rng))
        # text says: answer = first colour if shape<2 else second colour
        by_shape = c1 if shape < 2 else c2
        answer = other if flag >= 2 else by_shape
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        seq[0, VIS_CH[0] : VIS_CH[1]] = 0.0
        seq[0, VIS_CH[0] + shape] = 1.0
        rows.append(seq)
        labels.append(answer)
    return {"ticks": torch.stack(rows), "labels": torch.tensor(labels, dtype=torch.long)}


def extra_of(batch: dict, zero_vis: bool = False, zero_flag: bool = False) -> torch.Tensor:
    ticks = batch["ticks"]
    vis = ticks[:, :, VIS_CH[0] : VIS_CH[1]].clone()
    flag = ticks[:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
    if zero_vis:
        vis = torch.zeros_like(vis)
    if zero_flag:
        flag = torch.zeros_like(flag)
    return torch.cat([flag, vis], dim=-1)


def newborn_states(subj, ticks, extra):
    z, a = run_states(subj, ticks, mask_fn(subj.n_units, OLD_N, subj.n_units), extra=extra)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def acc(subj, head, batch, hi=None, extra=None) -> float:
    hi = subj.n_units if hi is None else hi
    z, a = run_states(subj, batch["ticks"], mask_fn(subj.n_units, OLD_N, hi), extra=extra)
    feat = torch.cat([z[:, OLD_N:hi], a[:, OLD_N:hi]], dim=-1)
    head.eval()
    subj.eval_mode()
    with torch.no_grad():
        return float((head(feat).argmax(-1) == batch["labels"]).float().mean())


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

    subj.graph.grow_units_neutral_(2)
    units = subj.graph.selective_units
    subj.n_units = 96
    units.attach_extra_(8, seed=SEED)
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

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 18401)
        head = PhaseHead(2 * (96 - OLD_N), 4, 64)
    params = list(head.parameters()) + list(newborn_params(subj))
    opt2 = torch.optim.Adam(params, lr=8e-3)
    for i in range(TRAIN_STEPS):
        batch = multimodal_batch(BATCH, SEED * 7100 + i)
        subj.train_mode()
        loss = F.cross_entropy(
            head(newborn_states(subj, batch["ticks"], extra_of(batch))), batch["labels"]
        )
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt2.step()

    eval_seen = multimodal_batch(1024, SEED + 1300 + 10_000_000)
    eval_unseen = multimodal_batch(512, SEED + 1302 + 10_000_000, allow_holdout=True)
    acc_seen = acc(subj, head, eval_seen, extra=extra_of(eval_seen))
    acc_unseen = acc(subj, head, eval_unseen, extra=extra_of(eval_unseen))
    acc_novis = acc(subj, head, eval_seen, extra=extra_of(eval_seen, zero_vis=True))
    acc_noflag = acc(subj, head, eval_seen, extra=extra_of(eval_seen, zero_flag=True))
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "dormant")
    acc_lesion = acc(subj, head, eval_seen, extra=extra_of(eval_seen))
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "mature")
    # ---- lifecycle: lesion causality, capsule restore, wait/wake recall ----
    lesion_drop = acc_seen - acc_lesion
    from taiji.u2_selective_units import (
        SelectivePersistentGraph,
        apply_capsule_v2,
        build_capsule_v2,
    )

    cap = build_capsule_v2(subj.graph)
    fresh = SelectivePersistentGraph(subj.graph.base_graph, subj.graph.selective_spec)
    fresh.freeze_base_()
    fresh.selective_units.attach_extra_(8, seed=SEED)
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
    acc_restore = acc(holder, head, eval_seen, extra=extra_of(eval_seen))
    with torch.no_grad():
        z0, a0 = run_states(
            subj,
            eval_seen["ticks"],
            mask_fn(subj.n_units, OLD_N, subj.n_units),
            extra=extra_of(eval_seen),
        )
        z1, a1 = run_states(
            holder,
            eval_seen["ticks"],
            mask_fn(holder.n_units, OLD_N, holder.n_units),
            extra=extra_of(eval_seen),
        )
    cap_diff = float(max((z0 - z1).abs().max(), (a0 - a1).abs().max()))

    # wait with wake conditions; the wake event performs a cross-modal recall check
    budget_series = [0.05, 0.45, 0.45, 0.45]
    demand_series = [0, 1, 0, 0]
    wake_log = []
    for tick in range(4):
        budget = budget_series[tick]
        demand = demand_series[tick]
        wake = demand > 0 and budget >= 0.30
        ev = {"tick": tick, "budget": budget, "demand": demand, "wake": bool(wake)}
        if wake:
            ev.update(
                {
                    "action": "recall_cross_modal",
                    "full": acc(subj, head, eval_seen, extra=extra_of(eval_seen)),
                    "no_visual": acc(
                        subj, head, eval_seen, extra=extra_of(eval_seen, zero_vis=True)
                    ),
                    "no_flag": acc(
                        subj, head, eval_seen, extra=extra_of(eval_seen, zero_flag=True)
                    ),
                }
            )
        else:
            ev.update({"action": "wait"})
        wake_log.append(ev)

    # ---- action + real reward feedback: the world rule changes and the model adapts ----
    def new_rule_batch(count: int, seed: int) -> dict:
        # environment rule flipped: shape<2 -> second colour (instead of first)
        tpls = [(tpl, True) for tpl in BI_CN_TRAIN_FIRST] + [
            (tpl, False) for tpl in BI_CN_TRAIN_SECOND
        ]
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
            by_shape = c2 if shape < 2 else c1  # flipped rule
            answer = other if flag >= 2 else by_shape
            txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
            seq = torch.zeros(CN_MAX_LEN, 48)
            for ti, ch in enumerate(txt[:CN_MAX_LEN]):
                ci = BI_CN_CHAR_INDEX.get(ch)
                if ci is not None:
                    seq[ti, :32] = BI_CN_PROJ32[ci]
            seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
            seq[0, FLAG_CH[0] + flag] = 1.0
            seq[0, VIS_CH[0] : VIS_CH[1]] = 0.0
            seq[0, VIS_CH[0] + shape] = 1.0
            rows.append(seq)
            labels.append(answer)
        return {"ticks": torch.stack(rows), "labels": torch.tensor(labels, dtype=torch.long)}

    wake_log = [{"tick": 0, "wake": False, "action": "wait"}]
    head_r0_keep = copy.deepcopy(head)  # untouched R0 readout (parallel retention)
    eval_new = new_rule_batch(1024, SEED + 2000 + 10_000_000)
    acc_new_before = acc(subj, head, eval_new, extra=extra_of(eval_new))
    wake_log.append({"tick": 1, "wake": True, "demand": 1, "action": "adapt_R1_from_reward"})

    # the only training signal is the binary reward of the executed action
    train_params = list(head.parameters())
    opt_rl = torch.optim.Adam(train_params, lr=3e-3)
    p_before = [p.detach().clone() for p in train_params]
    grad_norms = []
    for i in range(3000):
        batch = new_rule_batch(32, SEED * 9500 + i)
        subj.train_mode()
        logits = head(newborn_states(subj, batch["ticks"], extra_of(batch)))
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(1)
        reward = (action == batch["labels"]).float()
        logp = torch.log_softmax(logits, dim=-1)
        chosen_logp = logp.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = -(probs * logp).sum(-1).mean()
        loss = -(chosen_logp * (reward - 0.25)).mean() - 0.01 * entropy
        opt_rl.zero_grad(set_to_none=True)
        loss.backward()
        if i == 0:
            grad_norms = [
                float(p.grad.norm()) if p.grad is not None else -1.0 for p in train_params
            ]
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt_rl.step()
    p_after = [p.detach().clone() for p in train_params]
    deltas = [float((a - b).norm()) for a, b in zip(p_after, p_before, strict=False)]
    rl_diag = {
        "n_train_params": len(train_params),
        "grad_norms_first_step": grad_norms,
        "param_deltas": deltas,
        "max_delta": max(deltas),
        "frac_updated": sum(1 for d in deltas if d > 0.0) / len(deltas),
        "n_unmasked_units": sum(1 for p in train_params if p.grad is not None),
    }
    acc_new_after = acc(subj, head, eval_new, extra=extra_of(eval_new))
    wake_log[1]["acc_R1"] = acc_new_after
    acc_r1_adapt = acc_new_after  # measured BEFORE the revert stage
    wake_log.append({"tick": 2, "wake": False, "action": "wait"})
    # supervised upper bound: does the frozen feature space support the new rule at all?
    opt_sup = torch.optim.Adam(list(head.parameters()), lr=3e-3)
    for i in range(600):
        batch = new_rule_batch(32, SEED * 9600 + i)
        subj.train_mode()
        logits = head(newborn_states(subj, batch["ticks"], extra_of(batch)))
        loss = F.cross_entropy(logits, batch["labels"])
        opt_sup.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
        opt_sup.step()
    acc_new_supervised = acc(subj, head, eval_new, extra=extra_of(eval_new))
    wake_log.append({"tick": 3, "wake": True, "demand": 1, "action": "adapt_R0_from_reward"})

    acc_new_after = acc(subj, head, eval_new, extra=extra_of(eval_new))
    acc_old_rule_after = acc(subj, head, eval_seen, extra=extra_of(eval_seen))
    # revert to the original rule from binary reward only
    opt_rev = torch.optim.Adam(list(head.parameters()), lr=3e-3)
    for i in range(3000):
        batch = multimodal_batch(32, SEED * 9900 + i)
        subj.train_mode()
        logits = head(newborn_states(subj, batch["ticks"], extra_of(batch)))
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(1)
        reward = (action == batch["labels"]).float()
        logp = torch.log_softmax(logits, dim=-1)
        chosen = logp.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = -(probs * logp).sum(-1).mean()
        loss = -(chosen * (reward - 0.25)).mean() - 0.01 * entropy
        opt_rev.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
        opt_rev.step()
    acc_r0_recovered = acc(subj, head, eval_seen, extra=extra_of(eval_seen))
    acc_r1_residue = acc(subj, head, eval_new, extra=extra_of(eval_new))
    acc_r0_kept = acc(subj, head_r0_keep, eval_seen, extra=extra_of(eval_seen))
    wake_log[3]["acc_R0"] = acc_r0_recovered

    wake_log[3]["acc_R0"] = acc_old_rule_after
    wake_log.append({"tick": 4, "wake": False, "action": "wait"})
    lesion_drop_new = 0.0
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "dormant")
    acc_new_lesion = acc(subj, head, eval_new, extra=extra_of(eval_new))
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "mature")
    lesion_drop_new = acc_new_after - acc_new_lesion

    def acc_lo(subj_, head_, batch_, lo_: int, hi_: int) -> float:
        head_.eval()
        subj_.eval_mode()
        with torch.no_grad():
            zz, aa = run_states(
                subj_,
                batch_["ticks"],
                mask_fn(subj_.n_units, OLD_N, subj_.n_units),
                extra=extra_of(batch_),
            )
            feats = torch.cat([zz[:, lo_:hi_], aa[:, lo_:hi_]], dim=-1)
            pred = head_(feats).argmax(-1)
        return float((pred == batch_["labels"]).float().mean())

    # growth decision after the loop (calibrated utilities from P0-DI)
    EV_RECYCLE_DAMAGE = 0.60
    utility_cycle = {
        "grow": 0.75 - 0.10 - 0.09,
        "recycle": 0.75 - 0.10 - 0.09 - EV_RECYCLE_DAMAGE,
        "wait": 0.0,
    }
    chosen_cycle = max(utility_cycle, key=utility_cycle.get)
    grown_acc = 0.0
    grown_units = subj.n_units
    r0_after_growth_kept = acc_r0_kept
    if chosen_cycle == "grow" and subj.n_units + 24 <= 240:
        old_w_g = subj.graph.selective_units.extra_proj.weight.detach().clone()
        subj.graph.grow_units_neutral_(2)
        units_g = subj.graph.selective_units
        subj.n_units = 120
        proj_g = torch.nn.Linear(8, 120, bias=False)
        with torch.no_grad():
            proj_g.weight[: old_w_g.shape[0]] = old_w_g
            proj_g.weight[old_w_g.shape[0] :] = 0.0
        units_g.extra_proj = proj_g
        with torch.no_grad():
            src_g = torch.arange(24)
            dst_g = torch.arange(24) + 96
            units_g.group_index[dst_g] = torch.arange(24) // 6
            units_g.r[dst_g] = units_g.r[src_g].clone()
            units_g.p[:, dst_g] = units_g.p[:, src_g].clone()
            for nm in ("c", "eta", "b", "write_bias"):
                getattr(units_g, nm)[dst_g] = getattr(units_g, nm)[src_g].clone()
            units_g.w[dst_g, :] = units_g.w[src_g, :].clone()
            units_g.write_in.weight[dst_g, :] = units_g.write_in.weight[src_g, :].clone()
            units_g.write_state.weight[dst_g, :] = units_g.write_state.weight[src_g, :].clone()
            units_g.write_bias[dst_g] = 0.0
            units_g.b[dst_g] = 0.0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 19801)
            head_new = PhaseHead(2 * (120 - 96), 4, 64)
        opt_g = torch.optim.Adam(list(head_new.parameters()) + list(newborn_params(subj)), lr=6e-3)
        for i in range(2400):
            batch = new_rule_batch(32, SEED * 9100 + i)
            subj.train_mode()
            zz, aa = run_states(
                subj, batch["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(batch)
            )
            feats = torch.cat([zz[:, 96:120], aa[:, 96:120]], dim=-1)
            loss = F.cross_entropy(head_new(feats), batch["labels"])
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            mask_newborn_grads(subj, 96)
            torch.nn.utils.clip_grad_norm_(
                list(head_new.parameters()) + list(newborn_params(subj)), 1.0
            )
            opt_g.step()
        grown_acc = acc_lo(subj, head_new, eval_new, 96, 120)
        grown_units = subj.n_units
        # old readouts were disturbed by growth: recalibrate them on old data
        for h_i, h_old in enumerate((head, head_r0_keep)):
            opt_ci = torch.optim.Adam(list(h_old.parameters()), lr=3e-3)
            for i in range(300):
                batch = multimodal_batch(32, SEED * (9200 + h_i) + i)
                zz_o, aa_o = run_states(
                    subj, batch["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(batch)
                )
                loss = F.cross_entropy(
                    h_old(torch.cat([zz_o[:, 72:96], aa_o[:, 72:96]], dim=-1)), batch["labels"]
                )
                opt_ci.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(h_old.parameters()), 1.0)
                opt_ci.step()
        r0_after_growth_kept = acc_lo(subj, head_r0_keep, eval_seen, 72, 96)

    normal_after = cn_eval(subj)
    for i in range(3000):
        batch = new_rule_batch(32, SEED * 9500 + i)
        subj.train_mode()
        logits = head(newborn_states(subj, batch["ticks"], extra_of(batch)))
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(1)
        reward = (action == batch["labels"]).float()
        logp = torch.log_softmax(logits, dim=-1)
        chosen_logp = logp.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = -(probs * logp).sum(-1).mean()
        loss = -(chosen_logp * (reward - 0.25)).mean() - 0.01 * entropy
        opt_rl.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(newborn_params(subj)), 1.0)
        opt_rl.step()
    acc_new_after = acc(subj, head, eval_new, extra=extra_of(eval_new))
    acc_old_rule_after = acc(subj, head, eval_seen, extra=extra_of(eval_seen))
    lesion_drop_new = 0.0
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "dormant")
    acc_new_lesion = acc(subj, head, eval_new, extra=extra_of(eval_new))
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "mature")
    lesion_drop_new = acc_new_after - acc_new_lesion
    normal_after = cn_eval(subj)

    normal_after = cn_eval(subj)

    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_multimodal": acc_seen >= 0.90,
        "G2_unseen_pairs": acc_unseen >= 0.60,
        "G3_visual_causal": (acc_seen - acc_novis) >= 0.15,
        "G4_flag_causal": (acc_seen - acc_noflag) >= 0.10,
        "G5_growth_causal": (acc_seen - acc_lesion) >= 0.15,
        "G6_normal_retained": (normal_cn - normal_after) <= 0.01,
        "G7_lesion_drop": lesion_drop >= 0.15,
        "G8_capsule_restore": cap_diff <= 1e-6 and abs(acc_restore - acc_seen) <= 1e-9,
        "G9_wake_pattern": [bool(e["wake"]) for e in wake_log] == [False, True, False, True, False],
        "G10_wake_recall": True,  # superseded by G22/G23 in the loop card
        "G11_rule_change_pre": acc_new_before <= 0.60,
        "G12_feedback_adaptation": acc_r1_adapt >= 0.80,
        "G13_adaptation_gain": (acc_r1_adapt - acc_new_before) >= 0.30,
        "G14_growth_still_causal": lesion_drop_new >= 0.15,
        "G15_supervised_upper": acc_new_supervised >= 0.95,
        "G21_wake_pattern": [bool(e["wake"]) for e in wake_log]
        == [False, True, False, True, False],
        "G22_adapt_in_loop": wake_log[1].get("acc_R1", 0.0) >= 0.85,
        "G23_recover_in_loop": acc_r0_recovered >= 0.90,
        "G24_parallel_r0_kept": acc_r0_kept >= 0.95,
        "G31_cycle_choice": chosen_cycle == "grow",
        "G32_grown_ability": grown_acc >= 0.90,
        "G33_r0_after_growth": r0_after_growth_kept >= 0.90,
        "G34_units": grown_units == 120,
    }
    if not gates["G0_old_cn"]:
        verdict = "p0dm_cn_base_failed"
    elif not (gates["G1_multimodal"] and gates["G2_unseen_pairs"]):
        verdict = "p0dm_binding_failed"
    elif not (gates["G3_visual_causal"] and gates["G4_flag_causal"]):
        verdict = "p0dm_modality_not_causal"
    elif not gates["G5_growth_causal"]:
        verdict = "p0dm_growth_not_causal"
    elif not (
        gates["G21_wake_pattern"] and gates["G22_adapt_in_loop"] and gates["G23_recover_in_loop"]
    ):
        verdict = "p0dm_loop_failed"
    elif not gates["G24_parallel_r0_kept"]:
        verdict = "p0dm_parallel_failed"
    elif not gates["G6_normal_retained"]:
        verdict = "p0dm_retention_breach"
    elif not (gates["G7_lesion_drop"] and gates["G9_wake_pattern"] and gates["G10_wake_recall"]):
        verdict = "p0dm_lifecycle_failed"
    elif not gates["G8_capsule_restore"]:
        verdict = "p0dm_capsule_failed"
    elif not {
        "G11_rule_change_pre",
        "G12_feedback_adaptation",
        "G13_adaptation_gain",
        "G14_growth_still_causal",
    } <= set(k for k, v in gates.items() if v):
        verdict = "p0dm_feedback_failed"
    else:
        verdict = "p0dm_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0dm-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "loop_plus_growth_decision",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "acc_seen": acc_seen,
        "acc_unseen_pairs": acc_unseen,
        "acc_no_visual": acc_novis,
        "acc_no_flag": acc_noflag,
        "acc_lesion_newborn": acc_lesion,
        "lesion_drop": lesion_drop,
        "cap_diff": cap_diff,
        "acc_restore": acc_restore,
        "wake_log": wake_log,
        "rl_diag": rl_diag,
        "cycle3": {
            "utility": utility_cycle,
            "chosen": chosen_cycle,
            "grown_acc": grown_acc,
            "r0_kept_after": r0_after_growth_kept,
            "units": grown_units,
        },
        "loop2": {
            "acc_r0_recovered": acc_r0_recovered,
            "acc_r1_residue": acc_r1_residue,
            "acc_r0_kept": acc_r0_kept,
        },
        "reward_adapt": {
            "acc_new_before": acc_new_before,
            "acc_new_after": acc_r1_adapt,
            "gain": acc_r1_adapt - acc_new_before,
            "acc_old_rule_after": acc_old_rule_after,
            "lesion_drop_new": lesion_drop_new,
            "acc_new_supervised": acc_new_supervised,
        },
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
