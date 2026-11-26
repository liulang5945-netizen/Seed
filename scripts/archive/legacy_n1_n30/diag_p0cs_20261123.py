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
OUT = ROOT / f"reports/taiji_p0cs_seed{SEED}.json"
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
        mask_newborn_grads(subj, 96)
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

    eval_new = new_rule_batch(1024, SEED + 2000 + 10_000_000)
    acc_new_before = acc(subj, head, eval_new, extra=extra_of(eval_new))

    head_r0 = copy.deepcopy(head)  # keep the original-rule readout untouched
    # grow a dedicated cohort for the new rule; original cohorts stay frozen
    old_w = subj.graph.selective_units.extra_proj.weight.detach().clone()
    subj.graph.grow_units_neutral_(2)
    units_b = subj.graph.selective_units
    subj.n_units = 120
    proj_b = torch.nn.Linear(8, 120, bias=False)
    with torch.no_grad():
        proj_b.weight[: old_w.shape[0]] = old_w
        proj_b.weight[old_w.shape[0] :] = 0.0
    units_b.extra_proj = proj_b
    with torch.no_grad():
        src_idx = torch.arange(24)
        dst_idx = torch.arange(24) + 96
        units_b.group_index[dst_idx] = torch.arange(24) // 6
        units_b.r[dst_idx] = units_b.r[src_idx].clone()
        units_b.p[:, dst_idx] = units_b.p[:, src_idx].clone()
        for nm in ("c", "eta", "b", "write_bias"):
            getattr(units_b, nm)[dst_idx] = getattr(units_b, nm)[src_idx].clone()
        units_b.w[dst_idx, :] = units_b.w[src_idx, :].clone()
        units_b.write_in.weight[dst_idx, :] = units_b.write_in.weight[src_idx, :].clone()
        units_b.write_state.weight[dst_idx, :] = units_b.write_state.weight[src_idx, :].clone()
        units_b.write_bias[dst_idx] = 0.0
        units_b.b[dst_idx] = 0.0

    def cohort_feat(subj_, ticks_, extra_, lo_: int, hi_: int) -> torch.Tensor:
        zz, aa = run_states(
            subj_, ticks_, mask_fn(subj_.n_units, OLD_N, subj_.n_units), extra=extra_
        )
        return torch.cat([zz[:, lo_:hi_], aa[:, lo_:hi_]], dim=-1)

    def acc_cohort(subj_, head_, batch_, lo_: int, hi_: int) -> float:
        head_.eval()
        subj_.eval_mode()
        with torch.no_grad():
            pred = head_(cohort_feat(subj_, batch_["ticks"], extra_of(batch_), lo_, hi_)).argmax(-1)
        return float((pred == batch_["labels"]).float().mean())

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 18501)
        head_b = PhaseHead(2 * (120 - 96), 4, 64)
    params_b = list(head_b.parameters()) + list(subj.graph.selective_units.parameters())

    # the only training signal is the binary reward of the executed action
    train_params = params_b
    opt_rl = torch.optim.Adam(train_params, lr=3e-3)
    p_before = [p.detach().clone() for p in train_params]
    grad_norms = []
    for i in range(3000):
        batch = new_rule_batch(32, SEED * 9500 + i)
        subj.train_mode()
        logits = head_b(cohort_feat(subj, batch["ticks"], extra_of(batch), 96, 120))
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
        mask_newborn_grads(subj, 96)
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
    acc_new_after = acc_cohort(subj, head_b, eval_new, 96, 120)
    # supervised upper bound: does the frozen feature space support the new rule at all?
    opt_sup = torch.optim.Adam(list(head_b.parameters()), lr=3e-3)
    for i in range(600):
        batch = new_rule_batch(32, SEED * 9600 + i)
        subj.train_mode()
        logits = head_b(cohort_feat(subj, batch["ticks"], extra_of(batch), 96, 120))
        loss = F.cross_entropy(logits, batch["labels"])
        opt_sup.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
        opt_sup.step()
    acc_new_supervised = acc_cohort(subj, head_b, eval_new, 96, 120)

    acc_new_after = acc_cohort(subj, head_b, eval_new, 96, 120)
    acc_old_rule_after = acc_cohort(subj, head_r0, eval_seen, 72, 96)
    lesion_drop_new = 0.0
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "dormant")
    acc_new_lesion = acc_cohort(subj, head_b, eval_new, 96, 120)
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "mature")
    lesion_drop_new = acc_new_after - acc_new_lesion
    normal_after = cn_eval(subj)
    r0_parallel = acc_cohort(subj, head_r0, eval_seen, 72, 96)
    r1_parallel = acc_cohort(subj, head_b, eval_new, 96, 120)
    subj.graph.selective_units.set_unit_lifecycle_(range(96, 120), "dormant")
    r0_lesionB = acc_cohort(subj, head_r0, eval_seen, 72, 96)
    r1_lesionB = acc_cohort(subj, head_b, eval_new, 96, 120)
    subj.graph.selective_units.set_unit_lifecycle_(range(96, 120), "mature")
    subj.graph.selective_units.set_unit_lifecycle_(range(72, 96), "dormant")
    r0_lesionA = acc_cohort(subj, head_r0, eval_seen, 72, 96)
    subj.graph.selective_units.set_unit_lifecycle_(range(72, 96), "mature")
    lesionB_on_r1 = r1_parallel - r1_lesionB
    lesionB_on_r0 = r0_parallel - r0_lesionB
    lesionA_on_r0 = r0_parallel - r0_lesionA

    for i in range(3000):
        batch = new_rule_batch(32, SEED * 9500 + i)
        subj.train_mode()
        logits = head_b(cohort_feat(subj, batch["ticks"], extra_of(batch), 96, 120))
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(1)
        reward = (action == batch["labels"]).float()
        logp = torch.log_softmax(logits, dim=-1)
        chosen_logp = logp.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = -(probs * logp).sum(-1).mean()
        loss = -(chosen_logp * (reward - 0.25)).mean() - 0.01 * entropy
        opt_rl.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, 96)
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(newborn_params(subj)), 1.0)
        opt_rl.step()
    acc_new_after = acc_cohort(subj, head_b, eval_new, 96, 120)
    acc_old_rule_after = acc_cohort(subj, head_r0, eval_seen, 72, 96)
    lesion_drop_new = 0.0
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "dormant")
    acc_new_lesion = acc_cohort(subj, head_b, eval_new, 96, 120)
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
        "G9_wake_pattern": [bool(e["wake"]) for e in wake_log] == [False, True, False, False],
        "G10_wake_recall": wake_log[1].get("full", 0.0) >= 0.90,
        "G11_rule_change_pre": acc_new_before <= 0.60,
        "G12_feedback_adaptation": acc_new_after >= 0.80,
        "G13_adaptation_gain": (acc_new_after - acc_new_before) >= 0.30,
        "G14_growth_still_causal": lesion_drop_new >= 0.15,
        "G15_supervised_upper": acc_new_supervised >= 0.95,
        "G16_r0_unit_parallel": r0_parallel >= 0.90,
        "G17_r1_unit_parallel": r1_parallel >= 0.85,
        "G18_dissociation": lesionB_on_r1 >= 0.15
        and lesionB_on_r0 <= 0.05
        and lesionA_on_r0 >= 0.15,
    }
    if not gates["G0_old_cn"]:
        verdict = "p0cs_cn_base_failed"
    elif not (gates["G1_multimodal"] and gates["G2_unseen_pairs"]):
        verdict = "p0cs_binding_failed"
    elif not (gates["G3_visual_causal"] and gates["G4_flag_causal"]):
        verdict = "p0cs_modality_not_causal"
    elif not gates["G5_growth_causal"]:
        verdict = "p0cs_growth_not_causal"
    elif not gates["G6_normal_retained"]:
        verdict = "p0cs_retention_breach"
    elif not (gates["G7_lesion_drop"] and gates["G9_wake_pattern"] and gates["G10_wake_recall"]):
        verdict = "p0cs_lifecycle_failed"
    elif not gates["G8_capsule_restore"]:
        verdict = "p0cs_capsule_failed"
    elif not (
        gates["G16_r0_parallel"]
        and gates["G17_r0_parallel_lesion"]
        and gates["G18_r1_parallel_lesion"]
    ):
        verdict = "p0cs_dissociation_failed"
    elif not {
        "G11_rule_change_pre",
        "G12_feedback_adaptation",
        "G13_adaptation_gain",
        "G14_growth_still_causal",
    } <= set(k for k, v in gates.items() if v):
        verdict = "p0cs_feedback_failed"
    else:
        verdict = "p0cs_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0cs-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "unit_level_parallel_cohorts",
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
        "reward_adapt": {
            "acc_new_before": acc_new_before,
            "acc_new_after": acc_new_after,
            "gain": acc_new_after - acc_new_before,
            "acc_old_rule_after": acc_old_rule_after,
            "lesion_drop_new": lesion_drop_new,
            "acc_new_supervised": acc_new_supervised,
            "cohorts": {
                "r0": r0_parallel,
                "r1": r1_parallel,
                "lesionB_on_r1": lesionB_on_r1,
                "lesionB_on_r0": lesionB_on_r0,
                "lesionA_on_r0": lesionA_on_r0,
            },
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
