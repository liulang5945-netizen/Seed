"""N31.2.02: failure-mode breakdown of 8-colour unseen pairs (final planned card of line N31.2).

Exploration card, zero check/final, exploration seeds only.  Same pipeline as N31.2.01
(leak-free colour relabelling, HOLDOUT filter after the permutation); adds:
  * per-sample metadata (flag, template side, normal / other colour) for every eval sample;
  * a breakdown of unseen- and seen-pair predictions: accuracy by flag==0 / flag>0 and by
    template side, and the fraction of predictions equal to the normal colour, the other
    colour, c1, c2, or neither;
  * model-free rule baselines on the same eval sets (always-normal, always-other,
    always-c1, always-c2), to test whether unseen accuracy is a structured rule.
Motivation: N31.2.01 aug and P0-BI base gave identical unseen accuracy on seeds 20261121
(311/512) and 20261122 (375/512) despite different training trajectories.

Usage: python scripts/training/n31_2_02_unseen_breakdown.py <seed> <base|aug>
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
ARM = sys.argv[2] if len(sys.argv) > 2 else "base"
AUG = ARM == "aug"
OUT = ROOT / f"reports/n31_2_02_{ARM}_seed{SEED}.json"
CONFIRMATION_SEEDS = {20261201, 20261202, 20261203, 20261211, 20261212, 20261213}
if SEED in CONFIRMATION_SEEDS:
    raise SystemExit("N31.2.02 is an exploration card: confirmation seeds are forbidden")
PHASE_B_STEPS = 2400


def revision8(
    count: int, seed: int, allow_holdout: bool = False, perm: list[int] | None = None
) -> dict:
    """8-colour flip task (same template/flag scheme as revision_split, all 8 colours).

    ``perm`` applies a label-consistent colour relabelling (data augmentation): the
    sentence colours and the answer are both mapped through the same permutation.
    """
    first, second = BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    all_tpls = [(t, True) for t in first] + [(t, False) for t in second]
    rows, flip_labels, pairs, meta = [], [], [], []
    flag_rng = torch.Generator().manual_seed(seed + 999_999)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, CN_CLASSES, (1,), generator=gen))
        c2 = int(torch.randint(0, CN_CLASSES - 1, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        idx += 1
        if perm is not None:
            c1, c2 = perm[c1], perm[c2]
        # leak-free: the HOLDOUT filter is applied to the pair the model actually sees
        ok = ((c1, c2) in HOLDOUT) if allow_holdout else ((c1, c2) not in HOLDOUT)
        if not ok:
            continue
        tpl, pick_first = all_tpls[len(rows) % len(all_tpls)]
        normal = c1 if pick_first else c2
        other = c2 if pick_first else c1
        flag = int(torch.randint(0, 4, (1,), generator=flag_rng))
        answer = other if flag > 0 else normal
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        rows.append(seq)
        flip_labels.append(answer)
        pairs.append((c1, c2))
        meta.append({"flag": flag, "first": pick_first, "normal": normal, "other": other})
    return {
        "ticks": torch.stack(rows),
        "flip_labels": torch.tensor(flip_labels, dtype=torch.long),
        "pairs": pairs,
        "meta": meta,
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


def predict(subj, head, batch: dict, hi: int | None = None) -> torch.Tensor:
    hi = subj.n_units if hi is None else hi
    z, a = run_states(subj, batch["ticks"], mask_fn(subj.n_units, OLD_N, hi), extra=extra_of(batch))
    feat = torch.cat([z[:, OLD_N:hi], a[:, OLD_N:hi]], dim=-1)
    head.eval()
    subj.eval_mode()
    with torch.no_grad():
        return head(feat).argmax(-1)


def breakdown(batch: dict, pred: torch.Tensor) -> dict:
    p = pred.tolist()
    y = batch["flip_labels"].tolist()
    m = batch["meta"]
    c1 = [pr[0] for pr in batch["pairs"]]
    c2 = [pr[1] for pr in batch["pairs"]]
    n = len(y)

    def frac(cond) -> float:
        return sum(1 for i in range(n) if cond(i)) / n

    def acc_where(sel) -> float | None:
        idx = [i for i in range(n) if sel(i)]
        return (sum(1 for i in idx if p[i] == y[i]) / len(idx)) if idx else None

    return {
        "n": n,
        "acc": frac(lambda i: p[i] == y[i]),
        "acc_flag0": acc_where(lambda i: m[i]["flag"] == 0),
        "acc_flag_pos": acc_where(lambda i: m[i]["flag"] > 0),
        "acc_first": acc_where(lambda i: m[i]["first"]),
        "acc_second": acc_where(lambda i: not m[i]["first"]),
        "pred_is_normal": frac(lambda i: p[i] == m[i]["normal"]),
        "pred_is_other": frac(lambda i: p[i] == m[i]["other"]),
        "pred_is_c1": frac(lambda i: p[i] == c1[i]),
        "pred_is_c2": frac(lambda i: p[i] == c2[i]),
        "pred_is_neither": frac(lambda i: p[i] not in (c1[i], c2[i])),
        "rule_always_normal": frac(lambda i: m[i]["normal"] == y[i]),
        "rule_always_other": frac(lambda i: m[i]["other"] == y[i]),
        "rule_always_c1": frac(lambda i: c1[i] == y[i]),
        "rule_always_c2": frac(lambda i: c2[i] == y[i]),
    }


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
    test_birth = revision_split(4096, SEED + 1300 + 10_000_000)
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
    for _k in range(30):
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
    holdout_seen_in_training = 0
    for i in range(PHASE_B_STEPS):
        perm_b = torch.randperm(CN_CLASSES).tolist() if (AUG and i % 2 == 0) else None
        batch8 = revision8(24, SEED * 7200 + i, perm=perm_b)
        holdout_seen_in_training += sum(1 for pr in batch8["pairs"] if pr in HOLDOUT)
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
    bd_unseen = breakdown(eval8_unseen, predict(subj, head8, eval8_unseen))
    bd_seen = breakdown(eval8, predict(subj, head8, eval8))

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

    normal_after = cn_eval(subj)
    # gate classes (STAGE_GATES §2.2-4): E = experimental, R = engineering regression,
    # S = guaranteed by construction (frozen / grad-masked)
    gates = {
        "R_leak_free": holdout_seen_in_training == 0,
        "S_old_cn_retained": (normal_cn - normal_after) <= 0.01,
        "R_wake_exact": abs(acc8_wake - acc8) <= 1e-9 and abs(acc4_wake - acc4_after) <= 1e-9,
        "R_capsule_exact": capsule_diff <= 1e-6
        and abs(acc8_capsule - acc8) <= 1e-9
        and abs(acc4_capsule - acc4_after) <= 1e-9,
        "E_pressure": (cn_probe["test"] - null_cn_mean) <= 0.10
        and (birth_probe["test"] - null_birth_mean) <= 0.10,
        "E_cell06_unseen": acc8_unseen >= 0.80,
        "E_cell06_seen": acc8 >= 0.95,
        "E_cell06_old4": acc4_after >= 0.85,
        "E_growth_lesion": (acc8 - acc8_lesion) >= 0.15,
    }
    if not gates["R_leak_free"]:
        verdict = "n31_2_02_data_invalid"
    elif not (gates["R_wake_exact"] and gates["R_capsule_exact"] and gates["S_old_cn_retained"]):
        verdict = "n31_2_02_engineering_failure"
    elif not gates["E_pressure"]:
        verdict = "n31_2_02_inconclusive"
    elif gates["E_cell06_unseen"] and gates["E_cell06_seen"] and gates["E_cell06_old4"]:
        verdict = "n31_2_02_signal_positive"
    else:
        verdict = "n31_2_02_signal_negative"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-n31-2-02-v1",
        "card": "N31.2.02",
        "arm": ARM,
        "holdout_seen_in_training": holdout_seen_in_training,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "unseen_pairs_failure_breakdown",
        "breakdown_unseen": bd_unseen,
        "breakdown_seen": bd_seen,
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
        "acc8_capsule": acc8_capsule,
        "acc4_capsule": acc4_capsule,
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": cost},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
        "run_conditions": {"threads": 1},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
