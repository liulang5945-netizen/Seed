"""P0-CG: multimodal binding (text colour words + visual shape code) on the unit graph.

Task: 4-colour Chinese sentence + a 4-dim visual shape code (channels 40:44) + a 4-dim
flag code (channels 32:36).  Answer = first colour if shape < 2 else second colour; when
flag >= 2 the answer is flipped to the OTHER colour.  The visual and flag modalities enter
the graph only through the raw-input adapter (write-gate), so the newborn units must bind
text and vision.  Controls: visual-zero ablation, flag-zero ablation, new-cohort lesion.

Exploratory diagnostic; zero check/final.
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
OUT = ROOT / f"reports/taiji_p0ch_seed{SEED}.json"
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
    normal_after = cn_eval(subj)

    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_multimodal": acc_seen >= 0.90,
        "G2_unseen_pairs": acc_unseen >= 0.60,
        "G3_visual_causal": (acc_seen - acc_novis) >= 0.15,
        "G4_flag_causal": (acc_seen - acc_noflag) >= 0.10,
        "G5_growth_causal": (acc_seen - acc_lesion) >= 0.15,
        "G6_normal_retained": (normal_cn - normal_after) <= 0.01,
    }
    if not gates["G0_old_cn"]:
        verdict = "p0ch_cn_base_failed"
    elif not (gates["G1_multimodal"] and gates["G2_unseen_pairs"]):
        verdict = "p0ch_binding_failed"
    elif not (gates["G3_visual_causal"] and gates["G4_flag_causal"]):
        verdict = "p0ch_modality_not_causal"
    elif not gates["G5_growth_causal"]:
        verdict = "p0ch_growth_not_causal"
    elif not gates["G6_normal_retained"]:
        verdict = "p0ch_retention_breach"
    else:
        verdict = "p0ch_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0ch-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "multimodal_binding_hardened",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "acc_seen": acc_seen,
        "acc_unseen_pairs": acc_unseen,
        "acc_no_visual": acc_novis,
        "acc_no_flag": acc_noflag,
        "acc_lesion_newborn": acc_lesion,
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
