"""P0-DP: 8-colour unseen-pair attack (pre-registered DESIGN-8COLOR-UNSEEN_20261124).

Arm base: 8-colour flip task with the standard multimodal inputs.
Arm aug : same, plus colour-relabel augmentation (random permutation of the 8 colour
          identities applied consistently to the sentence and the answer).  The task
          rule is the same, so the augmentation is label-consistent by construction and
          forces colour-agnostic relation learning instead of pair memorisation.

Usage: python scripts/training/diag_p0dq_8color_aug_20261124.py <seed> <base|aug>
Gate (pre-registered): unseen-pair accuracy >= 0.80, per-seed judgement.
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

from scripts.training.diag_p0ci_20261123 import (
    BI_CN_CHAR_INDEX,
    BI_CN_PROJ32,
    CN_COLOURS,
    CN_MAX_LEN,
    FLAG_CH,
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

ROOT = Path(r"E:\Seed")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261123
ARM = sys.argv[2] if len(sys.argv) > 2 else "base"
OUT = ROOT / f"reports/taiji_p0dq_{ARM}_seed{SEED}.json"
HOLDOUT8 = ((0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6))


def batch8(count: int, seed: int, allow_holdout: bool = False, augment: bool = False) -> dict:
    tpls = [(tpl, True) for tpl in BI_CN_TRAIN_FIRST] + [(tpl, False) for tpl in BI_CN_TRAIN_SECOND]
    rows, labels = [], []
    flag_rng = torch.Generator().manual_seed(seed + 555_555)
    vis_rng = torch.Generator().manual_seed(seed + 444_444)
    perm_rng = torch.Generator().manual_seed(seed + 333_333)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, 8, (1,), generator=gen))
        c2 = int(torch.randint(0, 7, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = ((c1, c2) in HOLDOUT8) if allow_holdout else ((c1, c2) not in HOLDOUT8)
        idx += 1
        if not ok:
            continue
        flag = int(torch.randint(0, 4, (1,), generator=flag_rng))
        shape = int(torch.randint(0, 4, (1,), generator=vis_rng))
        by_shape = c1 if shape < 2 else c2
        flipped = c2 if shape < 2 else c1
        answer = flipped if flag >= 2 else by_shape
        if augment:
            perm = torch.randperm(8, generator=perm_rng).tolist()
            c1, c2, answer = perm[c1], perm[c2], perm[answer]
        tpl, _ = tpls[len(rows) % len(tpls)]
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


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=1e-4)
    subj.train_mode()
    for step in range(1200):
        batch = _build_bidirectional_cn_32dim(
            48, SEED * 6000 + step, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
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
    subj.graph.grow_units_neutral_(2)
    subj.n_units = 120
    subj.graph.selective_units.attach_extra_(8, seed=SEED)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 20001)
        head = PhaseHead(2 * (120 - 72), 8, 64)
    params = list(head.parameters()) + list(newborn_params(subj))
    opt2 = torch.optim.Adam(params, lr=8e-3)
    for i in range(6000):
        batch = batch8(48, SEED * 7100 + i, augment=(ARM == "aug" and i % 2 == 0))
        subj.train_mode()
        z, a = run_states(subj, batch["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(batch))
        feats = torch.cat([z[:, 72:120], a[:, 72:120]], dim=-1)
        loss = F.cross_entropy(head(feats), batch["labels"])
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt2.step()

    eval_seen = batch8(1024, SEED + 3000 + 10_000_000, allow_holdout=False)
    eval_unseen = batch8(512, SEED + 3100 + 10_000_000, allow_holdout=True)

    def eval_acc(batch, aug: bool = False) -> float:
        b = batch
        if aug:
            b = batch8(batch["ticks"].shape[0], SEED + 3200 + 10_000_000, allow_holdout=True)
        z, a = run_states(subj, b["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(b))
        head.eval()
        subj.eval_mode()
        with torch.no_grad():
            pred = head(torch.cat([z[:, 72:120], a[:, 72:120]], dim=-1)).argmax(-1)
        return float((pred == b["labels"]).float().mean())

    seen = eval_acc(eval_seen)
    unseen = eval_acc(eval_unseen)
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, 120), "dormant")
    z, a = run_states(subj, eval_seen["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(eval_seen))
    head.eval()
    with torch.no_grad():
        lesion_seen = float(
            (
                head(torch.cat([z[:, 72:120], a[:, 72:120]], dim=-1)).argmax(-1)
                == eval_seen["labels"]
            )
            .float()
            .mean()
        )
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, 120), "mature")
    normal_after = cn_eval(subj)

    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_seen": seen >= 0.90,
        "G2_unseen": unseen >= 0.80,
        "G3_lesion": (seen - lesion_seen) >= 0.15,
        "G4_normal_retained": (normal_cn - normal_after) <= 0.01,
    }
    verdict = "p0dq_verified" if all(gates.values()) else "p0dq_failed"
    payload = {
        "schema": "taiji-p0dq-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "arm": ARM,
        "task": "8color_unseen_pairs",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "seen": seen,
        "unseen": unseen,
        "lesion_seen": lesion_seen,
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
