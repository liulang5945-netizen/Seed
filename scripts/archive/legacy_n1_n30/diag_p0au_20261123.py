"""P0-AU: reference — how well does the trained CN base do on UNSEEN colour pairs?

Train the cn block with the stage1b recipe (1200 steps, pairs excluding HOLDOUT), then
measure normal-task accuracy on:
  seen pairs (train / fit templates),
  unseen pairs (HOLDOUT colour pairs; same templates),
so we know whether "unseen colour pairs" is a reasonable generalisation target for the
whole system.  Exploratory diagnostic; zero check/final.
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
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0au_diag_20261123.json"
SEED = 20261120
CN_STEPS = 1200
BATCH = 48


def normal_batch(count: int, seed: int, allow_holdout: bool, templates: str = "train") -> dict:
    first = BI_CN_TRAIN_FIRST
    second = BI_CN_TRAIN_SECOND
    all_tpls = [(t, True) for t in first] + [(t, False) for t in second]
    rows, labels = [], []
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
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        rows.append(seq)
        labels.append(c1 if pick_first else c2)
    return {"ticks": torch.stack(rows), "labels": torch.tensor(labels, dtype=torch.long)}


def cn_acc(subj, batch) -> float:
    sm = (batch["ticks"].abs().sum(-1) > 0).float()
    subj.eval_mode()
    with torch.no_grad():
        logits = subj.logits("cn", batch["ticks"], sm)
        return float((logits.argmax(-1) == batch["labels"]).float().mean())


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=1e-4)
    subj.train_mode()
    for step in range(CN_STEPS):
        batch = normal_batch(BATCH, SEED * 6000 + step, allow_holdout=False)
        loss = F.cross_entropy(
            subj.logits("cn", batch["ticks"], (batch["ticks"].abs().sum(-1) > 0).float()),
            batch["labels"],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
    results = {
        "seen_train_templates": cn_acc(subj, normal_batch(512, SEED + 900 + 10_000_000, False)),
        "unseen_pairs_train_templates": cn_acc(
            subj, normal_batch(512, SEED + 901 + 10_000_000, True)
        ),
    }
    gates = {
        "G0_seen": results["seen_train_templates"] >= 0.85,
        "G1_unseen_high": results["unseen_pairs_train_templates"] >= 0.80,
        "G2_unseen_low": results["unseen_pairs_train_templates"] <= 0.55,
    }
    if not gates["G0_seen"]:
        verdict = "p0au_cn_train_failed"
    elif gates["G1_unseen_high"]:
        verdict = "p0au_unseen_feasible"
    elif gates["G2_unseen_low"]:
        verdict = "p0au_unseen_infeasible"
    else:
        verdict = "p0au_unseen_partial"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0au-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "cn_steps": CN_STEPS,
        "acc": results,
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": cost},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"acc": results, "gates": gates, "verdict": verdict, "cost": payload["cost"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
