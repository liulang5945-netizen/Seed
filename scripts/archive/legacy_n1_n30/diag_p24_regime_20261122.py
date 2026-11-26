"""Diagnostic (zero check): P24 per-regime accuracy for Subject72 under single-task training.

Purpose: ground the 24-S.F1 pre-registration. 24-S.0b-0f ran P24 at chance level
(0.22-0.47) because P24 was undertrained inside the 8-session mixed protocol; this
diagnostic checks whether the same 24-unit P24 route reaches per-regime competence
when trained single-task, and whether the "outdated" (stale->trusted UPDATE) regime
is causally read from the update ticks.

Zero check/final. Standing authorization; owner may revise.
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

from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p24_stream
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}
SEED = 20261120
B = 48
FIT = 256
FIT_OFFSET = 10_000_000


def step(opt, loss, subj, keep):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    from scripts.training.diag_protect_smoke_20261121 import mask_grads

    mask_grads(subj, keep)
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()


def eval_p24(subj, seed):
    subj.eval_mode()
    fit = _make_p24_stream(FIT, seed + 212 + FIT_OFFSET)
    with torch.no_grad():
        lg = subj.logits("p24", fit["ticks"])
        val_ok = (lg[:, :4].argmax(-1) == fit["val_labels"]).float()
        conf_ok = (lg[:, 4] > 0.0).float() == fit["conflict_labels"]
        reg = fit["regimes"]
        out = {
            "overall": float(val_ok.mean()),
            "by_regime": {int(r): float(val_ok[reg == r].mean()) for r in range(4)},
            "conflict_accuracy": float(conf_ok.float().mean()),
            "conflict_fp": float((lg[reg != 3, 4] > 0.0).float().mean()),
            "conflict_recall": float((lg[reg == 3, 4] > 0.0).float().mean()),
        }
        # causal probe: zero the S3 update-event ticks (4..8) for the query entity
        ticks_noupd = fit["ticks"].clone()
        ticks_noupd[:, 4:9, :] = 0.0
        lg2 = subj.logits("p24", ticks_noupd)
        for r in range(4):
            sel = reg == r
            out[f"regime{r}_noupdate"] = float(
                (lg2[sel, :4].argmax(-1) == fit["val_labels"][sel]).float().mean()
            )
        # causal probe: zero the S1-2 base ticks (0..4) for the query entity
        ticks_nobase = fit["ticks"].clone()
        ticks_nobase[:, :4, :] = 0.0
        lg3 = subj.logits("p24", ticks_nobase)
        for r in range(4):
            sel = reg == r
            out[f"regime{r}_nobase"] = float(
                (lg3[sel, :4].argmax(-1) == fit["val_labels"][sel]).float().mean()
            )
    return out


def main():
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in BLOCKS.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for _n, h in subj.phase_heads.items():
        for p in h.parameters():
            p.requires_grad_(True)
    subj.train_mode()
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=1e-4)
    marks = {}
    for k in range(1400):
        b = _make_p24_stream(B, SEED * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], b["conflict_labels"]
        )
        step(opt, loss, subj, "p24")
        if k + 1 in (300, 600, 900, 1200, 1400):
            marks[k + 1] = eval_p24(subj, SEED)
            print(
                json.dumps(
                    {
                        "step": k + 1,
                        "overall": round(marks[k + 1]["overall"], 4),
                        "by_regime": {
                            kk: round(vv, 4) for kk, vv in marks[k + 1]["by_regime"].items()
                        },
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    payload = {
        "schema": "taiji-u2-diag-p24-regime-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "units": subj.n_units,
        "blocks": BLOCKS,
        "marks": marks,
        "run_conditions": {"threads": torch.get_num_threads()},
        "discipline": {"check_reads": 0, "final_reads": 0},
        "authority": "owner:standing-authorization:2026-11-20#recommendation-default",
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    Path(r"E:\Seed\reports\diag_p24_regime_20261122.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"cpu": payload["cost"]["cpu"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
