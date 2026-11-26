"""Diagnostic (zero check): can base P24 training already resolve a sequential
trusted stale-then-correct report ("regime 4")?

Grounds the 24-S.F1 pre-registration: if the base model already answers the newest
trusted statement, the "correctable outdated" capability is in-stream and F1 must
test a harder rule; if it answers the stale/first claim, F1 has a real target.

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

from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    _make_p24_stream,
)
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}
SEED = 20261120
B = 48
FIT = 256
FIT_OFFSET = 10_000_000


def make_seq_correct(count: int, seed: int) -> dict[str, torch.Tensor]:
    """Two trusted S3 statements on the query entity: first stale v, then v_new.

    Truth = the later (newest) trusted claim.  Base training (regimes 0-3) never
    contains a trusted-trusted sequential disagreement with a resolved answer.
    """
    horizon = 14
    g = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(count, horizon, 48, dtype=torch.float32)
    ticks[:, : horizon - 1, 16:24] = torch.randn(count, horizon - 1, 8, generator=g) * 0.2
    base_vals = torch.randint(0, 4, (count, 2), generator=g)
    e_queries = torch.randint(0, 2, (count,), generator=g)
    val_labels = torch.zeros(count, dtype=torch.long)
    stale_vals = torch.zeros(count, dtype=torch.long)
    base_of_query = torch.zeros(count, dtype=torch.long)
    for b in range(count):
        eq = int(e_queries[b])
        v_base = int(base_vals[b, eq])
        other = 1 - eq
        base_of_query[b] = v_base
        order_s12 = torch.randperm(4, generator=g).tolist()
        for idx_pos, t in enumerate(order_s12):
            if idx_pos < 2:
                ticks[b, t, idx_pos] = 1.0
                ticks[b, t, 4 + int(base_vals[b, idx_pos])] = 1.0
                ticks[b, t, 8] = 1.0
                ticks[b, t, 12] = 1.0
            else:
                ticks[b, t, other] = 1.0
                ticks[b, t, 4 + int(torch.randint(0, 4, (1,), generator=g))] = 1.0
                ticks[b, t, 9] = 1.0
                ticks[b, t, 13] = 1.0
        s3_perm = torch.randperm(5, generator=g).tolist()
        t_first, t_second = sorted((4 + s3_perm[0], 4 + s3_perm[1]))
        v_stale = (v_base + 1 + int(torch.randint(0, 3, (1,), generator=g))) % 4
        v_new = (v_stale + 1 + int(torch.randint(0, 2, (1,), generator=g))) % 4
        stale_vals[b] = v_stale
        ticks[b, t_first, eq] = 1.0
        ticks[b, t_first, 4 + v_stale] = 1.0
        ticks[b, t_first, 9] = 1.0  # UPDATE marker
        ticks[b, t_first, 12] = 1.0  # trusted
        ticks[b, t_second, eq] = 1.0
        ticks[b, t_second, 4 + v_new] = 1.0
        ticks[b, t_second, 9] = 1.0
        ticks[b, t_second, 12] = 1.0
        val_labels[b] = v_new
        ticks[b, 13, eq] = 1.0
        ticks[b, 13, 11] = 1.0
    return {
        "ticks": ticks,
        "val_labels": val_labels,
        "stale_vals": stale_vals,
        "base_of_query": base_of_query,
    }


def step(opt, loss, subj, keep):
    from scripts.training.diag_protect_smoke_20261121 import mask_grads

    opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subj, keep)
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()


def eval_seq(subj, data):
    subj.eval_mode()
    with torch.no_grad():
        lg = subj.logits("p24", data["ticks"])
        pred = lg[:, :4].argmax(-1)
    n = pred.shape[0]
    return {
        "acc_new": float((pred == data["val_labels"]).float().mean()),
        "pct_new": float((pred == data["val_labels"]).float().mean()),
        "pct_stale": float((pred == data["stale_vals"]).float().mean()),
        "pct_base": float((pred == data["base_of_query"]).float().mean()),
        "n": n,
    }


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
    probe = make_seq_correct(FIT, SEED + 333 + FIT_OFFSET)
    for k in range(1200):
        b = _make_p24_stream(B, SEED * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], b["conflict_labels"]
        )
        step(opt, loss, subj, "p24")
        if k + 1 in (600, 900, 1200):
            marks[k + 1] = eval_seq(subj, probe)
            print(
                json.dumps(
                    {
                        "step": k + 1,
                        **{kk: round(vv, 4) for kk, vv in marks[k + 1].items() if kk != "n"},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    payload = {
        "schema": "taiji-u2-diag-p24-seq-correct-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "marks": marks,
        "run_conditions": {"threads": torch.get_num_threads()},
        "discipline": {"check_reads": 0, "final_reads": 0},
        "authority": "owner:standing-authorization:2026-11-20#recommendation-default",
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    Path(r"E:\Seed\reports\diag_p24_seq_correct_20261122.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"cpu": payload["cost"]["cpu"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
