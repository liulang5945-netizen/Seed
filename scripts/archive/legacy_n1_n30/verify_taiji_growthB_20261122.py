"""Growth B: unit credit allocator (weight re-averaging after growth), 1 seed, <=400 CPU, 0 check."""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

SEED = 20261120
B = 48
BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}


def train_base(seed):
    subj = Subject72(seed)
    subj.set_masks(True)
    for n, (lo, hi) in BLOCKS.items():
        subj.graph.set_task_route(n, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for n, h in subj.phase_heads.items():
        for p in h.parameters():
            p.requires_grad_(n == "cn")
    subj.train_mode()
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=0.0)
    for k in range(1000):
        b = _build_bidirectional_cn_32dim(B, seed * 6000 + k, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND)
        sm = (b["ticks"].abs().sum(-1) > 0).float()
        step(opt, F.cross_entropy(subj.logits("cn", b["ticks"], sm), b["labels"]), subj, "cn")
    for p in subj.graph.cross_channel_mix.parameters():
        p.requires_grad_(False)
    for n in ("p24", "p27"):
        for p in subj.phase_heads[n].parameters():
            p.requires_grad_(True)
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
    for k in range(700):
        b = _make_p24_stream(B, seed * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        step(
            opt,
            F.cross_entropy(lg[:, :4], b["val_labels"])
            + F.binary_cross_entropy_with_logits(lg[:, 4], b["conflict_labels"]),
            subj,
            "p24",
        )
    for k in range(900):
        b = _make_p27_physics_batch(B, seed * 41000 + k)
        step(opt, F.cross_entropy(subj.logits("p27", b["ticks"]), b["labels"]), subj, "p27")
    return subj


def step(opt, loss, subj, keep):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subj, keep)
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()


def logits_mask(subj, ticks, name, mask):
    m = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], subj.n_units)
    out = subj.graph(_channels_seq(ticks), m, subj.graph.initial_state(batch_size=ticks.shape[0]))
    z, a = out["next_state"]["z"], out["next_state"]["a"]
    h = subj.phase_heads[name]
    n = h.net[0].in_features // 2
    return h(torch.cat([z[..., :n], a[..., :n]], dim=-1))


def run_arm(seed, allocate):
    torch.set_num_threads(1)
    subj = train_base(seed)
    subj.eval_mode()
    fit27 = _make_p27_physics_batch(128, seed + 81 + 10_000_000)
    with torch.no_grad():
        base = float(
            (subj.logits("p27", fit27["ticks"]).argmax(-1) == fit27["labels"]).float().mean()
        )
    old_head = subj.phase_heads["p27"]
    old_n = subj.graph.selective_units.n_units
    subj.graph.grow_units_neutral_(2)
    newn = subj.graph.selective_units.n_units
    subj.n_units = newn
    head = PhaseHead(2 * newn, 4, 64)
    with torch.no_grad():
        head.net[0].weight[:, : 2 * old_n].copy_(old_head.net[0].weight)
        head.net[0].bias.copy_(old_head.net[0].bias)
        # copy remaining layers by name
        for i in range(1, len(head.net)):
            if hasattr(old_head.net[i], "weight"):
                head.net[i].weight.copy_(old_head.net[i].weight)
                head.net[i].bias.copy_(old_head.net[i].bias)
        if allocate:
            col = old_head.net[0].weight.mean(dim=1, keepdim=True) * 0.5
            head.net[0].weight[:, 2 * old_n :] = col
        else:
            head.net[0].weight[:, 2 * old_n :] = 0.0
    subj.phase_heads["p27"] = head
    mask = torch.zeros(newn)
    mask[48:72] = 1.0
    mask[old_n:] = 1.0
    subj.masks["p27"] = mask
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    for p in head.parameters():
        p.requires_grad_(True)
    hopt = torch.optim.Adam(head.parameters(), lr=7e-3, weight_decay=0.0)
    gopt = torch.optim.Adam(subj.graph.parameters(), lr=3e-3, weight_decay=0.0)
    subj.train_mode()
    for k in range(700):
        b = _make_p27_physics_batch(B, seed * 41000 + 100000 + k)
        loss = F.cross_entropy(logits_mask(subj, b["ticks"], "p27", mask), b["labels"])
        hopt.zero_grad(set_to_none=True)
        gopt.zero_grad(set_to_none=True)
        loss.backward()
        for _n, p in subj.graph.selective_units.named_parameters():
            if p.grad is None or p.ndim == 0:
                continue
            if p.shape[0] == newn:
                p.grad[:48] = 0.0
            if p.ndim == 2 and p.shape[1] == newn:
                p.grad[:, :48] = 0.0
        torch.nn.utils.clip_grad_norm_(
            [p for g in (hopt, gopt) for grp in g.param_groups for p in grp["params"]], 1.0
        )
        hopt.step()
        gopt.step()
    subj.eval_mode()
    with torch.no_grad():
        after = float(
            (logits_mask(subj, fit27["ticks"], "p27", mask).argmax(-1) == fit27["labels"])
            .float()
            .mean()
        )
        zero_cols = head.net[0].weight.clone()
        head.net[0].weight[:, 2 * old_n :] = 0.0
        les = after - float(
            (logits_mask(subj, fit27["ticks"], "p27", mask).argmax(-1) == fit27["labels"])
            .float()
            .mean()
        )
        head.net[0].weight.copy_(zero_cols)
        fitcn = _build_bidirectional_cn_32dim(
            256, seed + 700 + 10_000_000, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
        )
        fit24 = _make_p24_stream(128, seed + 212 + 10_000_000)
        cn = float(
            (
                logits_mask(
                    subj, fitcn["ticks"], "cn", torch.cat([torch.ones(24), torch.zeros(newn - 24)])
                ).argmax(-1)
                == fitcn["labels"]
            )
            .float()
            .mean()
        )
        p24 = float(
            (
                logits_mask(
                    subj,
                    fit24["ticks"],
                    "p24",
                    torch.cat([torch.zeros(24), torch.ones(24), torch.zeros(newn - 48)]),
                )[:, :4].argmax(-1)
                == fit24["val_labels"]
            )
            .float()
            .mean()
        )
    return {
        "allocate": allocate,
        "base_p27": base,
        "after_p27": after,
        "lesion": les,
        "cn": cn,
        "p24": p24,
    }


def main():
    started = time.process_time()
    rows = [run_arm(SEED, True), run_arm(SEED, False)]
    ok = (
        rows[0]["lesion"] >= 0.10
        and rows[0]["after_p27"] >= rows[0]["base_p27"] - 0.05
        and rows[0]["lesion"] > rows[1]["lesion"]
    )
    payload = {
        "schema": "taiji-u2-growthB-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "rows": rows,
        "verdict": "growthB_verified" if ok else "growthB_partial",
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        },
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    Path(r"E:\Seed\reports\taiji_u2_growthB_20261122.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "rows": [
                    {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                    for r in rows
                ],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
