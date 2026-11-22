"""Growth A: demand-driven growth for a weak capability (1 seed, <=400 CPU, 0 check)."""

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

BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}
SEED = 20261120
B = 48


def step(opt, loss, subj, keep):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subj, keep)
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()


def _mask_for(subj, lo, hi):
    m = torch.zeros(subj.n_units)
    m[lo:hi] = 1.0
    return m


def remap_old(subj, ticks, name, mask):
    m = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], subj.n_units)
    out = subj.graph(_channels_seq(ticks), m, subj.graph.initial_state(batch_size=ticks.shape[0]))
    z, a = out["next_state"]["z"], out["next_state"]["a"]
    h = subj.phase_heads[name]
    n = h.net[0].in_features // 2
    return h(torch.cat([z[..., :n], a[..., :n]], dim=-1))


def logits_new(subj, ticks, name, mask):
    m = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], subj.n_units)
    out = subj.graph(_channels_seq(ticks), m, subj.graph.initial_state(batch_size=ticks.shape[0]))
    z, a = out["next_state"]["z"], out["next_state"]["a"]
    h = subj.phase_heads[name]
    n = h.net[0].in_features // 2
    return h(torch.cat([z[..., :n], a[..., :n]], dim=-1))


def np_ticks(batch):
    t = batch["ticks"].clone()
    t[:, :6, 8:11] = 0.0
    return t


def main():
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
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
        b = _build_bidirectional_cn_32dim(B, SEED * 6000 + k, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND)
        sm = (b["ticks"].abs().sum(-1) > 0).float()
        step(opt, F.cross_entropy(subj.logits("cn", b["ticks"], sm), b["labels"]), subj, "cn")
    for p in subj.graph.cross_channel_mix.parameters():
        p.requires_grad_(False)
    for n in ("p24", "p27"):
        for p in subj.phase_heads[n].parameters():
            p.requires_grad_(True)
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
    for k in range(700):
        b = _make_p24_stream(B, SEED * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        step(
            opt,
            F.cross_entropy(lg[:, :4], b["val_labels"])
            + F.binary_cross_entropy_with_logits(lg[:, 4], b["conflict_labels"]),
            subj,
            "p24",
        )
    for k in range(900):
        b = _make_p27_physics_batch(B, SEED * 41000 + k)
        step(opt, F.cross_entropy(subj.logits("p27", b["ticks"]), b["labels"]), subj, "p27")
    subj.eval_mode()
    fit27 = _make_p27_physics_batch(128, SEED + 81 + 10_000_000)
    fit24 = _make_p24_stream(128, SEED + 212 + 10_000_000)
    fitcn = _build_bidirectional_cn_32dim(
        256, SEED + 700 + 10_000_000, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    with torch.no_grad():
        base_p27 = float(
            (subj.logits("p27", fit27["ticks"]).argmax(-1) == fit27["labels"]).float().mean()
        )
        base_cn = float(
            (
                subj.logits(
                    "cn", fitcn["ticks"], (fitcn["ticks"].abs().sum(-1) > 0).float()
                ).argmax(-1)
                == fitcn["labels"]
            )
            .float()
            .mean()
        )
        lg24 = subj.logits("p24", fit24["ticks"])
        base_p24 = float((lg24[:, :4].argmax(-1) == fit24["val_labels"]).float().mean())
        weak = np_ticks(fit27)
        base_weak = float((subj.logits("p27", weak).argmax(-1) == fit27["labels"]).float().mean())
    old_n = subj.graph.selective_units.n_units
    subj.graph.grow_units_neutral_(3)
    newn = subj.graph.selective_units.n_units
    subj.n_units = newn
    for name in ("cn", "p24"):
        pad = torch.zeros(newn)
        pad[:old_n] = subj.masks[name]
        subj.masks[name] = pad
    new_mask = torch.zeros(newn)
    new_mask[old_n:] = 1.0
    subj.phase_heads["p27np"] = PhaseHead(2 * newn, 4, 64)
    for p in subj.phase_heads["p27np"].parameters():
        p.requires_grad_(True)
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    hopt = torch.optim.Adam(subj.phase_heads["p27np"].parameters(), lr=7e-3, weight_decay=0.0)
    gopt = torch.optim.Adam(subj.graph.parameters(), lr=5e-3, weight_decay=0.0)
    subj.train_mode()
    for k in range(1000):
        b = _make_p27_physics_batch(B, SEED * 41000 + 100000 + k)
        x = np_ticks(b)
        loss = F.cross_entropy(logits_new(subj, x, "p27np", new_mask), b["labels"])
        hopt.zero_grad(set_to_none=True)
        gopt.zero_grad(set_to_none=True)
        loss.backward()
        for _n, p in subj.graph.selective_units.named_parameters():
            if p.grad is None or p.ndim == 0:
                continue
            if p.shape[0] == newn:
                p.grad[:old_n] = 0.0
        rg = subj.graph.read_gate
        if rg.weight.grad is not None:
            for i in range(rg.weight.grad.shape[0]):
                if not (old_n <= i < newn or newn + old_n <= i < newn + newn):
                    rg.weight.grad[i] = 0.0
                    rg.bias.grad[i] = 0.0
        torch.nn.utils.clip_grad_norm_(
            [p for g in (hopt, gopt) for grp in g.param_groups for p in grp["params"]], 1.0
        )
        hopt.step()
        gopt.step()
    subj.eval_mode()
    with torch.no_grad():
        weak_after = float(
            (logits_new(subj, weak, "p27np", new_mask).argmax(-1) == fit27["labels"]).float().mean()
        )
        zero_mask = torch.zeros(newn)
        weak_zero = float(
            (logits_new(subj, weak, "p27np", zero_mask).argmax(-1) == fit27["labels"])
            .float()
            .mean()
        )
        old_p27 = float(
            (
                remap_old(subj, fit27["ticks"], "p27", _mask_for(subj, 48, 72)).argmax(-1)
                == fit27["labels"]
            )
            .float()
            .mean()
        )
        old_cn = float(
            (
                remap_old(subj, fitcn["ticks"], "cn", _mask_for(subj, 0, 24)).argmax(-1)
                == fitcn["labels"]
            )
            .float()
            .mean()
        )
        old_p24 = float(
            (
                remap_old(subj, fit24["ticks"], "p24", _mask_for(subj, 24, 48))[:, :4].argmax(-1)
                == fit24["val_labels"]
            )
            .float()
            .mean()
        )
    damage = max(0.0, base_p27 - old_p27, base_cn - old_cn, base_p24 - old_p24)
    lesion = weak_after - weak_zero
    gain = weak_after - base_weak
    verdict = (
        "growthA_verified"
        if (gain >= 0.05 and lesion >= 0.10 and damage <= 0.05 and weak_after >= 0.70)
        else "growthA_partial"
    )
    payload = {
        "schema": "taiji-u2-growthA-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "base_weak": base_weak,
        "weak_after": weak_after,
        "gain": gain,
        "lesion": lesion,
        "old_damage": damage,
        "old_p27": old_p27,
        "old_cn": old_cn,
        "old_p24": old_p24,
        "verdict": verdict,
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        },
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    Path(r"E:\Seed\reports\taiji_u2_growthA_20261122.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in payload.items()
                if k
                in ("base_weak", "weak_after", "gain", "lesion", "old_damage", "verdict", "cost")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
