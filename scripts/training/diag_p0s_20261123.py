"""P0-S: converged CN base + re-test of channel-group suppression (P0-Q on a trained base).

Train Subject72 cn block with the stage1b recipe (_build_bidirectional_cn_32dim, 1200
steps, step_mask) until normal Chinese accuracy >= 0.85, then inject a 4-way one-hot flag
into candidate channel groups (0:4 control, 32:36, 40:44) at every tick and probe the CN
state decodability (MLP 1500 steps).
Gate: base >= 0.85; a suppressed group has probe <= 0.55.
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

from scripts.training.verify_taiji_23s2a_gain_20261122 import BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
from scripts.training.verify_taiji_24s_f1c_20261122 import _forward_states
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    FIT_OFFSET,
    _build_bidirectional_cn_32dim,
)
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0s_diag_20261123.json"
SEED = 20261120
CN_STEPS = 1200
BATCH = 48
GROUPS = {"char_0_4": (0, 4), "grp_32_36": (32, 36), "grp_40_44": (40, 44)}


def flag_batch(n: int, seed: int, group: tuple[int, int]) -> dict:
    base = _build_bidirectional_cn_32dim(n, seed, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND)
    ticks = base["ticks"].clone()
    g = torch.Generator().manual_seed(seed + 31337)
    flags = torch.randint(0, 4, (n,), generator=g)
    lo, hi = group
    ticks[:, :, lo:hi] = 0.0
    for i in range(n):
        ticks[i, :, lo + int(flags[i])] = 1.0
    return {"ticks": ticks, "flags": flags}


def cn_states48(subj, ticks) -> torch.Tensor:
    mask = torch.zeros(72)
    mask[:24] = 1.0
    subj.eval_mode()
    with torch.no_grad():
        z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, :24], a[:, :24]], dim=-1)


def probe(feat: torch.Tensor, labels: torch.Tensor) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 16401)
        net = torch.nn.Sequential(
            torch.nn.Linear(feat.shape[-1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)
        )
    opt = torch.optim.Adam(net.parameters(), lr=7e-3)
    for _ in range(1500):
        idx = torch.randint(0, feat.shape[0], (64,))
        loss = F.cross_entropy(net(feat[idx]), labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return {"train": float((net(feat).argmax(-1) == labels).float().mean())}


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
    fit = _build_bidirectional_cn_32dim(
        256, SEED + 700 + FIT_OFFSET, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    )
    subj.eval_mode()
    with torch.no_grad():
        cn_acc = float(
            (
                subj.logits("cn", fit["ticks"], (fit["ticks"].abs().sum(-1) > 0).float()).argmax(-1)
                == fit["labels"]
            )
            .float()
            .mean()
        )
    probes = {}
    for name, group in GROUPS.items():
        tr = flag_batch(512, SEED + 1500 + 10_000_000, group)
        probes[name] = probe(cn_states48(subj, tr["ticks"]), tr["flags"])
    gates = {
        "G0_cn_converged": cn_acc >= 0.85,
        "G1_suppressed_32_36": probes["grp_32_36"]["train"] <= 0.55,
        "G2_suppressed_40_44": probes["grp_40_44"]["train"] <= 0.55,
        "G3_control_transmits": probes["char_0_4"]["train"] >= 0.85,
    }
    if not gates["G0_cn_converged"]:
        verdict = "p0s_cn_not_converged"
    elif not gates["G3_control_transmits"]:
        verdict = "p0s_control_failed"
    elif not (gates["G1_suppressed_32_36"] and gates["G2_suppressed_40_44"]):
        verdict = "p0s_not_suppressed"
    else:
        verdict = "p0s_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0s-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "cn_steps": CN_STEPS,
        "cn_acc": cn_acc,
        "probes": probes,
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": cost},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "cn_acc": cn_acc,
                "probes": probes,
                "gates": gates,
                "verdict": verdict,
                "cost": payload["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
