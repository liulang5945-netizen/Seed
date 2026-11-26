"""Joint revised criterion: simultaneous CN+P-B training, bounded-loss acceptance (1 seed, <=400 CPU, 0 check)."""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
    _make_p24_stream,
)
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

BLOCKS = {"cn": (0, 24), "p24": (24, 48)}
SEED = 20261120
B = 48


def step(opt, loss, subj, keep):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subj, keep)
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()


def mk():
    s = Subject72(SEED)
    s.set_masks(True)
    for n, (lo, hi) in BLOCKS.items():
        s.graph.set_task_route(n, list(range(lo, hi)))
    for p in s.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for n, h in s.phase_heads.items():
        for p in h.parameters():
            p.requires_grad_(n == "cn")
    s.train_mode()
    return s


def train(s, phases, steps):
    opt = torch.optim.Adam(s.params(), lr=6e-3, weight_decay=0.0)
    for k in range(steps):
        ph = phases[k % len(phases)]
        if ph == "cn":
            b = _build_bidirectional_cn_32dim(
                B, SEED * 6000 + k, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
            )
            sm = (b["ticks"].abs().sum(-1) > 0).float()
            loss = F.cross_entropy(s.logits("cn", b["ticks"], sm), b["labels"])
        else:
            b = _make_p24_stream(B, SEED * 25000 + k)
            lg = s.logits("p24", b["ticks"])
            loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
                lg[:, 4], b["conflict_labels"]
            )
        step(opt, loss, s, ph)


def acc(s):
    s.eval_mode()
    fitcn = _build_bidirectional_cn_32dim(
        256, SEED + 700 + 10_000_000, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    fit24 = _make_p24_stream(128, SEED + 212 + 10_000_000)
    with torch.no_grad():
        cn = float(
            (
                s.logits("cn", fitcn["ticks"], (fitcn["ticks"].abs().sum(-1) > 0).float()).argmax(
                    -1
                )
                == fitcn["labels"]
            )
            .float()
            .mean()
        )
        lg = s.logits("p24", fit24["ticks"])
        p24 = float((lg[:, :4].argmax(-1) == fit24["val_labels"]).float().mean())
        conf = float(((lg[:, 4] > 0).float() == fit24["conflict_labels"]).float().mean())
    return {"cn": cn, "p24": p24, "conflict": conf}


def main():
    started = time.process_time()
    torch.set_num_threads(1)
    solo_cn = mk()
    train(solo_cn, ["cn"], 600)
    cn_ref = acc(solo_cn)
    solo_p = mk()
    train(solo_p, ["p24"], 600)
    p_ref = acc(solo_p)
    joint = mk()
    train(joint, ["cn", "p24"], 1200)
    j = acc(joint)
    deg = {
        "cn": cn_ref["cn"] - j["cn"],
        "p24": p_ref["p24"] - j["p24"],
        "conflict": p_ref["conflict"] - j["conflict"],
    }
    floors = {
        "cn": j["cn"] >= max(0.85, cn_ref["cn"] - 0.10),
        "p24": j["p24"] >= max(0.75, p_ref["p24"] - 0.10),
        "conflict": j["conflict"] >= max(0.85, p_ref["conflict"] - 0.10),
    }
    no_collapse = min(j["cn"], j["p24"], j["conflict"]) >= 0.50
    bounded = max(deg.values()) <= 0.10
    verdict = (
        "joint_bounded_verified"
        if all(floors.values()) and no_collapse and bounded
        else "joint_bounded_partial"
    )
    payload = {
        "schema": "taiji-u2-joint-bounded-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "cn_ref": cn_ref,
        "p24_ref": p_ref,
        "joint": j,
        "degradation": deg,
        "floors": floors,
        "no_collapse": no_collapse,
        "bounded": bounded,
        "verdict": verdict,
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        },
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    Path(r"E:\Seed\reports\taiji_u2_joint_bounded_20261122.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "cn_ref": cn_ref,
                "p24_ref": p_ref,
                "joint": j,
                "degradation": {k: round(v, 4) for k, v in deg.items()},
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
