"""23-S.2a gain pilot: shared vs frozen vs routed (1 seed, <=600 CPU, 0 check)."""

import hashlib
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
    BI_CN_CHAR_INDEX,
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_PROJ32,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    CN_COLOURS,
    CN_MAX_LEN,
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

CN_CLASSES = len(CN_COLOURS)
HOLDOUT = {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)}


def cn_split(count, seed, split):
    first = BI_CN_TRAIN_FIRST if split == "train" else BI_CN_FIT_FIRST
    second = BI_CN_TRAIN_SECOND if split == "train" else BI_CN_FIT_SECOND
    all_tpls = [*[(t, True) for t in first], *[(t, False) for t in second]]
    rows, labels, idx, attempts = [], [], 0, 0
    while len(rows) < count and attempts < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, CN_CLASSES, (1,), generator=gen))
        c2 = int(torch.randint(0, CN_CLASSES - 1, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        pair = (c1, c2)
        ok = pair in HOLDOUT if split == "unseen" else pair not in HOLDOUT
        idx += 1
        attempts += 1
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


BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}
SEED = 20261120
B = 48
CN_STEPS, P24_STEPS, P27_STEPS = 1000, 900, 1300


def step(opt, loss, subj, keep, protect=True):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if protect:
        mask_grads(subj, keep)
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()


def eval_cn(subj, seed, split):
    subj.eval_mode()
    fit = cn_split(256, seed + (700 if split == "seen" else 900) + 10_000_000, split)
    with torch.no_grad():
        sm = (fit["ticks"].abs().sum(-1) > 0).float()
        return float(
            (subj.logits("cn", fit["ticks"], sm).argmax(-1) == fit["labels"]).float().mean()
        )


def build(seed, arm):
    subj = Subject72(seed)
    if arm == "shared":
        m = torch.ones(subj.n_units)
        subj.masks = {k: m.clone() for k in BLOCKS}
    else:
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
    for k in range(CN_STEPS):
        b = cn_split(B, seed * 6000 + k, "train")
        sm = (b["ticks"].abs().sum(-1) > 0).float()
        step(
            opt,
            F.cross_entropy(subj.logits("cn", b["ticks"], sm), b["labels"]),
            subj,
            "cn",
            arm != "shared",
        )
    if arm == "frozen":
        for p in subj.graph.selective_units.parameters():
            p.requires_grad_(False)
        for p in subj.graph.cross_channel_mix.parameters():
            p.requires_grad_(False)
    else:
        for p in subj.graph.cross_channel_mix.parameters():
            p.requires_grad_(False)
    for n in ("p24", "p27"):
        for p in subj.phase_heads[n].parameters():
            p.requires_grad_(True)
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
    for k in range(P24_STEPS):
        b = _make_p24_stream(B, seed * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], b["conflict_labels"]
        )
        step(opt, loss, subj, "p24", arm != "shared")
    fit24 = _make_p24_stream(128, seed + 212 + 10_000_000)
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
    for k in range(P27_STEPS):
        b = _make_p27_physics_batch(B, seed * 41000 + k)
        step(
            opt,
            F.cross_entropy(subj.logits("p27", b["ticks"]), b["labels"]),
            subj,
            "p27",
            arm != "shared",
        )
    subj.eval_mode()
    fit27 = _make_p27_physics_batch(128, seed + 81 + 10_000_000)
    with torch.no_grad():
        p24 = float(
            (subj.logits("p24", fit24["ticks"])[:, :4].argmax(-1) == fit24["val_labels"])
            .float()
            .mean()
        )
        p27 = float(
            (subj.logits("p27", fit27["ticks"]).argmax(-1) == fit27["labels"]).float().mean()
        )
    unseen = eval_cn(subj, seed, "unseen")
    return {"arm": arm, "cn_unseen": unseen, "p24": p24, "p27": p27, "q": min(unseen, p24, p27)}


def main():
    started = time.process_time()
    crit = json.loads(
        Path(r"E:\Seed\plans/manifests/taiji_u2_23s2a_criteria_20261122.json").read_text(
            encoding="utf-8"
        )
    )
    body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
    canon = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canon).hexdigest() == crit["criteria_sha256"]
    rows = [build(SEED, arm) for arm in ("shared", "frozen")]
    q_r = 0.835938  # 23-S.1b seed 20261120: min(1.0, 0.9922, 0.8359)
    gain = q_r - max(r["q"] for r in rows)
    verdict = "23s2a_gain_positive" if gain >= crit["gain_threshold"] else "23s2a_gain_absent"
    payload = {
        "schema": "taiji-u2-23s2a-gain-v1",
        "criteria_sha256": crit["criteria_sha256"],
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "q_routed": q_r,
        "rows": rows,
        "gain": gain,
        "threshold": crit["gain_threshold"],
        "verdict": verdict,
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        },
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    Path(r"E:\Seed\reports\taiji_u2_23s2a_gain_20261122.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "q_routed": q_r,
                "rows": rows,
                "gain": round(gain, 4),
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
