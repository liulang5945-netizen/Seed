"""P0-O: Chinese revision task with a new evidence-source token (first cut).

Base: Subject72 with the cn block (0:24) trained 800 steps on the standard 8-colour Chinese
task.  Revision task: the same sentences, but the last tick carries a NEW source token in
channels 8:16 (a fixed random 8-vector when the revision is active, zeros otherwise); when
active the correct answer is the OTHER colour of the pair (a flip), so the frozen cn head
cannot answer correctly.

Newborn units: grow_units_neutral_(2) -> 96 units; their write_in/r are zeroed at birth
(blind), then trained 200 interactions with an 8-way head over newborn states.

Checks: birth probe on newborn states, old cn-state probe, revision test accuracy, readout
reset (8 interactions), reset of learned input connections, normal-CN retention.
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

from scripts.training.verify_taiji_23s2a_gain_20261122 import (
    BI_CN_CHAR_INDEX,
    BI_CN_PROJ32,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    CN_CLASSES,
    CN_COLOURS,
    CN_MAX_LEN,
    HOLDOUT,
    cn_split,
)
from scripts.training.verify_taiji_24s_f1c_20261122 import (
    _forward_states,
    mask_newborn_grads,
    newborn_params,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0o_diag_20261123.json"
SEED = 20261120
OLD_N = 72
CN_STEPS = 800
REV_STEPS = 200
BATCH = 48
TARGET = 0.75
FLAG_CH = (8, 16)


def _flag_vector() -> torch.Tensor:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 15101)
        return torch.randn(FLAG_CH[1] - FLAG_CH[0])


FLAG_V = _flag_vector()


def revision_split(count: int, seed: int, split: str) -> dict:
    first = BI_CN_TRAIN_FIRST if split == "train" else BI_CN_TRAIN_SECOND
    all_tpls = [(t, True) for t in first] + [(t, False) for t in BI_CN_TRAIN_SECOND]
    rows, labels, flags = [], [], []
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, CN_CLASSES, (1,), generator=gen))
        c2 = int(torch.randint(0, CN_CLASSES - 1, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        pair = (c1, c2)
        ok = pair in HOLDOUT if split == "unseen" else pair not in HOLDOUT
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
        normal = c1 if pick_first else c2
        other = c2 if pick_first else c1
        fgen = torch.Generator().manual_seed(seed + 77_777 + idx)
        flag = int(torch.randint(0, 2, (1,), generator=fgen))
        if flag:
            seq[CN_MAX_LEN - 1, FLAG_CH[0] : FLAG_CH[1]] = FLAG_V
        rows.append(seq)
        labels.append(other if flag else normal)
        flags.append(flag)
    return {
        "ticks": torch.stack(rows),
        "labels": torch.tensor(labels, dtype=torch.long),
        "flags": torch.tensor(flags, dtype=torch.long),
    }


def mask96(lo: int, hi: int) -> torch.Tensor:
    m = torch.zeros(96)
    m[lo:hi] = 1.0
    return m


def states(subj, ticks, mask, newborn_only: bool = False):
    subj.eval_mode()
    with torch.no_grad():
        z, a = _forward_states(subj, ticks, mask)
    if newborn_only:
        return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)
    return torch.cat([z, a], dim=-1)


def eval_head(head, subj, ticks, labels, mask) -> float:
    subj.eval_mode()
    with torch.no_grad():
        pred = head(states(subj, ticks, mask, newborn_only=True)).argmax(-1)
    return float((pred == labels).float().mean())


def cn_states144(subj, ticks):
    subj.eval_mode()
    with torch.no_grad():
        z, a = _forward_states(subj, ticks, mask96(0, 24))
    zc = torch.zeros(ticks.shape[0], OLD_N)
    ac = torch.zeros(ticks.shape[0], OLD_N)
    zc[:, :24] = z[:, :24]
    ac[:, :24] = a[:, :24]
    return torch.cat([zc, ac], dim=-1)


def probe(feat: torch.Tensor, labels: torch.Tensor) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 15201)
        net = torch.nn.Sequential(
            torch.nn.Linear(feat.shape[-1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 8)
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
        train = float((net(feat).argmax(-1) == labels).float().mean())
    return {"train": train}


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=1e-4)
    subj.train_mode()
    for step in range(CN_STEPS):
        batch = cn_split(BATCH, SEED * 6000 + step, "train")
        loss = F.cross_entropy(
            subj.logits("cn", batch["ticks"], (batch["ticks"].abs().sum(-1) > 0).float()),
            batch["labels"],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
    normal = cn_split(256, SEED + 900 + 10_000_000, "seen")
    normal_acc = float(
        (
            subj.logits("cn", normal["ticks"], (normal["ticks"].abs().sum(-1) > 0).float()).argmax(
                -1
            )
            == normal["labels"]
        )
        .float()
        .mean()
    )

    subj.graph.grow_units_neutral_(2)
    units = subj.graph.selective_units
    subj.n_units = 96
    for name in ("cn", "p24", "p27"):
        pad = torch.zeros(96)
        pad[:OLD_N] = subj.masks[name][:OLD_N]
        subj.masks[name] = pad
    with torch.no_grad():
        units.write_in.weight[OLD_N:] = 0.0
        if hasattr(units, "r"):
            units.r[OLD_N:] = 0.0

    test = revision_split(256, SEED + 1300 + 10_000_000, "seen")
    new_mask = mask96(OLD_N, 96)
    cn_mask = mask96(0, 24)

    birth_feat = states(subj, test["ticks"], new_mask, newborn_only=True)
    birth_probe = probe(birth_feat, test["labels"])
    cn_feat = states(subj, test["ticks"], cn_mask)[:, : 2 * 24]
    cn_probe = probe(cn_feat, test["labels"])
    subj.eval_mode()
    with torch.no_grad():
        old_head_rev = float(
            (subj.phase_heads["cn"](cn_states144(subj, test["ticks"])).argmax(-1) == test["labels"])
            .float()
            .mean()
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 15301)
        head = PhaseHead(2 * (96 - OLD_N), 8, 64)
    params = list(head.parameters()) + list(newborn_params(subj))
    opt2 = torch.optim.Adam(params, lr=6e-3)
    for i in range(REV_STEPS):
        batch = revision_split(BATCH, SEED * 7000 + i, "train")
        subj.train_mode()
        z, a = _forward_states(subj, batch["ticks"], new_mask)
        logits = head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1))
        loss = F.cross_entropy(logits, batch["labels"])
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt2.step()
    rev_acc = eval_head(head, subj, test["ticks"], test["labels"], new_mask)

    # transfer: readout reset to a fresh head, 8 interactions
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 15302)
        head2 = PhaseHead(2 * (96 - OLD_N), 8, 64)
    for p in newborn_params(subj):
        p.requires_grad_(False)
    opt3 = torch.optim.Adam(head2.parameters(), lr=6e-3)
    for i in range(8):
        batch = revision_split(BATCH, SEED * 7000 + 5000 + i, "train")
        z, a = _forward_states(subj, batch["ticks"], new_mask)
        loss = F.cross_entropy(
            head2(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)), batch["labels"]
        )
        opt3.zero_grad(set_to_none=True)
        loss.backward()
        opt3.step()
    transfer = eval_head(head2, subj, test["ticks"], test["labels"], new_mask)

    # reset learned input connections
    with torch.no_grad():
        units.write_in.weight[OLD_N:] = 0.0
        if hasattr(units, "r"):
            units.r[OLD_N:] = 0.0
    head3 = PhaseHead(2 * (96 - OLD_N), 8, 64)
    opt4 = torch.optim.Adam(head3.parameters(), lr=6e-3)
    for i in range(8):
        batch = revision_split(BATCH, SEED * 7000 + 5100 + i, "train")
        z, a = _forward_states(subj, batch["ticks"], new_mask)
        loss = F.cross_entropy(
            head3(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)), batch["labels"]
        )
        opt4.zero_grad(set_to_none=True)
        loss.backward()
        opt4.step()
    reset_acc = eval_head(head3, subj, test["ticks"], test["labels"], new_mask)
    subj.eval_mode()
    with torch.no_grad():
        normal_after = float(
            (
                subj.phase_heads["cn"](cn_states144(subj, normal["ticks"])).argmax(-1)
                == normal["labels"]
            )
            .float()
            .mean()
        )

    gates = {
        "G0_old_cn": normal_acc >= 0.85,
        "G1_old_head_rev": old_head_rev <= 0.55,
        "G2_birth_probe": birth_probe["train"] <= 0.55,
        "G3_rev_acc": rev_acc >= TARGET,
        "G4_transfer": transfer >= TARGET,
        "G5_reset_drop": (rev_acc - reset_acc) >= 0.15,
        "G6_retention": (normal_acc - normal_after) <= 0.01,
    }
    if not gates["G0_old_cn"]:
        verdict = "p0o_cn_base_failed"
    elif not gates["G1_old_head_rev"]:
        verdict = "p0o_old_head_solves"
    elif not gates["G2_birth_probe"]:
        verdict = "p0o_not_pressured"
    elif not gates["G3_rev_acc"]:
        verdict = "p0o_not_learnable"
    elif not gates["G4_transfer"]:
        verdict = "p0o_no_transfer"
    elif not gates["G5_reset_drop"]:
        verdict = "p0o_connections_not_needed"
    elif not gates["G6_retention"]:
        verdict = "p0o_retention_breach"
    else:
        verdict = "p0o_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0o-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "normal_cn": normal_acc,
        "normal_cn_after": normal_after,
        "old_head_rev": old_head_rev,
        "birth_probe": birth_probe,
        "cn_probe": cn_probe,
        "rev_acc": rev_acc,
        "transfer": transfer,
        "reset": reset_acc,
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": cost},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "normal_cn",
                    "normal_cn_after",
                    "old_head_rev",
                    "birth_probe",
                    "cn_probe",
                    "rev_acc",
                    "transfer",
                    "reset",
                    "gates",
                    "verdict",
                    "cost",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
