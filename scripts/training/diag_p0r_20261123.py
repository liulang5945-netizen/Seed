"""P0-R: Chinese revision task with a new evidence-source token in a SUPPRESSED channel.

Base: Subject72 cn block trained 800 steps on the standard 8-colour Chinese task.
Revision: the same sentence, but the last tick carries a 2-dim source flag in the
SUPPRESSED channels 32:36 ([1,0]=normal, [0,1]=revision).  When revision is active the
correct answer flips to the OTHER colour of the pair (normal answer -> other).
Newborn units (grow 24, write_in/r zeroed at birth) read the flag ONLY through the unit-
level raw-input adapter (attach_extra_), trained for 200 interactions with an 8-way head.

Checks: old cn accuracy (unified step_mask), cn-state probe on revision labels, newborn
birth probe, revision accuracy, readout reset (8 interactions), adapter reset (drop),
old-cn retention.  Exploratory diagnostic; zero check/final.
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
from scripts.training.verify_taiji_24s_f1c_20261122 import mask_newborn_grads, newborn_params
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0r_diag_20261123.json"
SEED = 20261120
OLD_N = 72
CN_STEPS = 800
REV_STEPS = 200
BATCH = 48
TARGET = 0.75
FLAG_CH = (32, 34)


def revision_split(count: int, seed: int, split: str) -> dict:
    first = BI_CN_TRAIN_FIRST
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
            seq[CN_MAX_LEN - 1, FLAG_CH[0]] = 0.0
            seq[CN_MAX_LEN - 1, FLAG_CH[0] + 1] = 1.0
        else:
            seq[CN_MAX_LEN - 1, FLAG_CH[0]] = 1.0
            seq[CN_MAX_LEN - 1, FLAG_CH[0] + 1] = 0.0
        rows.append(seq)
        labels.append(other if flag else normal)
        flags.append(flag)
    return {
        "ticks": torch.stack(rows),
        "labels": torch.tensor(labels, dtype=torch.long),
        "flags": torch.tensor(flags, dtype=torch.long),
    }


def pad_mask(units: int, lo: int, hi: int) -> torch.Tensor:
    m = torch.zeros(units)
    m[lo:hi] = 1.0
    return m


def run_states(subj, ticks, mask, extra=None, step_mask=None):
    gates = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], mask.numel())
    out = subj.graph(
        _channels_seq(ticks),
        gates,
        subj.graph.initial_state(batch_size=ticks.shape[0]),
        step_mask=step_mask,
        extra=extra,
    )
    return out["next_state"]["z"], out["next_state"]["a"]


def cn_state_vec(subj, ticks, step_mask=None) -> torch.Tensor:
    mask = pad_mask(subj.n_units, 0, 24)
    subj.eval_mode()
    with torch.no_grad():
        z, a = run_states(subj, ticks, mask, step_mask=step_mask)
    zc = torch.zeros(ticks.shape[0], OLD_N)
    ac = torch.zeros(ticks.shape[0], OLD_N)
    zc[:, :24] = z[:, :24]
    ac[:, :24] = a[:, :24]
    return torch.cat([zc, ac], dim=-1)


def cn_eval(subj) -> float:
    normal = cn_split(256, SEED + 900 + 10_000_000, "seen")
    sm = (normal["ticks"].abs().sum(-1) > 0).float()
    subj.eval_mode()
    with torch.no_grad():
        logits = subj.phase_heads["cn"](cn_state_vec(subj, normal["ticks"], step_mask=sm))
        return float((logits.argmax(-1) == normal["labels"]).float().mean())


def probe(feat: torch.Tensor, labels: torch.Tensor) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 16201)
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
        return {"train": float((net(feat).argmax(-1) == labels).float().mean())}


def newborn_states(subj, ticks, extra) -> torch.Tensor:
    mask = pad_mask(subj.n_units, OLD_N, 96)
    z, a = run_states(subj, ticks, mask, extra=extra)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


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
    normal_cn = cn_eval(subj)

    subj.graph.grow_units_neutral_(2)
    units = subj.graph.selective_units
    subj.n_units = 96
    units.attach_extra_(2, seed=SEED)
    with torch.no_grad():
        units.write_in.weight[OLD_N:] = 0.0
        if hasattr(units, "r"):
            units.r[OLD_N:] = 0.0

    test = revision_split(256, SEED + 1300 + 10_000_000, "seen")
    extra_test = test["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
    cn_probe = probe(
        cn_state_vec(subj, test["ticks"], step_mask=(test["ticks"].abs().sum(-1) > 0).float()),
        test["labels"],
    )
    with torch.no_grad():
        birth_feat = newborn_states(subj, test["ticks"], extra_test).detach()
    birth_probe = probe(birth_feat, test["labels"])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 16301)
        head = PhaseHead(2 * (96 - OLD_N), 8, 64)
    params = list(head.parameters()) + list(newborn_params(subj))
    opt2 = torch.optim.Adam(params, lr=6e-3)
    for i in range(REV_STEPS):
        batch = revision_split(BATCH, SEED * 7000 + i, "train")
        b_extra = batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        subj.train_mode()
        logits = head(newborn_states(subj, batch["ticks"], b_extra))
        loss = F.cross_entropy(logits, batch["labels"])
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt2.step()
    subj.eval_mode()
    with torch.no_grad():
        rev_acc = float(
            (head(newborn_states(subj, test["ticks"], extra_test)).argmax(-1) == test["labels"])
            .float()
            .mean()
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 16302)
        head2 = PhaseHead(2 * (96 - OLD_N), 8, 64)
    for p in newborn_params(subj):
        p.requires_grad_(False)
    opt3 = torch.optim.Adam(head2.parameters(), lr=6e-3)
    for i in range(8):
        batch = revision_split(BATCH, SEED * 7000 + 5000 + i, "train")
        b_extra = batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        logits = head2(newborn_states(subj, batch["ticks"], b_extra))
        loss = F.cross_entropy(logits, batch["labels"])
        opt3.zero_grad(set_to_none=True)
        loss.backward()
        opt3.step()
    subj.eval_mode()
    with torch.no_grad():
        transfer = float(
            (head2(newborn_states(subj, test["ticks"], extra_test)).argmax(-1) == test["labels"])
            .float()
            .mean()
        )

    with torch.no_grad():
        units.extra_proj.weight[OLD_N:] = 0.0
    head3 = PhaseHead(2 * (96 - OLD_N), 8, 64)
    opt4 = torch.optim.Adam(head3.parameters(), lr=6e-3)
    for i in range(8):
        batch = revision_split(BATCH, SEED * 7000 + 5100 + i, "train")
        b_extra = batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        logits = head3(newborn_states(subj, batch["ticks"], b_extra))
        loss = F.cross_entropy(logits, batch["labels"])
        opt4.zero_grad(set_to_none=True)
        loss.backward()
        opt4.step()
    subj.eval_mode()
    with torch.no_grad():
        reset_acc = float(
            (head3(newborn_states(subj, test["ticks"], extra_test)).argmax(-1) == test["labels"])
            .float()
            .mean()
        )
    normal_after = cn_eval(subj)

    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_pressure": cn_probe["train"] <= 0.55 and birth_probe["train"] <= 0.55,
        "G2_rev": rev_acc >= TARGET,
        "G3_transfer": transfer >= TARGET,
        "G4_adapter_drop": (rev_acc - reset_acc) >= 0.15,
        "G5_retention": (normal_cn - normal_after) <= 0.01,
    }
    if not gates["G0_old_cn"]:
        verdict = "p0r_cn_base_failed"
    elif not gates["G1_pressure"]:
        verdict = "p0r_not_pressured"
    elif not gates["G2_rev"]:
        verdict = "p0r_not_learnable"
    elif not gates["G3_transfer"]:
        verdict = "p0r_no_transfer"
    elif not gates["G4_adapter_drop"]:
        verdict = "p0r_adapter_not_needed"
    elif not gates["G5_retention"]:
        verdict = "p0r_retention_breach"
    else:
        verdict = "p0r_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0r-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "cn_probe": cn_probe,
        "birth_probe": birth_probe,
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
                    "cn_probe",
                    "birth_probe",
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
