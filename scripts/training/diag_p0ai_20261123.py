"""P0-Y: does the newborn cohort learn to DETECT a revision from the new source channel?

Same pressure-qualified setting as P0-X (balanced 4-dim flag on tick 0 in the suppressed
channels 32:36; cn base 1200 steps, held-out probes).  Task simplified to binary detection:
label = 1 when the flag class is in {2,3} (revision active), else 0.  The newborn units keep
write_in and read the flag only through the unit-level adapter (attach_extra_).

Checks: old cn accuracy, held-out cn-state probe, held-out birth probe, detection accuracy,
readout reset (8 interactions), adapter reset drop, old-cn retention.
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
)
from scripts.training.verify_taiji_24s_f1c_20261122 import mask_newborn_grads, newborn_params
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_cognitive_loop_20261121 import _build_bidirectional_cn_32dim
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0ai_diag_20261123.json"
SEED = 20261120
OLD_N = 72
CN_STEPS = 1200
DET_STEPS = 800
FLIP_STEPS = 1600
CONSOL_STEPS = 1500
BATCH = 48
TARGET = 0.90
FLAG_CH = (32, 36)


def revision_split(count: int, seed: int) -> dict:
    all_tpls = [(t, True) for t in BI_CN_TRAIN_FIRST] + [(t, False) for t in BI_CN_TRAIN_SECOND]
    rows, flags, flip_labels = [], [], []
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, CN_CLASSES, (1,), generator=gen))
        c2 = int(torch.randint(0, CN_CLASSES - 1, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = (c1, c2) not in HOLDOUT
        idx += 1
        if not ok:
            continue
        tpl, _pick = all_tpls[len(rows) % len(all_tpls)]
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        normal = c1 if _pick else c2
        other = c2 if _pick else c1
        fgen = torch.Generator().manual_seed(seed + 77_777 + idx)
        flag = int(torch.randint(0, 4, (1,), generator=fgen))
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        rows.append(seq)
        flags.append(1 if flag >= 2 else 0)
        flip_labels.append(other if flag >= 2 else normal)
    return {
        "ticks": torch.stack(rows),
        "labels": torch.tensor(flags, dtype=torch.long),
        "flags": torch.tensor(flags, dtype=torch.long),
        "flip_labels": torch.tensor(flip_labels, dtype=torch.long),
    }


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


def mask_fn(units: int, lo: int, hi: int) -> torch.Tensor:
    m = torch.zeros(units)
    m[lo:hi] = 1.0
    return m


def cn_feature(subj, ticks, step_mask=None) -> torch.Tensor:
    subj.eval_mode()
    with torch.no_grad():
        z, a = run_states(subj, ticks, mask_fn(subj.n_units, 0, 24), step_mask=step_mask)
    return torch.cat([z[:, :OLD_N], a[:, :OLD_N]], dim=-1)


def cn_eval(subj) -> float:
    fit = _build_bidirectional_cn_32dim(
        256, SEED + 900 + 10_000_000, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    )
    sm = (fit["ticks"].abs().sum(-1) > 0).float()
    subj.eval_mode()
    with torch.no_grad():
        logits = subj.phase_heads["cn"](cn_feature(subj, fit["ticks"], step_mask=sm))
        return float((logits.argmax(-1) == fit["labels"]).float().mean())


def probe(feat: torch.Tensor, labels: torch.Tensor, classes: int = 2) -> dict:
    n_train = int(feat.shape[0] * 0.75)
    tr_x, tr_y = feat[:n_train], labels[:n_train]
    te_x, te_y = feat[n_train:], labels[n_train:]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17501)
        net = torch.nn.Sequential(
            torch.nn.Linear(feat.shape[-1], 64), torch.nn.Tanh(), torch.nn.Linear(64, classes)
        )
    opt = torch.optim.Adam(net.parameters(), lr=7e-3)
    for _ in range(1500):
        idx = torch.randint(0, tr_x.shape[0], (64,))
        loss = F.cross_entropy(net(tr_x[idx]), tr_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return {
            "train": float((net(tr_x).argmax(-1) == tr_y).float().mean()),
            "test": float((net(te_x).argmax(-1) == te_y).float().mean()),
        }


def newborn_states(subj, ticks, extra) -> torch.Tensor:
    z, a = run_states(subj, ticks, mask_fn(subj.n_units, OLD_N, subj.n_units), extra=extra)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


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
    normal_cn = cn_eval(subj)

    subj.graph.grow_units_neutral_(2)
    units = subj.graph.selective_units
    subj.n_units = 96
    units.attach_extra_(4, seed=SEED)

    # inherit the cn block's sentence-reading weights into the newborn units
    with torch.no_grad():
        src_idx = torch.arange(24)
        dst_idx = torch.arange(24) + OLD_N
        units.group_index[dst_idx] = torch.arange(24) // 6  # cn groups 0..3
        ar = torch.arange(12).unsqueeze(1)
        group_idx = units.group_index
        units.group_mask.copy_((group_idx.unsqueeze(0) == ar).to(units.group_mask.dtype))
        if units.spec.cross_group_edges:
            edge = 1.0 - torch.eye(96, dtype=units.edge_mask.dtype)
        else:
            same = (group_idx.unsqueeze(1) == group_idx.unsqueeze(0)).to(units.edge_mask.dtype)
            edge = same * (1.0 - torch.eye(96, dtype=units.edge_mask.dtype))
        units.edge_mask.copy_(edge)
        units.r[dst_idx] = units.r[src_idx].clone()
        units.p[:, dst_idx] = units.p[:, src_idx].clone()
        for name in ("c", "eta", "b", "write_bias"):
            target = getattr(units, name)
            target[dst_idx] = target[src_idx].clone()
        units.w[dst_idx, :] = units.w[src_idx, :].clone()
        units.write_in.weight[dst_idx, :] = units.write_in.weight[src_idx, :].clone()
        units.write_state.weight[dst_idx, :] = units.write_state.weight[src_idx, :].clone()
        # open the inherited write gate: without this the adapter (write-gate only) is muted
        units.write_bias[dst_idx] = 0.0
        units.b[dst_idx] = 0.0

    test = revision_split(512, SEED + 1300 + 10_000_000)
    sm = (test["ticks"].abs().sum(-1) > 0).float()
    cn_probe = probe(cn_feature(subj, test["ticks"], step_mask=sm), test["labels"])
    extra_test = test["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
    with torch.no_grad():
        birth_feat = newborn_states(subj, test["ticks"], extra_test).detach()
    birth_probe = probe(birth_feat, test["labels"])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17601)
        head = PhaseHead(2 * (96 - OLD_N), 2, 64)
    params = list(head.parameters()) + list(newborn_params(subj))
    opt2 = torch.optim.Adam(params, lr=6e-3)
    for i in range(DET_STEPS):
        batch = revision_split(BATCH, SEED * 7000 + i)
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
        detect_acc_at_phase = float(
            (head(newborn_states(subj, test["ticks"], extra_test)).argmax(-1) == test["labels"])
            .float()
            .mean()
        )

    # phase 2 (curriculum): learn the colour FLIP on top of the detection-trained units
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17701)
        head_flip = PhaseHead(2 * (96 - OLD_N), 8, 64)
    params_flip = list(head_flip.parameters()) + list(newborn_params(subj))
    opt_flip = torch.optim.Adam(params_flip, lr=6e-3)
    for i in range(FLIP_STEPS):
        batch = revision_split(BATCH, SEED * 7100 + i)
        b_extra = batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        subj.train_mode()
        logits = head_flip(newborn_states(subj, batch["ticks"], b_extra))
        loss = F.cross_entropy(logits, batch["flip_labels"])
        opt_flip.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params_flip, 1.0)
        opt_flip.step()
    # consolidation: freeze the readout, keep training units+adapter so the function
    # moves into the unit states (then a fresh 8-step readout should suffice)
    for p in head_flip.parameters():
        p.requires_grad_(False)
    units_only = list(newborn_params(subj))
    opt_cons = torch.optim.Adam(units_only, lr=6e-3)
    for i in range(CONSOL_STEPS):
        batch = revision_split(BATCH, SEED * 7200 + i)
        b_extra = batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        subj.train_mode()
        logits = head_flip(newborn_states(subj, batch["ticks"], b_extra))
        loss = F.cross_entropy(logits, batch["flip_labels"])
        opt_cons.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(units_only, 1.0)
        opt_cons.step()
    # closed-form linear probe on the consolidated unit states (no training)
    with torch.no_grad():
        probe_batch = revision_split(1024, SEED + 1800 + 10_000_000)
        probe_extra = probe_batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        feat = newborn_states(subj, probe_batch["ticks"], probe_extra).detach()
        feat = torch.cat([feat, torch.ones(feat.shape[0], 1)], dim=-1)
        y = F.one_hot(probe_batch["flip_labels"], 8).float()
        w = torch.linalg.lstsq(feat, y).solution
        test_batch = revision_split(256, SEED + 1801 + 10_000_000)
        test_extra = test_batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        feat_te = newborn_states(subj, test_batch["ticks"], test_extra).detach()
        feat_te = torch.cat([feat_te, torch.ones(feat_te.shape[0], 1)], dim=-1)
        linear_probe = float(((feat_te @ w).argmax(-1) == test_batch["flip_labels"]).float().mean())
    subj.eval_mode()
    with torch.no_grad():
        flip_acc = float(
            (
                head_flip(newborn_states(subj, test["ticks"], extra_test)).argmax(-1)
                == test["flip_labels"]
            )
            .float()
            .mean()
        )
    # flip transfer: fresh 8-way readout, 8 interactions
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17702)
        head_flip2 = PhaseHead(2 * (96 - OLD_N), 8, 64)
    for p in newborn_params(subj):
        p.requires_grad_(False)
    opt_f2 = torch.optim.Adam(head_flip2.parameters(), lr=6e-3)
    for i in range(8):
        batch = revision_split(BATCH, SEED * 7100 + 5000 + i)
        b_extra = batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        logits = head_flip2(newborn_states(subj, batch["ticks"], b_extra))
        loss = F.cross_entropy(logits, batch["flip_labels"])
        opt_f2.zero_grad(set_to_none=True)
        loss.backward()
        opt_f2.step()
    subj.eval_mode()
    with torch.no_grad():
        flip_transfer = float(
            (
                head_flip2(newborn_states(subj, test["ticks"], extra_test)).argmax(-1)
                == test["flip_labels"]
            )
            .float()
            .mean()
        )
    # pre-registered secondary transfer arm: 16 interactions, fresh head
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17703)
        head_flip16 = PhaseHead(2 * (96 - OLD_N), 8, 64)
    opt_f16 = torch.optim.Adam(head_flip16.parameters(), lr=6e-3)
    for i in range(16):
        batch = revision_split(BATCH, SEED * 7100 + 6000 + i)
        b_extra = batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        logits = head_flip16(newborn_states(subj, batch["ticks"], b_extra))
        loss = F.cross_entropy(logits, batch["flip_labels"])
        opt_f16.zero_grad(set_to_none=True)
        loss.backward()
        opt_f16.step()
    subj.eval_mode()
    with torch.no_grad():
        flip_transfer16 = float(
            (
                head_flip16(newborn_states(subj, test["ticks"], extra_test)).argmax(-1)
                == test["flip_labels"]
            )
            .float()
            .mean()
        )
    with torch.no_grad():
        units.extra_proj.weight[OLD_N:] = 0.0
    head_flip3 = PhaseHead(2 * (96 - OLD_N), 8, 64)
    opt_f3 = torch.optim.Adam(head_flip3.parameters(), lr=6e-3)
    for i in range(8):
        batch = revision_split(BATCH, SEED * 7100 + 5100 + i)
        b_extra = batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
        logits = head_flip3(newborn_states(subj, batch["ticks"], b_extra))
        loss = F.cross_entropy(logits, batch["flip_labels"])
        opt_f3.zero_grad(set_to_none=True)
        loss.backward()
        opt_f3.step()
    subj.eval_mode()
    with torch.no_grad():
        flip_reset = float(
            (
                head_flip3(newborn_states(subj, test["ticks"], extra_test)).argmax(-1)
                == test["flip_labels"]
            )
            .float()
            .mean()
        )
    normal_after = cn_eval(subj)
    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_pressure": cn_probe["test"] <= 0.55 and birth_probe["test"] <= 0.55,
        "G2_detect": detect_acc_at_phase >= 0.90,
        "G3_flip": flip_acc >= 0.75,
        "G4_flip_transfer": flip_transfer >= 0.75,
        "G4b_flip_transfer16": flip_transfer16 >= 0.75,
        "G5_adapter_drop": (flip_acc - flip_reset) >= 0.15,
        "G6_retention": (normal_cn - normal_after) <= 0.01,
    }
    if not gates["G0_old_cn"]:
        verdict = "p0ai_cn_base_failed"
    elif not gates["G1_pressure"]:
        verdict = "p0ai_not_pressured"
    elif not gates["G2_detect"]:
        verdict = "p0ai_detect_failed"

    elif not gates["G3_flip"]:
        verdict = "p0ai_flip_not_learnable"
    elif not gates["G4_flip_transfer"]:
        verdict = "p0ai_flip_no_transfer"
    elif not gates["G5_adapter_drop"]:
        verdict = "p0ai_adapter_not_needed"
    elif not gates["G6_retention"]:
        verdict = "p0ai_retention_breach"
    else:
        verdict = "p0ai_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0ai-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "revision_flip_consolidation",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "cn_probe": cn_probe,
        "birth_probe": birth_probe,
        "detect_acc": detect_acc_at_phase,
        "detect_acc_stale": (
            float(
                (head(newborn_states(subj, test["ticks"], extra_test)).argmax(-1) == test["labels"])
                .float()
                .mean()
            )
            if DET_STEPS > 0
            else 0.0
        ),
        "flip_acc": flip_acc,
        "flip_transfer": flip_transfer,
        "linear_probe": linear_probe,
        "flip_transfer16": flip_transfer16,
        "flip_reset": flip_reset,
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": cost},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
