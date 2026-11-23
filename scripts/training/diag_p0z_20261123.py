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
from taiji.u2_selective_units import SelectivePersistentGraph, apply_capsule_v2, build_capsule_v2

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0z_diag_20261123.json"
SEED = 20261120
OLD_N = 72
CN_STEPS = 1200
REV_STEPS = 500
BATCH = 48
TARGET = 0.90
FLAG_CH = (32, 36)


class _G:
    """Minimal subject-like wrapper around a bare graph for the shared helpers."""

    def __init__(self, graph, n_units: int) -> None:
        self.graph = graph
        self.n_units = n_units


def revision_split(count: int, seed: int) -> dict:
    all_tpls = [(t, True) for t in BI_CN_TRAIN_FIRST] + [(t, False) for t in BI_CN_TRAIN_SECOND]
    rows, flags = [], []
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
        fgen = torch.Generator().manual_seed(seed + 77_777 + idx)
        flag = int(torch.randint(0, 4, (1,), generator=fgen))
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        rows.append(seq)
        flags.append(1 if flag >= 2 else 0)
    return {
        "ticks": torch.stack(rows),
        "labels": torch.tensor(flags, dtype=torch.long),
        "flags": torch.tensor(flags, dtype=torch.long),
    }


def run_states(subj, ticks, mask, extra=None, step_mask=None):
    graph = getattr(subj, "graph", subj)
    gates = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], mask.numel())
    out = graph(
        _channels_seq(ticks),
        gates,
        graph.initial_state(batch_size=ticks.shape[0]),
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
    graph = getattr(subj, "graph", subj)
    n = getattr(subj, "n_units", None) or graph.selective_units.n_units
    z, a = run_states(subj, ticks, mask_fn(n, OLD_N, n), extra=extra)
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
    for i in range(REV_STEPS):
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
        rev_acc = float(
            (head(newborn_states(subj, test["ticks"], extra_test)).argmax(-1) == test["labels"])
            .float()
            .mean()
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17602)
        head2 = PhaseHead(2 * (96 - OLD_N), 2, 64)
    for p in newborn_params(subj):
        p.requires_grad_(False)
    opt3 = torch.optim.Adam(head2.parameters(), lr=6e-3)
    for i in range(8):
        batch = revision_split(BATCH, SEED * 7000 + 5000 + i)
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
    head3 = PhaseHead(2 * (96 - OLD_N), 2, 64)
    opt4 = torch.optim.Adam(head3.parameters(), lr=6e-3)
    for i in range(8):
        batch = revision_split(BATCH, SEED * 7000 + 5100 + i)
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
    # lifecycle: dormancy must disable the detection, wake must restore it exactly
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "dormant")
    subj.eval_mode()
    with torch.no_grad():
        dormant_acc = float(
            (head(newborn_states(subj, test["ticks"], extra_test)).argmax(-1) == test["labels"])
            .float()
            .mean()
        )
    subj.graph.selective_units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "mature")
    subj.eval_mode()
    with torch.no_grad():
        wake_acc = float(
            (head(newborn_states(subj, test["ticks"], extra_test)).argmax(-1) == test["labels"])
            .float()
            .mean()
        )
    # capsule round trip
    cap = build_capsule_v2(subj.graph)
    fresh = SelectivePersistentGraph(subj.graph.base_graph, subj.graph.selective_spec)
    fresh.freeze_base_()
    fresh.selective_units.attach_extra_(4, seed=SEED)
    apply_capsule_v2(fresh, cap)
    with torch.no_grad():
        z0, a0 = run_states(
            subj, test["ticks"], mask_fn(subj.n_units, OLD_N, subj.n_units), extra=extra_test
        )
        z1, a1 = run_states(
            fresh, test["ticks"], mask_fn(subj.n_units, OLD_N, subj.n_units), extra=extra_test
        )
    capsule_diff = float(max((z0 - z1).abs().max(), (a0 - a1).abs().max()))
    with torch.no_grad():
        capsule_acc = float(
            (
                head(newborn_states(_G(fresh, subj.n_units), test["ticks"], extra_test)).argmax(-1)
                == test["labels"]
            )
            .float()
            .mean()
        )
    normal_after = cn_eval(subj)

    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_pressure": cn_probe["test"] <= 0.55 and birth_probe["test"] <= 0.55,
        "G2_detect": rev_acc >= TARGET,
        "G3_transfer": transfer >= TARGET,
        "G4_adapter_drop": (rev_acc - reset_acc) >= 0.15,
        "G5_retention": (normal_cn - normal_after) <= 0.01,
        "G6_dormant": dormant_acc <= 0.60,
        "G7_wake": abs(wake_acc - rev_acc) <= 1e-9,
        "G8_capsule": capsule_diff <= 1e-6 and abs(capsule_acc - rev_acc) <= 1e-9,
    }
    if not gates["G0_old_cn"]:
        verdict = "p0z_cn_base_failed"
    elif not gates["G1_pressure"]:
        verdict = "p0z_not_pressured"
    elif not gates["G2_detect"]:
        verdict = "p0z_not_learnable"
    elif not gates["G3_transfer"]:
        verdict = "p0z_no_transfer"
    elif not gates["G4_adapter_drop"]:
        verdict = "p0z_adapter_not_needed"
    elif not gates["G5_retention"]:
        verdict = "p0z_retention_breach"
    elif not gates["G6_dormant"]:
        verdict = "p0z_dormancy_ineffective"
    elif not gates["G7_wake"]:
        verdict = "p0z_wake_not_restored"
    elif not gates["G8_capsule"]:
        verdict = "p0z_capsule_mismatch"
    else:
        verdict = "p0z_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0z-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "revision_presence_binary",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "cn_probe": cn_probe,
        "birth_probe": birth_probe,
        "detect_acc": rev_acc,
        "transfer": transfer,
        "reset": reset_acc,
        "dormant": dormant_acc,
        "wake": wake_acc,
        "capsule_diff": capsule_diff,
        "capsule_acc": capsule_acc,
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
                    "detect_acc",
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
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
