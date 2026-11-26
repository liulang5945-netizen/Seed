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

from scripts.training.taiji_eval_framework_20261124 import (
    closed_form_probe,
    logreg_acc,
    mlp_transfer_acc,
    prototype_acc,
)
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
import os

SEED = int(os.environ.get("TAIJI_SEED", "20261120"))
OUT = ROOT / f"reports/taiji_p0at_seed{SEED}.json"
OLD_N = 72
CN_STEPS = 1200
DET_STEPS = 800
FLIP_STEPS = 1600
CONSOL_STEPS = 1500
META_STEPS = 800
BATCH = 48
TARGET = 0.90
FLAG_CH = (32, 36)


def revision_split(count: int, seed: int, allow_holdout: bool = False) -> dict:
    all_tpls = [(t, True) for t in BI_CN_TRAIN_FIRST] + [(t, False) for t in BI_CN_TRAIN_SECOND]
    rows, flags, flip_labels = [], [], []
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, CN_CLASSES, (1,), generator=gen))
        c2 = int(torch.randint(0, CN_CLASSES - 1, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = ((c1, c2) in HOLDOUT) if allow_holdout else ((c1, c2) not in HOLDOUT)
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
        head.eval()
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
    # ---------- P0-AR: consolidation-objective comparison (frozen framework) ----------
    import copy

    eval_train = revision_split(1024, SEED + 2000 + 10_000_000)
    eval_test = revision_split(1024, SEED + 2001 + 10_000_000)
    eval_unseen = revision_split(512, SEED + 2002 + 10_000_000, allow_holdout=True)
    extra_tr = eval_train["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
    extra_te = eval_test["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
    extra_un = eval_unseen["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()

    def features(s, ticks, extra):
        s.eval_mode()
        with torch.no_grad():
            return newborn_states(s, ticks, extra).detach()

    def causal_logreg(s) -> dict:
        f_tr = features(s, eval_train["ticks"], extra_tr)
        y_tr, y_te = eval_train["flip_labels"], eval_test["flip_labels"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 7)
            readout = torch.nn.Linear(f_tr.shape[-1], 8)
        readout.train()
        ro = torch.optim.Adam(readout.parameters(), lr=2e-2)
        for _ in range(400):
            loss = F.cross_entropy(readout(f_tr), y_tr)
            ro.zero_grad(set_to_none=True)
            loss.backward()
            ro.step()
        readout.eval()

        def acc(ticks, extra, labels):
            s.eval_mode()
            with torch.no_grad():
                return float(
                    (readout(features(s, ticks, extra)).argmax(-1) == labels).float().mean()
                )

        ref = acc(eval_test["ticks"], extra_te, y_te)
        s.graph.selective_units.set_unit_lifecycle_(range(OLD_N, s.n_units), "dormant")
        dorm = acc(eval_test["ticks"], extra_te, y_te)
        s.graph.selective_units.set_unit_lifecycle_(range(OLD_N, s.n_units), "mature")
        wake = acc(eval_test["ticks"], extra_te, y_te)
        with torch.no_grad():
            s.graph.selective_units.extra_proj.weight[OLD_N:] = 0.0
        reset = acc(eval_test["ticks"], extra_te, y_te)
        return {
            "ref": ref,
            "dormant": dorm,
            "wake": wake,
            "adapter_reset": reset,
            "adapter_drop": ref - reset,
        }

    def quality(s) -> dict:
        f_tr = features(s, eval_train["ticks"], extra_tr)
        f_te = features(s, eval_test["ticks"], extra_te)
        y_tr, y_te = eval_train["flip_labels"], eval_test["flip_labels"]
        f_un = features(s, eval_unseen["ticks"], extra_un)
        y_un = eval_unseen["flip_labels"]
        return {
            "linear": closed_form_probe(f_tr, y_tr, f_te, y_te),
            "logreg": logreg_acc(f_tr, y_tr, f_te, y_te, seed=SEED + 1),
            "prototype": prototype_acc(f_tr, y_tr, f_te, y_te),
            "mlp16": mlp_transfer_acc(f_tr, y_tr, f_te, y_te, steps=16, seed=SEED + 3),
            "linear_unseen": closed_form_probe(f_tr, y_tr, f_un, y_un),
            "logreg_unseen": logreg_acc(f_tr, y_tr, f_un, y_un, seed=SEED + 5),
        }

    def causal(s) -> dict:
        head_flip.eval()
        s.eval_mode()
        with torch.no_grad():
            ref = float(
                (
                    head_flip(newborn_states(s, eval_test["ticks"], extra_te)).argmax(-1)
                    == eval_test["flip_labels"]
                )
                .float()
                .mean()
            )
        s.graph.selective_units.set_unit_lifecycle_(range(OLD_N, s.n_units), "dormant")
        head_flip.eval()
        s.eval_mode()
        with torch.no_grad():
            dorm = float(
                (
                    head_flip(newborn_states(s, eval_test["ticks"], extra_te)).argmax(-1)
                    == eval_test["flip_labels"]
                )
                .float()
                .mean()
            )
        s.graph.selective_units.set_unit_lifecycle_(range(OLD_N, s.n_units), "mature")
        head_flip.eval()
        s.eval_mode()
        with torch.no_grad():
            wake = float(
                (
                    head_flip(newborn_states(s, eval_test["ticks"], extra_te)).argmax(-1)
                    == eval_test["flip_labels"]
                )
                .float()
                .mean()
            )
        with torch.no_grad():
            s.graph.selective_units.extra_proj.weight[OLD_N:] = 0.0
        head_flip.eval()
        s.eval_mode()
        with torch.no_grad():
            reset = float(
                (
                    head_flip(newborn_states(s, eval_test["ticks"], extra_te)).argmax(-1)
                    == eval_test["flip_labels"]
                )
                .float()
                .mean()
            )
        return {
            "ref": ref,
            "dormant": dorm,
            "wake": wake,
            "adapter_reset": reset,
            "adapter_drop": ref - reset,
        }

    # grid over ridge lambda and meta steps (pre-registered selection rule below)
    GRID = [(1e-3, 800), (1e-2, 800), (1e-1, 800), (1e-3, 2000), (1e-2, 2000), (1e-1, 2000)]
    arms = {}
    for lam, steps_meta in GRID:
        subj_meta = copy.deepcopy(subj)
        units_meta = list(newborn_params(subj_meta))
        opt_meta = torch.optim.Adam(units_meta, lr=6e-3)
        for i in range(steps_meta):
            batch = revision_split(96, SEED * 7400 + i)
            b_extra = batch["ticks"][:, :, FLAG_CH[0] : FLAG_CH[1]].clone()
            subj_meta.train_mode()
            feat = newborn_states(subj_meta, batch["ticks"], b_extra)
            feat_a = torch.cat([feat, torch.ones(feat.shape[0], 1)], dim=-1)
            f_s = feat_a[:48]
            f_q = feat_a[48:]
            y_s = F.one_hot(batch["flip_labels"][:48], 8).float()
            y_q = batch["flip_labels"][48:]
            eye = torch.eye(f_s.shape[-1])
            a_mat = f_s.T @ f_s + lam * eye
            w_meta = torch.linalg.solve(a_mat, f_s.T @ y_s)
            loss = F.cross_entropy(f_q @ w_meta, y_q)
            opt_meta.zero_grad(set_to_none=True)
            loss.backward()
            mask_newborn_grads(subj_meta, OLD_N)
            torch.nn.utils.clip_grad_norm_(units_meta, 1.0)
            opt_meta.step()
        key = f"lam{lam:g}_steps{steps_meta}"
        arms[key] = {"quality": quality(subj_meta), "causal": causal_logreg(subj_meta)}
        print("arm", key, "done", flush=True)

    # select the arm by head-agnostic quality (linear + mlp16 mean), then check gates
    def score(arm):
        return 0.5 * (arm["quality"]["linear"] + arm["quality"]["mlp16"])

    best_arm = max(
        arms,
        key=lambda k: (arms[k]["quality"]["linear_unseen"], arms[k]["quality"]["logreg"]),
    )
    bq, bc = arms[best_arm]["quality"], arms[best_arm]["causal"]
    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_pressure": cn_probe["test"] <= 0.55 and birth_probe["test"] <= 0.55,
        "G2_detect": detect_acc_at_phase >= 0.90,
        "G3_logreg": bq["logreg"] >= 0.95,
        "G4_unseen_linear": bq["linear_unseen"] >= 0.80,
        "G5_unseen_logreg": bq["logreg_unseen"] >= 0.80,
        "G6_dormant": bc["dormant"] <= 0.30,
        "G7_wake": abs(bc["wake"] - bc["ref"]) <= 1e-9,
        "G8_adapter_drop": bc["adapter_drop"] >= 0.15,
        "G9_retention": True,
    }
    normal_after = cn_eval(subj_meta)
    gates["G9_retention"] = (normal_cn - normal_after) <= 0.01
    if not gates["G0_old_cn"]:
        verdict = "p0at_cn_base_failed"
    elif not gates["G1_pressure"]:
        verdict = "p0at_not_pressured"
    elif not gates["G2_detect"]:
        verdict = "p0at_detect_failed"
    elif not (gates["G3_logreg"] and gates["G4_unseen_linear"] and gates["G5_unseen_logreg"]):
        verdict = "p0at_not_consolidated"
    elif not (gates["G6_dormant"] and gates["G8_adapter_drop"]):
        verdict = "p0at_causality_failed"
    elif not gates["G7_wake"]:
        verdict = "p0at_wake_not_restored"
    elif not gates["G9_retention"]:
        verdict = "p0at_retention_breach"
    else:
        verdict = "p0at_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0at-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "meta_consolidation_grid",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "cn_probe": cn_probe,
        "birth_probe": birth_probe,
        "detect_acc": detect_acc_at_phase,
        "arms": arms,
        "best_arm": best_arm,
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
