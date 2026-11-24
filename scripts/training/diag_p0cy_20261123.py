"""P0-CT: grow first, then train two rule-specific cohorts in parallel.

CN base -> two growths (72->96->120) before any rule training; cohort A (72:96) is read by
head_r0 for the original rule R0, cohort B (96:120) by head_r1 for the flipped rule R1.
Joint supervised training, then lesion dissociation checks.  Exploratory; zero check/final.
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

from scripts.training.diag_p0ci_20261123 import (  # type: ignore
    BI_CN_CHAR_INDEX,
    BI_CN_PROJ32,
    CN_COLOURS,
    CN_MAX_LEN,
    FLAG_CH,
    HOLDOUT4,
    OLD_N,
    VIS_CH,
    PhaseHead,
    _build_bidirectional_cn_32dim,
    cn_eval,
    extra_of,
    mask_fn,
    mask_newborn_grads,
    multimodal_batch,
    newborn_params,
    run_states,
)
from scripts.training.verify_taiji_23s2a_gain_20261122 import BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261120
OUT = ROOT / f"reports/taiji_p0cy_seed{SEED}.json"
TRAIN_STEPS = 3000


def new_rule_batch(count: int, seed: int) -> dict:
    tpls = [(tpl, True) for tpl in BI_CN_TRAIN_FIRST] + [(tpl, False) for tpl in BI_CN_TRAIN_SECOND]
    rows, labels = [], []
    f_rng = torch.Generator().manual_seed(seed + 555_555)
    v_rng = torch.Generator().manual_seed(seed + 444_444)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, 4, (1,), generator=gen))
        c2 = int(torch.randint(0, 3, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = (c1, c2) not in HOLDOUT4
        idx += 1
        if not ok:
            continue
        tpl, pick_first = tpls[len(rows) % len(tpls)]
        other = c2 if pick_first else c1
        flag = int(torch.randint(0, 4, (1,), generator=f_rng))
        shape = int(torch.randint(0, 4, (1,), generator=v_rng))
        by_shape = c2 if shape < 2 else c1  # flipped rule
        answer = other if flag >= 2 else by_shape
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        seq[0, VIS_CH[0] : VIS_CH[1]] = 0.0
        seq[0, VIS_CH[0] + shape] = 1.0
        rows.append(seq)
        labels.append(answer)
    return {"ticks": torch.stack(rows), "labels": torch.tensor(labels, dtype=torch.long)}


def new_rule3_batch(count: int, seed: int) -> dict:
    """Third rule: flag convention inverted (other colour when flag < 2)."""
    tpls = [(tpl, True) for tpl in BI_CN_TRAIN_FIRST] + [(tpl, False) for tpl in BI_CN_TRAIN_SECOND]
    rows, labels = [], []
    f_rng = torch.Generator().manual_seed(seed + 555_555)
    v_rng = torch.Generator().manual_seed(seed + 444_444)
    idx = 0
    while len(rows) < count and idx < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, 4, (1,), generator=gen))
        c2 = int(torch.randint(0, 3, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        ok = (c1, c2) not in HOLDOUT4
        idx += 1
        if not ok:
            continue
        tpl, pick_first = tpls[len(rows) % len(tpls)]
        other = c2 if pick_first else c1
        flag = int(torch.randint(0, 4, (1,), generator=f_rng))
        shape = int(torch.randint(0, 4, (1,), generator=v_rng))
        by_shape = c1 if shape < 2 else c2
        answer = other if flag < 2 else by_shape
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        seq[0, FLAG_CH[0] : FLAG_CH[1]] = 0.0
        seq[0, FLAG_CH[0] + flag] = 1.0
        seq[0, VIS_CH[0] : VIS_CH[1]] = 0.0
        seq[0, VIS_CH[0] + shape] = 1.0
        rows.append(seq)
        labels.append(answer)
    return {"ticks": torch.stack(rows), "labels": torch.tensor(labels, dtype=torch.long)}


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=1e-4)
    subj.train_mode()
    for step in range(1200):
        batch = _build_bidirectional_cn_32dim(
            BATCH_SIZE, SEED * 6000 + step, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
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

    for _ in range(3):
        subj.graph.grow_units_neutral_(2)
    subj.n_units = 144
    units = subj.graph.selective_units
    units.attach_extra_(8, seed=SEED)
    with torch.no_grad():
        src = torch.arange(24)
        for base in (72, 96, 120):
            dst = torch.arange(24) + base
            units.group_index[dst] = torch.arange(24) // 6
            units.r[dst] = units.r[src].clone()
            units.p[:, dst] = units.p[:, src].clone()
            for nm in ("c", "eta", "b", "write_bias"):
                getattr(units, nm)[dst] = getattr(units, nm)[src].clone()
            units.w[dst, :] = units.w[src, :].clone()
            units.write_in.weight[dst, :] = units.write_in.weight[src, :].clone()
            units.write_state.weight[dst, :] = units.write_state.weight[src, :].clone()
            units.write_bias[dst] = 0.0
            units.b[dst] = 0.0

    def feat(subj_, ticks_, extra_, lo_: int, hi_: int) -> torch.Tensor:
        zz, aa = run_states(
            subj_, ticks_, mask_fn(subj_.n_units, OLD_N, subj_.n_units), extra=extra_
        )
        return torch.cat([zz[:, lo_:hi_], aa[:, lo_:hi_]], dim=-1)

    def acc_cohort(subj_, head_, batch_, lo_: int, hi_: int) -> float:
        head_.eval()
        subj_.eval_mode()
        with torch.no_grad():
            pred = head_(feat(subj_, batch_["ticks"], extra_of(batch_), lo_, hi_)).argmax(-1)
        return float((pred == batch_["labels"]).float().mean())

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 18601)
        head_r0 = PhaseHead(2 * (96 - 72), 4, 64)
        head_r1 = PhaseHead(2 * (120 - 96), 4, 64)
        head_r2 = PhaseHead(2 * (144 - 120), 4, 64)
    params = (
        list(head_r0.parameters())
        + list(head_r1.parameters())
        + list(head_r2.parameters())
        + list(newborn_params(subj))
    )
    opt2 = torch.optim.Adam(params, lr=8e-3)
    for i in range(TRAIN_STEPS):
        b0 = multimodal_batch(24, SEED * 7000 + i)
        b1 = new_rule_batch(24, SEED * 7100 + i)
        subj.train_mode()
        loss = F.cross_entropy(
            head_r0(feat(subj, b0["ticks"], extra_of(b0), 72, 96)), b0["labels"]
        ) + F.cross_entropy(head_r1(feat(subj, b1["ticks"], extra_of(b1), 96, 120)), b1["labels"])
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, 72)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt2.step()

    # stage 2: train R2 alone on cohort C, gradients limited to rows >= 120
    opt_r2 = torch.optim.Adam(
        list(head_r2.parameters()) + list(subj.graph.selective_units.parameters()), lr=6e-3
    )
    for i in range(2400):
        b2 = new_rule3_batch(32, SEED * 7300 + i)
        subj.train_mode()
        loss = F.cross_entropy(
            head_r2(feat(subj, b2["ticks"], extra_of(b2), 120, 144)), b2["labels"]
        )
        opt_r2.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, 120)
        torch.nn.utils.clip_grad_norm_(
            list(head_r2.parameters()) + list(subj.graph.selective_units.parameters()), 1.0
        )
        opt_r2.step()

    eval_r0 = multimodal_batch(1024, SEED + 2100 + 10_000_000)
    eval_r1 = new_rule_batch(1024, SEED + 2200 + 10_000_000)
    eval_r2 = new_rule3_batch(1024, SEED + 2250 + 10_000_000)
    r0 = acc_cohort(subj, head_r0, eval_r0, 72, 96)
    r1 = acc_cohort(subj, head_r1, eval_r1, 96, 120)
    r2 = acc_cohort(subj, head_r2, eval_r2, 120, 144)
    r0_staged = acc_cohort(subj, head_r0, eval_r0, 72, 96)
    r1_staged = acc_cohort(subj, head_r1, eval_r1, 96, 120)
    subj.graph.selective_units.set_unit_lifecycle_(range(72, 96), "dormant")
    r0_lesA = acc_cohort(subj, head_r0, eval_r0, 72, 96)
    r1_lesA = acc_cohort(subj, head_r1, eval_r1, 96, 120)
    r2_lesA = acc_cohort(subj, head_r2, eval_r2, 120, 144)
    subj.graph.selective_units.set_unit_lifecycle_(range(72, 96), "mature")
    subj.graph.selective_units.set_unit_lifecycle_(range(96, 120), "dormant")
    r0_lesB = acc_cohort(subj, head_r0, eval_r0, 72, 96)
    r1_lesB = acc_cohort(subj, head_r1, eval_r1, 96, 120)
    r2_lesB = acc_cohort(subj, head_r2, eval_r2, 120, 144)
    subj.graph.selective_units.set_unit_lifecycle_(range(96, 120), "mature")
    subj.graph.selective_units.set_unit_lifecycle_(range(120, 144), "dormant")
    r0_lesC = acc_cohort(subj, head_r0, eval_r0, 72, 96)
    r1_lesC = acc_cohort(subj, head_r1, eval_r1, 96, 120)
    r2_lesC = acc_cohort(subj, head_r2, eval_r2, 120, 144)
    subj.graph.selective_units.set_unit_lifecycle_(range(120, 144), "mature")
    from taiji.u2_selective_units import (
        SelectivePersistentGraph,
        apply_capsule_v2,
        build_capsule_v2,
    )

    cap = build_capsule_v2(subj.graph)
    fresh = SelectivePersistentGraph(subj.graph.base_graph, subj.graph.selective_spec)
    fresh.freeze_base_()
    fresh.selective_units.attach_extra_(8, seed=SEED)
    apply_capsule_v2(fresh, cap)

    class _Holder:
        def __init__(self, graph_, num_units_):
            self.graph = graph_
            self.n_units = num_units_

        def eval_mode(self):
            pass

        def train_mode(self):
            pass

    holder = _Holder(fresh, subj.n_units)
    r0_restore = acc_cohort(holder, head_r0, eval_r0, 72, 96)
    r1_restore = acc_cohort(holder, head_r1, eval_r1, 96, 120)
    r2_restore = acc_cohort(holder, head_r2, eval_r2, 120, 144)
    with torch.no_grad():
        z0, a0 = run_states(
            subj,
            eval_r0["ticks"],
            mask_fn(subj.n_units, OLD_N, subj.n_units),
            extra=extra_of(eval_r0),
        )
        z1, a1 = run_states(
            holder,
            eval_r0["ticks"],
            mask_fn(holder.n_units, OLD_N, holder.n_units),
            extra=extra_of(eval_r0),
        )
    cap_diff = float(max((z0 - z1).abs().max(), (a0 - a1).abs().max()))
    normal_after = cn_eval(subj)

    gates = {
        "G0_old_cn": normal_cn >= 0.85,
        "G1_r0": r0 >= 0.90,
        "G2_r1": r1 >= 0.90,
        "G2b_r2": r2 >= 0.90,
        "G2c_staged_retention": r0_staged >= 0.95 and r1_staged >= 0.95,
        "G3_lesA_r0": (r0 - r0_lesA) >= 0.15,
        "G4_lesB_r1": (r1 - r1_lesB) >= 0.15,
        "G5_normal_retained": (normal_cn - normal_after) <= 0.01,
        "G6_capsule": cap_diff <= 1e-6
        and abs(r0_restore - r0) <= 1e-9
        and abs(r1_restore - r1) <= 1e-9
        and abs(r2_restore - r2) <= 1e-9,
        "G7_lesC_r2": (r2 - r2_lesC) >= 0.15,
    }
    if not (gates["G1_r0"] and gates["G2_r1"] and gates["G2b_r2"]):
        verdict = "p0cy_parallel_rules_failed"
    elif not (gates["G3_lesA_r0"] and gates["G4_lesB_r1"]):
        verdict = "p0cy_dissociation_failed"
    elif not gates["G0_old_cn"] or not gates["G5_normal_retained"]:
        verdict = "p0cy_retention_failed"
    else:
        verdict = "p0cy_verified"
    payload = {
        "schema": "taiji-p0cy-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "staged_r2_strengthened",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "r0_staged": r0_staged,
        "r1_staged": r1_staged,
        "lesA": {"r0": r0_lesA, "r1": r1_lesA, "r2": r2_lesA},
        "lesB": {"r0": r0_lesB, "r1": r1_lesB, "r2": r2_lesB},
        "lesC": {"r0": r0_lesC, "r1": r1_lesC, "r2": r2_lesC},
        "capsule": {
            "r0_restore": r0_restore,
            "r1_restore": r1_restore,
            "r2_restore": r2_restore,
            "cap_diff": cap_diff,
        },
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


BATCH_SIZE = 48
if __name__ == "__main__":
    raise SystemExit(main())
