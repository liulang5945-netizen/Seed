"""LC-1: unit lifecycle -- reversible dormancy/wake, retention-verified retirement and
inheritance-based recycling of retired units.

Subject: Subject72 seed 20261120 with p24/p27 blocks isolated (each block reads only its
own units plus inputs), per-task 48-dim heads, wd=0 (Adam L2 must not leak through
masked zero-gradient rows).  p24 500 steps, p27 600 steps.

Lifecycle:
  * dormant p27 block -> p27 lost / p24 unchanged; dormant half of p24 block -> dose drop;
  * wake -> accuracies restored bitwise;
  * retire p27 block -> p24 retention + capsule verified, retired count = 24;
  * recycle retired block by inheriting the p24 block (budding-style 1:1 by group and
    position) + unit_generation += 1 + immature -> revision-source task (8 interactions)
    -> mature; new-task causality via recycled-block dormancy; p24 retention bitwise.

Single seed, <=400 CPU, zero check/final. Standing authorization; owner may revise.
"""

from __future__ import annotations

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
from scripts.training.verify_taiji_24s_f1_20261122 import make_revision_stream
from scripts.training.verify_taiji_24s_f1c_20261122 import capsule_state_diff
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import SELECTIVE_LIFECYCLE

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_lc1_criteria_20261123.json"
OUT = ROOT / "reports/taiji_u2_lc1_20261123.json"
SEED = 20261120
FIT = 256
FIT_OFFSET = 10_000_000
BLOCK = {"p24": (24, 48), "p27": (48, 72)}
RECYCLED = list(range(48, 72))
REV_MASK_UNITS = list(range(24, 48)) + RECYCLED
LIFECYCLE_ID = {name: i for i, name in enumerate(SELECTIVE_LIFECYCLE)}


# ---------------------------------------------------------------------------
# subject setup / task readouts
# ---------------------------------------------------------------------------
def own_logits(subj, task: str, ticks: torch.Tensor, mask_units=None) -> torch.Tensor:
    if mask_units is None:
        mask = subj.masks[task]
    else:
        mask = torch.zeros(subj.n_units)
        mask[mask_units] = 1.0
    gates = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], subj.n_units)
    out = subj.graph(
        _channels_seq(ticks), gates, subj.graph.initial_state(batch_size=ticks.shape[0])
    )
    lo, hi = BLOCK.get(task, (48, 72))
    z, a = out["next_state"]["z"][:, lo:hi], out["next_state"]["a"][:, lo:hi]
    head_name = f"{task}u" if task in BLOCK else "rev"
    return subj.phase_heads[head_name](torch.cat([z, a], dim=-1))


def acc(subj, task: str, data, label_key: str, mask_units=None) -> float:
    subj.eval_mode()
    with torch.no_grad():
        pred = own_logits(subj, task, data["ticks"], mask_units).argmax(-1)
    return float((pred == data[label_key]).float().mean())


def build_subject(crit: dict) -> Subject72:
    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 401)
        subj.phase_heads["p24u"] = PhaseHead(48, 4, 64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 402)
        subj.phase_heads["p27u"] = PhaseHead(48, 4, 64)
    for p in subj.phase_heads["p24u"].parameters():
        p.requires_grad_(True)
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        units = subj.graph.selective_units
        for rows, other in (
            (slice(24, 48), (slice(0, 24), slice(48, 72))),
            (slice(48, 72), (slice(0, 24), slice(24, 48))),
        ):
            for cols in other:
                units.w[rows, cols] = 0.0
                units.write_state.weight[rows, cols] = 0.0
    lr, wd = crit["protocol"]["lr"], crit["protocol"]["wd"]
    opt24 = torch.optim.Adam(subj.params(), lr=lr, weight_decay=wd)
    for k in range(crit["protocol"]["p24_steps"]):
        b = _make_p24_stream(crit["protocol"]["batch"], SEED * 25000 + k)
        loss = F.cross_entropy(own_logits(subj, "p24", b["ticks"]), b["val_labels"])
        opt24.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "p24")
        torch.nn.utils.clip_grad_norm_([p for g in opt24.param_groups for p in g["params"]], 1.0)
        opt24.step()
    for p in subj.graph.cross_channel_mix.parameters():
        p.requires_grad_(False)
    for p in subj.phase_heads["p24u"].parameters():
        p.requires_grad_(False)
    for p in subj.phase_heads["p27u"].parameters():
        p.requires_grad_(True)
    opt27 = torch.optim.Adam(subj.params(), lr=lr, weight_decay=wd)
    for k in range(crit["protocol"]["p27_steps"]):
        b = _make_p27_physics_batch(crit["protocol"]["batch"], SEED * 41000 + k)
        loss = F.cross_entropy(own_logits(subj, "p27", b["ticks"]), b["labels"])
        opt27.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "p27")
        torch.nn.utils.clip_grad_norm_([p for g in opt27.param_groups for p in g["params"]], 1.0)
        opt27.step()
    subj.eval_mode()
    return subj


def lifecycle_counts(subj) -> dict:
    values = [int(x) for x in subj.graph.selective_units.unit_lifecycle.tolist()]
    return {name: values.count(idx) for name, idx in LIFECYCLE_ID.items()}


def set_lifecycle(subj, indices, state: str) -> None:
    subj.graph.selective_units.set_unit_lifecycle_(indices, state)


def inherit_p24_into_recycled(subj) -> float:
    """Re-differentiate the retired block by inheriting the p24 block (budding-style)."""
    units = subj.graph.selective_units
    newn = units.n_units
    new = list(RECYCLED)
    src = list(range(24, 48))
    with torch.no_grad():
        gi = units.group_index
        gi[new] = (torch.arange(24) // 6) + 4
        ar = torch.arange(12).unsqueeze(1)
        units.group_mask.copy_((gi.unsqueeze(0) == ar).to(units.group_mask.dtype))
        if units.spec.cross_group_edges:
            edge = 1.0 - torch.eye(newn, dtype=units.edge_mask.dtype)
        else:
            same = (gi.unsqueeze(1) == gi.unsqueeze(0)).to(units.edge_mask.dtype)
            edge = same * (1.0 - torch.eye(newn, dtype=units.edge_mask.dtype))
        units.edge_mask.copy_(edge)
        units.r[new] = units.r[src].clone()
        units.p[:, new] = units.p[:, src].clone()
        for name in ("c", "eta", "b", "write_bias"):
            t = getattr(units, name)
            t[new] = t[src].clone()
        units.w[new, :] = units.w[src, :].clone()
        units.write_in.weight[new, :] = units.write_in.weight[src, :].clone()
        units.write_state.weight[new, :] = units.write_state.weight[src, :].clone()
        units.unit_generation[torch.tensor(new)] += 1


def _states(subj, ticks, mask_units):
    mask = torch.zeros(subj.n_units)
    if mask_units is None:
        mask[REV_MASK_UNITS] = 1.0
    else:
        mask[mask_units] = 1.0
    gates = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], subj.n_units)
    out = subj.graph(
        _channels_seq(ticks), gates, subj.graph.initial_state(batch_size=ticks.shape[0])
    )
    return out["next_state"]["z"], out["next_state"]["a"]


def main() -> int:
    started = time.process_time()
    crit = json.loads(CRITERIA.read_text(encoding="utf-8"))
    body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
    digest = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == crit["criteria_sha256"], "criteria_sha256_mismatch"
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    subj = build_subject(crit)
    p24_test = _make_p24_stream(FIT, SEED + crit["protocol"]["p24_test_seed_offset"] + FIT_OFFSET)
    p27_test = _make_p27_physics_batch(
        FIT, SEED + crit["protocol"]["p27_test_seed_offset"] + FIT_OFFSET
    )
    rev_test = make_revision_stream(FIT, SEED + 444 + FIT_OFFSET)
    a0 = acc(subj, "p24", p24_test, "val_labels")
    b0 = acc(subj, "p27", p27_test, "labels")
    cap_base = capsule_state_diff(subj, p24_test["ticks"][:32], torch.ones(subj.n_units))
    gen_before = subj.graph.selective_units.unit_generation.clone()
    stages = {
        "base": {"a0": a0, "b0": b0, "capsule": cap_base, "lifecycle": lifecycle_counts(subj)}
    }
    gate0 = a0 >= crit["gates"]["G0_p24_acc_min"] and b0 >= crit["gates"]["G0_p27_acc_min"]
    payload = {
        "schema": "taiji-u2-lc1-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "base": {"p24_acc": a0, "p27_acc": b0, "capsule_diff": cap_base},
    }

    if not gate0:
        payload.update(
            {
                "gates": {"G0": False},
                "verdict": "lc1_baseline_insufficient",
                "cost": {"cpu": round(time.process_time() - started, 2)},
                "discipline": {"check_reads": 0, "final_reads": 0},
            }
        )
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"verdict": payload["verdict"]}, ensure_ascii=False))
        return 0

    # ---------- 1. dormancy / wake ----------
    set_lifecycle(subj, RECYCLED, "dormant")
    b_dorm = acc(subj, "p27", p27_test, "labels")
    a_keep = acc(subj, "p24", p24_test, "val_labels")
    dorm_counts = lifecycle_counts(subj)
    set_lifecycle(subj, RECYCLED, "mature")
    b_wake = acc(subj, "p27", p27_test, "labels")
    a_wake = acc(subj, "p24", p24_test, "val_labels")
    cap_wake = capsule_state_diff(subj, p24_test["ticks"][:32], torch.ones(subj.n_units))

    partial = crit["protocol"]["p24_dormant_units"]
    set_lifecycle(subj, list(range(partial[0], partial[1])), "dormant")
    a_part = acc(subj, "p24", p24_test, "val_labels")
    b_keep = acc(subj, "p27", p27_test, "labels")
    set_lifecycle(subj, list(range(partial[0], partial[1])), "mature")
    a_wake2 = acc(subj, "p24", p24_test, "val_labels")
    stages["dormancy"] = {
        "p27_dormant_acc": b_dorm,
        "p24_after_p27_dormant": a_keep,
        "p27_after_wake": b_wake,
        "p24_after_wake": a_wake,
        "p24_half_dormant_acc": a_part,
        "p27_during_p24_half_dormant": b_keep,
        "p24_after_second_wake": a_wake2,
        "capsule_after_wake": cap_wake,
        "lifecycle_dormant": dorm_counts,
    }

    # ---------- 2. retirement with retention verification ----------
    set_lifecycle(subj, RECYCLED, "retired")
    a_ret = acc(subj, "p24", p24_test, "val_labels")
    b_ret = acc(subj, "p27", p27_test, "labels")
    cap_ret = capsule_state_diff(subj, p24_test["ticks"][:32], torch.ones(subj.n_units))
    retired_counts = lifecycle_counts(subj)
    stages["retirement"] = {
        "p24_acc": a_ret,
        "p27_acc": b_ret,
        "capsule_diff": cap_ret,
        "lifecycle": retired_counts,
    }

    # ---------- 3. recycling: re-differentiate retired block by inheriting p24 ----------
    inherit_p24_into_recycled(subj)
    set_lifecycle(subj, RECYCLED, "immature")
    with torch.no_grad():
        z_rec, a_rec = _states(subj, rev_test["ticks"][:64], REV_MASK_UNITS)
        parent_equal = float(
            max(
                (z_rec[:, 48:72] - z_rec[:, 24:48]).abs().max(),
                (a_rec[:, 48:72] - a_rec[:, 24:48]).abs().max(),
            )
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 9001)
        subj.phase_heads["rev"] = PhaseHead(48, 4, 64)
    for p in subj.phase_heads["rev"].parameters():
        p.requires_grad_(True)
    subj.train_mode()
    head_opt = torch.optim.Adam(
        subj.phase_heads["rev"].parameters(), lr=crit["protocol"]["recycle_head_lr"]
    )
    graph_opt = torch.optim.Adam(
        [p for p in subj.graph.selective_units.parameters() if p.requires_grad],
        lr=crit["protocol"]["recycle_graph_lr"],
    )
    curve = []
    for i in range(crit["protocol"]["recycle_revision_interactions"]):
        b = make_revision_stream(crit["protocol"]["recycle_revision_batch"], SEED * 77000 + i)
        subj.phase_heads["rev"].train()
        logits = own_logits(subj, "rev", b["ticks"], REV_MASK_UNITS)
        loss = F.cross_entropy(logits, b["val_labels"])
        head_opt.zero_grad(set_to_none=True)
        graph_opt.zero_grad(set_to_none=True)
        loss.backward()
        for _name, p in subj.graph.selective_units.named_parameters():
            if p.grad is None:
                continue
            if p.shape[0] == subj.n_units:
                p.grad[:48] = 0.0
                p.grad[72:] = 0.0
            if p.ndim == 2 and p.shape[1] == subj.n_units:
                p.grad[:, :48] = 0.0
                p.grad[:, 72:] = 0.0
        torch.nn.utils.clip_grad_norm_(
            list(subj.phase_heads["rev"].parameters())
            + [p for g in graph_opt.param_groups for p in g["params"]],
            1.0,
        )
        head_opt.step()
        graph_opt.step()
        subj.eval_mode()
        with torch.no_grad():
            pred = own_logits(subj, "rev", rev_test["ticks"], REV_MASK_UNITS).argmax(-1)
        curve.append(float((pred == rev_test["val_labels"]).float().mean()))
        subj.train_mode()
    subj.eval_mode()
    rev_acc = curve[-1]
    # causality: dormant recycled block must remove the new task
    set_lifecycle(subj, RECYCLED, "dormant")
    rev_dorm = acc(subj, "rev", rev_test, "val_labels", REV_MASK_UNITS)
    set_lifecycle(subj, RECYCLED, "mature")
    rev_wake = acc(subj, "rev", rev_test, "val_labels", REV_MASK_UNITS)
    # promotion invariance for recycled units
    promo = subj.graph.selective_units
    with torch.no_grad():
        z_before, a_before = _states(subj, rev_test["ticks"][:32], REV_MASK_UNITS)
        promo.promote_immature_to_mature_()
        z_after, a_after = _states(subj, rev_test["ticks"][:32], REV_MASK_UNITS)
        promotion_diff = float(
            max((z_before - z_after).abs().max(), (a_before - a_after).abs().max())
        )
    a_rec = acc(subj, "p24", p24_test, "val_labels")
    cap_rec = capsule_state_diff(subj, rev_test["ticks"][:32], torch.ones(subj.n_units))
    gen_after = subj.graph.selective_units.unit_generation
    gen_increment = int((gen_after - gen_before).max())
    stages["recycle"] = {
        "curve": curve,
        "revision_acc": rev_acc,
        "revision_acc_dormant": rev_dorm,
        "revision_acc_after_wake": rev_wake,
        "parent_state_equality": parent_equal,
        "p24_retention_acc": a_rec,
        "generation_increment": gen_increment,
        "lifecycle": lifecycle_counts(subj),
        "promotion_forward_diff": promotion_diff,
        "capsule_diff": cap_rec,
    }

    p24_drop = a0 - a_rec
    gates = {
        "G0": bool(gate0),
        "G1": b_dorm <= crit["gates"]["G1_dormant_p27_acc_max"]
        and (a0 - a_keep) <= crit["gates"]["G1_p24_drop_max"]
        and (a0 - a_part) >= crit["gates"]["G1_partial_p24_drop_min"],
        "G2": abs(b_wake - b0) <= crit["gates"]["G2_equality_tol"]
        and abs(a_wake - a0) <= crit["gates"]["G2_equality_tol"]
        and abs(a_wake2 - a0) <= crit["gates"]["G2_equality_tol"]
        and cap_wake <= crit["gates"]["G5_capsule_diff_max"]
        and promotion_diff <= crit["gates"]["G5_promotion_diff_max"],
        "G3": (a0 - a_ret) <= crit["gates"]["G3_p24_drop_max"]
        and b_ret <= crit["gates"]["G3_dormant_p27_acc_max"]
        and cap_ret <= crit["gates"]["G5_capsule_diff_max"]
        and retired_counts.get("retired", 0) == 24,
        "G4": rev_acc >= crit["gates"]["G4_revision_acc_min"]
        and p24_drop <= crit["gates"]["G4_p24_drop_max"]
        and gen_increment == crit["gates"]["G4_generation_increment"]
        and cap_rec <= crit["gates"]["G5_capsule_diff_max"]
        and rev_dorm <= crit["gates"]["G4_recycled_dormant_acc_max"]
        and promotion_diff <= crit["gates"]["G5_promotion_diff_max"],
    }
    if not gates["G0"]:
        verdict = "lc1_baseline_insufficient"
    elif not gates["G2"]:
        verdict = "lc1_dormancy_irreversible"
    elif not gates["G3"]:
        verdict = "lc1_retirement_breach"
    elif not gates["G4"]:
        verdict = "lc1_recycle_partial"
    elif not gates["G1"]:
        verdict = "lc1_dormancy_weak"
    elif all(gates.values()):
        verdict = "lc1_verified"
    else:
        verdict = "lc1_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload.update(
        {
            "stages": stages,
            "gates": gates,
            "verdict": verdict,
            "verdict_inputs": {
                "a0": a0,
                "b0": b0,
                "b_dorm": b_dorm,
                "a_keep": a_keep,
                "a_part": a_part,
                "b_wake": b_wake,
                "a_wake": a_wake,
                "a_ret": a_ret,
                "b_ret": b_ret,
                "rev_acc": rev_acc,
                "rev_dorm": rev_dorm,
                "p24_drop": p24_drop,
                "gen_increment": gen_increment,
            },
            "run_conditions": {"threads": torch.get_num_threads()},
            "discipline": {"check_reads": 0, "final_reads": 0},
            "cost": {"cpu": cost},
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "verdict_inputs": payload["verdict_inputs"],
                "recycle": {
                    "curve": [round(x, 4) for x in curve],
                    "parent_state_equality": round(parent_equal, 6),
                    "rev_dorm": round(rev_dorm, 4),
                    "promotion_forward_diff": promotion_diff,
                    "capsule": cap_rec,
                },
                "cost": cost,
                "criteria_sha256": crit["criteria_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
