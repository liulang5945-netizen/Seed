"""24-S.F1b: retention-protected representation-level outdated correction.

Same candidate/protocol as 24-S.F1 (new revision source channel 14 on top of the
P24 5-session stream), but correction interactions are protected against forgetting
the four base regimes:
  * P_protected_replay: each interaction = 96 revision + 96 old-regime samples,
  * N_no_protection:   96 revision only (F1 arm A equivalent),
  * L_anchor_only:     96 revision only + L2 anchor to base parameters (lambda=1.0),
  * C_frozen:          no updates.

Primary gate at 8 interactions (F1 established that budget); 4-interaction reading is
reported but does not relax the F1 gate.  Single seed, <=300 CPU, zero check/final.
Standing authorization; owner may revise.
"""

from __future__ import annotations

import copy
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
from scripts.training.verify_taiji_24s_f1_20261122 import (
    BLOCKS,
    FIT,
    FIT_OFFSET,
    SEED,
    base_step,
    capsule_roundtrip,
    eval_base,
    eval_rev,
    make_revision_stream,
)
from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p24_stream
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_24s_f1b_criteria_20261122.json"
OUT = ROOT / "reports/taiji_u2_24s_f1b_20261122.json"


def set_arm_trainable(subj, mode: str) -> None:
    for p in subj.graph.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(False)
    train = mode in ("P_protected_replay", "N_no_protection", "L_anchor_only")
    for p in subj.phase_heads["p24"].parameters():
        p.requires_grad_(train)
    if train:
        for p in subj.graph.selective_units.parameters():
            p.requires_grad_(True)
        for p in subj.graph.read_gate.parameters():
            p.requires_grad_(True)
    subj.train_mode()


def arm_parameters(subj, mode: str) -> dict[str, torch.nn.Parameter]:
    params = {f"head.{n}": p for n, p in subj.phase_heads["p24"].named_parameters()}
    if mode in ("P_protected_replay", "N_no_protection", "L_anchor_only"):
        with torch.no_grad():
            for n, p in subj.graph.selective_units.named_parameters():
                params[f"units.{n}"] = p
            for n, p in subj.graph.read_gate.named_parameters():
                params[f"gate.{n}"] = p
    return params


def make_old_batch(count: int, seed: int) -> dict[str, torch.Tensor]:
    return _make_p24_stream(count, seed)


def interaction_batch(mode: str, i: int, crit: dict) -> dict[str, torch.Tensor]:
    rev = make_revision_stream(crit["protocol"]["revision_batch"], SEED * 77000 + i)
    if mode != "P_protected_replay":
        return {
            "ticks": rev["ticks"],
            "val_labels": rev["val_labels"],
            "conflict_labels": torch.zeros(rev["val_labels"].shape[0]),
        }
    old = make_old_batch(crit["protocol"]["replay_old_batch"], SEED * 88000 + i)
    return {
        "ticks": torch.cat([rev["ticks"], old["ticks"]], dim=0),
        "val_labels": torch.cat([rev["val_labels"], old["val_labels"]], dim=0),
        "conflict_labels": torch.cat(
            [torch.zeros(rev["val_labels"].shape[0]), old["conflict_labels"]], dim=0
        ),
    }


def run_arm(base_subj, mode: str, rev_probe, crit) -> dict:
    subj = copy.deepcopy(base_subj)
    set_arm_trainable(subj, mode)
    params = arm_parameters(subj, mode)
    anchor = {name: p.detach().clone() for name, p in params.items()}
    graph_train = [
        p for name, p in params.items() if name.startswith("units.") or name.startswith("gate.")
    ]
    head_train = [p for name, p in params.items() if name.startswith("head.")]
    opt = torch.optim.Adam(graph_train, lr=crit["protocol"]["graph_lr"]) if graph_train else None
    head_opt = torch.optim.Adam(head_train, lr=crit["protocol"]["head_lr"]) if head_train else None
    event = []
    for k in range(crit["protocol"]["event_batches"]):
        eb = make_revision_stream(crit["protocol"]["event_batch"], SEED * 66000 + k)
        event.append(eval_rev(subj, eb)["acc_new"])
    base_anchor = eval_base(subj)
    rev_curve = {key: value[: len(value) // 2] for key, value in rev_probe.items()}
    n_int = crit["protocol"]["correction_interactions"]
    curve = []
    for i in range(n_int):
        b = interaction_batch(mode, i, crit)
        full_probe = (i + 1) in (crit["protocol"]["interaction_eval_4"], n_int)
        if mode != "C_frozen":
            subj.graph.eval()
            subj.phase_heads["p24"].train()
            logits = subj.logits("p24", b["ticks"])
            loss = F.cross_entropy(
                logits[:, :4], b["val_labels"]
            ) + F.binary_cross_entropy_with_logits(logits[:, 4], b["conflict_labels"])
            if mode == "L_anchor_only":
                penalty = sum(((p - anchor[name]) ** 2).sum() for name, p in params.items())
                loss = loss + crit["protocol"]["anchor_lambda"] * penalty
            head_opt.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            mask_grads(subj, "p24")
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
            head_opt.step()
            opt.step()
        rev = eval_rev(subj, rev_probe if full_probe else rev_curve)
        row = {
            "interaction": i + 1,
            "acc_new": rev["acc_new"],
            "acc_stale": rev["acc_stale"],
        }
        if full_probe:
            base_now = eval_base(subj)
            row["base_overall"] = base_now["overall"]
            row["base_per_regime"] = base_now["per_regime"]
        curve.append(row)
    final_base = curve[-1].get("base_per_regime") or base_anchor["per_regime"]
    retention_drop = max(0.0, max(base_anchor["per_regime"][r] - final_base[r] for r in range(4)))
    cap_diff = capsule_roundtrip(subj, rev_probe["ticks"][:32])
    return {
        "mode": mode,
        "event_acc_new": event,
        "base_anchor": base_anchor,
        "curve": curve,
        "acc_at_4": curve[min(crit["protocol"]["interaction_eval_4"], len(curve)) - 1]["acc_new"],
        "acc_at_8": curve[-1]["acc_new"],
        "retention_drop": retention_drop,
        "capsule_diff": cap_diff,
    }


def main() -> int:
    started = time.process_time()
    crit = json.loads(CRITERIA.read_text(encoding="utf-8"))
    body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
    canon = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canon).hexdigest()
    assert digest == crit["criteria_sha256"], "criteria_sha256_mismatch"
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in BLOCKS.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(True)
    subj.train_mode()
    opt = torch.optim.Adam(
        subj.params(), lr=crit["protocol"]["base_lr"], weight_decay=crit["protocol"]["base_wd"]
    )
    for k in range(crit["protocol"]["base_steps"]):
        b = _make_p24_stream(crit["protocol"]["base_batch"], SEED * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], b["conflict_labels"]
        )
        base_step(opt, loss, subj)
    base_eval = eval_base(subj)
    rev_probe = make_revision_stream(FIT, SEED + 444 + FIT_OFFSET)
    base_rev = eval_rev(subj, rev_probe)
    cap_base = capsule_roundtrip(subj, rev_probe["ticks"][:32])
    gate0 = (
        base_eval["overall"] >= crit["gates"]["G0_base_overall_min"]
        and base_eval["per_regime"][0] >= crit["gates"]["G0_base_outdated_min"]
    )
    payload = {
        "schema": "taiji-u2-24s-f1b-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "base": base_eval,
        "base_revision": base_rev,
        "capsule_diff_after_base": cap_base,
        "event": {"updates": crit["protocol"]["event_updates"]},
    }
    if not gate0:
        payload.update(
            {
                "gates": {"G0": False},
                "verdict": "24s_f1b_baseline_insufficient",
                "cost": {"cpu": round(time.process_time() - started, 2)},
                "discipline": {"check_reads": 0, "final_reads": 0},
            }
        )
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"verdict": payload["verdict"], "base": base_eval}, ensure_ascii=False))
        return 0

    arms = {
        mode: run_arm(subj, mode, rev_probe, crit)
        for mode in ("P_protected_replay", "N_no_protection", "L_anchor_only", "C_frozen")
    }
    p = arms["P_protected_replay"]
    c = arms["C_frozen"]
    gates = {
        "G0": bool(gate0),
        "G1": p["acc_at_8"] >= crit["gates"]["G1_acc8_min"],
        "G2": (p["acc_at_8"] - c["acc_at_8"]) >= crit["gates"]["G2_vs_frozen_min"],
        "G3": p["retention_drop"] <= crit["gates"]["G3_retention_drop_max"],
        "G4": p["capsule_diff"] <= crit["gates"]["G4_capsule_diff_max"],
    }
    if all(gates.values()):
        verdict = "24s_f1b_verified"
    elif not gates["G1"]:
        verdict = "24s_f1b_partial"
    elif not gates["G3"]:
        verdict = "24s_f1b_interference"
    else:
        verdict = "24s_f1b_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload.update(
        {
            "arms": arms,
            "gates": gates,
            "verdict": verdict,
            "verdict_inputs": {
                "acc_at_8_P": p["acc_at_8"],
                "acc_at_8_N": arms["N_no_protection"]["acc_at_8"],
                "acc_at_8_L": arms["L_anchor_only"]["acc_at_8"],
                "acc_at_8_C": c["acc_at_8"],
                "acc_at_4_P": p["acc_at_4"],
                "retention_drop_P": p["retention_drop"],
                "retention_drop_N": arms["N_no_protection"]["retention_drop"],
                "retention_drop_L": arms["L_anchor_only"]["retention_drop"],
                "capsule_diff_P": p["capsule_diff"],
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
                "base_overall": base_eval["overall"],
                "base_rev": base_rev,
                "acc_curve": {m: [round(x["acc_new"], 4) for x in arms[m]["curve"]] for m in arms},
                "acc_at_8": {m: round(arms[m]["acc_at_8"], 4) for m in arms},
                "retention_drop": {m: round(arms[m]["retention_drop"], 4) for m in arms},
                "capsule_diff": {"base": cap_base, "P": p["capsule_diff"]},
                "cpu": cost,
                "criteria_sha256": crit["criteria_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
