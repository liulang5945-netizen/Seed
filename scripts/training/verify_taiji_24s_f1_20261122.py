"""24-S.F1: representation-level outdated correction (new revision source precedence).

Protocol (pre-registered, criteria_sha256 checked before any read):
  * base: single-task P24 training 1200 steps on Subject72 (24-unit p24 route);
  * event: 10 unlabeled exposures of the revision stream (0 parameter updates);
  * correction: 4 labeled interactions (8 recorded by curve), 4 arms:
      A_representation = p24 unit block + p24 head trainable,
      B_head_only      = p24 head only,
      C_frozen         = no parameter updates,
      D_permuted_labels= same as A but feedback labels rotated +1,
  * held-out revision accuracy is the "outdated subset" metric.

Single seed, <=400 CPU, zero check/final. Standing authorization; owner may revise.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p24_stream
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import (
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_24s_f1_criteria_20261122.json"
OUT = ROOT / "reports/taiji_u2_24s_f1_20261122.json"

BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}
SEED = 20261120
FIT = 256
FIT_OFFSET = 10_000_000
REV_SOURCE = 14  # new revision-authority source channel (unused in base training)


# ---------------------------------------------------------------------------
# revision stream: old trusted source(12) repeats v_base, revision source(14) corrects
# ---------------------------------------------------------------------------
def make_revision_stream(count: int, seed: int) -> dict[str, torch.Tensor]:
    horizon = 14
    g = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(count, horizon, 48, dtype=torch.float32)
    ticks[:, : horizon - 1, 16:24] = torch.randn(count, horizon - 1, 8, generator=g) * 0.2
    base_vals = torch.randint(0, 4, (count, 2), generator=g)
    e_queries = torch.randint(0, 2, (count,), generator=g)
    val_labels = torch.zeros(count, dtype=torch.long)
    stale_labels = torch.zeros(count, dtype=torch.long)
    for b in range(count):
        eq = int(e_queries[b])
        v_base = int(base_vals[b, eq])
        stale_labels[b] = v_base
        other = 1 - eq
        order_s12 = torch.randperm(4, generator=g).tolist()
        for idx_pos, t in enumerate(order_s12):
            if idx_pos < 2:
                ticks[b, t, idx_pos] = 1.0
                ticks[b, t, 4 + int(base_vals[b, idx_pos])] = 1.0
                ticks[b, t, 8] = 1.0
                ticks[b, t, 12] = 1.0
            else:
                ticks[b, t, other] = 1.0
                ticks[b, t, 4 + int(torch.randint(0, 4, (1,), generator=g))] = 1.0
                ticks[b, t, 9] = 1.0
                ticks[b, t, 13] = 1.0
        s3 = torch.randperm(5, generator=g).tolist()
        t_old, t_new = sorted((4 + s3[0], 4 + s3[1]))
        # old trusted source repeats the base value (stale claim)
        ticks[b, t_old, eq] = 1.0
        ticks[b, t_old, 4 + v_base] = 1.0
        ticks[b, t_old, 9] = 1.0
        ticks[b, t_old, 12] = 1.0
        # revision authority gives the corrected value
        v_new = (v_base + 1 + int(torch.randint(0, 3, (1,), generator=g))) % 4
        val_labels[b] = v_new
        ticks[b, t_new, eq] = 1.0
        ticks[b, t_new, 4 + v_new] = 1.0
        ticks[b, t_new, 9] = 1.0
        ticks[b, t_new, REV_SOURCE] = 1.0
        ticks[b, 13, eq] = 1.0
        ticks[b, 13, 11] = 1.0
    return {"ticks": ticks, "val_labels": val_labels, "stale_labels": stale_labels}


# ---------------------------------------------------------------------------
# training / eval helpers
# ---------------------------------------------------------------------------
def base_step(opt, loss, subj):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subj, "p24")
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()


def eval_base(subj) -> dict:
    subj.eval_mode()
    fit = _make_p24_stream(FIT, SEED + 212 + FIT_OFFSET)
    with torch.no_grad():
        lg = subj.logits("p24", fit["ticks"])
        ok = (lg[:, :4].argmax(-1) == fit["val_labels"]).float()
        conf_ok = (lg[:, 4] > 0.0).float() == fit["conflict_labels"]
        reg = fit["regimes"]
        per = {int(r): float(ok[reg == r].mean()) for r in range(4)}
        return {
            "overall": float(ok.mean()),
            "per_regime": per,
            "conflict_accuracy": float(conf_ok.float().mean()),
            "conflict_recall_regime3": float((lg[reg == 3, 4] > 0.0).float().mean()),
        }


def eval_rev(subj, data) -> dict:
    subj.eval_mode()
    with torch.no_grad():
        pred = subj.logits("p24", data["ticks"])[:, :4].argmax(-1)
    return {
        "acc_new": float((pred == data["val_labels"]).float().mean()),
        "acc_stale": float((pred == data["stale_labels"]).float().mean()),
    }


def set_trainable(subj, mode: str) -> None:
    for p in subj.graph.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(False)
    for p in subj.phase_heads["p24"].parameters():
        p.requires_grad_(mode in ("A_representation", "B_head_only", "D_permuted_labels"))
    if mode in ("A_representation", "D_permuted_labels"):
        for p in subj.graph.selective_units.parameters():
            p.requires_grad_(True)
        for p in subj.graph.read_gate.parameters():
            p.requires_grad_(True)
    subj.train_mode()


def arm_step(subj, mode, batch, head_opt, graph_opt) -> None:
    labels = batch["val_labels"]
    if mode == "D_permuted_labels":
        labels = (labels + 1) % 4
    head = subj.phase_heads["p24"]
    head.train()
    subj.graph.eval()
    logits = subj.logits("p24", batch["ticks"])
    loss = F.cross_entropy(logits[:, :4], labels) + F.binary_cross_entropy_with_logits(
        logits[:, 4], torch.zeros_like(logits[:, 4])
    )
    if mode == "C_frozen" or head_opt is None:
        return
    if mode == "B_head_only":
        head_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        head_opt.step()
        return
    head_opt.zero_grad(set_to_none=True)
    graph_opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subj, "p24")
    params = list(head.parameters()) + [p for p in graph_opt.param_groups[0]["params"]]
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    head_opt.step()
    graph_opt.step()


def capsule_roundtrip(subj, ticks) -> float:
    with torch.no_grad():
        out_a = subj.graph(
            _channels_seq(ticks),
            torch.ones(ticks.shape[0], ticks.shape[1], subj.n_units),
            subj.graph.initial_state(batch_size=ticks.shape[0]),
        )["readout"]
    cap = build_capsule_v2(subj.graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "cap.json")
        loaded = load_capsule_v2(path)
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subj.graph.base_graph), subj.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    with torch.no_grad():
        out_b = fresh(
            _channels_seq(ticks),
            torch.ones(ticks.shape[0], ticks.shape[1], subj.n_units),
            fresh.initial_state(batch_size=ticks.shape[0]),
        )["readout"]
    return float((out_a - out_b).abs().max())


def run_arm(base_subj, mode: str, rev_probe, crit) -> dict:
    subj = copy.deepcopy(base_subj)
    set_trainable(subj, mode)
    head_params = [p for p in subj.phase_heads["p24"].parameters() if p.requires_grad]
    graph_params = [p for p in subj.graph.parameters() if p.requires_grad]
    head_opt = (
        torch.optim.Adam(head_params, lr=crit["protocol"]["head_lr"]) if head_params else None
    )
    graph_opt = (
        torch.optim.Adam(graph_params, lr=crit["protocol"]["graph_lr"]) if graph_params else None
    )
    # event: 10 unlabeled exposures, zero parameter updates
    event = []
    for k in range(crit["protocol"]["event_batches"]):
        eb = make_revision_stream(crit["protocol"]["event_batch"], SEED * 66000 + k)
        event.append(eval_rev(subj, eb)["acc_new"])
    base_anchor = eval_base(subj)
    rev_curve = {key: value[: len(value) // 2] for key, value in rev_probe.items()}
    curve = []
    n_int = crit["protocol"]["correction_interactions_extended"]
    for i in range(n_int):
        b = make_revision_stream(crit["protocol"]["correction_batch"], SEED * 77000 + i)
        arm_step(subj, mode, b, head_opt, graph_opt)
        full_probe = (i + 1) in (4, n_int)
        rev = eval_rev(subj, rev_probe if full_probe else rev_curve)
        row = {
            "interaction": i + 1,
            "acc_new": rev["acc_new"],
            "acc_stale": rev["acc_stale"],
            "probe_size": (
                int(rev_probe["ticks"].shape[0]) if full_probe else int(rev_curve["ticks"].shape[0])
            ),
        }
        if full_probe:
            base_now = eval_base(subj)
            row["base_overall"] = base_now["overall"]
            row["base_per_regime"] = base_now["per_regime"]
        curve.append(row)
    for row in curve:
        if "base_per_regime" not in row:
            row["base_per_regime"] = curve[-1].get("base_per_regime") or base_anchor["per_regime"]
    # lesion: zero the p24 unit mask and re-measure the revision subset
    lesion = {}
    saved = subj.masks["p24"].clone()
    subj.masks["p24"].zero_()
    lesion["acc_new_after"] = eval_rev(subj, rev_probe)["acc_new"]
    subj.masks["p24"].copy_(saved)
    lesion["acc_new_before"] = curve[-1]["acc_new"]
    lesion["drop"] = lesion["acc_new_before"] - lesion["acc_new_after"]
    # identity/restore consistency of the updated candidate
    probe_ticks = rev_probe["ticks"][:32]
    cap_diff = capsule_roundtrip(subj, probe_ticks)
    retention_drop = max(
        0.0,
        max(base_anchor["per_regime"][r] - curve[-1]["base_per_regime"][r] for r in range(4)),
    )
    return {
        "mode": mode,
        "event_acc_new": event,
        "base_anchor": base_anchor,
        "curve": curve,
        "acc_at_4": curve[min(3, len(curve) - 1)]["acc_new"],
        "acc_at_8": curve[-1]["acc_new"],
        "lesion": lesion,
        "capsule_diff": cap_diff,
        "retention_drop": retention_drop,
        "parameter_updates": 0 if mode == "C_frozen" else n_int,
    }


def main() -> int:
    started = time.process_time()
    crit = json.loads(CRITERIA.read_text(encoding="utf-8"))
    body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
    canon = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canon).hexdigest()
    assert digest == crit["criteria_sha256"], "criteria_sha256_mismatch"
    torch.set_num_threads(1)

    # ---------------- base subject: single-task P24 competence ----------------
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
        "schema": "taiji-u2-24s-f1-v1",
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
                "verdict": "24s_f1_baseline_insufficient",
                "cost": {"cpu": round(time.process_time() - started, 2)},
                "discipline": {"check_reads": 0, "final_reads": 0},
            }
        )
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"verdict": payload["verdict"], "base": base_eval}, ensure_ascii=False))
        return 0

    arms = {
        mode: run_arm(subj, mode, rev_probe, crit)
        for mode in ("A_representation", "B_head_only", "C_frozen", "D_permuted_labels")
    }
    a, b, c, d = (
        arms["A_representation"],
        arms["B_head_only"],
        arms["C_frozen"],
        arms["D_permuted_labels"],
    )
    gates = {
        "G0": bool(gate0),
        "G1": a["acc_at_4"] >= crit["gates"]["G1_acc4_min"],
        "G2_vs_frozen": (a["acc_at_4"] - c["acc_at_4"]) >= crit["gates"]["G2_vs_frozen_min"],
        "G2_vs_permuted": (a["acc_at_4"] - d["acc_at_4"]) >= crit["gates"]["G2_vs_permuted_min"],
        "G3": a["lesion"]["drop"] >= crit["gates"]["G3_lesion_drop_min"],
        "G4_retention": a["retention_drop"] <= crit["gates"]["G4_retention_drop_max"],
        "G4_capsule": a["capsule_diff"] <= crit["gates"]["G4_capsule_diff_max"],
    }
    if all(gates.values()):
        verdict = "24s_f1_verified"
    elif a["acc_at_8"] >= crit["gates"]["G1_acc4_min"]:
        verdict = "24s_f1_partial_slow"
    else:
        verdict = "24s_f1_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload.update(
        {
            "arms": arms,
            "gates": gates,
            "verdict": verdict,
            "verdict_inputs": {
                "acc_at_4_A": a["acc_at_4"],
                "acc_at_4_C": c["acc_at_4"],
                "acc_at_4_D": d["acc_at_4"],
                "acc_at_8_A": a["acc_at_8"],
                "acc_at_8_B": b["acc_at_8"],
                "lesion_drop_A": a["lesion"]["drop"],
                "retention_drop_A": a["retention_drop"],
                "capsule_diff_A": a["capsule_diff"],
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
                "lesion_A": {k: round(v, 4) for k, v in a["lesion"].items()},
                "retention_drop_A": round(a["retention_drop"], 4),
                "capsule_diff": {"base": cap_base, "A": a["capsule_diff"]},
                "cpu": cost,
                "criteria_sha256": crit["criteria_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
