"""24-S.F1c: growth-borne outdated correction (newborn units + dedicated readout path).

Same base candidate/stream as 24-S.F1, but the new revision-source rule is carried by
24 neutral newborn units (72->96, exact inheritance of old parameters) plus a dedicated
`p24rev` readout head; the old p24 block/head stay frozen.  Correction = 8 labeled
interactions training only newborn unit parameters and the new head (the 4-interaction
reading is the primary gate).  Zero retention damage is expected by construction and is
verified bitwise, together with newborn-unit causality, capsule identity and lifecycle
promotion invariance.

Single seed, <=500 CPU, zero check/final. Standing authorization; owner may revise.
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

from scripts.training.verify_taiji_24s_f1_20261122 import (
    BLOCKS,
    FIT,
    FIT_OFFSET,
    SEED,
    base_step,
    make_revision_stream,
)
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p24_stream
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import (
    SELECTIVE_LIFECYCLE,
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_24s_f1c_criteria_20261122.json"
OUT = ROOT / "reports/taiji_u2_24s_f1c_20261122.json"
OLD_N = 72
P24_OLD = list(range(24, 48))
UNIT_ROW_PARAMS = ("r", "c", "eta", "b", "write_bias", "extra_proj.weight")


def _forward_states(subj, ticks, mask):
    gates = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], mask.numel())
    out = subj.graph(
        _channels_seq(ticks), gates, subj.graph.initial_state(batch_size=ticks.shape[0])
    )
    return out["next_state"]["z"], out["next_state"]["a"]


def rev_logits(subj, ticks, rev_mask):
    z, a = _forward_states(subj, ticks, rev_mask)
    return subj.phase_heads["p24rev"](torch.cat([z, a], dim=-1))


def rev_logits_without_newborn(subj, ticks, rev_mask):
    z, a = _forward_states(subj, ticks, rev_mask)
    z[:, OLD_N:] = 0.0
    a[:, OLD_N:] = 0.0
    return subj.phase_heads["p24rev"](torch.cat([z, a], dim=-1))


def old_logits(subj, ticks, old_mask):
    z, a = _forward_states(subj, ticks, old_mask)
    return subj.phase_heads["p24"](torch.cat([z[:, :OLD_N], a[:, :OLD_N]], dim=-1))


def eval_rev(subj, data, rev_mask) -> dict:
    subj.eval_mode()
    with torch.no_grad():
        pred = rev_logits(subj, data["ticks"], rev_mask)[:, :4].argmax(-1)
    return {
        "acc_new": float((pred == data["val_labels"]).float().mean()),
        "acc_stale": float((pred == data["stale_labels"]).float().mean()),
    }


def eval_base_old(subj, old_mask) -> dict:
    subj.eval_mode()
    fit = _make_p24_stream(FIT, SEED + 212 + FIT_OFFSET)
    with torch.no_grad():
        lg = old_logits(subj, fit["ticks"], old_mask)
        ok = (lg[:, :4].argmax(-1) == fit["val_labels"]).float()
        conf = (lg[:, 4] > 0.0).float() == fit["conflict_labels"]
        reg = fit["regimes"]
        return {
            "overall": float(ok.mean()),
            "per_regime": {int(r): float(ok[reg == r].mean()) for r in range(4)},
            "conflict_accuracy": float(conf.float().mean()),
        }


def mask_newborn_grads(subj, old_n: int) -> None:
    for name, p in subj.graph.selective_units.named_parameters():
        if p.grad is None:
            continue
        if name in UNIT_ROW_PARAMS:
            p.grad[:old_n] = 0.0
        elif name == "p":
            p.grad[:, :old_n] = 0.0
        elif name in ("w", "write_state.weight"):
            p.grad[:old_n, :] = 0.0
            p.grad[:, :old_n] = 0.0
        elif name == "write_in.weight":
            p.grad[:old_n, :] = 0.0


def newborn_params(subj) -> list[torch.nn.Parameter]:
    return [p for p in subj.graph.selective_units.parameters() if p.requires_grad]


def old_slice(name: str, tensor: torch.Tensor, old_n: int) -> torch.Tensor:
    if name in UNIT_ROW_PARAMS:
        return tensor[:old_n]
    if name == "p":
        return tensor[:, :old_n]
    if name == "w":
        return tensor[:old_n, :old_n]
    if name == "write_in.weight":
        return tensor[:old_n, :]
    if name == "write_state.weight":
        return tensor[:old_n, :old_n]
    raise KeyError(name)


def inheritance_diff(subj, old_sd: dict[str, torch.Tensor]) -> float:
    current = {name: p.detach() for name, p in subj.graph.selective_units.named_parameters()}
    return max(
        float((old_slice(name, current[name], OLD_N) - old_sd[name]).abs().max()) for name in old_sd
    )


def birth_snapshot(subj, old_n: int) -> dict[str, torch.Tensor]:
    return {
        name: copy.deepcopy(p.detach()[old_n:].clone())
        for name, p in subj.graph.selective_units.named_parameters()
    }


def load_birth(subj, snap: dict[str, torch.Tensor], old_n: int) -> None:
    with torch.no_grad():
        for name, p in subj.graph.selective_units.named_parameters():
            p[old_n:].copy_(snap[name])


def capsule_state_diff(subj, ticks, mask) -> float:
    with torch.no_grad():
        z0, a0 = _forward_states(subj, ticks, mask)
    cap = build_capsule_v2(subj.graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "cap.json")
        loaded = load_capsule_v2(path)
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subj.graph.base_graph), subj.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    gates = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], mask.numel())
    with torch.no_grad():
        out = fresh(_channels_seq(ticks), gates, fresh.initial_state(batch_size=ticks.shape[0]))
    z1, a1 = out["next_state"]["z"], out["next_state"]["a"]
    return float(max((z0 - z1).abs().max(), (a0 - a1).abs().max()))


def make_opts(subj, mode, crit):
    if mode == "C1_no_correction":
        return None, None
    head_opt = torch.optim.Adam(
        subj.phase_heads["p24rev"].parameters(), lr=crit["protocol"]["head_lr"]
    )
    graph_opt = torch.optim.Adam(newborn_params(subj), lr=crit["protocol"]["graph_lr"])
    return head_opt, graph_opt


def train_one(subj, mode, i, rev_mask, crit, head_opt, graph_opt):
    if mode == "C1_no_correction":
        return
    b = make_revision_stream(crit["protocol"]["revision_batch"], SEED * 77000 + i)
    labels = b["val_labels"]
    if mode == "C2_permuted_labels":
        labels = (labels + 1) % 4
    subj.phase_heads["p24rev"].train()
    logits = rev_logits(subj, b["ticks"], rev_mask)
    loss = F.cross_entropy(logits[:, :4], labels) + F.binary_cross_entropy_with_logits(
        logits[:, 4], torch.zeros_like(logits[:, 4])
    )
    head_opt.zero_grad(set_to_none=True)
    graph_opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_newborn_grads(subj, OLD_N)
    torch.nn.utils.clip_grad_norm_(
        list(subj.phase_heads["p24rev"].parameters()) + newborn_params(subj), 1.0
    )
    head_opt.step()
    graph_opt.step()


def run_arm(base_subj, mode, rev_probe, rev_mask, old_mask, crit):
    subj = copy.deepcopy(base_subj)
    head_opt, graph_opt = make_opts(subj, mode, crit)
    base_anchor = eval_base_old(subj, old_mask)
    rev_curve = {key: value[: len(value) // 2] for key, value in rev_probe.items()}
    curve = []
    n_int = crit["protocol"]["correction_interactions"]
    for i in range(n_int):
        train_one(subj, mode, i, rev_mask, crit, head_opt, graph_opt)
        full_probe = (i + 1) in (crit["protocol"]["interaction_eval_4"], n_int)
        rev = eval_rev(subj, rev_probe if full_probe else rev_curve, rev_mask)
        row = {"interaction": i + 1, "acc_new": rev["acc_new"], "acc_stale": rev["acc_stale"]}
        if full_probe:
            base_now = eval_base_old(subj, old_mask)
            row["base_overall"] = base_now["overall"]
            row["base_per_regime"] = base_now["per_regime"]
        curve.append(row)
    final_base = curve[-1].get("base_per_regime") or base_anchor["per_regime"]
    retention_drop = max(0.0, max(base_anchor["per_regime"][r] - final_base[r] for r in range(4)))
    return (
        {
            "mode": mode,
            "base_anchor": base_anchor,
            "curve": curve,
            "acc_at_4": curve[crit["protocol"]["interaction_eval_4"] - 1]["acc_new"],
            "acc_at_8": curve[-1]["acc_new"],
            "retention_drop": retention_drop,
        },
        subj,
    )


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

    # ---------- base (identical to 24-S.F1) ----------
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
    pre_growth_mask = torch.tensor([1.0 if i in P24_OLD else 0.0 for i in range(OLD_N)])
    base_eval = eval_base_old(subj, pre_growth_mask)
    gate0 = (
        base_eval["overall"] >= crit["gates"]["G0_base_overall_min"]
        and base_eval["per_regime"][0] >= crit["gates"]["G0_base_outdated_min"]
    )
    payload = {
        "schema": "taiji-u2-24s-f1c-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "base": base_eval,
    }
    if not gate0:
        payload.update(
            {
                "gates": {"G0": False},
                "verdict": "24s_f1c_baseline_insufficient",
                "cost": {"cpu": round(time.process_time() - started, 2)},
                "discipline": {"check_reads": 0, "final_reads": 0},
            }
        )
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"verdict": payload["verdict"], "base": base_eval}, ensure_ascii=False))
        return 0

    # ---------- growth: +24 neutral newborn units (72 -> 96) ----------
    old_sd = {name: p.detach().clone() for name, p in subj.graph.selective_units.named_parameters()}
    new_ids = subj.graph.grow_units_neutral_(crit["protocol"]["grow_additional_per_group"])
    newn = subj.graph.selective_units.n_units
    subj.n_units = newn
    for name in ("cn", "p24", "p27"):
        pad = torch.zeros(newn)
        pad[:OLD_N] = subj.masks[name]
        subj.masks[name] = pad
    rev_mask = torch.zeros(newn)
    rev_mask[P24_OLD] = 1.0
    rev_mask[OLD_N:] = 1.0
    old_mask = torch.zeros(newn)
    old_mask[P24_OLD] = 1.0
    subj.graph.set_task_route("p24", P24_OLD + new_ids)
    inherit_growth = inheritance_diff(subj, old_sd)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 1017)
        subj.phase_heads["p24rev"] = PhaseHead(2 * newn, 5, 128)
    for p in subj.graph.parameters():
        p.requires_grad_(False)
    for name, head in subj.phase_heads.items():
        for p in head.parameters():
            p.requires_grad_(name == "p24rev")
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    birth = birth_snapshot(subj, OLD_N)
    rev_probe = make_revision_stream(FIT, SEED + 444 + FIT_OFFSET)
    event = []
    for k in range(crit["protocol"]["event_batches"]):
        eb = make_revision_stream(crit["protocol"]["event_batch"], SEED * 66000 + k)
        event.append(eval_rev(subj, eb, rev_mask)["acc_new"])
    post_growth_base = eval_base_old(subj, old_mask)
    post_growth_rev = eval_rev(subj, rev_probe, rev_mask)

    arms = {}
    subjects = {}
    for mode in ("P_proper_feedback", "C1_no_correction", "C2_permuted_labels"):
        arms[mode], subjects[mode] = run_arm(subj, mode, rev_probe, rev_mask, old_mask, crit)
    trained = subjects["P_proper_feedback"]
    inherit_after_training = inheritance_diff(trained, old_sd)
    trained_rev = eval_rev(trained, rev_probe, rev_mask)["acc_new"]
    gated_off = eval_rev(trained, rev_probe, old_mask)["acc_new"]
    with torch.no_grad():
        readout_zero_pred = rev_logits_without_newborn(trained, rev_probe["ticks"], rev_mask)[
            :, :4
        ].argmax(-1)
    readout_zero = float((readout_zero_pred == rev_probe["val_labels"]).float().mean())
    ablation = copy.deepcopy(trained)
    load_birth(ablation, birth, OLD_N)
    birth_reset = eval_rev(ablation, rev_probe, rev_mask)["acc_new"]
    lesion = {
        "trained_acc": trained_rev,
        "newborn_gated_off": gated_off,
        "newborn_readout_zero": readout_zero,
        "newborn_reset_to_birth": birth_reset,
        "drop_birth_reset": trained_rev - birth_reset,
    }
    # ---------- lifecycle promotion invariance ----------
    promo = copy.deepcopy(trained)
    with torch.no_grad():
        z_before, a_before = _forward_states(promo, rev_probe["ticks"][:32], rev_mask)
        promo.graph.selective_units.promote_immature_to_mature_()
        z_after, a_after = _forward_states(promo, rev_probe["ticks"][:32], rev_mask)
        promotion_diff = float(
            max((z_before - z_after).abs().max(), (a_before - a_after).abs().max())
        )
    lifecycle = [int(x) for x in promo.graph.selective_units.unit_lifecycle.tolist()]
    mature_id = SELECTIVE_LIFECYCLE.index("mature")
    immature_id = SELECTIVE_LIFECYCLE.index("immature")
    cap_diff = capsule_state_diff(promo, rev_probe["ticks"][:32], rev_mask)
    p, c1, c2 = (
        arms["P_proper_feedback"],
        arms["C1_no_correction"],
        arms["C2_permuted_labels"],
    )
    gates = {
        "G0": bool(gate0),
        "G1": p["acc_at_4"] >= crit["gates"]["G1_acc4_min"],
        "G2_vs_no_correction": (p["acc_at_4"] - c1["acc_at_4"])
        >= crit["gates"]["G2_vs_no_correction_min"],
        "G2_vs_permuted": (p["acc_at_4"] - c2["acc_at_4"]) >= crit["gates"]["G2_vs_permuted_min"],
        "G3": p["retention_drop"] <= crit["gates"]["G3_retention_drop_max"],
        "G3_inheritance": inherit_after_training == 0.0,
        "G4": lesion["drop_birth_reset"] >= crit["gates"]["G4_lesion_drop_min"],
        "G5": cap_diff <= crit["gates"]["G5_capsule_diff_max"],
        "G6_promotion": promotion_diff <= 1e-9,
    }
    if all(gates.values()):
        verdict = "24s_f1c_verified"
    elif not gates["G1"] and p["acc_at_8"] >= crit["gates"]["G1_acc4_min"]:
        verdict = "24s_f1c_partial_slow"
    elif not gates["G1"]:
        verdict = "24s_f1c_partial"
    elif not gates["G3"]:
        verdict = "24s_f1c_interference"
    else:
        verdict = "24s_f1c_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload.update(
        {
            "growth": {
                "added_units": len(new_ids),
                "n_units": newn,
                "inheritance_max_diff_at_growth": inherit_growth,
                "inheritance_max_diff_after_training": inherit_after_training,
                "post_growth_base": post_growth_base,
                "post_growth_revision": post_growth_rev,
                "event_acc_new": event,
            },
            "arms": arms,
            "lesion": lesion,
            "lifecycle": {
                "promotion_forward_diff": promotion_diff,
                "mature_units_after_promotion": sum(1 for x in lifecycle if x == mature_id),
                "immature_units_after_promotion": sum(1 for x in lifecycle if x == immature_id),
                "total_units": newn,
            },
            "capsule_state_diff": cap_diff,
            "gates": gates,
            "verdict": verdict,
            "verdict_inputs": {
                "acc_at_4_P": p["acc_at_4"],
                "acc_at_8_P": p["acc_at_8"],
                "acc_at_4_C1": c1["acc_at_4"],
                "acc_at_4_C2": c2["acc_at_4"],
                "acc_at_8_C1": c1["acc_at_8"],
                "retention_drop_P": p["retention_drop"],
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
                "growth": {
                    "added": len(new_ids),
                    "inheritance_growth": inherit_growth,
                    "inheritance_trained": inherit_after_training,
                    "post_growth_base": post_growth_base["overall"],
                    "post_growth_rev": post_growth_rev["acc_new"],
                },
                "curve": {m: [round(x["acc_new"], 4) for x in arms[m]["curve"]] for m in arms},
                "lesion": {k: round(v, 4) for k, v in lesion.items()},
                "retention_drop_P": round(p["retention_drop"], 4),
                "promotion_forward_diff": promotion_diff,
                "capsule_state_diff": cap_diff,
                "cpu": cost,
                "criteria_sha256": crit["criteria_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
