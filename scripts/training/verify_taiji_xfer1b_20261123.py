"""XFER-1b: cross-session compositional transfer (session A on seen colour pairs ->
session B on held-out pairs), same subject identity via capsule_v2.

Protocol (pre-registered, criteria_sha256 checked before any read):
  * session A: CN seen-pair training 600 steps; A_end accuracy on a separate seen test split;
  * capsule save/load -> fresh graph forward diff (identity/restore consistency);
  * session B: 250 steps on unseen-pair data from the restored same candidate (warm);
  * fresh control: new same-seed subject without A experience, same 250 steps;
  * retention: A seen accuracy after session B, drop <= 0.05.

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
from scripts.training.verify_taiji_23s2a_gain_20261122 import cn_split
from scripts.training.verify_taiji_24s_f1c_20261122 import capsule_state_diff
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import (
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_xfer1b_criteria_20261123.json"
OUT = ROOT / "reports/taiji_u2_xfer1b_20261123.json"
SEED = 20261120
FIT_OFFSET = 10_000_000
BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}


def build_subject() -> Subject72:
    torch.manual_seed(SEED)
    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in BLOCKS.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for name, head in subj.phase_heads.items():
        for p in head.parameters():
            p.requires_grad_(name == "cn")
    subj.train_mode()
    return subj


def cn_acc(subj, ticks, labels) -> float:
    subj.eval_mode()
    with torch.no_grad():
        sm = (ticks.abs().sum(-1) > 0).float()
        out = subj.logits("cn", ticks, sm).argmax(-1)
    return float((out == labels).float().mean())


def train_cn(subj, split: str, steps: int, *, curve_every: int = 0, curve_fn=None) -> list:
    opt = torch.optim.Adam([p for p in subj.params() if p.requires_grad], lr=6e-3, weight_decay=0.0)
    curve: list = []
    for k in range(steps):
        b = cn_split(48, SEED * 6000 + k + (0 if split == "seen" else 10_000), split)
        sm = (b["ticks"].abs().sum(-1) > 0).float()
        loss = F.cross_entropy(subj.logits("cn", b["ticks"], sm), b["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "cn")
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        if curve_every and curve_fn is not None and (k + 1) % curve_every == 0:
            curve.append({"step": k + 1, "acc": curve_fn()})
    return curve


def main() -> int:
    started = time.process_time()
    crit = json.loads(CRITERIA.read_text(encoding="utf-8"))
    body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
    digest = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == crit["criteria_sha256"], "criteria_sha256_mismatch"
    torch.set_num_threads(1)

    p = crit["protocol"]
    seen_test = cn_split(
        p["test_count"], SEED + p["test_seed_offset_seen"] + p["fit_offset"], "seen"
    )
    unseen_test = cn_split(
        p["test_count"], SEED + p["test_seed_offset_unseen"] + p["fit_offset"], "unseen"
    )

    # ---------- session A ----------
    subj_a = build_subject()
    train_cn(subj_a, "seen", p["session_a_steps"])
    a_end = cn_acc(subj_a, seen_test["ticks"], seen_test["labels"])
    cap_diff = capsule_state_diff(subj_a, seen_test["ticks"][:32], torch.ones(subj_a.n_units))

    # ---------- capsule cross-session ----------
    cap = build_capsule_v2(subj_a.graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "cap.json")
        loaded = load_capsule_v2(path)
    restored = SelectivePersistentGraph(
        copy.deepcopy(subj_a.graph.base_graph), subj_a.graph.selective_spec
    )
    restored.freeze_base_()
    apply_capsule_v2(restored, copy.deepcopy(loaded))

    # ---------- session B: warm (same candidate) ----------
    warm = copy.deepcopy(subj_a)
    warm.train_mode()
    b_curve = train_cn(
        warm,
        "unseen",
        p["session_b_steps"],
        curve_every=p["curve_every"],
        curve_fn=lambda: cn_acc(warm, unseen_test["ticks"], unseen_test["labels"]),
    )
    b_warm = cn_acc(warm, unseen_test["ticks"], unseen_test["labels"])
    a_after_b = cn_acc(warm, seen_test["ticks"], seen_test["labels"])

    # ---------- session B: fresh control (same seed, no A) ----------
    fresh = build_subject()
    fb_curve = train_cn(
        fresh,
        "unseen",
        p["session_b_steps"],
        curve_every=p["curve_every"],
        curve_fn=lambda: cn_acc(fresh, unseen_test["ticks"], unseen_test["labels"]),
    )
    b_fresh = cn_acc(fresh, unseen_test["ticks"], unseen_test["labels"])

    gates = {
        "G0": a_end >= crit["gates"]["G0_a_end_min"],
        "G1": b_warm >= a_end - crit["gates"]["G1_level_gap_max"],
        "G2": (b_warm - b_fresh) >= crit["gates"]["G2_gain_min"],
        "G3": (a_end - a_after_b) <= crit["gates"]["G3_retention_drop_max"]
        and cap_diff <= crit["gates"]["G3_capsule_diff_max"],
    }
    if not gates["G0"]:
        verdict = "xfer1b_baseline_insufficient"
    elif not gates["G2"]:
        verdict = "xfer1b_no_transfer"
    elif not gates["G1"]:
        verdict = "xfer1b_level_low"
    elif not gates["G3"]:
        verdict = "xfer1b_retention_breach"
    elif all(gates.values()):
        verdict = "xfer1b_verified"
    else:
        verdict = "xfer1b_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload = {
        "schema": "taiji-u2-xfer1-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "session_a": {"steps": p["session_a_steps"], "seen_acc_end": a_end},
        "cross_session": {"capsule_diff": cap_diff},
        "session_b": {
            "steps": p["session_b_steps"],
            "warm_unseen_acc": b_warm,
            "fresh_unseen_acc": b_fresh,
            "gain": b_warm - b_fresh,
            "level_gap": a_end - b_warm,
            "warm_curve": b_curve,
            "fresh_curve": fb_curve,
        },
        "retention": {"seen_acc_after_b": a_after_b, "drop": a_end - a_after_b},
        "gates": gates,
        "verdict": verdict,
        "run_conditions": {"threads": torch.get_num_threads()},
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": cost},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "a_end": round(a_end, 4),
                "capsule_diff": cap_diff,
                "b_warm": round(b_warm, 4),
                "b_fresh": round(b_fresh, 4),
                "gain": round(b_warm - b_fresh, 4),
                "level_gap": round(a_end - b_warm, 4),
                "retention_drop": round(a_end - a_after_b, 4),
                "warm_curve": [(r["step"], round(r["acc"], 4)) for r in b_curve],
                "fresh_curve": [(r["step"], round(r["acc"], 4)) for r in fb_curve],
                "cpu": cost,
                "criteria_sha256": crit["criteria_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
