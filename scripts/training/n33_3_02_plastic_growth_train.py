"""N33.3.02 (exploration): task-agnostic continual plasticity & on-demand growth in training.

Pre-registered: plans/reference/DESIGN-N33-3-02_20261125.md (D-20261125-588).
Compares `growth_plastic` (8 -> on-demand neutral birth 32 + selective plasticity + SessionStore)
against `fixed_small` (8) and `prealloc_equal` (32 preallocated) on `taiji-continual-growth-v1`.
Zero check/final/confirm reads.

Usage:
  python scripts/training/n33_3_02_plastic_growth_train.py --smoke
  python scripts/training/n33_3_02_plastic_growth_train.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from scripts.training.n31_guard import preflight  # noqa: E402
from taiji.choice_simulator import seed_from  # noqa: E402
from taiji.continual_growth_task import IN_DIM, NUM_CLASSES, TASK_VERSION, make_split  # noqa: E402
from taiji.plastic_growth import (
    ARCH_VERSION,
    GrowthPressureController,
    PlasticGrowthNet,
)  # noqa: E402
from taiji.session_store import SessionStore  # noqa: E402

CARD = "N33.3.02"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
TRAIN_SEEDS = (3001, 3002, 3003)
SPOT_CHECK_SEED = 3001
P1_STEPS = 500
P2_STEPS = 1200
SMOKE_STEPS = 40
CHECK_EVERY = 100
BATCH = 128
LR = 5e-3
CAP_CPU = 60.0
OUT = ROOT / "reports" / "n33_3_02_plastic_growth.json"


def _eval_acc(net: PlasticGrowthNet, split: dict[str, torch.Tensor]) -> float:
    net.eval()
    with torch.no_grad():
        preds = net(split["x"]).argmax(dim=-1)
        return float((preds == split["y"]).to(torch.float32).mean().item())


def _store_handoff(
    net: PlasticGrowthNet,
    store: SessionStore,
    subject_id: str,
    x_probe: torch.Tensor,
) -> tuple[PlasticGrowthNet, float]:
    with torch.no_grad():
        y_before = net(x_probe)
    rec = net.to_session_record(subject_id, pending=[{"id": "p2", "phase": 2}])
    store.save(rec)
    loaded = PlasticGrowthNet.from_session_record(store.load(subject_id))
    with torch.no_grad():
        diff = float((y_before - loaded(x_probe)).abs().max())
    return loaded, diff


def run_arm(
    seed: int,
    arm: str,
    data: dict[str, dict[str, dict[str, torch.Tensor]]],
    *,
    p1_steps: int,
    p2_steps: int,
    deadline: float,
) -> tuple[dict[str, Any], PlasticGrowthNet]:
    """Run Phase 1 -> SessionStore handoff -> Phase 2 for one arm."""
    assert arm in ("fixed_small", "growth_plastic", "prealloc_equal")
    init_h = 32 if arm == "prealloc_equal" else 8
    net = PlasticGrowthNet(IN_DIM, init_h, NUM_CLASSES, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen_p1 = torch.Generator().manual_seed(seed_from(CARD, seed, "p1-batches"))
    gen_p2 = torch.Generator().manual_seed(seed_from(CARD, seed, "p2-batches"))
    ctrl = GrowthPressureController(window=3, plateau_eps=0.015, error_threshold=0.15, grow_k=24)

    p1_tr, p1_va = data["phase1"]["train"], data["phase1"]["val"]
    p2_tr, p2_va = data["phase2"]["train"], data["phase2"]["val"]

    # Phase 1
    net.train()
    for step in range(1, p1_steps + 1):
        if time.process_time() > deadline:
            raise TimeoutError(f"{arm} seed={seed} p1 step={step}")
        idx = torch.randint(0, p1_tr["x"].shape[0], (BATCH,), generator=gen_p1)
        loss = F.cross_entropy(net(p1_tr["x"][idx]), p1_tr["y"][idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite p1 {arm} seed={seed}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if arm == "growth_plastic" and step % CHECK_EVERY == 0:
            p1_err = 1.0 - _eval_acc(net, p1_va)
            ctrl.observe(p1_err)
            net.train()

    p1_before = _eval_acc(net, p1_va)
    p1_growth_events = ctrl.growth_events

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(tmpdir)
        net, handoff_diff = _store_handoff(net, store, f"{arm}-{seed}-p1", p1_va["x"][:32])

        if arm == "growth_plastic":
            net.mature_()
            ctrl.history.clear()

        opt = torch.optim.Adam(net.parameters(), lr=LR)
        half = BATCH // 2
        birth_diff = 0.0
        new_units: list[int] = []

        net.train()
        for step in range(1, p2_steps + 1):
            if time.process_time() > deadline:
                raise TimeoutError(f"{arm} seed={seed} p2 step={step}")
            i2 = torch.randint(0, p2_tr["x"].shape[0], (half,), generator=gen_p2)
            i1 = torch.randint(0, p1_tr["x"].shape[0], (half,), generator=gen_p1)
            xb = torch.cat([p2_tr["x"][i2], p1_tr["x"][i1]], dim=0)
            yb = torch.cat([p2_tr["y"][i2], p1_tr["y"][i1]], dim=0)
            loss = F.cross_entropy(net(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite p2 {arm} seed={seed}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if arm == "growth_plastic":
                net.apply_selective_plasticity_grad_mask(freeze_mature=True)
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()

            if arm == "growth_plastic" and step % CHECK_EVERY == 0:
                p2_err = 1.0 - _eval_acc(net, p2_va)
                grow_k = ctrl.observe(p2_err)
                if grow_k > 0:
                    net.eval()
                    with torch.no_grad():
                        y0 = net(p1_va["x"][:32])
                        new_units = net.grow_neutral_(grow_k, seed=seed_from(CARD, seed, "grow"))
                        birth_diff = float((y0 - net(p1_va["x"][:32])).abs().max())
                    opt = torch.optim.Adam(net.parameters(), lr=LR)
                net.train()

        p1_after = _eval_acc(net, p1_va)
        p2_acc = _eval_acc(net, p2_va)
        combined = round(0.5 * (p1_after + p2_acc), 6)

        dormancy_drop = 0.0
        wake_diff = 0.0
        if arm == "growth_plastic" and new_units:
            saved = net.set_dormant_(new_units)
            dormant_p2 = _eval_acc(net, p2_va)
            dormancy_drop = round(p2_acc - dormant_p2, 6)
            net.wake_(saved)
            wake_diff = abs(_eval_acc(net, p2_va) - p2_acc)

        net, post_diff = _store_handoff(net, store, f"{arm}-{seed}-p2", p2_va["x"][:32])

    return (
        {
            "arm": arm,
            "final_hidden": net.hidden_dim,
            "n_params": net.n_params(),
            "p1_before_acc": round(p1_before, 6),
            "p1_after_acc": round(p1_after, 6),
            "p1_retention_drop": round(p1_before - p1_after, 6),
            "p2_acc": round(p2_acc, 6),
            "combined_acc": combined,
            "p1_growth_events": p1_growth_events,
            "p2_growth_events": ctrl.growth_events - p1_growth_events,
            "birth_perturbation": birth_diff,
            "p2_dormancy_drop": dormancy_drop,
            "wake_diff": wake_diff,
            "store_roundtrip_exact": bool(handoff_diff == 0.0 and post_diff == 0.0),
        },
        net,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    design_file = preflight(CARD, list(TRAIN_SEEDS), record=False).name
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    p1_steps = SMOKE_STEPS if args.smoke else P1_STEPS
    p2_steps = SMOKE_STEPS if args.smoke else P2_STEPS

    per_seed: dict[str, Any] = {}
    repro_exact = False
    try:
        for seed in TRAIN_SEEDS:
            s_t0 = time.process_time()
            data = make_split(seed)
            small_res, _ = run_arm(
                seed, "fixed_small", data, p1_steps=p1_steps, p2_steps=p2_steps, deadline=deadline
            )
            grow_res, grow_net = run_arm(
                seed,
                "growth_plastic",
                data,
                p1_steps=p1_steps,
                p2_steps=p2_steps,
                deadline=deadline,
            )
            pre_res, _ = run_arm(
                seed,
                "prealloc_equal",
                data,
                p1_steps=p1_steps,
                p2_steps=p2_steps,
                deadline=deadline,
            )
            if seed == SPOT_CHECK_SEED:
                _, grow_net2 = run_arm(
                    seed,
                    "growth_plastic",
                    data,
                    p1_steps=p1_steps,
                    p2_steps=p2_steps,
                    deadline=deadline,
                )
                repro_exact = all(
                    torch.equal(a, b)
                    for a, b in zip(grow_net.parameters(), grow_net2.parameters(), strict=True)
                ) and torch.equal(grow_net.stage, grow_net2.stage)

            vs_small = round(grow_res["p2_acc"] - small_res["p2_acc"], 6)
            vs_prealloc = round(grow_res["combined_acc"] - pre_res["combined_acc"], 6)
            g_eng = bool(
                grow_res["store_roundtrip_exact"]
                and small_res["store_roundtrip_exact"]
                and pre_res["store_roundtrip_exact"]
                and grow_res["birth_perturbation"] <= 1e-6
                and grow_res["wake_diff"] == 0.0
                and grow_res["p1_growth_events"] == 0
                and (args.smoke or grow_res["p2_growth_events"] == 1)
            )
            g_press = bool(
                grow_res["p2_acc"] >= 0.85
                and vs_small >= 0.12
                and grow_res["p2_dormancy_drop"] >= 0.15
                and grow_res["p1_retention_drop"] <= 0.02
            )
            per_seed[str(seed)] = {
                "fixed_small": small_res,
                "growth_plastic": grow_res,
                "prealloc_equal": pre_res,
                "p2_gain_over_small": vs_small,
                "combined_diff_vs_prealloc": vs_prealloc,
                "g_eng": g_eng,
                "g_pressure": g_press,
                "cpu": round(time.process_time() - s_t0, 2),
            }

        all_eng = repro_exact and all(v["g_eng"] for v in per_seed.values())
        all_press = all(v["g_pressure"] for v in per_seed.values())
        diffs = [v["combined_diff_vs_prealloc"] for v in per_seed.values()]
        if not all_eng:
            exit_ = "engineering_failure"
        elif args.smoke:
            exit_ = "smoke"
        elif not all_press:
            exit_ = "growth_pressure_not_met"
        elif all(d >= 0.05 for d in diffs):
            exit_ = "growth_adopt"
        elif all(d >= -0.02 for d in diffs):
            exit_ = "growth_required_inplace"
        else:
            exit_ = "growth_required_revise"
    except (TimeoutError, FloatingPointError) as exc:
        exit_ = "engineering_failure"
        per_seed["error"] = str(exc)

    payload = {
        "schema": SCHEMA,
        "card": CARD,
        "design": design_file,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "arch_version": ARCH_VERSION,
        "task_version": TASK_VERSION,
        "steps": {"p1": p1_steps, "p2": p2_steps},
        "train_seeds": list(TRAIN_SEEDS),
        "repro_spot_check": {"seed": SPOT_CHECK_SEED, "exact": bool(repro_exact)},
        "per_seed": per_seed,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "exit": exit_,
        "repro_spot_check": payload["repro_spot_check"],
        "cost": payload["cost"],
    }
    if all(isinstance(v, dict) and "growth_plastic" in v for v in per_seed.values()):
        summary["per_seed"] = {
            s: {
                "small_p2": v["fixed_small"]["p2_acc"],
                "grow_p1_drop": v["growth_plastic"]["p1_retention_drop"],
                "grow_p2": v["growth_plastic"]["p2_acc"],
                "grow_comb": v["growth_plastic"]["combined_acc"],
                "prealloc_comb": v["prealloc_equal"]["combined_acc"],
                "vs_prealloc": v["combined_diff_vs_prealloc"],
                "dormancy_drop": v["growth_plastic"]["p2_dormancy_drop"],
                "g_eng": v["g_eng"],
                "g_pressure": v["g_pressure"],
            }
            for s, v in per_seed.items()
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if exit_ != "engineering_failure" else 3


if __name__ == "__main__":
    raise SystemExit(main())
