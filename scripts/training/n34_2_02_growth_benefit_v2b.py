"""N34.2.02 (exploration): growth net-benefit v2, calibration fix.

Pre-registered: plans/reference/DESIGN-N34-2-02_20261126.md (D-20261126-626).\nSingle variable vs N34.2.01: main prealloc_big LR 5e-3 -> 1e-2 (single LR, matching post-growth LR; no grid).
Single contrast of interest: growth_plastic (8 -> 32 on pressure) vs prealloc_big (32 from the
start, single LR, so no extra hyper-parameter budget). Task: taiji-continual-growth-v2 (harder
P1 with label noise; 4-pair parity mod 3 P2) so that preallocation cannot saturate the task the
way it did in N33.3 (net benefit was exactly 0.0 there).

Usage (after admission):
  python scripts/training/n34_2_01_growth_benefit_v2.py --smoke
  python scripts/training/n34_2_01_growth_benefit_v2.py
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
from scripts.training.n33_3_02_plastic_growth_train import (  # noqa: E402
    BATCH,
    P1_STEPS,
    P2_STEPS,
    SMOKE_STEPS,
    _eval_acc,
    _store_handoff,
)
from taiji.choice_simulator import seed_from  # noqa: E402
from taiji.continual_growth_task_v2 import (
    IN_DIM,
    NUM_CLASSES,
    TASK_VERSION,
    make_split,
)  # noqa: E402
from taiji.plastic_growth import GrowthPressureController, PlasticGrowthNet  # noqa: E402
from taiji.session_store import SessionStore  # noqa: E402

CARD = "N34.2.02"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
DESIGN = "DESIGN-N34-2-02_20261126.md"
CHECK_EVERY = 25
LR_BASE = 5e-3
LR_IMMATURE = 1e-2
TRAIN_SEEDS = (4501, 4502, 4503)
SPOT_CHECK_SEED = 4501
GROWTH_TARGET = 32
CAP_CPU = 120.0
BENEFIT_GATE = 0.02
RETENTION_FLOOR = -0.02
CAL_FIXED_P2_MAX = 0.80
CAL_PREALLOC_P2_MIN = 0.90
OUT = ROOT / "reports" / "n34_2_02_growth_benefit_v2b.json"


def run_arm(
    seed: int,
    arm: str,
    data: dict[str, dict[str, dict[str, torch.Tensor]]],
    *,
    lr: float,
    lr_post_growth: float,
    p1_steps: int,
    p2_steps: int,
    check_every: int,
    deadline: float,
) -> tuple[dict[str, Any], PlasticGrowthNet]:
    assert arm in ("fixed_small", "growth_plastic", "prealloc_big", "prealloc_big_lr2")
    init_h = GROWTH_TARGET if arm.startswith("prealloc") else 8
    net = PlasticGrowthNet(IN_DIM, init_h, NUM_CLASSES, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen_p1 = torch.Generator().manual_seed(seed_from(CARD, seed, "p1-batches"))
    gen_p2 = torch.Generator().manual_seed(seed_from(CARD, seed, "p2-batches"))
    ctrl = GrowthPressureController(window=3, plateau_eps=0.015, error_threshold=0.15, grow_k=24)

    p1_tr, p1_va = data["phase1"]["train"], data["phase1"]["val"]
    p2_tr, p2_va = data["phase2"]["train"], data["phase2"]["val"]

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
        if arm == "growth_plastic" and step % check_every == 0:
            ctrl.observe(1.0 - _eval_acc(net, p1_va))
            net.train()

    p1_before = _eval_acc(net, p1_va)
    p1_growth_events = ctrl.growth_events

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(tmpdir)
        net, handoff_diff = _store_handoff(net, store, f"{arm}-{seed}-p1", p1_va["x"][:32])
        if arm == "growth_plastic":
            net.mature_()
            ctrl.history.clear()

        opt = torch.optim.Adam(net.parameters(), lr=lr)
        half = BATCH // 2
        birth_diff = 0.0
        grew_at_step: int | None = None
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
            if arm == "growth_plastic" and step % check_every == 0:
                grow_k = ctrl.observe(1.0 - _eval_acc(net, p2_va))
                if grow_k > 0:
                    grew_at_step = step
                    net.eval()
                    with torch.no_grad():
                        y0 = net(p1_va["x"][:32])
                        new_units = net.grow_neutral_(grow_k, seed=seed_from(CARD, seed, "grow"))
                        birth_diff = float((y0 - net(p1_va["x"][:32])).abs().max())
                    opt = torch.optim.Adam(net.parameters(), lr=lr_post_growth)
                net.train()

        p1_after = _eval_acc(net, p1_va)
        p2_acc = _eval_acc(net, p2_va)
        combined = round(0.5 * (p1_after + p2_acc), 6)

        dormancy_drop = 0.0
        wake_diff = 0.0
        if arm == "growth_plastic" and new_units:
            saved = net.set_dormant_(new_units)
            dormancy_drop = round(p2_acc - _eval_acc(net, p2_va), 6)
            net.wake_(saved)
            wake_diff = abs(_eval_acc(net, p2_va) - p2_acc)

        net, post_diff = _store_handoff(net, store, f"{arm}-{seed}-p2", p2_va["x"][:32])

    return (
        {
            "arm": arm,
            "lr": lr,
            "lr_post_growth": lr_post_growth if arm == "growth_plastic" else lr,
            "final_hidden": net.hidden_dim,
            "n_params": net.n_params(),
            "p1_before_acc": round(p1_before, 6),
            "p1_after_acc": round(p1_after, 6),
            "p1_retention_drop": round(p1_before - p1_after, 6),
            "p2_acc": round(p2_acc, 6),
            "combined_acc": combined,
            "p1_growth_events": p1_growth_events,
            "p2_growth_events": ctrl.growth_events - p1_growth_events,
            "grew_at_step": grew_at_step,
            "birth_perturbation": birth_diff,
            "p2_dormancy_drop": dormancy_drop,
            "wake_diff": wake_diff,
            "store_roundtrip_exact": bool(handoff_diff == 0.0 and post_diff == 0.0),
        },
        net,
    )


def _arm_specs() -> list[tuple[str, float, float]]:
    return [
        ("fixed_small", LR_BASE, LR_BASE),
        ("growth_plastic", LR_BASE, LR_IMMATURE),
        ("prealloc_big", LR_IMMATURE, LR_IMMATURE),
        ("prealloc_big_lr2", LR_IMMATURE, LR_IMMATURE),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    design = preflight(CARD, list(TRAIN_SEEDS), record=False).name
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    p1_steps = SMOKE_STEPS if args.smoke else P1_STEPS
    p2_steps = SMOKE_STEPS if args.smoke else P2_STEPS

    per_seed: dict[str, Any] = {}
    repro_exact = False
    try:
        for seed in TRAIN_SEEDS:
            data = make_split(seed)
            runs: dict[str, Any] = {}
            nets: dict[str, PlasticGrowthNet] = {}
            for arm, lr, lr_post in _arm_specs():
                res, net = run_arm(
                    seed,
                    arm,
                    data,
                    lr=lr,
                    lr_post_growth=lr_post,
                    p1_steps=p1_steps,
                    p2_steps=p2_steps,
                    check_every=CHECK_EVERY,
                    deadline=deadline,
                )
                runs[arm] = res
                nets[arm] = net
            if seed == SPOT_CHECK_SEED and not args.smoke:
                _, spot_net = run_arm(
                    seed,
                    "growth_plastic",
                    data,
                    lr=LR_BASE,
                    lr_post_growth=LR_IMMATURE,
                    p1_steps=p1_steps,
                    p2_steps=p2_steps,
                    check_every=CHECK_EVERY,
                    deadline=deadline,
                )
                repro_exact = all(
                    torch.equal(a, b)
                    for a, b in zip(
                        nets["growth_plastic"].parameters(), spot_net.parameters(), strict=True
                    )
                )
            calib = (
                runs["fixed_small"]["p2_acc"] < CAL_FIXED_P2_MAX
                and runs["prealloc_big"]["p2_acc"] >= CAL_PREALLOC_P2_MIN
                and runs["growth_plastic"]["p2_growth_events"] == 1
            )
            benefit = round(
                runs["growth_plastic"]["combined_acc"] - runs["prealloc_big"]["combined_acc"], 6
            )
            retention = round(
                runs["growth_plastic"]["p1_retention_drop"]
                - runs["prealloc_big"]["p1_retention_drop"],
                6,
            )
            per_seed[str(seed)] = {
                "arms": runs,
                "calibration_ok": bool(calib),
                "benefit": benefit,
                "retention_diff": retention,
                "benefit_pass": bool(calib and benefit >= BENEFIT_GATE),
                "retention_pass": bool(calib and retention >= RETENTION_FLOOR),
            }
    except (TimeoutError, FloatingPointError) as exc:
        payload = {
            "schema": SCHEMA,
            "card": CARD,
            "design": design,
            "exit": "engineering_failure",
            "reason": str(exc),
            "per_seed": per_seed,
            "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"exit": "engineering_failure", "reason": str(exc)}, ensure_ascii=False))
        return 1

    calib_ok = all(v["calibration_ok"] for v in per_seed.values())
    benefit_ok = calib_ok and all(v["benefit_pass"] for v in per_seed.values())
    retention_ok = calib_ok and all(v["retention_pass"] for v in per_seed.values())
    if not calib_ok:
        exit_ = "engineering_failure"
    elif benefit_ok and retention_ok:
        exit_ = "net_benefit_positive"
    elif all(not v["benefit_pass"] for v in per_seed.values()) and all(
        v["benefit"] <= -BENEFIT_GATE for v in per_seed.values()
    ):
        exit_ = "net_benefit_negative"
    else:
        exit_ = "parity"

    payload = {
        "schema": SCHEMA,
        "card": CARD,
        "design": design,
        "created_utc": datetime.now(UTC).isoformat(),
        "task_version": TASK_VERSION,
        "train_seeds": list(TRAIN_SEEDS),
        "p1_steps": p1_steps,
        "p2_steps": p2_steps,
        "benefit_gate": BENEFIT_GATE,
        "retention_floor": RETENTION_FLOOR,
        "per_seed": per_seed,
        "repro_spot_check": {"seed": SPOT_CHECK_SEED, "exact": bool(repro_exact)},
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "scripted_used_in_training": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "per_seed": {
                    k: {"benefit": v["benefit"], "retention": v["retention_diff"]}
                    for k, v in per_seed.items()
                },
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ in ("net_benefit_positive", "parity", "net_benefit_negative") else 1


if __name__ == "__main__":
    raise SystemExit(main())
