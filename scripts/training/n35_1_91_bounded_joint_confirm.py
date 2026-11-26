"""N35.1.91 (bounded confirmation): Option B1 joint confirmation on backup seeds 20261911-13.

Pre-registered: plans/reference/DESIGN-N35-1-91_20261126.md (D-20261126-679).
Runs bounded closed-vocab & multi-module joint confirmation (`long_run` 5-module 100-session
durability + `n34_3` state-dependent growth decision + `n33_2` cross-session persistent state
use + `n33_4` in-model recursive development) on unread backup seeds 20261911-13.
Open-vocabulary natural Chinese is explicitly `deferred_post_N35`. Zero check/final reads.

Usage:
  python scripts/training/n35_1_91_bounded_joint_confirm.py --smoke
  python scripts/training/n35_1_91_bounded_joint_confirm.py
Output:
  reports/n35_1_91_bounded_joint_confirm.json (formal mode only)
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

from scripts.training.n31_guard import find_design, preflight  # noqa: E402
from scripts.training.n33_2_02_cross_session_use import (  # noqa: E402
    GATE_COUNTERFACTUAL_DROP,
    GATE_RESOLVE_RATE,
    GATE_SCRIPTED_RATIO,
    GATE_STATELESS_GAP,
    SMOKE_EPISODES,
    greedy,
    train_policy_det,
)
from scripts.training.n33_2_90_confirm import (  # noqa: E402
    CONFIRM_EVAL_SEEDS as N33_2_EVAL_SEEDS,
)
from scripts.training.n33_2_90_confirm import (  # noqa: E402
    EPISODES as N33_2_EPISODES,
)
from scripts.training.n33_2_90_confirm import calibrate_confirm_env  # noqa: E402
from scripts.training.n33_4_01_recursive_env_check import (  # noqa: E402
    EVAL_SEEDS as N33_4_EVAL_SEEDS,
)
from scripts.training.n33_4_01_recursive_env_check import calibrate_recursive_env  # noqa: E402
from scripts.training.n34_5_01_long_run_joint import _eval_gates  # noqa: E402
from scripts.training.n34_9_90_confirm_battery import _run_n34_3  # noqa: E402
from taiji.cross_session_env import evaluate_policy_with_store  # noqa: E402
from taiji.long_run_trained_subject import RunConfig, run_long_run_subject  # noqa: E402
from taiji.recursive_dev import (  # noqa: E402
    greedy_recursive_policy,
    train_consequence_model,
    train_policy_in_model,
)
from taiji.recursive_dev_env import (  # noqa: E402
    RecursiveDevEnv,
    collect_logged_experience,
    evaluate_recursive_with_store,
    seed_from,
)

CARD = "N35.1.91"
CONFIRM_SEEDS = (20261911, 20261912, 20261913)
SMOKE_SEEDS = (6931, 6932, 6933)
CAP_CPU = 450.0
OUT = ROOT / "reports" / "n35_1_91_bounded_joint_confirm.json"


def _run_n33_2(seeds: tuple[int, ...], deadline: float, *, smoke: bool) -> dict[str, Any]:
    t0 = time.process_time()
    episodes = SMOKE_EPISODES if smoke else N33_2_EPISODES
    calib = calibrate_confirm_env(N33_2_EVAL_SEEDS)
    if not calib["c0_pass"]:
        return {"exit": "engineering_failure", "calibration": calib, "per_seed": {}}
    scripted_mean = calib["scripted_normal"]["mean_return"]
    per_seed: dict[str, Any] = {}
    repro_exact = False
    store_ok = True
    spot = seeds[0]
    for seed in seeds:
        pol_pers = train_policy_det(seed, stateless=False, episodes=episodes, deadline=deadline)
        pol_stat = train_policy_det(seed, stateless=True, episodes=episodes, deadline=deadline)
        if seed == spot:
            pol_pers2 = train_policy_det(
                seed, stateless=False, episodes=episodes, deadline=deadline
            )
            repro_exact = all(
                torch.equal(a, b)
                for a, b in zip(pol_pers.parameters(), pol_pers2.parameters(), strict=True)
            )
        ev_norm = evaluate_policy_with_store(greedy(pol_pers), N33_2_EVAL_SEEDS, mode="normal")
        ev_zero = evaluate_policy_with_store(greedy(pol_pers), N33_2_EVAL_SEEDS, mode="zero_cross")
        ev_swap = evaluate_policy_with_store(greedy(pol_pers), N33_2_EVAL_SEEDS, mode="swap_cross")
        ev_stat = evaluate_policy_with_store(greedy(pol_stat), N33_2_EVAL_SEEDS, mode="zero_cross")
        store_ok = (
            store_ok
            and ev_norm["store_roundtrip_exact"]
            and ev_norm["post_s2_persist_exact"]
            and ev_zero["store_roundtrip_exact"]
            and ev_swap["store_roundtrip_exact"]
            and ev_stat["store_roundtrip_exact"]
        )
        gap_stat = ev_norm["mean_return"] - ev_stat["mean_return"]
        drop_zero = ev_norm["mean_return"] - ev_zero["mean_return"]
        drop_swap = ev_norm["mean_return"] - ev_swap["mean_return"]
        g1 = (
            gap_stat >= GATE_STATELESS_GAP
            and ev_norm["mean_return"] >= GATE_SCRIPTED_RATIO * scripted_mean
            and ev_norm["resolve_rate"] >= GATE_RESOLVE_RATE
        )
        g2 = drop_zero >= GATE_COUNTERFACTUAL_DROP and drop_swap >= GATE_COUNTERFACTUAL_DROP
        per_seed[str(seed)] = {
            "mean_return": ev_norm["mean_return"],
            "resolve_rate": ev_norm["resolve_rate"],
            "stateless_return": ev_stat["mean_return"],
            "gap_stateless": round(gap_stat, 4),
            "drop_zero": round(drop_zero, 4),
            "drop_swap": round(drop_swap, 4),
            "g1": bool(g1),
            "g2": bool(g2),
        }
    passed = bool(repro_exact and store_ok and all(v["g1"] and v["g2"] for v in per_seed.values()))
    return {
        "exit": "confirm_positive" if passed else "confirm_partial",
        "repro_exact": bool(repro_exact),
        "store_ok": bool(store_ok),
        "per_seed": per_seed,
        "cpu": round(time.process_time() - t0, 2),
    }


def _run_n33_4(seeds: tuple[int, ...], deadline: float, *, smoke: bool) -> dict[str, Any]:
    t0 = time.process_time()
    cm_steps = 30 if smoke else 6000
    pol_eps = 32 if smoke else 1600
    calib = calibrate_recursive_env(N33_4_EVAL_SEEDS)
    if not calib["c0_pass"]:
        return {"exit": "engineering_failure", "calibration": calib, "per_seed": {}}
    oracle_gain = calib["oracle"]["mean_net_gain"]
    per_seed: dict[str, Any] = {}
    repro_exact = False
    store_exact = True
    spot = seeds[0]
    for seed in seeds:
        log_tr_seeds = [seed * 1000 + i for i in range(1500)]
        log_va_seeds = [seed * 1000 + 2000 + i for i in range(100)]
        tr_exp = collect_logged_experience(log_tr_seeds, seed_from("N35.1.91", seed, "log-tr"))
        va_exp = collect_logged_experience(log_va_seeds, seed_from("N35.1.91", seed, "log-va"))
        init_states = torch.stack([RecursiveDevEnv(s).reset() for s in log_tr_seeds])
        arms: dict[str, Any] = {}
        full_cm = full_pol = None
        for mode in ("full", "masked", "permuted"):
            cm, cm_stats = train_consequence_model(
                tr_exp,
                va_exp,
                seed,
                mode=mode,
                steps=cm_steps,
                outcome_weight_scale=True,
                hidden=256,
                deadline=deadline,
            )
            pol = train_policy_in_model(
                cm, init_states, seed, mode=mode, episodes=pol_eps, deadline=deadline
            )
            ev = evaluate_recursive_with_store(
                greedy_recursive_policy(pol, cm, mask_preds=(mode == "masked")),
                N33_4_EVAL_SEEDS,
            )
            store_exact = store_exact and bool(ev["store_roundtrip_exact"])
            arms[mode] = {"cmodel": cm_stats, "eval": ev}
            if mode == "full":
                full_cm, full_pol = cm, pol
        if seed == spot and full_cm is not None and full_pol is not None:
            cm2, _ = train_consequence_model(
                tr_exp,
                va_exp,
                seed,
                mode="full",
                steps=cm_steps,
                outcome_weight_scale=True,
                hidden=256,
                deadline=deadline,
            )
            pol2 = train_policy_in_model(
                cm2, init_states, seed, mode="full", episodes=pol_eps, deadline=deadline
            )
            repro_exact = all(
                torch.equal(a, b)
                for a, b in zip(full_cm.parameters(), cm2.parameters(), strict=True)
            ) and all(
                torch.equal(a, b)
                for a, b in zip(full_pol.parameters(), pol2.parameters(), strict=True)
            )
        fg = arms["full"]["eval"]["mean_net_gain"]
        mg = arms["masked"]["eval"]["mean_net_gain"]
        pg = arms["permuted"]["eval"]["mean_net_gain"]
        g1 = (
            arms["full"]["cmodel"]["delta_u_r2"] >= 0.85
            and arms["permuted"]["cmodel"]["delta_u_r2"] <= 0.10
        )
        g2 = (
            fg >= 0.15
            and fg >= 0.80 * oracle_gain
            and arms["full"]["eval"]["mean_improving_steps"] >= 3.0
        )
        g3 = (fg - mg) >= 0.02 and (fg - pg) >= 0.02
        per_seed[str(seed)] = {
            "full_r2": arms["full"]["cmodel"]["delta_u_r2"],
            "permuted_r2": arms["permuted"]["cmodel"]["delta_u_r2"],
            "full_gain": fg,
            "masked_gain": mg,
            "permuted_gain": pg,
            "improving_steps": arms["full"]["eval"]["mean_improving_steps"],
            "g1": bool(g1),
            "g2": bool(g2),
            "g3": bool(g3),
        }
    passed = bool(
        repro_exact
        and store_exact
        and all(v["g1"] and v["g2"] and v["g3"] for v in per_seed.values())
    )
    return {
        "exit": "confirm_positive" if passed else "confirm_partial",
        "repro_exact": bool(repro_exact),
        "store_exact": bool(store_exact),
        "per_seed": per_seed,
        "cpu": round(time.process_time() - t0, 2),
    }


def _r0_b1_checks(
    long_run: dict[str, Any],
    spot_match: bool,
    n34_3: dict[str, Any],
    n33_2: dict[str, Any],
    n33_4: dict[str, Any],
    *,
    n_sessions: int,
    n_events: int,
) -> dict[str, bool]:
    base = _eval_gates(long_run, spot_match, n_sessions=n_sessions, n_events=n_events)
    r2 = base["g3_physics_causal_endurance"] and all(
        v["physics"]["cross_family"]["collision_recall"] >= 0.95
        and v["physics"]["cross_family"]["collision_recall"]
        - v["physics"]["barrier_ablation"]["collision_recall_drop"]
        <= 0.02
        for v in long_run.values()
    )
    r3 = base["g4_plastic_growth_retention"] and all(
        v["growth"]["p1_retention_drop"] <= 0.02 and v["growth"]["p2_acc"] >= 0.95
        for v in long_run.values()
    )
    r5 = base["g6_coupled_uncertainty_calibration"] and all(
        v["uncertainty"]["coupled_ece"] <= 0.06 for v in long_run.values()
    )
    return {
        "r0_b1_1_closed_chinese": base["g2_chinese_endurance"],
        "r0_b1_2_physics_cross_family": r2,
        "r0_b1_3_growth_retention": r3,
        "r0_b1_4_credibility": base["g5_credibility_drift_tracking"],
        "r0_b1_5_uncertainty": r5,
        "r0_b1_6_durability_repro": base["g1_durability_and_recovery"]
        and base["g7_reproducibility"],
        "r0_b1_7_growth_decision": n34_3.get("exit") == "confirm_positive",
        "r0_b1_8_cross_session": n33_2.get("exit") == "confirm_positive",
        "r0_b1_9_recursive_and_discipline": n33_4.get("exit") == "confirm_positive",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        design_file = find_design(CARD).name
        seeds = SMOKE_SEEDS
        cfg = RunConfig(
            n_sessions=12,
            events_per_session=20,
            crash_sessions=(3, 8),
            p1_end_session=4,
            p2_end_session=10,
            dormancy_session=11,
            ch_train_end_session=6,
            ch_updates_per_session=2,
            phy_steps_early=10,
            phy_steps_mid=10,
            grow_p1_steps=10,
            grow_p2_steps=10,
            grow_maint_steps=2,
        )
    else:
        design_file = preflight(CARD, list(CONFIRM_SEEDS), record=True).name
        seeds = CONFIRM_SEEDS
        cfg = RunConfig()

    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    spot = seeds[0]
    long_run: dict[str, Any] = {}
    spot_match = False
    n34_3: dict[str, Any] = {}
    n33_2: dict[str, Any] = {}
    n33_4: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    error_msg: str | None = None

    try:
        for seed in seeds:
            with tempfile.TemporaryDirectory() as tmp1:
                res = run_long_run_subject(tmp1, seed, deadline, cfg=cfg)
            if seed == spot and not args.smoke:
                with tempfile.TemporaryDirectory() as tmp2:
                    again = run_long_run_subject(tmp2, seed, deadline, cfg=cfg)
                spot_match = (
                    res["durability"]["chain_head"] == again["durability"]["chain_head"]
                    and res["durability"]["record_digest"] == again["durability"]["record_digest"]
                )
            elif seed == spot and args.smoke:
                spot_match = True
            long_run[str(seed)] = res

        n34_3 = _run_n34_3(seeds, deadline, smoke=args.smoke)
        n33_2 = _run_n33_2(seeds, deadline, smoke=args.smoke)
        n33_4 = _run_n33_4(seeds, deadline, smoke=args.smoke)
        checks = _r0_b1_checks(
            long_run,
            spot_match,
            n34_3,
            n33_2,
            n33_4,
            n_sessions=cfg.n_sessions,
            n_events=cfg.n_sessions * cfg.events_per_session,
        )
        if args.smoke:
            exit_ = "smoke"
        elif all(checks.values()):
            exit_ = "confirm_positive"
        else:
            exit_ = "confirm_partial"
    except (TimeoutError, FloatingPointError, RuntimeError, KeyError) as exc:
        exit_ = "engineering_failure"
        error_msg = f"{type(exc).__name__}: {exc}"

    cpu = round(time.process_time() - started, 2)
    payload: dict[str, Any] = {
        "schema": "taiji-n35-1-91-v1",
        "card": CARD,
        "design": design_file,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "seeds": list(seeds),
        "long_run": long_run,
        "n34_3": n34_3,
        "n33_2": n33_2,
        "n33_4": n33_4,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": cpu, "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0 if args.smoke else len(CONFIRM_SEEDS),
            "confirmation_seeds": [] if args.smoke else list(CONFIRM_SEEDS),
            "open_chinese_scope": "deferred_post_N35",
            "oracle_used_by_subject": False,
        },
    }
    if error_msg is not None:
        payload["error"] = error_msg
    if not args.smoke:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"exit": exit_, "checks": checks, "cpu": cpu}, ensure_ascii=False))
    return 0 if exit_ in ("confirm_positive", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
