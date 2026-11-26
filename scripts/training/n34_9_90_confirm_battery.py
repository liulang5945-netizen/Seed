"""N34.9.90 (confirmation): six-axis formal confirmation battery on seeds 20261801-03.

Pre-registered: plans/reference/DESIGN-N34-9-90_20261126.md (D-20261126-658).
Freezes all six verified exploration configs (`ecl`, `unc`, `phy`, `n34_2`, `n34_3`, `nch`) and
runs a single one-time confirmation pass on `20261801, 20261802, 20261803` (`record=True`).
Backup seeds `20261811-13` and sealed `check`/`final` splits remain untouched.

Usage:
  python scripts/training/n34_9_90_confirm_battery.py --smoke
  python scripts/training/n34_9_90_confirm_battery.py
Output:
  reports/n34_9_90_confirm_battery.json (formal mode only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import n34_2_04_growth_benefit_v2d as _n34_2  # noqa: E402
from scripts.training import n34_3_03_state_dependent_growth as _n34_3  # noqa: E402
from scripts.training import nch_2_01_continuous_chinese as _nch  # noqa: E402
from scripts.training import phy_2_01_causal_transfer as _phy  # noqa: E402
from scripts.training.n31_guard import find_design, preflight  # noqa: E402
from taiji import continual_growth_task_v2 as _cg2  # noqa: E402
from taiji import credibility_learning as _ecl  # noqa: E402
from taiji import growth_decision_learning_v2 as _gdl2  # noqa: E402
from taiji import multimodal_physics_env_v2 as _phys2  # noqa: E402
from taiji import uncertainty_learning as _unc  # noqa: E402
from taiji.plastic_growth import PlasticGrowthNet  # noqa: E402

BENEFIT_GATE = _n34_2.BENEFIT_GATE
CAL_FIXED_P2_MAX = _n34_2.CAL_FIXED_P2_MAX
CAL_PREALLOC_P2_MIN = _n34_2.CAL_PREALLOC_P2_MIN
RETENTION_FLOOR = _n34_2.RETENTION_FLOOR
N34_2_LR_BASE = _n34_2.LR_BASE
N34_2_LR_IMM = _n34_2.LR_IMMATURE
N34_2_P1 = _n34_2.P1_STEPS
N34_2_P2 = _n34_2.P2_STEPS
run_n34_2_arm = _n34_2.run_arm

N_EVAL_PER_TYPE = _n34_3.N_EVAL_PER_TYPE
N_TRAIN_PER_TYPE = _n34_3.N_TRAIN_PER_TYPE
N34_3_UPDATES = _n34_3.N_UPDATES
_build_snapshots = _n34_3._build_snapshots
_calibration_from_table = _n34_3._calibration_from_table
_eval_table = _n34_3._eval_table
_evaluate_all = _n34_3._evaluate_all

NCH_EVAL_SEEDS = _nch.EVAL_SEEDS
NCH_HELDOUT_SEEDS = _nch.HELDOUT_SEEDS
NCH_UPDATES = _nch.UPDATES
nch_calibrate = _nch._calibrate
nch_eval_arm = _nch._eval_arm
nch_train_arm = _nch.train_arm

PHY_EVAL_EP = _phy.EVAL_EPISODES
PHY_STEPS = _phy.STEPS
PHY_TRAIN_EP = _phy.TRAIN_EPISODES
phy_calibrate = _phy._calibrate

make_v2_split = _cg2.make_split

ECL_N_ITEMS = _ecl.N_ITEMS
ECL_SOURCES = _ecl.SOURCES
ECL_TRUE_ACC = _ecl.TRUE_ACC
BetaCredibility = _ecl.BetaCredibility
ClaimStream = _ecl.ClaimStream
OracleAccuracy = _ecl.OracleAccuracy
StaticPrior = _ecl.StaticPrior
ecl_eval = _ecl.evaluate

train_actor_critic = _gdl2.train_actor_critic

ablate_barrier_collision_recall = _phys2.ablate_barrier_collision_recall
collect_v2 = _phys2.collect_v2
eval_count_baseline_v2 = _phys2.eval_count_baseline_v2
eval_metrics_v2 = _phys2.eval_metrics_v2
train_net_v2 = _phys2.train_net_v2

UNC_N_EVAL = _unc.N_EVAL
UNC_N_TRAIN = _unc.N_TRAIN
BaseScore = _unc.BaseScore
LearnedConfidence = _unc.LearnedConfidence
TemperatureCalibrator = _unc.TemperatureCalibrator
unc_eval = _unc.evaluate
unc_make_data = _unc.make_data

CARD = "N34.9.90"
CONFIRM_SEEDS = (20261801, 20261802, 20261803)
BACKUP_SEEDS = (20261811, 20261812, 20261813)
SMOKE_SEEDS = (6901, 6902, 6903)
CAP_CPU = 280.0
OUT = ROOT / "reports" / "n34_9_90_confirm_battery.json"


def _run_ecl(seeds: tuple[int, ...]) -> dict[str, Any]:
    t0 = time.process_time()
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        stream = ClaimStream(seed)
        items = stream.generate(ECL_N_ITEMS)
        static = ecl_eval(StaticPrior(), items)
        beta = ecl_eval(BetaCredibility(forgetting=0.995), items)
        oracle = ecl_eval(OracleAccuracy(stream), items)
        no_shift = ClaimStream(seed + 100, shift=False).generate(4000)
        beta_conv = BetaCredibility()
        for src, corr in no_shift:
            beta_conv.update(src, corr)
        conv_err = max(abs(beta_conv.predict(src, 0) - ECL_TRUE_ACC[src]) for src in ECL_SOURCES)
        det = ecl_eval(BetaCredibility(forgetting=0.995), items)["brier"] == beta["brier"]
        per_seed[str(seed)] = {
            "static": static,
            "beta": beta,
            "oracle": oracle,
            "beta_vs_static_brier": round(beta["brier"] - static["brier"], 4),
            "beta_vs_static_post": round(beta["post_abs_error"] - static["post_abs_error"], 4),
            "no_shift_convergence_err": round(conv_err, 4),
            "deterministic": det,
        }
    checks = {
        "g1_brier_better": all(v["beta_vs_static_brier"] <= -0.02 for v in per_seed.values()),
        "g2_post_shift_better": all(v["beta_vs_static_post"] <= -0.05 for v in per_seed.values()),
        "g3_converges": all(v["no_shift_convergence_err"] <= 0.10 for v in per_seed.values()),
        "g4_deterministic": all(v["deterministic"] for v in per_seed.values()),
    }
    exit_ = "confirm_positive" if all(checks.values()) else "confirm_partial"
    return {
        "exit": exit_,
        "checks": checks,
        "per_seed": per_seed,
        "cpu": round(time.process_time() - t0, 2),
    }


def _run_unc(seeds: tuple[int, ...], *, smoke: bool) -> dict[str, Any]:
    t0 = time.process_time()
    per_seed: dict[str, Any] = {}
    spot = seeds[0]
    for seed in seeds:
        x_tr, y_tr, _ = unc_make_data(seed, 400 if smoke else UNC_N_TRAIN)
        x_ev, y_ev, true_p = unc_make_data(seed + 100, 200 if smoke else UNC_N_EVAL)
        steps = 80 if smoke else 800
        learned = LearnedConfidence.train(x_tr, y_tr, seed=seed, steps=steps)
        static = torch.full_like(y_ev, 0.5)
        base = BaseScore()
        cal = TemperatureCalibrator()
        cal.fit(x_tr, y_tr)
        metrics = {
            "static": unc_eval(static, y_ev),
            "base": unc_eval(base.predict(x_ev), y_ev),
            "temperature": unc_eval(cal.predict(x_ev), y_ev),
            "learned": unc_eval(learned.predict(x_ev), y_ev),
            "oracle": unc_eval(true_p, y_ev),
        }
        exact = None
        if seed == spot:
            again = LearnedConfidence.train(x_tr, y_tr, seed=seed, steps=steps)
            exact = bool(torch.equal(learned.w, again.w) and torch.equal(learned.b, again.b))
        per_seed[str(seed)] = {
            "metrics": metrics,
            "learned_vs_static_brier": round(
                metrics["learned"]["brier"] - metrics["static"]["brier"], 4
            ),
            "temperature_vs_base_ece": round(
                metrics["temperature"]["ece"] - metrics["base"]["ece"], 4
            ),
            "weights_exact": exact,
        }
    checks = {
        "g1_learned_beats_static_brier": all(
            v["learned_vs_static_brier"] <= -0.05 for v in per_seed.values()
        ),
        "g2_learned_ece_low": all(
            v["metrics"]["learned"]["ece"] <= 0.10 for v in per_seed.values()
        ),
        "g3_temperature_improves_base": all(
            v["temperature_vs_base_ece"] <= 0.0 for v in per_seed.values()
        ),
        "g4_oracle_control": all(v["metrics"]["oracle"]["ece"] <= 0.05 for v in per_seed.values()),
        "g5_deterministic": all(
            v["weights_exact"] is True for v in per_seed.values() if v["weights_exact"] is not None
        ),
    }
    exit_ = "confirm_positive" if all(checks.values()) else "confirm_partial"
    return {
        "exit": exit_,
        "checks": checks,
        "per_seed": per_seed,
        "cpu": round(time.process_time() - t0, 2),
    }


def _run_phy(seeds: tuple[int, ...], *, smoke: bool) -> dict[str, Any]:
    t0 = time.process_time()
    n_tr = 60 if smoke else PHY_TRAIN_EP
    n_ev = 30 if smoke else PHY_EVAL_EP
    steps = 120 if smoke else PHY_STEPS
    calib = phy_calibrate(n_tr)
    per_seed: dict[str, Any] = {}
    spot = seeds[0]
    for seed in seeds:
        x_tr, y_tr = collect_v2(seed, "ramp_bumper", n_tr)
        x_in, y_in = collect_v2(seed + 100, "ramp_bumper", n_ev)
        x_cr, y_cr = collect_v2(seed + 200, "rough_collision", n_ev)
        net = train_net_v2(x_tr, y_tr, seed=seed, steps=steps)
        m_in = eval_metrics_v2(net, x_in, y_in)
        m_cr = eval_metrics_v2(net, x_cr, y_cr)
        b_in = eval_count_baseline_v2(seed, "ramp_bumper", seed + 100, "ramp_bumper", n_tr, n_ev)
        b_cr = eval_count_baseline_v2(
            seed, "ramp_bumper", seed + 200, "rough_collision", n_tr, n_ev
        )
        abl = ablate_barrier_collision_recall(net, x_cr, y_cr)
        exact = None
        if seed == spot:
            again = train_net_v2(x_tr, y_tr, seed=seed, steps=steps)
            exact = all(
                torch.equal(a, b) for a, b in zip(net.parameters(), again.parameters(), strict=True)
            )
        per_seed[str(seed)] = {
            "in_family": m_in,
            "cross_family": m_cr,
            "count_in_family": b_in,
            "count_cross_family": b_cr,
            "barrier_ablation": abl,
            "weights_exact": exact,
        }
    checks = {
        "c0_calibration": bool(calib["c0_pass"]),
        "g1_in_family": all(
            v["in_family"]["accuracy"] >= 0.90 and v["in_family"]["collision_recall"] >= 0.85
            for v in per_seed.values()
        ),
        "g2_cross_family": all(
            v["cross_family"]["accuracy"] >= 0.80
            and v["cross_family"]["collision_recall"] >= 0.70
            and v["cross_family"]["macro_f1"] >= 0.65
            for v in per_seed.values()
        ),
        "g3_beats_count_both": all(
            v["in_family"]["accuracy"] >= v["count_in_family"]["accuracy"] + 0.05
            and v["cross_family"]["accuracy"] >= v["count_cross_family"]["accuracy"] + 0.10
            for v in per_seed.values()
        ),
        "g4_barrier_ablation": all(
            v["barrier_ablation"]["collision_recall_drop"] >= 0.35 for v in per_seed.values()
        ),
        "g5_deterministic": all(
            v["weights_exact"] is True for v in per_seed.values() if v["weights_exact"] is not None
        ),
    }
    exit_ = "confirm_positive" if all(checks.values()) else "confirm_partial"
    return {
        "exit": exit_,
        "calibration": calib,
        "checks": checks,
        "per_seed": per_seed,
        "cpu": round(time.process_time() - t0, 2),
    }


def _run_n34_2(seeds: tuple[int, ...], deadline: float, *, smoke: bool) -> dict[str, Any]:
    t0 = time.process_time()
    p1_steps = 40 if smoke else N34_2_P1
    p2_steps = 40 if smoke else N34_2_P2
    arm_specs = [
        ("fixed_small", N34_2_LR_BASE, N34_2_LR_BASE),
        ("growth_plastic", N34_2_LR_BASE, N34_2_LR_IMM),
        ("prealloc_big", N34_2_LR_IMM, N34_2_LR_IMM),
    ]
    per_seed: dict[str, Any] = {}
    repro_exact = False
    spot = seeds[0]
    for seed in seeds:
        data = make_v2_split(seed, p2_pairs=3)
        runs: dict[str, Any] = {}
        nets: dict[str, PlasticGrowthNet] = {}
        for arm, lr, lr_post in arm_specs:
            res, net = run_n34_2_arm(
                seed,
                arm,
                data,
                lr=lr,
                lr_post_growth=lr_post,
                p1_steps=p1_steps,
                p2_steps=p2_steps,
                check_every=25,
                deadline=deadline,
            )
            runs[arm] = res
            nets[arm] = net
        if seed == spot:
            _, spot_net = run_n34_2_arm(
                seed,
                "growth_plastic",
                data,
                lr=N34_2_LR_BASE,
                lr_post_growth=N34_2_LR_IMM,
                p1_steps=p1_steps,
                p2_steps=p2_steps,
                check_every=25,
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
            runs["growth_plastic"]["p1_retention_drop"] - runs["prealloc_big"]["p1_retention_drop"],
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
    calib_ok = all(v["calibration_ok"] for v in per_seed.values())
    benefit_ok = calib_ok and all(v["benefit_pass"] for v in per_seed.values())
    retention_ok = calib_ok and all(v["retention_pass"] for v in per_seed.values())
    if not (calib_ok and repro_exact):
        exit_ = "engineering_failure"
    elif benefit_ok and retention_ok:
        exit_ = "confirm_net_benefit_positive"
    elif all(v["benefit"] <= -BENEFIT_GATE for v in per_seed.values()):
        exit_ = "confirm_net_benefit_negative"
    else:
        exit_ = "confirm_parity"
    return {
        "exit": exit_,
        "repro_exact": bool(repro_exact),
        "per_seed": per_seed,
        "cpu": round(time.process_time() - t0, 2),
    }


def _run_n34_3(seeds: tuple[int, ...], deadline: float, *, smoke: bool) -> dict[str, Any]:
    t0 = time.process_time()
    n_tr = 2 if smoke else N_TRAIN_PER_TYPE
    n_ev = 2 if smoke else N_EVAL_PER_TYPE
    n_up = 4 if smoke else N34_3_UPDATES
    per_seed: dict[str, Any] = {}
    spot = seeds[0]
    for seed in seeds:
        tr_snaps = _build_snapshots(seed * 1000, n_tr, deadline)
        ev_snaps = _build_snapshots(seed * 1000 + 500, n_ev, deadline)
        ev_tbl = _eval_table(ev_snaps, deadline)
        calib = _calibration_from_table(ev_snaps, ev_tbl)
        pol = train_actor_critic(tr_snaps, seed=seed, deadline=deadline, n_updates=n_up)
        ev = _evaluate_all(pol, ev_snaps, ev_tbl)
        exact = None
        if seed == spot:
            again = train_actor_critic(tr_snaps, seed=seed, deadline=deadline, n_updates=n_up)
            exact = all(
                torch.equal(a, b) for a, b in zip(pol.parameters(), again.parameters(), strict=True)
            )
        per_seed[str(seed)] = {"calibration": calib, **ev, "weights_exact": exact}
    c0_all = all(v["calibration"]["c0_pass"] for v in per_seed.values())
    g1 = all(
        v["learned"]["mean_reward"]
        >= max(
            v["baselines"]["no_grow"],
            v["baselines"]["grow8"],
            v["baselines"]["grow16"],
            v["baselines"]["grow24"],
        )
        + 0.008
        and v["learned"]["mean_reward"] >= v["baselines"]["scripted_pressure"] - 0.01
        for v in per_seed.values()
    )
    g2 = all(
        (v["learned"]["mean_reward"] - v["baselines"]["no_grow"]) >= 0.025
        and (v["learned"]["mean_reward"] - v["baselines"]["no_grow"])
        >= 0.85 * (v["calibration"]["oracle_per_type"] - v["baselines"]["no_grow"]) - 1e-9
        for v in per_seed.values()
    )
    g3 = all(
        v["learned"]["easy_no_grow_rate"] >= 0.70 and v["learned"]["hard_grow_rate"] >= 0.70
        for v in per_seed.values()
    )
    g4 = all(
        v["weights_exact"] is True for v in per_seed.values() if v["weights_exact"] is not None
    )
    checks = {
        "c0_calibration": c0_all,
        "g1_beats_fixed_and_scripted": g1,
        "g2_oracle_calibrated_gain": g2,
        "g3_state_dependent": g3,
        "g4_deterministic": g4,
    }
    exit_ = "confirm_positive" if all(checks.values()) else "confirm_partial"
    return {
        "exit": exit_,
        "checks": checks,
        "per_seed": per_seed,
        "cpu": round(time.process_time() - t0, 2),
    }


def _run_nch(seeds: tuple[int, ...], deadline: float, *, smoke: bool) -> dict[str, Any]:
    t0 = time.process_time()
    updates = 8 if smoke else NCH_UPDATES
    calib = nch_calibrate()
    per_seed: dict[str, Any] = {}
    spot = seeds[0]
    for seed in seeds:
        pol_rec = nch_train_arm(seed, recurrent=True, updates=updates, deadline=deadline)
        pol_stat = nch_train_arm(seed, recurrent=False, updates=updates, deadline=deadline)
        exact = None
        if seed == spot:
            again = nch_train_arm(seed, recurrent=True, updates=updates, deadline=deadline)
            exact = all(
                torch.equal(a, b)
                for a, b in zip(pol_rec.parameters(), again.parameters(), strict=True)
            )
        per_seed[str(seed)] = {
            "gru_recurrent": {
                "train": nch_eval_arm(pol_rec, NCH_EVAL_SEEDS, heldout=False),
                "heldout": nch_eval_arm(pol_rec, NCH_HELDOUT_SEEDS, heldout=True),
            },
            "stateless_mlp": {
                "train": nch_eval_arm(pol_stat, NCH_EVAL_SEEDS, heldout=False),
                "heldout": nch_eval_arm(pol_stat, NCH_HELDOUT_SEEDS, heldout=True),
            },
            "weights_exact": exact,
        }
    checks = {
        "c0_calibration": bool(calib["c0_pass"]),
        "g1_train_recurrent": all(
            v["gru_recurrent"]["train"]["mean_step_acc"] >= 0.85
            and v["gru_recurrent"]["train"]["coref_acc"] >= 0.80
            for v in per_seed.values()
        ),
        "g2_heldout_recurrent": all(
            v["gru_recurrent"]["heldout"]["mean_step_acc"] >= 0.80
            and v["gru_recurrent"]["heldout"]["coref_acc"] >= 0.75
            for v in per_seed.values()
        ),
        "g3_memory_necessity": all(
            v["gru_recurrent"]["train"]["mean_step_acc"]
            >= v["stateless_mlp"]["train"]["mean_step_acc"] + 0.15
            and v["gru_recurrent"]["train"]["coref_acc"]
            >= v["stateless_mlp"]["train"]["coref_acc"] + 0.35
            and v["stateless_mlp"]["train"]["coref_acc"] <= 0.45
            for v in per_seed.values()
        ),
        "g4_deterministic": all(
            v["weights_exact"] is True for v in per_seed.values() if v["weights_exact"] is not None
        ),
    }
    exit_ = "confirm_positive" if all(checks.values()) else "confirm_partial"
    return {
        "exit": exit_,
        "calibration": calib,
        "checks": checks,
        "per_seed": per_seed,
        "cpu": round(time.process_time() - t0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        design_file = find_design(CARD).name
        seeds = SMOKE_SEEDS
    else:
        design_file = preflight(CARD, list(CONFIRM_SEEDS), record=True).name
        seeds = CONFIRM_SEEDS

    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU

    axes: dict[str, Any] = {}
    try:
        axes["ecl"] = _run_ecl(seeds)
        axes["unc"] = _run_unc(seeds, smoke=args.smoke)
        axes["phy"] = _run_phy(seeds, smoke=args.smoke)
        axes["n34_2"] = _run_n34_2(seeds, deadline, smoke=args.smoke)
        axes["n34_3"] = _run_n34_3(seeds, deadline, smoke=args.smoke)
        axes["nch"] = _run_nch(seeds, deadline, smoke=args.smoke)
        exit_ = (
            "smoke"
            if args.smoke
            else (
                "confirm_all_passed"
                if all(
                    axes[k]["exit"]
                    in ("confirm_positive", "confirm_parity", "confirm_net_benefit_positive")
                    for k in axes
                )
                else "confirm_mixed"
            )
        )
    except (TimeoutError, FloatingPointError, RuntimeError) as exc:
        exit_ = "engineering_failure"
        axes["error"] = str(exc)

    payload = {
        "schema": "taiji-n34-9-90-v1",
        "card": CARD,
        "design": design_file,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "seeds": list(seeds),
        "axes": axes,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0 if args.smoke else len(CONFIRM_SEEDS),
            "confirmation_seeds": [] if args.smoke else list(CONFIRM_SEEDS),
            "backup_seeds_unread": list(BACKUP_SEEDS),
            "oracle_used_by_subject": False,
        },
    }
    if not args.smoke:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "exit": exit_,
        "axis_exits": {k: v.get("exit") for k, v in axes.items() if isinstance(v, dict)},
        "axis_cpu": {k: v.get("cpu") for k, v in axes.items() if isinstance(v, dict)},
        "cpu": payload["cost"]["cpu"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if exit_ != "engineering_failure" else 3


if __name__ == "__main__":
    raise SystemExit(main())
