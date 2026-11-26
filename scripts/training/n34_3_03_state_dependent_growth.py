"""N34.3.03 (exploration): state-dependent learned growth decision v2 (actor-critic + fork).

Design: plans/reference/DESIGN-N34-3-03_20261126.md (D-20261126-654).
Fixes v1 defects: snapshot-fork episodes (no 2x probe waste), selective plasticity after
growth, and GrowthActorCritic with learned value baseline + batch-standardised advantages +
entropy regularisation. Exploration seeds 6611-6613; zero confirmation seed / check / final.

Usage: python scripts/training/n34_3_03_state_dependent_growth.py [--smoke]
Output: reports/n34_3_03_state_dependent_growth.json
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

from taiji.growth_decision_learning_v2 import (  # noqa: E402
    ACTIONS,
    GrowthActorCritic,
    WarmupSnapshot,
    finish_from_warmup,
    greedy_action_v2,
    prepare_warmup,
    train_actor_critic,
)

CARD = "N34.3.03"
DESIGN = "DESIGN-N34-3-03_20261126.md"
TRAIN_SEEDS = (6611, 6612, 6613)
SPOT_CHECK_SEED = 6611
N_TRAIN_PER_TYPE = 6
N_EVAL_PER_TYPE = 7
N_UPDATES = 60
SMOKE_TASKS_PER_TYPE = 2
SMOKE_UPDATES = 12
CAP_CPU = 120.0
OUT = ROOT / "reports" / "n34_3_03_state_dependent_growth.json"


def _build_snapshots(base_seed: int, n_per_type: int, deadline: float) -> list[WarmupSnapshot]:
    snaps: list[WarmupSnapshot] = []
    for i in range(n_per_type):
        snaps.append(prepare_warmup(base_seed + 2 * i, hard=False, deadline=deadline))
        snaps.append(prepare_warmup(base_seed + 2 * i + 1, hard=True, deadline=deadline))
    return snaps


def _eval_table(snaps: list[WarmupSnapshot], deadline: float) -> list[dict[str, dict[str, Any]]]:
    table: list[dict[str, dict[str, Any]]] = []
    for s in snaps:
        row = {act: finish_from_warmup(s, act, deadline) for act in ACTIONS}
        table.append(row)
    return table


def _calibration_from_table(
    snaps: list[WarmupSnapshot], table: list[dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    easy_rows = [row for s, row in zip(snaps, table, strict=True) if not s.hard]
    hard_rows = [row for s, row in zip(snaps, table, strict=True) if s.hard]
    easy_no_grow = sum(r["no_grow"]["reward"] for r in easy_rows) / len(easy_rows)
    easy_grow16 = sum(r["grow16"]["reward"] for r in easy_rows) / len(easy_rows)
    easy_grow24 = sum(r["grow24"]["reward"] for r in easy_rows) / len(easy_rows)
    hard_no_grow = sum(r["no_grow"]["reward"] for r in hard_rows) / len(hard_rows)
    hard_best_grow = max(
        sum(r[a]["reward"] for r in hard_rows) / len(hard_rows)
        for a in ("grow8", "grow16", "grow24")
    )
    fixed_means = {a: sum(r[a]["reward"] for r in table) / len(table) for a in ACTIONS}
    best_fixed = max(fixed_means.values())
    oracle_per_type = 0.5 * (easy_no_grow + hard_best_grow)
    c0_pass = (
        easy_no_grow >= max(easy_grow16, easy_grow24) + 0.015
        and hard_best_grow >= hard_no_grow + 0.06
        and oracle_per_type >= best_fixed + 0.012
    )
    return {
        "c0_pass": bool(c0_pass),
        "easy_no_grow": round(easy_no_grow, 4),
        "easy_max_high_grow": round(max(easy_grow16, easy_grow24), 4),
        "hard_no_grow": round(hard_no_grow, 4),
        "hard_best_grow": round(hard_best_grow, 4),
        "best_fixed": round(best_fixed, 4),
        "oracle_per_type": round(oracle_per_type, 4),
    }


def _evaluate_all(
    policy: GrowthActorCritic,
    snaps: list[WarmupSnapshot],
    table: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    baselines: dict[str, float] = {}
    for act in ACTIONS:
        baselines[act] = round(sum(r[act]["reward"] for r in table) / len(table), 4)
    scripted_rews = []
    for s, r in zip(snaps, table, strict=True):
        act = "grow24" if s.val_err > 0.15 else "no_grow"
        scripted_rews.append(r[act]["reward"])
    baselines["scripted_pressure"] = round(sum(scripted_rews) / len(scripted_rews), 4)

    learned_rews = []
    easy_acts: list[int] = []
    hard_acts: list[int] = []
    for s, r in zip(snaps, table, strict=True):
        a = greedy_action_v2(policy, s)
        learned_rews.append(r[ACTIONS[a]]["reward"])
        (hard_acts if s.hard else easy_acts).append(a)
    easy_no_grow_rate = sum(1 for a in easy_acts if a == 0) / max(1, len(easy_acts))
    hard_grow_rate = sum(1 for a in hard_acts if a in (1, 2, 3)) / max(1, len(hard_acts))
    return {
        "learned": {
            "mean_reward": round(sum(learned_rews) / len(learned_rews), 4),
            "easy_actions": easy_acts,
            "hard_actions": hard_acts,
            "easy_no_grow_rate": round(easy_no_grow_rate, 4),
            "hard_grow_rate": round(hard_grow_rate, 4),
        },
        "baselines": baselines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    n_train = SMOKE_TASKS_PER_TYPE if args.smoke else N_TRAIN_PER_TYPE
    n_eval = SMOKE_TASKS_PER_TYPE if args.smoke else N_EVAL_PER_TYPE
    n_updates = SMOKE_UPDATES if args.smoke else N_UPDATES

    per_seed: dict[str, Any] = {}
    checks: dict[str, bool] = {}

    try:
        for seed in TRAIN_SEEDS:
            t0 = time.process_time()
            train_snaps = _build_snapshots(seed * 1000, n_train, deadline)
            eval_snaps = _build_snapshots(seed * 1000 + 500, n_eval, deadline)
            eval_tbl = _eval_table(eval_snaps, deadline)
            calib = _calibration_from_table(eval_snaps, eval_tbl)
            if not calib["c0_pass"] and not args.smoke:
                per_seed[str(seed)] = {"calibration": calib, "c0_pass": False}
                raise RuntimeError(f"C0 calibration failed on seed={seed}: {calib}")

            pol = train_actor_critic(train_snaps, seed=seed, deadline=deadline, n_updates=n_updates)
            ev = _evaluate_all(pol, eval_snaps, eval_tbl)

            exact = None
            if seed == SPOT_CHECK_SEED and not args.smoke:
                again = train_actor_critic(
                    train_snaps, seed=seed, deadline=deadline, n_updates=n_updates
                )
                exact = all(
                    torch.equal(a, b)
                    for a, b in zip(pol.parameters(), again.parameters(), strict=True)
                )

            per_seed[str(seed)] = {
                "calibration": calib,
                **ev,
                "weights_exact": exact,
                "cpu": round(time.process_time() - t0, 2),
            }

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
            v["learned"]["mean_reward"] >= v["baselines"]["no_grow"] + 0.04
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
            "g2_beats_no_grow": g2,
            "g3_state_dependent": g3,
            "g4_deterministic": g4,
        }
        if args.smoke:
            exit_ = "smoke"
        elif all(checks.values()):
            exit_ = "growth_decision_v2_positive"
        else:
            exit_ = "growth_decision_v2_partial"
    except (TimeoutError, FloatingPointError, RuntimeError) as exc:
        exit_ = "engineering_failure"
        per_seed["error"] = str(exc)

    payload = {
        "schema": "taiji-n34-3-03-v1",
        "card": CARD,
        "design": DESIGN,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "train_seeds": list(TRAIN_SEEDS),
        "n_train_per_type": n_train,
        "n_eval_per_type": n_eval,
        "n_updates": n_updates,
        "per_seed": per_seed,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "scripted_rule_is_control_only": True,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit": exit_, "checks": checks, "per_seed": per_seed, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if exit_ in ("growth_decision_v2_positive", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
