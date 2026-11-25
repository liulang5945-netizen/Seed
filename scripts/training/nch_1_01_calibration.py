"""NCH.1.01 (zero training): calibration of the closed-vocabulary Chinese continuous-task harness.

Design: plans/reference/DESIGN-NCH-1-01_20261126.md (D-20261126-616).
Runs the scripted oracle (control), uniform random and 6 constant action policies on train and
held-out slot combinations. No model is trained; no confirmation seed / check / final is read.

Usage: python scripts/training/nch_1_01_calibration.py
Output: reports/nch_1_01_calibration.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.chinese_task_env import (  # noqa: E402
    ACTIONS,
    TASK_VERSION,
    ChineseTaskEnv,
    eval_policy,
    random_policy,
    scripted_oracle_policy,
)

CARD = "NCH.1.01"
TRAIN_SEEDS = list(range(4100, 4140))
HELDOUT_SEEDS = list(range(4140, 4180))
CAP_CPU = 5.0
OUT = ROOT / "reports" / "nch_1_01_calibration.json"


def constant_policy(action_id: int):
    return lambda env: action_id


def main() -> int:
    torch.set_num_threads(1)
    started = time.process_time()
    train_oracle = eval_policy(scripted_oracle_policy, TRAIN_SEEDS)
    held_oracle = eval_policy(scripted_oracle_policy, HELDOUT_SEEDS, heldout=True)
    train_random = eval_policy(random_policy, TRAIN_SEEDS)
    held_random = eval_policy(random_policy, HELDOUT_SEEDS, heldout=True)
    constants = {}
    for a in ACTIONS:
        aid = ACTIONS.index(a)
        c = eval_policy(constant_policy(aid), TRAIN_SEEDS)
        constants[a] = {
            "mean_step_acc": c["mean_step_acc"],
            "episode_success": c["episode_success"],
        }
    best_constant = max(v["mean_step_acc"] for v in constants.values())

    checks = {
        "oracle_train_high": train_oracle["mean_step_acc"] >= 0.99,
        "oracle_heldout_high": held_oracle["mean_step_acc"] >= 0.99,
        "random_low": train_random["mean_step_acc"] <= 0.30,
        "constants_below_random_plus_margin": best_constant <= train_random["mean_step_acc"] + 0.10,
        "heldout_episodes_use_heldout_combos": True,
        "no_task_id": set(ChineseTaskEnv(1).reset()) == {"tokens", "step", "turns", "pending"},
    }
    # explicitly verify the held-out split draws held-out combos
    from taiji.chinese_task_env import HELDOUT_COMBOS

    for seed in HELDOUT_SEEDS[:5]:
        env = ChineseTaskEnv(seed, heldout=True)
        env.reset()
        for turn in env.plan:
            combo_ok = any(
                (len(c) == 2 and turn["colour"] == c[0] and turn["place"] in (c[1], "原位"))
                or (
                    len(c) == 3
                    and turn["object"] == c[0]
                    and turn["colour"] == c[1]
                    and turn["place"] in (c[2], "原位")
                )
                for c in HELDOUT_COMBOS
            )
            if not combo_ok:
                checks["heldout_episodes_use_heldout_combos"] = False
                break

    exit_ = "env_ready" if all(checks.values()) else "env_failure"
    payload = {
        "schema": "taiji-nch-1-01-v1",
        "card": CARD,
        "design": "DESIGN-NCH-1-01_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "task_version": TASK_VERSION,
        "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1], len(TRAIN_SEEDS)],
        "heldout_seeds": [HELDOUT_SEEDS[0], HELDOUT_SEEDS[-1], len(HELDOUT_SEEDS)],
        "oracle_train": train_oracle,
        "oracle_heldout": held_oracle,
        "random_train": train_random,
        "random_heldout": held_random,
        "constants": constants,
        "best_constant_step_acc": best_constant,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "trained_policies": 0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "checks": checks,
                "oracle": train_oracle["mean_step_acc"],
                "heldout": held_oracle["mean_step_acc"],
                "random": train_random["mean_step_acc"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "env_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
