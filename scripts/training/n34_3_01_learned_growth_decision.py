"""N34.3.01 (exploration): learned growth decisions vs scripted pressure rule.

Design: plans/reference/DESIGN-N34-3-01_20261126.md (D-20261126-646).
The decision whether/how much to grow is made by a learned bandit policy (subjects' own policy),
rewarded by final combined accuracy minus a real per-unit cost. Controls: fixed growth sizes, no
growth, and the scripted pressure rule (grow8 if validation error > 0.15). Exploration seeds
6501-6503; no confirmation seed, check or final is read.

Usage: python scripts/training/n34_3_01_learned_growth_decision.py
Output: reports/n34_3_01_growth_decision.json
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

from taiji.growth_decision_learning import (  # noqa: E402
    ACTIONS,
    GrowthDecisionPolicy,
    greedy_action,
    run_episode,
    train_policy,
)

CARD = "N34.3.01"
TRAIN_SEEDS = (6501, 6502, 6503)
TRAIN_EPISODES = 50
EVAL_TASKS = 20
CAP_CPU = 300.0
OUT = ROOT / "reports" / "n34_3_01_growth_decision.json"


def _eval_policy(policy: GrowthDecisionPolicy, tasks: list[int], deadline: float) -> dict:
    rewards = []
    actions = []
    for task in tasks:
        probe = run_episode(task, "no_grow", deadline)
        a = greedy_action(policy, probe)
        res = run_episode(task, ACTIONS[a], deadline)
        rewards.append(res["reward"])
        actions.append(a)
    return {
        "mean_reward": round(sum(rewards) / len(rewards), 4),
        "actions": actions,
        "unique_actions": len(set(actions)),
    }


def _eval_fixed(action: str, tasks: list[int], deadline: float) -> float:
    rewards = [run_episode(task, action, deadline)["reward"] for task in tasks]
    return round(sum(rewards) / len(rewards), 4)


def _eval_scripted(tasks: list[int], deadline: float) -> float:
    rewards = []
    for task in tasks:
        probe = run_episode(task, "no_grow", deadline)
        action = "grow8" if probe["val_err"] > 0.15 else "no_grow"
        rewards.append(run_episode(task, action, deadline)["reward"])
    return round(sum(rewards) / len(rewards), 4)


def main() -> int:
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    per_seed: dict[str, dict] = {}
    for seed in TRAIN_SEEDS:
        task_train = [seed * 1000 + i for i in range(40)]
        task_eval = [seed * 1000 + 500 + i for i in range(EVAL_TASKS)]
        t0 = time.process_time()
        trained = train_policy(task_train, seed=seed, episodes=TRAIN_EPISODES, deadline=deadline)
        learned = _eval_policy(trained.policy, task_eval, deadline)
        baselines = {
            "no_grow": _eval_fixed("no_grow", task_eval, deadline),
            "grow8": _eval_fixed("grow8", task_eval, deadline),
            "grow16": _eval_fixed("grow16", task_eval, deadline),
            "grow24": _eval_fixed("grow24", task_eval, deadline),
            "scripted_pressure": _eval_scripted(task_eval, deadline),
        }
        exact = None
        if seed == TRAIN_SEEDS[0]:
            again = train_policy(task_train, seed=seed, episodes=TRAIN_EPISODES, deadline=deadline)
            exact = all(
                torch.equal(a, b)
                for a, b in zip(trained.policy.parameters(), again.policy.parameters(), strict=True)
            )
        per_seed[str(seed)] = {
            "learned": learned,
            "baselines": baselines,
            "train_reward_last20": round(sum(trained.rewards[-20:]) / 20, 4),
            "weights_exact": exact,
            "cpu": round(time.process_time() - t0, 2),
        }
    checks = {
        "g1_beats_scripted": all(
            v["learned"]["mean_reward"] >= v["baselines"]["scripted_pressure"] - 0.01
            for v in per_seed.values()
        ),
        "g2_beats_no_grow": all(
            v["learned"]["mean_reward"] >= v["baselines"]["no_grow"] + 0.05
            for v in per_seed.values()
        ),
        "g3_action_diversity": all(v["learned"]["unique_actions"] >= 2 for v in per_seed.values()),
        "g4_deterministic": all(
            v["weights_exact"] is True for v in per_seed.values() if v["weights_exact"] is not None
        ),
    }
    exit_ = (
        "growth_decision_learned_positive"
        if all(checks.values())
        else "growth_decision_learned_partial"
    )
    payload = {
        "schema": "taiji-n34-3-01-v1",
        "card": CARD,
        "design": "DESIGN-N34-3-01_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "train_seeds": list(TRAIN_SEEDS),
        "train_episodes": TRAIN_EPISODES,
        "eval_tasks": EVAL_TASKS,
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
    return 0 if exit_ == "growth_decision_learned_positive" else 3


if __name__ == "__main__":
    raise SystemExit(main())
