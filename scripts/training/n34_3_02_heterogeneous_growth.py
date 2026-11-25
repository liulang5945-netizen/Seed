"""N34.3.02 (exploration): state-dependent growth decisions with heterogeneous task difficulty.

Design: plans/reference/DESIGN-N34-3-02_20261126.md (D-20261126-648).
Single variable vs N34.3.01: tasks mix an easy prototype phase 2 (8 units suffice; growth only
costs) with the hard parity phase 2 (growth helps), so a state-dependent policy can beat every
fixed action. The learned bandit policy decides no_grow/grow8/grow16/grow24 from the warm-up
validation error; controls are fixed actions and the scripted pressure rule.

Usage: python scripts/training/n34_3_02_heterogeneous_growth.py
Output: reports/n34_3_02_growth_decision_hetero.json
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

CARD = "N34.3.02"
TRAIN_SEEDS = (6601, 6602, 6603)
TRAIN_EPISODES = 40
EVAL_TASKS_PER_TYPE = 15
CAP_CPU = 300.0
OUT = ROOT / "reports" / "n34_3_02_growth_decision_hetero.json"


def _eval_policy(
    policy: GrowthDecisionPolicy, tasks: list[tuple[int, bool]], deadline: float
) -> dict:
    rewards = []
    actions_easy: list[int] = []
    actions_hard: list[int] = []
    for task, hard in tasks:
        probe = run_episode(task, "no_grow", deadline, hard=hard)
        a = greedy_action(policy, probe)
        res = run_episode(task, ACTIONS[a], deadline, hard=hard)
        rewards.append(res["reward"])
        (actions_hard if hard else actions_easy).append(a)
    easy_grow = sum(1 for a in actions_easy if a in (2, 3)) / max(1, len(actions_easy))
    hard_grow = sum(1 for a in actions_hard if a in (2, 3)) / max(1, len(actions_hard))
    return {
        "mean_reward": round(sum(rewards) / len(rewards), 4),
        "easy_actions": actions_easy,
        "hard_actions": actions_hard,
        "easy_high_grow_rate": round(easy_grow, 4),
        "hard_high_grow_rate": round(hard_grow, 4),
    }


def _eval_fixed(action: str, tasks: list[tuple[int, bool]], deadline: float) -> float:
    rewards = [run_episode(task, action, deadline, hard=hard)["reward"] for task, hard in tasks]
    return round(sum(rewards) / len(rewards), 4)


def _eval_scripted(tasks: list[tuple[int, bool]], deadline: float) -> float:
    rewards = []
    for task, hard in tasks:
        probe = run_episode(task, "no_grow", deadline, hard=hard)
        action = "grow8" if probe["val_err"] > 0.15 else "no_grow"
        rewards.append(run_episode(task, action, deadline, hard=hard)["reward"])
    return round(sum(rewards) / len(rewards), 4)


def main() -> int:
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    per_seed: dict[str, dict] = {}
    for seed in TRAIN_SEEDS:
        train_specs = [(seed * 1000 + i, i % 2 == 1) for i in range(TRAIN_EPISODES)]
        eval_specs = [(seed * 1000 + 500 + i, i % 2 == 1) for i in range(2 * EVAL_TASKS_PER_TYPE)]
        t0 = time.process_time()
        trained = train_policy(train_specs, seed=seed, episodes=TRAIN_EPISODES, deadline=deadline)
        learned = _eval_policy(trained.policy, eval_specs, deadline)
        baselines = {
            "no_grow": _eval_fixed("no_grow", eval_specs, deadline),
            "grow8": _eval_fixed("grow8", eval_specs, deadline),
            "grow16": _eval_fixed("grow16", eval_specs, deadline),
            "grow24": _eval_fixed("grow24", eval_specs, deadline),
            "scripted_pressure": _eval_scripted(eval_specs, deadline),
        }
        exact = None
        if seed == TRAIN_SEEDS[0]:
            again = train_policy(train_specs, seed=seed, episodes=TRAIN_EPISODES, deadline=deadline)
            exact = all(
                torch.equal(a, b)
                for a, b in zip(trained.policy.parameters(), again.policy.parameters(), strict=True)
            )
        per_seed[str(seed)] = {
            "learned": learned,
            "baselines": baselines,
            "weights_exact": exact,
            "cpu": round(time.process_time() - t0, 2),
        }
    checks = {
        "g1_beats_scripted": all(
            v["learned"]["mean_reward"] >= v["baselines"]["scripted_pressure"] - 0.01
            for v in per_seed.values()
        ),
        "g2_beats_no_grow": all(
            v["learned"]["mean_reward"] >= v["baselines"]["no_grow"] + 0.02
            for v in per_seed.values()
        ),
        "g3_state_dependent": all(
            v["learned"]["easy_high_grow_rate"] <= 0.30
            and v["learned"]["hard_high_grow_rate"] >= 0.70
            for v in per_seed.values()
        ),
        "g4_deterministic": all(
            v["weights_exact"] is True for v in per_seed.values() if v["weights_exact"] is not None
        ),
    }
    exit_ = (
        "growth_decision_state_dependent_positive"
        if all(checks.values())
        else "growth_decision_state_dependent_partial"
    )
    payload = {
        "schema": "taiji-n34-3-02-v1",
        "card": CARD,
        "design": "DESIGN-N34-3-02_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "train_seeds": list(TRAIN_SEEDS),
        "train_episodes": TRAIN_EPISODES,
        "eval_tasks_per_type": EVAL_TASKS_PER_TYPE,
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
    return 0 if exit_ == "growth_decision_state_dependent_positive" else 3


if __name__ == "__main__":
    raise SystemExit(main())
