"""Online closed-loop autonomous choice over the C-task pool (UCB1, 0 check).

Episodes: partially observed cases from the 32 admitted pool worlds; the agent picks
one of 7 policies per case, receives only that policy's noisy utility (sigma=0.05)
and updates per-context UCB1.  Evaluation on 8 held-out worlds compares the greedy
per-context policy with the best fixed policy, the wait policy and random choice;
learning is checked by regret over the first vs last half of episodes.  Four gates
per D-20261119-122.  3 seeds; cap 200 CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.curate_s2_22_6_pool_20261120 import (  # noqa: E402
    policy_utilities,
    tick_stream,
)

OUT = ROOT / "reports/taiji_u2_online_choice_20261120.json"
POOL = ROOT / "reports/taiji_u2_22_6_pool_admission_20261120.json"
CAP_CPU = 200.0
SEEDS = (0, 1, 2)
EPOCHS = 8
HELDOUT_WORLDS = 8
EXPLORE = 1.0
NOISE = 0.05
BINS = 64
ACTIONS = (
    "answer_no_observe",
    "rest_then_answer",
    "observe_t0_answer",
    "observe_t7_answer",
    "observe_t14_answer",
    "observe_deadline_answer",
    "verify_t0_answer",
)
POSITIONS = ("左边", "中间", "右边", "未知")
WAIT_ACTION = 1
GATE_BEST_FIXED = -0.05
GATE_WAIT = 0.10
GATE_REGRET = 0.05
DIVERSITY_MIN = 3
WAIT_MAX = 0.8


def admissible_seeds() -> list[int]:
    report = json.loads(POOL.read_text(encoding="utf-8"))
    return [entry["seed"] for entry in report["admitted_worlds"] if entry.get("admissible")]


def build_cases(seeds: list[int]) -> list[dict]:
    cases = []
    for seed in seeds:
        ticks = tick_stream(seed)
        present = list(ticks[0]["observed"])
        for name in present:
            truth = ticks[-1]["truth"][name]
            observed = [tick["observed"][name] for tick in ticks]
            utility = policy_utilities({"truth": truth, "observed": observed})
            signature = tuple(ticks[0]["observed"][other] for other in present)
            code = hash((signature, name)) % BINS
            cases.append(
                {
                    "seed": seed,
                    "entity": name,
                    "context": code,
                    "utilities": np.array([utility[action] for action in ACTIONS], dtype=float),
                }
            )
    return cases


class Ucb:
    def __init__(self, bins: int, actions: int) -> None:
        self.counts = np.zeros((bins, actions))
        self.values = np.zeros((bins, actions))
        self.total = 0

    def select(self, context: int) -> int:
        counts = self.counts[context]
        if (counts == 0).any():
            return int(np.argmax(counts == 0))
        bonus = EXPLORE * np.sqrt(np.log(self.total + 1) / counts)
        return int(np.argmax(self.values[context] + bonus))

    def greedy(self, context: int) -> int:
        counts = self.counts[context]
        if (counts == 0).any():
            return int(np.argmax(counts == 0))
        return int(np.argmax(self.values[context]))

    def update(self, context: int, action: int, reward: float) -> None:
        self.total += 1
        self.counts[context, action] += 1
        n = self.counts[context, action]
        self.values[context, action] += (reward - self.values[context, action]) / n


def evaluate(ucb: Ucb, cases: list[dict], greedy: bool) -> float:
    values = []
    for case in cases:
        action = ucb.greedy(case["context"]) if greedy else ucb.select(case["context"])
        values.append(case["utilities"][action])
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    seeds = admissible_seeds()
    train_seeds = seeds[:-HELDOUT_WORLDS]
    heldout_seeds = seeds[-HELDOUT_WORLDS:]
    train_cases = build_cases(train_seeds)
    heldout_cases = build_cases(heldout_seeds)
    results = []
    for seed in SEEDS:
        generator = np.random.default_rng(20261120 + seed)
        ucb = Ucb(BINS, len(ACTIONS))
        episodes = []
        for _epoch in range(EPOCHS):
            order = generator.permutation(len(train_cases))
            for position in order:
                case = train_cases[int(position)]
                action = ucb.select(case["context"])
                reward = float(case["utilities"][action]) + float(generator.normal(0.0, NOISE))
                ucb.update(case["context"], action, reward)
                episodes.append((case["utilities"][action], float(case["utilities"].max())))
        online = evaluate(ucb, heldout_cases, greedy=True)
        heldout_matrix = np.stack([case["utilities"] for case in heldout_cases])
        fixed = heldout_matrix.mean(axis=0)
        best_fixed = float(fixed.max())
        wait = float(fixed[WAIT_ACTION])
        random_utility = float(heldout_matrix.mean())
        half = len(episodes) // 2
        regret_first = float(np.mean([best - value for value, best in episodes[:half]]))
        regret_last = float(np.mean([best - value for value, best in episodes[half:]]))
        actions_used = len(
            {
                ucb.counts[context].argmax()
                for context in range(BINS)
                if ucb.counts[context].sum() > 0
            }
        )
        wait_share = float(ucb.counts[:, WAIT_ACTION].sum() / max(1, ucb.counts.sum()))
        results.append(
            {
                "seed": seed,
                "online": round(online, 4),
                "best_fixed": round(best_fixed, 4),
                "wait": round(wait, 4),
                "random": round(random_utility, 4),
                "regret_first": round(regret_first, 4),
                "regret_last": round(regret_last, 4),
                "actions_used": actions_used,
                "wait_share": round(wait_share, 4),
            }
        )
    gates = {
        "beats_best_fixed": all(
            result["online"] >= result["best_fixed"] + GATE_BEST_FIXED for result in results
        ),
        "beats_wait": all(result["online"] >= result["wait"] + GATE_WAIT for result in results),
        "learning": all(
            result["regret_last"] <= result["regret_first"] - GATE_REGRET for result in results
        ),
        "diversity": all(
            result["actions_used"] >= DIVERSITY_MIN and result["wait_share"] <= WAIT_MAX
            for result in results
        ),
    }
    if all(gates.values()):
        verdict = "autonomous_choice_online_ready"
    elif gates["beats_best_fixed"] and gates["beats_wait"]:
        verdict = "autonomous_choice_online_partial"
    else:
        verdict = "autonomous_choice_online_absent"
    payload = {
        "schema": "taiji-u2-online-choice-v1",
        "card": "online UCB1 policy over the C-task pool with bandit feedback",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "train_worlds": len(train_seeds),
            "heldout_worlds": len(heldout_seeds),
            "epochs": EPOCHS,
            "noise": NOISE,
            "actions": len(ACTIONS),
            "gates": "online>=best_fixed-0.05; online>=wait+0.10; regret_last<=regret_first-0.05; actions>=3 & wait<=0.8",
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "autonomous_choice_online_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
