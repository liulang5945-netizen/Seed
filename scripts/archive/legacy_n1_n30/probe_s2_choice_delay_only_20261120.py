"""Delayed feedback and non-stationary drift in the synthetic choice environment (0 check).

Builds on D-20261119-127: 10 actions, cross-session trust dynamics, noisy partial
observation.  Added here: (a) utility feedback arrives K=3 slots later (bandit update
uses the delayed reward); (b) the utility mapping drifts every 8 sessions (action
values rotate), so old statistics become stale.  Held-out evaluation compares the
online policy with best fixed, wait and random under the same drift.  Gates: online
>= best_fixed - 0.05; online >= wait + 0.10; online >= random + 0.10; actions>=3 &
wait<=0.8.  3 seeds; cap 300 CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402

OUT = ROOT / "reports/taiji_u2_choice_delay_only_20261120.json"
CAP_CPU = 300.0
SEEDS = (0, 1, 2)
SESSIONS = 32
SLOTS = 32
EPOCHS = 8
DELAY = 3
DRIFT_EVERY = 10**9  # no drift in this variant
EXPLORE = 1.0
NOISE = 0.05
TRUST_FLIP = 0.20
PENDING_FLIP = 0.10
ACTIONS = (
    "continue",
    "recall",
    "learn",
    "verify",
    "ask",
    "wait",
    "report",
    "escalate",
    "sleep_wake",
    "defer",
)
WAIT_ACTION = 5
VERIFY_ACTION = 3
ASK_ACTION = 4
LEARN_ACTION = 2


def base_utilities(trust: int, pending: int, stale: int) -> np.ndarray:
    base = np.full(len(ACTIONS), 0.15)
    if not pending:
        base[5] = 0.65
        base[2] = 0.55
        base[9] = 0.50
        base[0] = 0.30
        return base
    if trust == 1 and stale == 1:
        base[[1, 3, 2, 0]] = [0.72, 0.90, 0.60, 0.45]
        return base
    if trust == 1 and stale == 0:
        base[[3, 0, 2, 1]] = [0.78, 0.62, 0.55, 0.45]
        return base
    if trust == 0 and stale == 1:
        base[[2, 4, 9, 0]] = [0.82, 0.72, 0.55, 0.30]
        return base
    base[[2, 4, 9, 0]] = [0.75, 0.68, 0.50, 0.35]
    return base


def utilities(trust: int, pending: int, stale: int, phase: int) -> np.ndarray:
    base = base_utilities(trust, pending, stale)
    return np.roll(base, -(phase * 3) % len(ACTIONS))


def next_trust(current: int, action: int, generator) -> int:
    if action == VERIFY_ACTION:
        return 1
    if action == ASK_ACTION:
        return 0 if generator.random() < 0.30 else current
    if action == LEARN_ACTION:
        return 1 if generator.random() < 0.30 else current
    return current


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
        self.values[context, action] += (reward - self.values[context, action]) / self.counts[
            context, action
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    results = []
    for seed in SEEDS:
        generator = np.random.default_rng(20261120 + seed)
        ucb = Ucb(8, len(ACTIONS))
        for epoch in range(EPOCHS):
            for session in range(SESSIONS):
                phase = (epoch * SESSIONS + session) // DRIFT_EVERY
                trust = int(generator.integers(0, 2))
                pending_delay: deque = deque()
                for _slot in range(SLOTS):
                    pending = int(generator.integers(0, 2))
                    stale = int(generator.integers(0, 2)) if pending else 0
                    observed = 1 - trust if generator.random() < TRUST_FLIP else trust
                    observed_pending = 1 - pending if generator.random() < PENDING_FLIP else pending
                    context = observed * 4 + observed_pending * 2 + stale
                    action = ucb.select(context)
                    reward = float(utilities(trust, pending, stale, phase)[action])
                    reward += float(generator.normal(0.0, NOISE))
                    pending_delay.append((context, action, reward))
                    if len(pending_delay) > DELAY:
                        old_context, old_action, old_reward = pending_delay.popleft()
                        ucb.update(old_context, old_action, old_reward)
                    trust = next_trust(trust, action, generator)
                while pending_delay:
                    old_context, old_action, old_reward = pending_delay.popleft()
                    ucb.update(old_context, old_action, old_reward)
        eval_rows = []
        for session in range(6):
            eval_generator = np.random.default_rng(9000 + seed * 100 + session)
            phase = session // 3
            trust = int(eval_generator.integers(0, 2))
            for _ in range(SLOTS):
                pending = int(eval_generator.integers(0, 2))
                stale = int(eval_generator.integers(0, 2)) if pending else 0
                observed = 1 - trust if eval_generator.random() < TRUST_FLIP else trust
                observed_pending = (
                    1 - pending if eval_generator.random() < PENDING_FLIP else pending
                )
                context = observed * 4 + observed_pending * 2 + stale
                action = ucb.greedy(context)
                eval_rows.append(
                    {
                        "utility": float(utilities(trust, pending, stale, phase)[action]),
                        "table": utilities(trust, pending, stale, phase),
                        "phase": phase,
                    }
                )
                trust = next_trust(trust, action, eval_generator)
        online = float(np.mean([row["utility"] for row in eval_rows]))
        fixed_values = []
        for action in range(len(ACTIONS)):
            fixed_values.append(float(np.mean([row["table"][action] for row in eval_rows])))
        best_fixed = float(np.max(fixed_values))
        wait = float(fixed_values[WAIT_ACTION])
        random_utility = float(np.mean([row["table"].mean() for row in eval_rows]))
        actions_used = int(sum(1 for context in range(8) if ucb.counts[context].sum() > 0))
        wait_share = float(ucb.counts[:, WAIT_ACTION].sum() / max(1, ucb.counts.sum()))
        results.append(
            {
                "seed": seed,
                "online": round(online, 4),
                "best_fixed": round(best_fixed, 4),
                "wait": round(wait, 4),
                "random": round(random_utility, 4),
                "actions_used": actions_used,
                "wait_share": round(wait_share, 4),
            }
        )
    gates = {
        "beats_best_fixed": all(
            result["online"] >= result["best_fixed"] - 0.05 for result in results
        ),
        "beats_wait": all(result["online"] >= result["wait"] + 0.10 for result in results),
        "beats_random": all(result["online"] >= result["random"] + 0.10 for result in results),
        "diversity": all(
            result["actions_used"] >= 3 and result["wait_share"] <= 0.8 for result in results
        ),
    }
    verdict = "online_choice_drift_ready" if all(gates.values()) else "online_choice_drift_absent"
    payload = {
        "schema": "taiji-u2-online-choice-drift-v1",
        "card": "delayed feedback K=3 only (stationary utility mapping)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "sessions": SESSIONS,
            "slots": SLOTS,
            "epochs": EPOCHS,
            "delay": DELAY,
            "drift_every": DRIFT_EVERY,
            "actions": len(ACTIONS),
            "gates": "online>=best_fixed-0.05; online>=wait+0.10; online>=random+0.10; actions>=3 & wait<=0.8",
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
    return 0 if verdict == "online_choice_drift_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
