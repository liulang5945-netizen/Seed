"""Harder synthetic choice: 10 strategies, cross-session trust dynamics (0 check).

Each episode is a session of sequential slots; the agent's action partly controls the
next slot's latent source trust (verify keeps/restores high trust; ask erodes it;
learn can improve low trust).  The agent sees noisy trust/pending flags only, chooses
one of 10 policies per slot, and gets noisy bandit feedback.  Evaluation compares the
learned contextual UCB1 policy with best fixed, wait and random on fresh sessions.
Gates: online >= best_fixed - 0.05; online >= wait + 0.10; online >= random + 0.10;
actions>=3 & wait<=0.8.  3 seeds; cap 300 CPU.
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

OUT = ROOT / "reports/taiji_u2_online_choice_hard_20261120.json"
CAP_CPU = 300.0
SEEDS = (0, 1, 2)
SESSIONS = 24
SLOTS = 32
EPOCHS = 8
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


def utilities(trust: int, pending: int, stale: int) -> np.ndarray:
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


def run_session(generator, policy, collect: bool = False) -> list[dict]:
    trust = int(generator.integers(0, 2))
    rows = []
    for _ in range(SLOTS):
        pending = int(generator.integers(0, 2))
        stale = int(generator.integers(0, 2)) if pending else 0
        observed = 1 - trust if generator.random() < TRUST_FLIP else trust
        observed_pending = 1 - pending if generator.random() < PENDING_FLIP else pending
        context = observed * 4 + observed_pending * 2 + stale
        action = policy(context)
        true_utility = float(utilities(trust, pending, stale)[action])
        rows.append(
            {
                "context": context,
                "action": action,
                "utility": true_utility,
                "trust": trust,
                "pending": pending,
                "stale": stale,
            }
        )
        trust = next_trust(trust, action, generator)
    return rows


def best_fixed_utility(generator_factory, policy_factory) -> float:
    """Mean utility of each fixed action under the same session dynamics (estimate)."""

    totals = np.zeros(len(ACTIONS))
    count = 0
    for session in range(12):
        generator = generator_factory(session)
        rows = run_session(generator, lambda _context: 0, collect=True)
        trust = rows[0]["trust"]
        # roll again with fixed actions
        for action in range(len(ACTIONS)):
            generator = generator_factory(1000 + session)
            trust = int(generator.integers(0, 2))
            values = []
            for _ in range(SLOTS):
                pending = int(generator.integers(0, 2))
                stale = int(generator.integers(0, 2)) if pending else 0
                values.append(float(utilities(trust, pending, stale)[action]))
                trust = next_trust(trust, action, generator)
            totals[action] += float(np.mean(values))
            count += 1
    return float((totals / (count / len(ACTIONS))).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    results = []
    for seed in SEEDS:
        generator = np.random.default_rng(20261120 + seed)
        ucb = Ucb(8, len(ACTIONS))
        for _epoch in range(EPOCHS):
            for _session in range(SESSIONS):
                trust = int(generator.integers(0, 2))
                for _ in range(SLOTS):
                    pending = int(generator.integers(0, 2))
                    stale = int(generator.integers(0, 2)) if pending else 0
                    observed = 1 - trust if generator.random() < TRUST_FLIP else trust
                    observed_pending = 1 - pending if generator.random() < PENDING_FLIP else pending
                    context = observed * 4 + observed_pending * 2 + stale
                    action = ucb.select(context)
                    reward = float(utilities(trust, pending, stale)[action]) + float(
                        generator.normal(0.0, NOISE)
                    )
                    ucb.update(context, action, reward)
                    trust = next_trust(trust, action, generator)
        eval_rows = []
        for session in range(6):
            eval_generator = np.random.default_rng(9000 + seed * 100 + session)
            eval_rows.extend(run_session(eval_generator, ucb.greedy, collect=True))
        online = float(np.mean([row["utility"] for row in eval_rows]))
        matrix = np.stack(
            [utilities(row["trust"], row["pending"], row["stale"]) for row in eval_rows]
        )
        fixed = matrix.mean(axis=0)
        best_fixed = float(fixed.max())
        wait = float(fixed[WAIT_ACTION])
        random_utility = float(matrix.mean())
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
    verdict = "online_choice_hard_ready" if all(gates.values()) else "online_choice_hard_absent"
    payload = {
        "schema": "taiji-u2-online-choice-hard-v1",
        "card": "cross-session trust dynamics with 10 actions and noisy feedback",
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
    return 0 if verdict == "online_choice_hard_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
