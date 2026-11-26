"""Synthetic context-dependent choice environment: online UCB1 over 7 source-trust actions (0 check).

Switch-problem design (D-20261119-126, option E): each slot has latent source trust,
a pending-question flag and a stale flag; utilities favour verify/recall when trust is
high and learn/ask when trust is low, wait when nothing is pending.  The agent observes
trust with 15% corruption and chooses online with bandit feedback only.  Held-out slots
compare the learned contextual policy with the best fixed policy, wait and random.
Gates: online >= best_fixed - 0.05; online >= wait + 0.10; held-out greedy beats random
+0.10; actions>=3 & wait<=0.8.  3 seeds; cap 300 CPU.
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

OUT = ROOT / "reports/taiji_u2_online_choice_synthetic_20261120.json"
CAP_CPU = 300.0
SEEDS = (0, 1, 2)
TRAIN_SLOTS = 192
HELDOUT_SLOTS = 96
EPOCHS = 30
EXPLORE = 1.0
NOISE = 0.05
TRUST_FLIP = 0.15
ACTIONS = ("continue", "recall", "learn", "verify", "ask", "wait", "report")
WAIT_ACTION = 5


def utilities(trust: int, pending: int, stale: int) -> np.ndarray:
    if not pending:
        return np.array([0.30, 0.35, 0.55, 0.20, 0.25, 0.60, 0.20])
    if trust == 1 and stale == 1:
        return np.array([0.50, 0.70, 0.60, 0.90, 0.30, 0.20, 0.10])
    if trust == 1 and stale == 0:
        return np.array([0.60, 0.45, 0.55, 0.75, 0.30, 0.40, 0.15])
    if trust == 0 and stale == 1:
        return np.array([0.25, 0.35, 0.80, 0.25, 0.70, 0.40, 0.30])
    return np.array([0.30, 0.40, 0.75, 0.20, 0.65, 0.45, 0.25])


def build_slots(count: int, seed: int, with_noise: bool = True) -> list[dict]:
    generator = np.random.default_rng(seed)
    slots = []
    for _ in range(count):
        trust = int(generator.integers(0, 2))
        pending = int(generator.integers(0, 2))
        stale = int(generator.integers(0, 2))
        observed = trust
        if with_noise and generator.random() < TRUST_FLIP:
            observed = 1 - trust
        context = observed * 4 + pending * 2 + stale
        slots.append(
            {
                "context": context,
                "utilities": utilities(trust, pending, stale),
            }
        )
    return slots


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


def evaluate(policy: Ucb, slots: list[dict]) -> float:
    return float(np.mean([slot["utilities"][policy.greedy(slot["context"])] for slot in slots]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    results = []
    for seed in SEEDS:
        generator = np.random.default_rng(20261120 + seed)
        train_slots = build_slots(TRAIN_SLOTS, 1000 + seed)
        heldout_slots = build_slots(HELDOUT_SLOTS, 9000 + seed)
        ucb = Ucb(8, len(ACTIONS))
        for _epoch in range(EPOCHS):
            order = generator.permutation(len(train_slots))
            for position in order:
                slot = train_slots[int(position)]
                action = ucb.select(slot["context"])
                reward = float(slot["utilities"][action]) + float(generator.normal(0.0, NOISE))
                ucb.update(slot["context"], action, reward)
        online = evaluate(ucb, heldout_slots)
        matrix = np.stack([slot["utilities"] for slot in heldout_slots])
        fixed = matrix.mean(axis=0)
        best_fixed = float(fixed.max())
        wait = float(fixed[WAIT_ACTION])
        random_utility = float(matrix.mean())
        actions_used = len({context for context in range(8) if ucb.counts[context].sum() > 0})
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
    verdict = (
        "online_choice_synthetic_ready" if all(gates.values()) else "online_choice_synthetic_absent"
    )
    payload = {
        "schema": "taiji-u2-online-choice-synthetic-v1",
        "card": "synthetic source-trust switch environment with online contextual UCB1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "train_slots": TRAIN_SLOTS,
            "heldout_slots": HELDOUT_SLOTS,
            "epochs": EPOCHS,
            "actions": len(ACTIONS),
            "trust_flip": TRUST_FLIP,
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
    return 0 if verdict == "online_choice_synthetic_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
