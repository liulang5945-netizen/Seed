"""N33.1.01 (zero training): simulator + scripted-baseline replay for the autonomous-choice fixture.

Pre-registered: plans/reference/DESIGN-N33-1-01_20261125.md (D-20261125-554).
Runs no learned policy and reads no confirmation seeds; it only checks that the fixture is
usable as an evaluation environment for N33.1 and records the baselines.

Usage: python scripts/training/n33_1_01_sim_baseline.py
Output: reports/n33_1_01_sim.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.choice_simulator import (  # noqa: E402
    CHOICES,
    HORIZON,
    ChoiceSessionEnv,
    evaluate_policy,
    random_policy,
    scripted_policy,
)

ROOT_DIR = ROOT
CARD = "N33.1.01"
SEEDS = list(range(40))
OUT = ROOT_DIR / "reports" / "n33_1_01_sim.json"


def action_histogram(policy) -> dict[str, int]:
    hist: Counter[str] = Counter()
    for seed in SEEDS:
        env = ChoiceSessionEnv(seed)
        state = env.reset()
        for _ in range(HORIZON):
            action = policy(state)
            hist[action] += 1
            state, _, done = env.step(action)
            if done:
                break
    return dict(hist)


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    scripted = evaluate_policy(scripted_policy, SEEDS)
    gen = torch.Generator().manual_seed(7)
    random = evaluate_policy(lambda s: random_policy(s, gen), SEEDS)
    hist = action_histogram(scripted_policy)
    deterministic = evaluate_policy(scripted_policy, [1, 2, 3]) == evaluate_policy(
        scripted_policy, [1, 2, 3]
    )
    share = {k: hist.get(k, 0) / max(1, sum(hist.values())) for k in CHOICES}
    used = [k for k in CHOICES if hist.get(k, 0) > 0]
    times_used_all = len(used) == len(CHOICES)
    dominant = max(share.values())
    improvement = scripted["mean_return"] - random["mean_return"]
    exit_ = (
        "sim_ready"
        if deterministic and times_used_all and improvement >= 1.0 and dominant <= 0.80
        else "presentation_failure"
    )
    payload = {
        "schema": "taiji-n33-1-01-v1",
        "card": CARD,
        "design": "DESIGN-N33-1-01_20261125.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "fixture": {"choices": list(CHOICES), "state_dim": 7, "horizon": HORIZON, "seeds": SEEDS},
        "baselines": {"scripted": scripted, "random": random, "improvement": round(improvement, 6)},
        "scripted_action_share": {k: round(v, 4) for k, v in share.items()},
        "actions_used": used,
        "deterministic_same_seed": deterministic,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": 20.0},
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
                "scripted": scripted,
                "random": random,
                "improvement": round(improvement, 3),
                "actions_used": used,
                "dominant_share": round(dominant, 3),
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "sim_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
