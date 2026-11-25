"""N33.4.05 (diagnostic): can the consequence model reach G1's delta_u R2 in principle?

Pre-registered: plans/reference/DESIGN-N33-4-05_20261126.md (D-20261126-605).
Trains only the consequence model, with a much larger net and much more data/steps than
N33.4.03/04, to probe the reachable ceiling of delta_u R2 under the same weighted loss.
No four-arm comparison, no policy training, no gate changes, no confirmation seeds.

Usage: python scripts/training/n33_4_05_ceiling_probe.py
Output: reports/n33_4_05_ceiling.json
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

from scripts.training.n33_4_01_recursive_env_check import EVAL_SEEDS  # noqa: E402
from scripts.training.n33_4_02_recursive_dev import collect_logged_experience  # noqa: E402
from taiji.choice_simulator import seed_from  # noqa: E402
from taiji.recursive_dev import train_consequence_model  # noqa: E402

CARD = "N33.4.05"
TRAIN_SEEDS = (4001, 4002, 4003)
LOG_EPISODES = 1500
VAL_EPISODES = 100
CM_STEPS = 6000
HIDDEN = 256
CAP_CPU = 60.0
OUT = ROOT / "reports" / "n33_4_05_ceiling.json"


def main() -> int:
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    per_seed: dict[str, dict] = {}
    try:
        for seed in TRAIN_SEEDS:
            log_train = [seed * 1000 + i for i in range(LOG_EPISODES)]
            log_val = [seed * 1000 + 900 + i for i in range(VAL_EPISODES)]
            tr = collect_logged_experience(log_train, seed_from(CARD, seed, "log-tr"))
            va = collect_logged_experience(log_val, seed_from(CARD, seed, "log-va"))
            t0 = time.process_time()
            cm, stats = train_consequence_model(
                tr,
                va,
                seed,
                mode="full",
                steps=CM_STEPS,
                outcome_weight_scale=True,
                hidden=HIDDEN,
                deadline=deadline,
            )
            per_seed[str(seed)] = {
                **stats,
                "params": sum(p.numel() for p in cm.parameters()),
                "cpu": round(time.process_time() - t0, 2),
            }
    except (TimeoutError, FloatingPointError) as exc:
        payload = {
            "schema": "taiji-n33-4-05-v1",
            "card": CARD,
            "exit": "engineering_failure",
            "reason": str(exc),
            "per_seed": per_seed,
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    best = max(v["delta_u_r2"] for v in per_seed.values())
    exit_ = (
        "ceiling_reachable"
        if best >= 0.90
        else ("ceiling_marginal" if best >= 0.85 else "ceiling_low")
    )
    payload = {
        "schema": "taiji-n33-4-05-v1",
        "card": CARD,
        "design": "DESIGN-N33-4-05_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "train_seeds": list(TRAIN_SEEDS),
        "log_episodes": LOG_EPISODES,
        "cm_steps": CM_STEPS,
        "hidden": HIDDEN,
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS)],
        "per_seed": per_seed,
        "best_r2": best,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0, "gate_changes": 0},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit": exit_, "best_r2": best, "per_seed": per_seed, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
