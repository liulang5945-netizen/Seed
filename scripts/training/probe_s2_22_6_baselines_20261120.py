"""Pre-registered scripted-baseline rollout for the 22.6.0 C sessions (zero training).

Runs the six scripted controls from the 22.6.0 design card over a fixed
development-fit pool (32 worlds x 3 sessions = 96 sessions, 16 decisions each)
on the frozen physical world, scoring utility = 1[answer==deadline truth] minus
action costs.  These are the "strongest fixed/reactive baselines" that 22.6.2
must beat later.  Zero optimizer steps, zero check/final reads, no signing.
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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import NAMES, RGB_COLORS  # noqa: E402
from scripts.training.validate_22_6_c_fixture_20261120 import (  # noqa: E402
    DEADLINE,
    OBSERVE_COST,
    TICKS,
    WAIT_COST,
    session,
)
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_6_baselines_20261120.json"
WORLDS = 32
SESSIONS = 3
SEED_START = 20262000
CAP_CPU = 120.0
VERIFY_COST = 0.10


def world_names(seed: int) -> list[str]:
    world = PhysicalWorld.reset(seed, {"body_count": 3})
    world.elapse("baseline-probe")
    return [NAMES[RGB_COLORS.index(tuple(body["color"]))] for body in world.snapshot()["bodies"]]


def policy_utilities(case: dict) -> dict[str, float]:
    truth = case["truth"]
    observed = case["observed"]

    def obs_answer(k: int) -> float:
        hit = 1.0 if observed[k] == truth else 0.0
        return hit - OBSERVE_COST - k * WAIT_COST

    utilities = {
        "answer_no_observe": 1.0 if truth == "未知" else 0.0,
        "rest_then_answer": (1.0 if truth == "未知" else 0.0) - DEADLINE * WAIT_COST,
        "observe_t0_answer": obs_answer(0),
        "observe_t7_answer": obs_answer(7),
        "observe_t14_answer": obs_answer(14),
        "observe_deadline_answer": obs_answer(DEADLINE),
        "verify_t0_answer": (1.0 if observed[0] == truth else 0.0) - OBSERVE_COST - VERIFY_COST,
    }
    return {name: round(value, 4) for name, value in utilities.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = []
    for world_index in range(WORLDS):
        seed = SEED_START + world_index
        # Sessions vary the queried entity inside the same world stream.
        targets = world_names(seed)
        for session_index in range(SESSIONS):
            if targets:
                target = targets[(world_index + session_index) % len(targets)]
                case = session(seed, target)
            else:
                case = session(seed)
            utilities = policy_utilities(case)
            rows.append(
                {
                    "world": world_index,
                    "seed": seed,
                    "session": session_index,
                    "target": case["target"],
                    "truth": case["truth"],
                    "utilities": utilities,
                }
            )
            if time.process_time() - started > CAP_CPU:
                break
        if time.process_time() - started > CAP_CPU:
            break
    names = list(rows[0]["utilities"]) if rows else []
    aggregate = {
        name: {
            "mean": round(float(np.mean([row["utilities"][name] for row in rows])), 4),
            "std": round(float(np.std([row["utilities"][name] for row in rows])), 4),
            "win_rate": round(
                float(
                    np.mean(
                        [row["utilities"][name] == max(row["utilities"].values()) for row in rows]
                    )
                ),
                4,
            ),
        }
        for name in names
    }
    best_policy = max(aggregate, key=lambda name: aggregate[name]["mean"])
    wait_value = round(
        float(
            np.mean(
                [
                    row["utilities"]["observe_deadline_answer"]
                    - row["utilities"]["observe_t0_answer"]
                    for row in rows
                ]
            )
        ),
        4,
    )
    verdict = "budget_stop" if time.process_time() - started > CAP_CPU else "baselines_ready"
    payload = {
        "schema": "taiji-u2-22-6-scripted-baselines-v1",
        "card": "22.6.0 C sessions, scripted fixed/reactive baselines",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "worlds": WORLDS,
            "sessions_per_world": SESSIONS,
            "decisions_per_session": TICKS,
            "deadline": DEADLINE,
            "costs": {"observe": OBSERVE_COST, "wait_or_rest": WAIT_COST, "verify": VERIFY_COST},
            "policies": names,
            "cap_cpu": CAP_CPU,
        },
        "aggregate": aggregate,
        "best_fixed_policy": {
            "name": best_policy,
            "mean": aggregate[best_policy]["mean"],
        },
        "wait_value": wait_value,
        "n_sessions": len(rows),
        "rows": rows,
        "verdict": verdict,
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "best_fixed_policy": payload["best_fixed_policy"],
                "wait_value": wait_value,
                "n_sessions": len(rows),
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
