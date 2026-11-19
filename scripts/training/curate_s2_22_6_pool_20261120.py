"""Pre-registered pool curation for the 22.6 C sessions (zero training).

Scans a fixed seed range, builds the 16-tick observation stream per world once,
and admits a world only if it satisfies the pool-level rules of
``D-20261119-23``:

* at least 1/3 of the world's sessions have ``observed_t0 != truth`` while
  ``observed_deadline == truth`` (stale-at-deadline evidence exists);
* ``wait_value >= 0.05`` over the world's sessions;
* the strongest scripted fixed policy averages ``<= 0.90`` on the world
  (headroom ``>= 0.10`` for the 22.6 ``>= 0.05`` net-gain criterion);
* deadline truths are non-degenerate (>= 2 distinct position words).

Then recomputes the scripted baselines over the curated pool.  Fewer than 32
admitted worlds in range is a pool-level ``fixture_invalid``.  Zero optimizer
steps, zero check/final reads, no signing.
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
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    NAMES,
    RGB_COLORS,
    locate_public,
    position_word,
)
from scripts.training.probe_s2_22_6_baselines_20261120 import policy_utilities  # noqa: E402
from scripts.training.validate_22_6_c_fixture_20261120 import (  # noqa: E402
    DEADLINE,
    TICKS,
)
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_6_pool_admission_20261120.json"
SEED_START = 20263000
SEED_STOP = 20264000
TARGET_WORLDS = 32
SESSIONS = 3
CAP_CPU = 240.0
MIN_STALE_FRACTION = 1.0 / 3.0
MIN_WAIT_VALUE = 0.05
MAX_BEST_FIXED = 0.90


def tick_stream(seed: int) -> list[dict]:
    world = PhysicalWorld.reset(seed, {"body_count": 3})
    ticks = []
    for index in range(TICKS):
        result = world.elapse(f"pool-{index}")
        bodies = world.snapshot()["bodies"]
        names = [NAMES[RGB_COLORS.index(tuple(body["color"]))] for body in bodies]
        truth = dict(
            zip(names, (position_word(body["position"][0]) for body in bodies), strict=True)
        )
        observed = {}
        for name in names:
            local = locate_public(result["raw_media"]["rgb"], name)
            observed[name] = "未知" if local is None else position_word(local)
        ticks.append({"truth": truth, "observed": observed})
    return ticks


def world_cases(seed: int) -> list[dict]:
    ticks = tick_stream(seed)
    names = list(ticks[0]["observed"])
    cases = []
    for target in names:
        cases.append(
            {
                "seed": seed,
                "target": target,
                "truth": ticks[DEADLINE]["truth"].get(target, "未知"),
                "observed": [tick["observed"].get(target, "未知") for tick in ticks],
            }
        )
    return cases


def world_stats(cases: list[dict]) -> dict:
    stale = sum(
        1
        for case in cases
        if case["observed"][0] != case["truth"] and case["observed"][DEADLINE] == case["truth"]
    )
    utilities = [policy_utilities(case) for case in cases]
    names = list(utilities[0])
    means = {name: float(np.mean([util[name] for util in utilities])) for name in names}
    best_fixed = max(means.values())
    wait_value = float(
        np.mean([util["observe_deadline_answer"] - util["observe_t0_answer"] for util in utilities])
    )
    truths = {case["truth"] for case in cases}
    return {
        "stale_sessions": stale,
        "stale_fraction": round(stale / len(cases), 4),
        "wait_value": round(wait_value, 4),
        "best_fixed": round(best_fixed, 4),
        "distinct_truths": len(truths),
        "admissible": bool(
            stale / len(cases) >= MIN_STALE_FRACTION
            and wait_value >= MIN_WAIT_VALUE
            and best_fixed <= MAX_BEST_FIXED
            and len(truths) >= 2
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    admitted: list[dict] = []
    scanned = 0
    for seed in range(SEED_START, SEED_STOP):
        scanned += 1
        cases = world_cases(seed)
        stats = world_stats(cases)
        if stats["admissible"]:
            admitted.append({"seed": seed, **stats})
        if len(admitted) >= TARGET_WORLDS:
            break
        if time.process_time() - started > CAP_CPU:
            break
    if len(admitted) < TARGET_WORLDS:
        verdict = "pool_inadequate"
    elif time.process_time() - started > CAP_CPU:
        verdict = "budget_stop"
    else:
        verdict = "pool_admitted"
    pool_sessions = []
    for entry in admitted:
        cases = world_cases(entry["seed"])
        for case in cases:
            pool_sessions.append(
                {**policy_utilities(case), "seed": entry["seed"], "target": case["target"]}
            )
    names = list(policy_utilities(world_cases(admitted[0]["seed"])[0]))
    aggregate = {
        name: {
            "mean": round(float(np.mean([row[name] for row in pool_sessions])), 4),
            "std": round(float(np.std([row[name] for row in pool_sessions])), 4),
        }
        for name in names
    }
    best_policy = max(aggregate, key=lambda name: aggregate[name]["mean"])
    strongest = aggregate[best_policy]["mean"]
    wait_value = round(
        float(
            np.mean(
                [row["observe_deadline_answer"] - row["observe_t0_answer"] for row in pool_sessions]
            )
        ),
        4,
    )
    stale_fraction = (
        round(float(np.mean([entry["stale_fraction"] for entry in admitted])), 4)
        if admitted
        else 0.0
    )
    payload = {
        "schema": "taiji-u2-22-6-pool-admission-v1",
        "card": "22.6 C-session pool curation (D-20261119-23 rules)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "rules": {
            "seed_range": [SEED_START, SEED_STOP],
            "target_worlds": TARGET_WORLDS,
            "sessions_per_world": SESSIONS,
            "min_stale_fraction": MIN_STALE_FRACTION,
            "min_wait_value": MIN_WAIT_VALUE,
            "max_best_fixed": MAX_BEST_FIXED,
            "cap_cpu": CAP_CPU,
        },
        "verdict": verdict,
        "scanned_worlds": scanned,
        "admitted_worlds": admitted,
        "curated_baselines": aggregate,
        "strongest_fixed": {"name": best_policy, "mean": strongest},
        "wait_value": wait_value,
        "mean_stale_fraction": stale_fraction,
        "headroom": round(1.0 - strongest, 4),
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "n_admitted": len(admitted),
                "scanned": scanned,
                "strongest_fixed": payload["strongest_fixed"],
                "wait_value": wait_value,
                "headroom": payload["headroom"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "pool_admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
