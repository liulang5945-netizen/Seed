"""Action-interface feasibility for the physical-consequence axis (zero check).

Before designing an action-consequence task, verify that the world's ``act``
command actually produces direction-controlled displacement of the targeted body
and preserves the others: for several seeds, target one body, push it along +x /
-x / +y, and compare with a same-seed ``elapse`` baseline.  The probe is a
capability-of-the-simulator check only; no training, no capability signing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

OUT = ROOT / "reports/taiji_u2_action_interface_20261120.json"
CAP_CPU = 60.0
SEEDS = (3537161815461848802, 3537161815461848803, 3537161815461848804)
DIRECTIONS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))


def positions(world) -> list[tuple[float, float]]:
    return [tuple(body["position"]) for body in world.snapshot()["bodies"]]


def one_step(seed: int, target_index: int, direction: tuple[float, float]) -> dict:
    baseline_world = PhysicalWorld.reset(seed, {"body_count": 3})
    baseline_world.elapse("public-scene")
    before = positions(baseline_world)
    baseline_world.elapse("public-next")
    baseline_after = positions(baseline_world)
    baseline_delta = [
        (round(after[0] - before[0], 6), round(after[1] - before[1], 6))
        for after, before in zip(baseline_after, before, strict=True)
    ]

    world = PhysicalWorld.reset(seed, {"body_count": 3})
    world.elapse("public-scene")
    before = positions(world)
    target = before[target_index]
    command = torch.zeros(15, dtype=torch.float64)
    command[0] = 1.0
    command[9], command[10] = target
    command[11], command[12] = direction
    command[13] = 1.0
    command[14] = 0.05
    world.step("push", command)
    after = positions(world)
    deltas = [
        (round(after[index][0] - before[index][0], 6), round(after[index][1] - before[index][1], 6))
        for index in range(3)
    ]
    return {
        "seed": seed,
        "target_index": target_index,
        "direction": list(direction),
        "before": before,
        "after": after,
        "delta": deltas,
        "baseline_delta": baseline_delta,
        "target_shift": round(
            (
                (after[target_index][0] - before[target_index][0]) ** 2
                + (after[target_index][1] - before[target_index][1]) ** 2
            )
            ** 0.5,
            6,
        ),
        "other_shift": round(
            max(
                (
                    (after[other][0] - before[other][0]) ** 2
                    + (after[other][1] - before[other][1]) ** 2
                )
                ** 0.5
                for other in range(3)
                if other != target_index
            ),
            6,
        ),
        "target_dx": round(after[target_index][0] - before[target_index][0], 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    results = []
    for seed in SEEDS:
        for target_index in (0, 1):
            for direction in DIRECTIONS:
                results.append(one_step(seed, target_index, direction))
    checks = {
        "target_displacement_above_0_05": all(entry["target_shift"] > 0.05 for entry in results),
        "other_bodies_below_0_02": all(entry["other_shift"] < 0.02 for entry in results),
        "direction_controls_sign": all(
            (entry["target_dx"] > 0.02 if entry["direction"][0] > 0 else True)
            and (entry["target_dx"] < -0.02 if entry["direction"][0] < 0 else True)
            for entry in results
        ),
    }
    verdict = "action_interface_ready" if all(checks.values()) else "action_interface_unusable"
    payload = {
        "schema": "taiji-u2-action-interface-v1",
        "card": "physical action-interface feasibility",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "targets": [0, 1],
            "directions": [list(direction) for direction in DIRECTIONS],
            "command": "act one-hot[0], target=command[9:11], direction=command[11:13], intensity=1, pulse=0.05",
        },
        "checks": checks,
        "results": results,
        "verdict": verdict,
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "checks": checks, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "action_interface_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
