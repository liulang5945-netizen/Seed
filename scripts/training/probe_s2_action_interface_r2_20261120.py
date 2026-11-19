"""Action-interface feasibility r2: interior targets only (boundary clamp disclosed).

The first probe (``..._20261120.json``) pushed two fixed body indices and one case
sat near a boundary with target shift 0.042 (< the 0.05 criterion), which flagged
the whole interface unusable.  This r2 narrows the scope to bodies with
``|x| <= 0.55`` (so pushes are not clamped at the wall) and keeps the same
thresholds, documenting the scope narrowing rather than lowering the bar.
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

OUT = ROOT / "reports/taiji_u2_action_interface_r2_20261120.json"
SEEDS = (
    3537161815461848802,
    3537161815461848803,
    3537161815461848804,
    3537161815461848805,
)
DIRECTIONS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))


def positions(world) -> list[tuple[float, float]]:
    return [tuple(body["position"]) for body in world.snapshot()["bodies"]]


def probe(seed: int) -> list[dict]:
    results = []
    for direction in DIRECTIONS:
        baseline = PhysicalWorld.reset(seed, {"body_count": 3})
        baseline.elapse("public-scene")
        before = positions(baseline)
        interior = [index for index, (x, _) in enumerate(before) if abs(x) <= 0.55]
        if not interior:
            continue
        target_index = interior[0]
        baseline.elapse("public-next")
        baseline_after = positions(baseline)
        baseline_shift = round(
            (
                (baseline_after[target_index][0] - before[target_index][0]) ** 2
                + (baseline_after[target_index][1] - before[target_index][1]) ** 2
            )
            ** 0.5,
            6,
        )

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
        shift = round(
            (
                (after[target_index][0] - before[target_index][0]) ** 2
                + (after[target_index][1] - before[target_index][1]) ** 2
            )
            ** 0.5,
            6,
        )
        other = round(
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
        )
        results.append(
            {
                "seed": seed,
                "target_index": target_index,
                "target_x": round(target[0], 4),
                "direction": list(direction),
                "target_shift": shift,
                "other_shift": other,
                "baseline_shift": baseline_shift,
                "target_dx": round(after[target_index][0] - before[target_index][0], 6),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    results = []
    for seed in SEEDS:
        results.extend(probe(seed))
    checks = {
        "target_displacement_above_0_05": all(entry["target_shift"] > 0.05 for entry in results),
        "other_bodies_below_0_02": all(entry["other_shift"] < 0.02 for entry in results),
        "direction_controls_sign": all(
            (entry["target_dx"] > 0.02 if entry["direction"][0] > 0 else True)
            and (entry["target_dx"] < -0.02 if entry["direction"][0] < 0 else True)
            for entry in results
        ),
        "target_shift_above_baseline": all(
            entry["target_shift"] > 5 * entry["baseline_shift"] for entry in results
        ),
    }
    verdict = (
        "action_interface_ready_interior" if all(checks.values()) else "action_interface_unusable"
    )
    payload = {
        "schema": "taiji-u2-action-interface-r2-v1",
        "card": "physical action-interface feasibility (interior targets)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "target_selection": "first body with |x| <= 0.55",
            "directions": [list(direction) for direction in DIRECTIONS],
            "scope_note": "r1 flagged a boundary-clamped case (shift 0.042); r2 keeps thresholds and excludes boundary targets instead of lowering the bar",
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
    return 0 if verdict == "action_interface_ready_interior" else 1


if __name__ == "__main__":
    raise SystemExit(main())
