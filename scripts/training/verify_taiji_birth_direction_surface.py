"""09.6.7 verification: is the consensus rule a platform or a spike?

Fifteen declared cells (parts x units_per_group), all published. The pass/fail
thresholds were preregistered before this run. Arm implementation is imported
from 09.6.4; only the grid and the platform criterion are new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_birth_direction as rule  # noqa: E402
import scripts.training.verify_taiji_living_fragment_pricing_r028 as pricing  # noqa: E402
from taiji import birth_direction as bd  # noqa: E402
from taiji import growth_credit as gc  # noqa: E402
from taiji import native_development as development  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402

PARTS_GRID = (2, 3, 4, 5, 6)
UNITS_GRID = (1, 2, 3)
RANDOM_SEEDS = (20261101, 20261112)
ABSOLUTE_GATE = 1e-4
RETENTION_GATE = 0.030
BASELINE_REFERENCE = 4.33e-7
BASELINE_LOW = 0.2
BASELINE_HIGH = 5.0
PLATFORM_CELLS = 6
PLATFORM_PARTS = 3
BUDGET_CPU_SECONDS = 1800
BUDGET_WALL_SECONDS = 3600


class Budget(Exception):
    pass


def model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def random_directions(parent, count: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    directions = {}
    for group in range(parent.config.groups):
        vectors = torch.randn(parent.P.shape[1], count, generator=generator, dtype=torch.float64)
        vectors = vectors / vectors.norm(dim=0, keepdim=True)
        directions[("random", group)] = [
            vectors[:, index].detach().clone() for index in range(count)
        ]
    return directions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("09.6.7 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()

    def guard() -> None:
        if time.process_time() - total_cpu > BUDGET_CPU_SECONDS:
            raise Budget("cpu budget exhausted")
        if time.perf_counter() - total_wall > BUDGET_WALL_SECONDS:
            raise Budget("wall budget exhausted")

    leak_hits = rule.leak_scan((ROOT / "taiji/birth_direction.py").read_text(encoding="utf-8"))
    torch.manual_seed(20261101)
    parent = LivingCoreReference(pricing.PRINCIPLE).double()
    parent_digest = model_digest(parent)
    book, fit_credits, verdicts = rule._birth_inputs(parent)

    cells: list[dict] = []
    random_arms: list[dict] = []
    baseline_row: dict | None = None
    budget_ok = True
    try:
        for units in UNITS_GRID:
            for parts in PARTS_GRID:
                guard()
                child, _ = gc.birth(
                    deepcopy(parent),
                    bd.consensus_directions(
                        deepcopy(parent), pricing.LONG, parts=parts, count=units
                    ),
                    rule._credits(parent),
                )
                row = rule.measure(child, rule.open_gates(bd.RULE_NAME), guard)
                cells.append({"parts": parts, "units_per_group": units, **row})
            for seed in RANDOM_SEEDS:
                guard()
                child, _ = gc.birth(
                    deepcopy(parent), random_directions(parent, units, seed), rule._credits(parent)
                )
                random_arms.append(
                    {
                        "units_per_group": units,
                        "seed": seed,
                        **rule.measure(child, rule.open_gates("random"), guard),
                    }
                )
            print(f"units {units} done cpu={time.process_time() - total_cpu:.1f}", flush=True)
        guard()
        baseline_child, baseline_provider, _plan = development.coverage_birth_arm(
            deepcopy(parent), book, fit_credits, verdicts
        )
        baseline_row = rule.measure(baseline_child, baseline_provider, guard)
    except Budget as exc:
        budget_ok = False
        cells.append({"status": "budget_stop", "reason": str(exc)})

    scored = [cell for cell in cells if "newborn_contribution" in cell]
    passing = [
        {"parts": cell["parts"], "units_per_group": cell["units_per_group"]}
        for cell in scored
        if cell["newborn_contribution"] >= ABSOLUTE_GATE
        and cell["retention_drift"] <= RETENTION_GATE
    ]
    passing_parts = sorted({row["parts"] for row in passing})
    random_breaches = [
        {
            "units_per_group": arm["units_per_group"],
            "seed": arm["seed"],
            "contribution": arm["newborn_contribution"],
        }
        for arm in random_arms
        if arm["newborn_contribution"] >= ABSOLUTE_GATE
    ]
    baseline_ratio = (
        None if baseline_row is None else baseline_row["newborn_contribution"] / BASELINE_REFERENCE
    )
    baseline_ok = bool(
        baseline_ratio is not None and BASELINE_LOW <= baseline_ratio <= BASELINE_HIGH
    )
    leak_free = all(count == 0 for count in leak_hits.values())

    if not leak_free:
        status = "leak_detected"
    elif not budget_ok:
        status = "budget_stop"
    elif not baseline_ok:
        status = "baseline_face_mismatch"
    elif random_breaches:
        status = "metric_not_discriminating"
    elif not passing:
        status = "no_cell_passes"
    elif len(passing) >= PLATFORM_CELLS and len(passing_parts) >= PLATFORM_PARTS:
        status = "platform_supported"
    else:
        status = "spike_only"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {"id": "no_leak_in_rule_source", "passed": leak_free, "observed": leak_hits},
        {"id": "baseline_face_reproduced", "passed": baseline_ok, "observed": baseline_ratio},
        {
            "id": "random_controls_below_gate",
            "passed": not random_breaches,
            "observed": random_breaches,
        },
        {
            "id": "grid_fully_published",
            "passed": len(scored) == len(PARTS_GRID) * len(UNITS_GRID),
            "observed": [len(scored), len(PARTS_GRID) * len(UNITS_GRID)],
        },
        {
            "id": "platform_supported",
            "passed": len(passing) >= PLATFORM_CELLS and len(passing_parts) >= PLATFORM_PARTS,
            "observed": {"cells": len(passing), "parts": passing_parts},
        },
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    report = {
        "schema": "taiji-birth-direction-surface-v1",
        "step": "09.6.7",
        "date": "2026-11-01",
        "status": status,
        "config": asdict(pricing.PRINCIPLE),
        "parent_model_digest": parent_digest,
        "declared": {
            "parts_grid": list(PARTS_GRID),
            "units_grid": list(UNITS_GRID),
            "random_seeds": list(RANDOM_SEEDS),
            "absolute_gate": ABSOLUTE_GATE,
            "retention_gate": RETENTION_GATE,
            "baseline_reference": BASELINE_REFERENCE,
            "baseline_band": [BASELINE_LOW, BASELINE_HIGH],
            "platform_cells": PLATFORM_CELLS,
            "platform_parts": PLATFORM_PARTS,
        },
        "cells": cells,
        "random_arms": random_arms,
        "baseline_fit_svd": baseline_row,
        "passing_cells": passing,
        "passing_parts": passing_parts,
        "baseline_ratio": baseline_ratio,
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "Absolute contribution on one deterministic subject over synthetic Chinese careers. "
            "The baseline arm allocates its own units, so it is independent of units_per_group and "
            "was run once instead of three times. Nothing here prices the birth cost or shows a "
            "capability, and 09.7 is not admitted."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/living_core.py",
                "taiji/growth_credit.py",
                "taiji/native_development.py",
                "taiji/birth_direction.py",
                "scripts/training/verify_taiji_birth_direction.py",
                "scripts/training/verify_taiji_birth_direction_surface.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "passing": len(passing),
                "passing_parts": passing_parts,
                "cells": [
                    {
                        "parts": cell["parts"],
                        "units": cell["units_per_group"],
                        "contrib": round(cell["newborn_contribution"], 9),
                        "drift": round(cell["retention_drift"], 6),
                    }
                    for cell in scored
                ],
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
