"""09.6.6 verification: replicate the consensus rule across its own window count.

The subject is deterministic by design, so the varied axis is the rule's `parts`
(the number of disjoint fit windows), not a seed. Arm implementation is imported
from 09.6.4 on purpose: only the axis and the criterion are new.
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

PARTS_GRID = (2, 3, 4)
RANDOM_SEEDS = (20261101, 20261112)
UNITS_PER_GROUP = 2
ABSOLUTE_GATE = 1e-4
RETENTION_GATE = 0.030
BASELINE_REFERENCE = 4.33e-7
BASELINE_LOW = 0.2
BASELINE_HIGH = 5.0
REQUIRED_SUPPORT = 2
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
        raise FileExistsError("09.6.6 report is write-once")

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

    rows = []
    budget_ok = True
    try:
        for parts in PARTS_GRID:
            entry = {"parts": parts}
            guard()
            consensus_child, _ = gc.birth(
                deepcopy(parent),
                bd.consensus_directions(
                    deepcopy(parent), pricing.LONG, parts=parts, count=UNITS_PER_GROUP
                ),
                rule._credits(parent),
            )
            entry["consensus"] = rule.measure(consensus_child, rule.open_gates(bd.RULE_NAME), guard)
            entry["random"] = []
            for seed in RANDOM_SEEDS:
                guard()
                random_child, _ = gc.birth(
                    deepcopy(parent),
                    random_directions(parent, UNITS_PER_GROUP, seed),
                    rule._credits(parent),
                )
                row = rule.measure(random_child, rule.open_gates("random"), guard)
                entry["random"].append({"seed": seed, **row})
            guard()
            baseline_child, baseline_provider, _plan = development.coverage_birth_arm(
                deepcopy(parent), book, fit_credits, verdicts
            )
            entry["baseline_fit_svd"] = rule.measure(baseline_child, baseline_provider, guard)
            rows.append(entry)
            print(f"parts {parts} done cpu={time.process_time() - total_cpu:.1f}", flush=True)
    except Budget as exc:
        budget_ok = False
        rows.append({"status": "budget_stop", "reason": str(exc)})

    scored = [row for row in rows if "consensus" in row]
    supported = [
        row["parts"]
        for row in scored
        if row["consensus"]["newborn_contribution"] >= ABSOLUTE_GATE
        and row["consensus"]["retention_drift"] <= RETENTION_GATE
    ]
    random_breaches = [
        {"parts": row["parts"], "seed": arm["seed"], "contribution": arm["newborn_contribution"]}
        for row in scored
        for arm in row["random"]
        if arm["newborn_contribution"] >= ABSOLUTE_GATE
    ]
    baseline_ratios = [
        row["baseline_fit_svd"]["newborn_contribution"] / BASELINE_REFERENCE for row in scored
    ]
    baseline_ok = bool(
        baseline_ratios and all(BASELINE_LOW <= ratio <= BASELINE_HIGH for ratio in baseline_ratios)
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
    elif len(supported) >= REQUIRED_SUPPORT:
        status = "hypothesis_h1_supported"
    elif supported:
        status = "hypothesis_h1_not_replicated"
    else:
        status = "hypothesis_h1_refuted"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {"id": "no_leak_in_rule_source", "passed": leak_free, "observed": leak_hits},
        {"id": "baseline_face_reproduced", "passed": baseline_ok, "observed": baseline_ratios},
        {
            "id": "random_control_below_gate",
            "passed": not random_breaches,
            "observed": random_breaches,
        },
        {
            "id": "hypothesis_h1_supported",
            "passed": len(supported) >= REQUIRED_SUPPORT and not random_breaches,
            "observed": supported,
        },
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    report = {
        "schema": "taiji-birth-direction-parts-v1",
        "step": "09.6.6",
        "date": "2026-11-01",
        "status": status,
        "config": asdict(pricing.PRINCIPLE),
        "parent_model_digest": parent_digest,
        "declared": {
            "parts_grid": list(PARTS_GRID),
            "random_seeds": list(RANDOM_SEEDS),
            "units_per_group": UNITS_PER_GROUP,
            "absolute_gate": ABSOLUTE_GATE,
            "retention_gate": RETENTION_GATE,
            "baseline_reference": BASELINE_REFERENCE,
            "baseline_band": [BASELINE_LOW, BASELINE_HIGH],
            "required_support": REQUIRED_SUPPORT,
            "varied_axis": "rule window count (parts); the subject is deterministic",
        },
        "rows": rows,
        "supported_parts": supported,
        "random_breaches": random_breaches,
        "baseline_ratios": baseline_ratios,
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "Absolute contribution on synthetic Chinese careers, one deterministic subject, "
            "replicated over the rule's window count only. It does not show that birth pays "
            "for its own cost, that any task capability exists, or that 09.7 is admitted."
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
                "scripts/training/verify_taiji_birth_direction_parts.py",
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
                "rows": [
                    {
                        "parts": row["parts"],
                        "consensus": round(row["consensus"]["newborn_contribution"], 12),
                        "random": [round(arm["newborn_contribution"], 12) for arm in row["random"]],
                        "baseline": round(row["baseline_fit_svd"]["newborn_contribution"], 12),
                        "drift": round(row["consensus"]["retention_drift"], 6),
                    }
                    for row in scored
                ],
                "parent_digest": parent_digest[:16],
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
