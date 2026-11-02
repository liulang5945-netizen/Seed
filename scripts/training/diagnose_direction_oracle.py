"""09.6.3b reachability bound: would ANY direction rule let newborn units pay?

09.6.3 showed the newborn rows do learn but contribute 4.33e-7, and R026 showed
credit-chosen directions are no better than random ones. This instrument asks the
prior question: if the birth directions were chosen by an oracle that looks at the
holdout gradient, would the newborn branches carry a measurable share at all?

The oracle is a reachability bound, never a method: it uses the evaluation set, so
nothing here can be shipped or claimed as a learning result.
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

import scripts.training.verify_taiji_living_fragment_pricing_r028 as pricing  # noqa: E402
from taiji import constrained_learning as cl  # noqa: E402
from taiji import growth_credit as gc  # noqa: E402
from taiji import native_development as development  # noqa: E402

SEED = 20261101
STEPS = 8
LR = 1e-5
UNITS_PER_GROUP = 2
NEWBORN_SHARE_GATE = 1e-4
ABSOLUTE_CONTRIBUTION_GATE = 1e-4
BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200


def open_gates(index: int) -> dict:
    """Fully open newborn gates for the single declared oracle branch."""
    return {"oracle": 1.0}


def oracle_directions(parent) -> tuple[dict, dict]:
    """Top singular vectors of the holdout gradient on P, per group, plus fake credits."""
    parent.zero_grad(set_to_none=True)
    loss = cl.career_nll(parent, pricing.HOLDOUT)
    loss.backward()
    groups = list(range(parent.config.groups))
    directions = {}
    for group in groups:
        gradient = parent.P.grad[group].detach().clone()
        _, _, vectors = torch.linalg.svd(gradient, full_matrices=False)
        count = min(UNITS_PER_GROUP, vectors.shape[1])
        directions[("oracle", group)] = [
            vectors[:, index].detach().clone() for index in range(count)
        ]
    credits = {"oracle": torch.stack([parent.P.grad[group].detach().clone() for group in groups])}
    parent.zero_grad(set_to_none=True)
    return directions, credits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("09.6.3b report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    torch.manual_seed(SEED)

    parent, _ = pricing.fresh(pricing.PRINCIPLE, "oracle-directions")
    directions, credits = oracle_directions(parent)
    child, _plan = gc.birth(deepcopy(parent), directions, credits)
    split = child.growth_parent_units
    units_total = (int(child.R.shape[1]) - int(split)) * parent.config.groups

    holdout_pre = development.evaluate_career_nll(child, pricing.HOLDOUT, open_gates)
    optimizer = torch.optim.Adam(child.parameters(), lr=LR)
    losses = development.declared_feedback_updates(
        child, pricing.LONG, optimizer, open_gates, steps=STEPS
    )
    holdout_post = development.evaluate_career_nll(child, pricing.HOLDOUT, open_gates)
    gc.lesion_trained(child)
    holdout_lesion = development.evaluate_career_nll(child, pricing.HOLDOUT)

    benefit = holdout_pre - holdout_post
    newborn_contribution = holdout_lesion - holdout_post
    share = (newborn_contribution / benefit) if benefit else None
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    report = {
        "schema": "taiji-direction-oracle-v1",
        "step": "09.6.3b",
        "date": "2026-11-01",
        "config": asdict(pricing.PRINCIPLE),
        "declared": {
            "steps": STEPS,
            "lr": LR,
            "units_per_group": UNITS_PER_GROUP,
            "newborn_share_gate": NEWBORN_SHARE_GATE,
            "absolute_contribution_gate": ABSOLUTE_CONTRIBUTION_GATE,
            "oracle_uses_holdout": True,
            "units_total": units_total,
        },
        "parent_units": split,
        "nll": {
            "holdout_pre": holdout_pre,
            "holdout_post": holdout_post,
            "holdout_after_lesion": holdout_lesion,
            "holdout_benefit": benefit,
            "newborn_contribution": newborn_contribution,
            "newborn_share": share,
        },
        "reaches_share_gate": bool(share is not None and share >= NEWBORN_SHARE_GATE),
        "reaches_absolute_gate": bool(newborn_contribution >= ABSOLUTE_CONTRIBUTION_GATE),
        "update_losses": losses,
        "reading": (
            "If an oracle direction still yields a newborn share below the gate, no direction "
            "rule can rescue coverage birth on this task family and the growth line is closed "
            "on this reference graph. If it clears the gate, the direction rule is the "
            "bottleneck and a new mechanism hypothesis is warranted."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "training_admitted": False,
        "method_claim": False,
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/living_core.py",
                "taiji/growth_credit.py",
                "taiji/native_development.py",
                "taiji/constrained_learning.py",
                "scripts/training/diagnose_direction_oracle.py",
                "scripts/training/verify_taiji_living_fragment_pricing_r028.py",
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
                "benefit": round(benefit, 9),
                "newborn_contribution": round(newborn_contribution, 9),
                "newborn_share": None if share is None else round(share, 9),
                "reaches_share_gate": report["reaches_share_gate"],
                "reaches_absolute_gate": report["reaches_absolute_gate"],
                "cpu_seconds": round(cpu_seconds, 2),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
