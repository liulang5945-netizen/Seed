"""09.6.3 diagnosis: why do coverage-birth newborn units carry no measurable benefit?

Four negative readings say the newborn branches do not pay. This instrument asks
the intermediate question instead of rerunning the hypothesis: do the newborn
parameters receive gradient at all, do they move, and how large is their output
contribution relative to the inherited parent rows?

Synthetic Chinese fixtures only; no sealed data, no product default, no training.
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
BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200


def birth_inputs(parent):
    book, _ = gc.collect_credit(parent, pricing.LONG)
    steps = len(book.deltas)
    fit_window, check_window = range(steps // 2), range(steps // 2, steps)
    tau_map = {
        name: gc.calibrate_tau_noise(book, branch, fit_window)
        for branch, name in enumerate(book.branch_ids)
    }
    fit_credits, verdicts = {}, {}
    for branch, name in enumerate(book.branch_ids):
        fit_credits[name] = gc.branch_credit(book, branch, fit_window)
        verdicts[name] = gc.branch_verdict(book, branch, tau_map[name], fit_window, check_window)
    return book, fit_credits, verdicts


def norm(tensor) -> float:
    return float(tensor.detach().norm())


def gradient_norms(model, journal, provider) -> dict:
    """||dL/dR|| and ||dL/dP|| split into parent rows/cols and newborn rows/cols."""
    model.zero_grad(set_to_none=True)
    loss = cl.career_nll(model, journal, provider)
    loss.backward()
    split = model.growth_parent_units
    report = {
        "loss": float(loss.detach()),
        "R_total": norm(model.R.grad),
        "P_total": norm(model.P.grad),
        "R_parent": norm(model.R.grad[:, :split, :]),
        "R_newborn": norm(model.R.grad[:, split:, :]),
        "P_parent": norm(model.P.grad[:, :, :split]),
        "P_newborn": norm(model.P.grad[:, :, split:]),
    }
    model.zero_grad(set_to_none=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("09.6.3 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    torch.manual_seed(SEED)

    parent, _ = pricing.fresh(pricing.PRINCIPLE, "diagnose-birth")
    book, fit_credits, verdicts = birth_inputs(parent)
    child, provider, plan = development.coverage_birth_arm(
        deepcopy(parent), book, fit_credits, verdicts
    )
    split = child.growth_parent_units

    holdout_before = development.evaluate_career_nll(child, pricing.HOLDOUT, provider)
    fit_before = development.evaluate_career_nll(child, pricing.LONG, provider)
    grad_holdout = gradient_norms(child, pricing.HOLDOUT, provider)
    grad_fit = gradient_norms(child, pricing.LONG, provider)

    birth_state = {
        "R_parent_norm": norm(child.R[:, :split, :]),
        "R_newborn_norm": norm(child.R[:, split:, :]),
        "P_parent_norm": norm(child.P[:, :, :split]),
        "P_newborn_norm": norm(child.P[:, :, split:]),
        "P_newborn_over_parent": norm(child.P[:, :, split:]) / norm(child.P[:, :, :split]),
    }

    optimizer = torch.optim.Adam(child.parameters(), lr=LR)
    losses = development.declared_feedback_updates(
        child, pricing.LONG, optimizer, provider, steps=STEPS
    )
    holdout_after = development.evaluate_career_nll(child, pricing.HOLDOUT, provider)
    trained = {
        "R_parent_norm": norm(child.R[:, :split, :]),
        "R_newborn_norm": norm(child.R[:, split:, :]),
        "P_parent_norm": norm(child.P[:, :, :split]),
        "P_newborn_norm": norm(child.P[:, :, split:]),
    }
    trained["R_newborn_abs_change"] = trained["R_newborn_norm"] - birth_state["R_newborn_norm"]
    trained["P_newborn_relative_change"] = trained["P_newborn_norm"] / max(
        birth_state["P_newborn_norm"], 1e-300
    )
    trained["R_newborn_started_at_zero"] = birth_state["R_newborn_norm"] == 0.0

    holdout_benefit = holdout_before - holdout_after
    gc.lesion_trained(child)
    holdout_lesion = development.evaluate_career_nll(child, pricing.HOLDOUT)
    newborn_contribution = holdout_lesion - holdout_after

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    report = {
        "schema": "taiji-birth-inertness-diagnosis-v1",
        "step": "09.6.3",
        "date": "2026-11-01",
        "config": asdict(pricing.PRINCIPLE),
        "declared": {"steps": STEPS, "lr": LR, "units_total": plan["units_total"]},
        "parent_units": split,
        "newborn_units": int(child.R.shape[1]) - int(split),
        "birth_parameter_norms": birth_state,
        "trained_parameter_norms": trained,
        "gradients_at_birth": {"holdout": grad_holdout, "fit": grad_fit},
        "newborn_to_parent_gradient_ratio": {
            "holdout_R": grad_holdout["R_newborn"] / max(grad_holdout["R_parent"], 1e-300),
            "fit_R": grad_fit["R_newborn"] / max(grad_fit["R_parent"], 1e-300),
        },
        "nll": {
            "fit_before": fit_before,
            "holdout_before": holdout_before,
            "holdout_after_training": holdout_after,
            "holdout_after_lesion": holdout_lesion,
            "holdout_benefit": holdout_benefit,
            "newborn_contribution": newborn_contribution,
        },
        "update_losses": losses,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "training_admitted": False,
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/living_core.py",
                "taiji/growth_credit.py",
                "taiji/native_development.py",
                "taiji/constrained_learning.py",
                "scripts/training/diagnose_birth_inertness.py",
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
                "birth_R_newborn": round(birth_state["R_newborn_norm"], 9),
                "trained_R_newborn": round(trained["R_newborn_norm"], 9),
                "R_newborn_abs_change": round(trained["R_newborn_abs_change"], 9),
                "P_newborn_over_parent": round(birth_state["P_newborn_over_parent"], 6),
                "grad_ratio_holdout": round(
                    report["newborn_to_parent_gradient_ratio"]["holdout_R"], 6
                ),
                "grad_ratio_fit": round(report["newborn_to_parent_gradient_ratio"]["fit_R"], 6),
                "holdout_benefit": round(holdout_benefit, 9),
                "newborn_contribution": round(newborn_contribution, 9),
                "cpu_seconds": round(cpu_seconds, 2),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
