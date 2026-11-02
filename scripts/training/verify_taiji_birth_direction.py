"""09.6.4 verification: does a non-leaking consensus direction rule clear the share gate?

Four arms in one run: consensus with K=2 and K=4 fit windows, the current fit-SVD
baseline, and a random-direction negative control. The four preregistered gates in
the contract all have to hold before any reading is reported as support for H1.
"""

from __future__ import annotations

import argparse
import ast
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
from taiji import birth_direction as bd  # noqa: E402
from taiji import growth_credit as gc  # noqa: E402
from taiji import native_development as development  # noqa: E402

SEED = 20261101
STEPS = 8
LR = 1e-5
UNITS_PER_GROUP = 2
PARTS_GRID = (2, 4)
SHARE_GATE = 1e-4
RETENTION_GATE = 0.030
BASELINE_SHARE = 2.146e-5
BASELINE_TOLERANCE = 0.20
BUDGET_CPU_SECONDS = 1800
BUDGET_WALL_SECONDS = 3600
LEAK_IDENTIFIERS = ("HOLDOUT", "RETENTION", "pricing", "case_eval")


class Budget(Exception):
    pass


def open_gates(label: str):
    def provider(index: int) -> dict:
        return {label: 1.0}

    return provider


def random_directions(parent, count: int) -> dict:
    generator = torch.Generator().manual_seed(SEED)
    directions = {}
    for group in range(parent.config.groups):
        size = parent.P.shape[1]
        vectors = torch.randn(size, count, generator=generator, dtype=torch.float64)
        vectors = vectors / vectors.norm(dim=0, keepdim=True)
        directions[("random", group)] = [
            vectors[:, index].detach().clone() for index in range(count)
        ]
    return directions


def measure(child, provider, guard) -> dict:
    holdout_pre = development.evaluate_career_nll(child, pricing.HOLDOUT, provider)
    retention_pre = development.evaluate_career_nll(child, pricing.RETENTION, provider)
    optimizer = torch.optim.Adam(child.parameters(), lr=LR)
    guard()
    losses, cpu, wall = pricing.timed(
        lambda: development.declared_feedback_updates(
            child, pricing.LONG, optimizer, provider, steps=STEPS
        )
    )
    holdout_post = development.evaluate_career_nll(child, pricing.HOLDOUT, provider)
    retention_post = development.evaluate_career_nll(child, pricing.RETENTION, provider)
    guard()
    report = gc.lesion_trained(child)
    holdout_lesion = development.evaluate_career_nll(child, pricing.HOLDOUT)
    benefit = holdout_pre - holdout_post
    contribution = holdout_lesion - holdout_post
    return {
        "holdout_pre": holdout_pre,
        "holdout_post": holdout_post,
        "holdout_after_lesion": holdout_lesion,
        "benefit": benefit,
        "newborn_contribution": contribution,
        "newborn_share": (contribution / benefit) if benefit else None,
        "retention_drift": abs(retention_post - retention_pre),
        "removed_units_per_group": report["removed_units_per_group"],
        "first_update_loss": losses[0],
        "last_update_loss": losses[-1],
        "cpu_seconds": cpu,
        "wall_seconds": wall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("09.6.4 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    torch.manual_seed(SEED)

    def guard() -> None:
        if time.process_time() - total_cpu > BUDGET_CPU_SECONDS:
            raise Budget("cpu budget exhausted")
        if time.perf_counter() - total_wall > BUDGET_WALL_SECONDS:
            raise Budget("wall budget exhausted")

    source = (ROOT / "taiji/birth_direction.py").read_text(encoding="utf-8")
    leak_hits = leak_scan(source)

    parent, _ = pricing.fresh(pricing.PRINCIPLE, "direction-rule")
    book, fit_credits, verdicts = _birth_inputs(parent)

    arms: dict[str, dict] = {}
    try:
        for parts in PARTS_GRID:
            guard()
            directions = bd.consensus_directions(
                deepcopy(parent), pricing.LONG, parts=parts, count=UNITS_PER_GROUP
            )
            child, _ = gc.birth(deepcopy(parent), directions, _credits(parent))
            arms[f"consensus_k{parts}"] = measure(child, open_gates(bd.RULE_NAME), guard)
        guard()
        baseline_child, baseline_provider, _plan = development.coverage_birth_arm(
            deepcopy(parent), book, fit_credits, verdicts
        )
        arms["healthy_baseline_fit_svd"] = measure(baseline_child, baseline_provider, guard)
        guard()
        random_child, _ = gc.birth(
            deepcopy(parent), random_directions(parent, UNITS_PER_GROUP), _credits(parent)
        )
        arms["random_control"] = measure(random_child, open_gates("random"), guard)
        budget_ok = True
    except Budget as exc:
        arms["status"] = {"budget_stop": str(exc)}
        budget_ok = False

    consensus = [arms.get(f"consensus_k{parts}") for parts in PARTS_GRID]
    supported = any(
        row
        and row["newborn_share"] is not None
        and row["newborn_share"] >= SHARE_GATE
        and row["retention_drift"] <= RETENTION_GATE
        for row in consensus
    )
    baseline_row = arms.get("healthy_baseline_fit_svd")
    baseline_ok = bool(
        baseline_row
        and baseline_row["newborn_share"] is not None
        and abs(baseline_row["newborn_share"] - BASELINE_SHARE)
        <= BASELINE_TOLERANCE * BASELINE_SHARE
    )
    random_row = arms.get("random_control")
    random_ok = bool(
        random_row
        and (random_row["newborn_share"] is None or random_row["newborn_share"] < SHARE_GATE)
    )
    leak_free = all(count == 0 for count in leak_hits.values())

    checks = [
        {"id": "no_leak_in_rule_source", "passed": leak_free, "observed": leak_hits},
        {
            "id": "baseline_face_reproduced",
            "passed": baseline_ok,
            "observed": None if not baseline_row else baseline_row["newborn_share"],
        },
        {
            "id": "random_control_below_gate",
            "passed": random_ok,
            "observed": None if not random_row else random_row["newborn_share"],
        },
        {
            "id": "hypothesis_h1_supported",
            "passed": supported,
            "observed": [None if row is None else row["newborn_share"] for row in consensus],
        },
    ]
    if not leak_free:
        status = "leak_detected"
    elif not budget_ok:
        status = "budget_stop"
    elif not baseline_ok:
        status = "baseline_face_mismatch"
    elif not random_ok:
        status = "metric_not_discriminating"
    elif supported:
        status = "hypothesis_h1_supported"
    else:
        status = "hypothesis_h1_refuted"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks.append(
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        }
    )
    report = {
        "schema": "taiji-birth-direction-v1",
        "step": "09.6.4",
        "date": "2026-11-01",
        "status": status,
        "config": asdict(pricing.PRINCIPLE),
        "declared": {
            "lr": LR,
            "steps": STEPS,
            "units_per_group": UNITS_PER_GROUP,
            "parts_grid": list(PARTS_GRID),
            "share_gate": SHARE_GATE,
            "retention_gate": RETENTION_GATE,
            "baseline_share": BASELINE_SHARE,
            "baseline_tolerance": BASELINE_TOLERANCE,
        },
        "arms": arms,
        "checks": checks,
        "absolute_gate_claim": False,
        "training_admitted": False,
        "limits": (
            "Share gate only. Reaching it means the newborn units carry a measurable "
            "fraction of the holdout benefit, not that the absolute benefit reaches the "
            "1e-4 NLL bar used elsewhere, and not that any capability was gained."
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
                "status": status,
                "shares": {
                    name: (
                        None if row.get("newborn_share") is None else round(row["newborn_share"], 9)
                    )
                    for name, row in arms.items()
                    if "newborn_share" in row
                },
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


def leak_scan(source: str) -> dict:
    """Identifier-level scan: prose may name the evaluation set, code must not.

    A substring scan matches the module's own prohibition sentence, so the check
    parses the AST and looks at names and attributes only.
    """
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            imports.update(f"{node.module}.{alias.name}" for alias in node.names)
    seen = names | attributes | imports
    return {token: int(token in seen) for token in LEAK_IDENTIFIERS}


def _birth_inputs(parent):
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


def _credits(parent):
    """Zero credit table: the rule under test supplies the directions, not the credit."""
    generator = torch.Generator().manual_seed(SEED)
    rows = []
    for _ in range(parent.config.groups):
        rows.append(
            torch.randn(
                parent.P.shape[1], parent.P.shape[2], generator=generator, dtype=torch.float64
            )
        )
    return {"consensus": torch.stack(rows), "random": torch.stack(rows)}


if __name__ == "__main__":
    raise SystemExit(main())
