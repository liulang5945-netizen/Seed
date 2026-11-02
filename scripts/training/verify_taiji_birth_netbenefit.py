"""09.6.8 verification: does birth beat not-birthing on the same budget?

Two arms on the same parent and the same eight declared updates: replay_only
(no birth) and consensus_birth (parts=5, units_per_group=3, the best 09.6.7 cell).
Birth and training costs are timed so the net benefit has a price next to it.
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

SEED = 20261101
PARTS = 5
UNITS_PER_GROUP = 3
STEPS = 8
LR = 1e-5
NET_GATE = 1e-4
RETENTION_GATE = 0.030
REFERENCE_CONTRIBUTION = 1.7374e-4
FACE_TOLERANCE = 0.20
BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200


class Budget(Exception):
    pass


def train_and_eval(model, provider, guard) -> dict:
    holdout_pre = development.evaluate_career_nll(model, pricing.HOLDOUT, provider)
    retention_pre = development.evaluate_career_nll(model, pricing.RETENTION, provider)
    parameters_pre = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    guard()
    losses, cpu, wall = pricing.timed(
        lambda: development.declared_feedback_updates(
            model, pricing.LONG, optimizer, provider, steps=STEPS
        )
    )
    holdout_post = development.evaluate_career_nll(model, pricing.HOLDOUT, provider)
    retention_post = development.evaluate_career_nll(model, pricing.RETENTION, provider)
    return {
        "holdout_pre": holdout_pre,
        "holdout_post": holdout_post,
        "delta_holdout": holdout_post - holdout_pre,
        "retention_drift": abs(retention_post - retention_pre),
        "parameters_before": parameters_pre,
        "parameters_after": sum(parameter.numel() for parameter in model.parameters()),
        "train_cpu_seconds": cpu,
        "train_wall_seconds": wall,
        "first_update_loss": losses[0],
        "last_update_loss": losses[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("09.6.8 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()

    def guard() -> None:
        if time.process_time() - total_cpu > BUDGET_CPU_SECONDS:
            raise Budget("cpu budget exhausted")
        if time.perf_counter() - total_wall > BUDGET_WALL_SECONDS:
            raise Budget("wall budget exhausted")

    torch.manual_seed(SEED)
    parent = LivingCoreReference(pricing.PRINCIPLE).double()
    digest = hashlib.sha256(
        b"".join(
            name.encode("utf-8") + parameter.detach().cpu().contiguous().numpy().tobytes()
            for name, parameter in sorted(parent.named_parameters())
        )
    ).hexdigest()

    budget_ok = True
    try:
        replay = train_and_eval(deepcopy(parent), None, guard)
        guard()

        def build_child():
            directions = bd.consensus_directions(
                deepcopy(parent), pricing.LONG, parts=PARTS, count=UNITS_PER_GROUP
            )
            return gc.birth(deepcopy(parent), directions, rule._credits(parent))[0]

        child, birth_cpu, birth_wall = pricing.timed(build_child)
        guard()
        birth = train_and_eval(child, rule.open_gates(bd.RULE_NAME), guard)
        guard()
        holdout_before_lesion = birth["holdout_post"]
        lesion_report = gc.lesion_trained(child)
        holdout_after_lesion = development.evaluate_career_nll(child, pricing.HOLDOUT)
        birth["newborn_contribution"] = holdout_after_lesion - holdout_before_lesion
        birth["lesion_report"] = lesion_report
        birth["birth_cpu_seconds"] = birth_cpu
        birth["birth_wall_seconds"] = birth_wall
        birth["birth_plus_train_cpu_seconds"] = birth_cpu + birth["train_cpu_seconds"]
    except Budget as exc:
        budget_ok = False
        replay = {"status": "budget_stop", "reason": str(exc)}
        birth = {"status": "budget_stop", "reason": str(exc)}

    net = None
    extra_cpu = None
    extra_cpu_per_gate = None
    if "delta_holdout" in replay and "delta_holdout" in birth:
        net = replay["delta_holdout"] - birth["delta_holdout"]
        extra_cpu = birth["birth_plus_train_cpu_seconds"] - replay["train_cpu_seconds"]
        extra_cpu_per_gate = (extra_cpu / (net / NET_GATE)) if net > 0 else None
    face_ok = bool(
        "newborn_contribution" in birth
        and abs(birth["newborn_contribution"] - REFERENCE_CONTRIBUTION)
        <= FACE_TOLERANCE * REFERENCE_CONTRIBUTION
    )
    retention_ok = bool(
        "retention_drift" in replay
        and "retention_drift" in birth
        and replay["retention_drift"] <= RETENTION_GATE
        and birth["retention_drift"] <= RETENTION_GATE
    )
    if not budget_ok:
        status = "budget_stop"
    elif not face_ok:
        status = "arm_face_mismatch"
    elif not retention_ok:
        status = "retention_breach"
    elif net is not None and net >= NET_GATE:
        status = "net_benefit_supported"
    else:
        status = "net_benefit_refuted"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {
            "id": "retention_within_gate",
            "passed": retention_ok,
            "observed": [replay.get("retention_drift"), birth.get("retention_drift")],
        },
        {
            "id": "birth_face_matches_09_6_7",
            "passed": face_ok,
            "observed": birth.get("newborn_contribution"),
        },
        {
            "id": "net_benefit_supported",
            "passed": net is not None and net >= NET_GATE,
            "observed": net,
        },
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    report = {
        "schema": "taiji-birth-netbenefit-v1",
        "step": "09.6.8",
        "date": "2026-11-01",
        "status": status,
        "config": asdict(pricing.PRINCIPLE),
        "parent_model_digest": digest,
        "declared": {
            "parts": PARTS,
            "units_per_group": UNITS_PER_GROUP,
            "steps": STEPS,
            "lr": LR,
            "net_gate": NET_GATE,
            "retention_gate": RETENTION_GATE,
            "reference_contribution": REFERENCE_CONTRIBUTION,
            "face_tolerance": FACE_TOLERANCE,
        },
        "arms": {"replay_only": replay, "consensus_birth": birth},
        "net_benefit": net,
        "extra_cpu_seconds": extra_cpu,
        "extra_cpu_per_gate_of_nll": extra_cpu_per_gate,
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "Net benefit on one deterministic subject over synthetic Chinese careers. The cost "
            "figures are a pricing input, not a threshold: no cost gate was declared. No task "
            "capability is shown and 09.7 is not admitted."
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
                "scripts/training/verify_taiji_birth_netbenefit.py",
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
                "replay_delta": None if net is None else round(replay["delta_holdout"], 9),
                "birth_delta": None if net is None else round(birth["delta_holdout"], 9),
                "net_benefit": None if net is None else round(net, 9),
                "newborn_contribution": (
                    None
                    if "newborn_contribution" not in birth
                    else round(birth["newborn_contribution"], 9)
                ),
                "birth_cpu": None if extra_cpu is None else round(birth["birth_cpu_seconds"], 3),
                "replay_train_cpu": None if net is None else round(replay["train_cpu_seconds"], 3),
                "birth_train_cpu": None if net is None else round(birth["train_cpu_seconds"], 3),
                "extra_cpu": None if extra_cpu is None else round(extra_cpu, 3),
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
