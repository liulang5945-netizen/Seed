"""09.7b verification: locate the longest checkpoint that still holds retention.

Same arms and schedule as 09.7, but the retention probe is taken at every
checkpoint instead of only at the end, so the horizon can be attributed.
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
import scripts.training.verify_taiji_netbenefit_second as second  # noqa: E402
from taiji import birth_direction as bd  # noqa: E402
from taiji import growth_credit as gc  # noqa: E402
from taiji import native_development as development  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402

SEED = 20261101
PARTS = 5
UNITS_PER_GROUP = 3
BLOCK = 8
BLOCKS = 4
LR = 1e-5
NET_GATE = 1e-4
RETENTION_GATE = 0.030
REFERENCE_NET_AT_8 = 1.73924e-4
FACE_TOLERANCE = 0.20
BUDGET_CPU_SECONDS = 1800
BUDGET_WALL_SECONDS = 3600
CAREERS = second.EVALUATION_CAREERS


class Budget(Exception):
    pass


def run_arm(model, provider, guard) -> dict:
    pre = {
        name: development.evaluate_career_nll(model, journal, provider)
        for name, journal in CAREERS.items()
    }
    retention_pre = development.evaluate_career_nll(model, pricing.RETENTION, provider)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    checkpoints = []
    train_cpu = 0.0
    for block in range(BLOCKS):
        guard()
        _losses, cpu, _wall = pricing.timed(
            lambda: development.declared_feedback_updates(
                model, pricing.LONG, optimizer, provider, steps=BLOCK
            )
        )
        train_cpu += cpu
        delta = {
            name: development.evaluate_career_nll(model, journal, provider) - pre[name]
            for name, journal in CAREERS.items()
        }
        retention = development.evaluate_career_nll(model, pricing.RETENTION, provider)
        checkpoints.append(
            {
                "steps": (block + 1) * BLOCK,
                "delta": delta,
                "retention_drift": abs(retention - retention_pre),
            }
        )
    return {"checkpoints": checkpoints, "train_cpu_seconds": train_cpu}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("09.7b report is write-once")

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
        replay = run_arm(deepcopy(parent), None, guard)
        guard()

        def build_child():
            directions = bd.consensus_directions(
                deepcopy(parent), pricing.LONG, parts=PARTS, count=UNITS_PER_GROUP
            )
            return gc.birth(deepcopy(parent), directions, rule._credits(parent))[0]

        child, birth_cpu, _birth_wall = pricing.timed(build_child)
        guard()
        birth = run_arm(child, rule.open_gates(bd.RULE_NAME), guard)
        birth["birth_cpu_seconds"] = birth_cpu
    except Budget as exc:
        budget_ok = False
        replay = {"status": "budget_stop", "reason": str(exc)}
        birth = {"status": "budget_stop", "reason": str(exc)}

    timeline = []
    if "checkpoints" in replay and "checkpoints" in birth:
        for replay_point, birth_point in zip(
            replay["checkpoints"], birth["checkpoints"], strict=True
        ):
            timeline.append(
                {
                    "steps": replay_point["steps"],
                    "replay_retention_drift": replay_point["retention_drift"],
                    "birth_retention_drift": birth_point["retention_drift"],
                    "net_benefit": {
                        name: replay_point["delta"][name] - birth_point["delta"][name]
                        for name in CAREERS
                    },
                }
            )

    sustainable = [
        point
        for point in timeline
        if point["replay_retention_drift"] <= RETENTION_GATE
        and point["birth_retention_drift"] <= RETENTION_GATE
    ]
    horizon = sustainable[-1] if sustainable else None
    first = timeline[0] if timeline else None
    face_ok = bool(
        first
        and all(
            abs(first["net_benefit"][name] - REFERENCE_NET_AT_8)
            <= FACE_TOLERANCE * REFERENCE_NET_AT_8
            for name in CAREERS
        )
        and first["replay_retention_drift"] <= RETENTION_GATE
        and first["birth_retention_drift"] <= RETENTION_GATE
    )
    if not budget_ok:
        status = "budget_stop"
    elif not face_ok:
        status = "arm_face_mismatch"
    elif horizon is None:
        status = "no_sustainable_horizon"
    elif all(value >= NET_GATE for value in horizon["net_benefit"].values()):
        status = "sustainable_horizon_found"
    else:
        status = "sustainable_but_no_net_benefit"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {
            "id": "eight_step_face_reproduced",
            "passed": face_ok,
            "observed": None if first is None else first["net_benefit"],
        },
        {
            "id": "sustainable_horizon_exists",
            "passed": horizon is not None,
            "observed": None if horizon is None else horizon["steps"],
        },
        {
            "id": "net_benefit_at_horizon",
            "passed": bool(
                horizon and all(value >= NET_GATE for value in horizon["net_benefit"].values())
            ),
            "observed": None if horizon is None else horizon["net_benefit"],
        },
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    report = {
        "schema": "taiji-fragment-horizon-v1",
        "step": "09.7b",
        "date": "2026-11-01",
        "status": status,
        "config": asdict(pricing.PRINCIPLE),
        "parent_model_digest": digest,
        "declared": {
            "parts": PARTS,
            "units_per_group": UNITS_PER_GROUP,
            "block": BLOCK,
            "blocks": BLOCKS,
            "lr": LR,
            "net_gate": NET_GATE,
            "retention_gate": RETENTION_GATE,
            "reference_net_at_8": REFERENCE_NET_AT_8,
            "face_tolerance": FACE_TOLERANCE,
        },
        "timeline": timeline,
        "sustainable_horizon_steps": None if horizon is None else horizon["steps"],
        "arms": {"replay_only": replay, "consensus_birth": birth},
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "The horizon is the longest declared checkpoint where both arms keep retention "
            "inside the gate on this deterministic subject. It is not a sustainable limit of "
            "the mechanism, not a capability and not training admission."
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
                "scripts/training/verify_taiji_fragment_horizon.py",
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
                "horizon": None if horizon is None else horizon["steps"],
                "timeline": [
                    {
                        "steps": point["steps"],
                        "replay_drift": round(point["replay_retention_drift"], 6),
                        "birth_drift": round(point["birth_retention_drift"], 6),
                        "net": {k: round(v, 9) for k, v in point["net_benefit"].items()},
                    }
                    for point in timeline
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
