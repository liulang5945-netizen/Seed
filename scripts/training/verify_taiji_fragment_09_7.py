"""09.7 verification: does the birth net benefit hold across a 32-step fragment?

Single declared candidate (parts=5, units=3). Two arms, four checkpoints, both
evaluation careers, retention probe kept separate from the evaluation faces.
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
CPU_PER_GATE = 6.6
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
    losses: list[float] = []
    checkpoints = []
    train_cpu = 0.0
    for block in range(BLOCKS):
        guard()
        block_losses, cpu, _wall = pricing.timed(
            lambda: development.declared_feedback_updates(
                model, pricing.LONG, optimizer, provider, steps=BLOCK
            )
        )
        losses.extend(block_losses)
        train_cpu += cpu
        current = {
            name: development.evaluate_career_nll(model, journal, provider)
            for name, journal in CAREERS.items()
        }
        checkpoints.append(
            {
                "steps": (block + 1) * BLOCK,
                "delta": {name: current[name] - pre[name] for name in CAREERS},
                "loss_last": block_losses[-1],
            }
        )
    retention_post = development.evaluate_career_nll(model, pricing.RETENTION, provider)
    return {
        "pre": pre,
        "checkpoints": checkpoints,
        "retention_drift": abs(retention_post - retention_pre),
        "train_cpu_seconds": train_cpu,
        "first_update_loss": losses[0],
        "last_update_loss": losses[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("09.7 report is write-once")

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
        birth["birth_plus_train_cpu_seconds"] = birth_cpu + birth["train_cpu_seconds"]
        guard()
        gc.lesion_trained(child)
        holdout_after_lesion = development.evaluate_career_nll(child, pricing.HOLDOUT)
        birth["holdout_after_lesion"] = holdout_after_lesion
    except Budget as exc:
        budget_ok = False
        replay = {"status": "budget_stop", "reason": str(exc)}
        birth = {"status": "budget_stop", "reason": str(exc)}

    net_by_checkpoint = []
    if "checkpoints" in replay and "checkpoints" in birth:
        for replay_point, birth_point in zip(
            replay["checkpoints"], birth["checkpoints"], strict=True
        ):
            net_by_checkpoint.append(
                {
                    "steps": replay_point["steps"],
                    "net_benefit": {
                        name: replay_point["delta"][name] - birth_point["delta"][name]
                        for name in CAREERS
                    },
                }
            )
    final_net = net_by_checkpoint[-1]["net_benefit"] if net_by_checkpoint else {}
    first_net = net_by_checkpoint[0]["net_benefit"] if net_by_checkpoint else {}
    held = bool(final_net) and all(value >= NET_GATE for value in final_net.values())
    face_ok = bool(first_net) and all(
        abs(first_net[name] - REFERENCE_NET_AT_8) <= FACE_TOLERANCE * REFERENCE_NET_AT_8
        for name in CAREERS
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
    elif held:
        status = "fragment_net_benefit_held"
    else:
        status = "fragment_net_benefit_decayed"

    extra_cpu = (
        birth["birth_plus_train_cpu_seconds"] - replay["train_cpu_seconds"]
        if "birth_plus_train_cpu_seconds" in birth and "train_cpu_seconds" in replay
        else None
    )
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {
            "id": "retention_within_gate",
            "passed": retention_ok,
            "observed": [replay.get("retention_drift"), birth.get("retention_drift")],
        },
        {"id": "eight_step_face_reproduced", "passed": face_ok, "observed": first_net},
        {"id": "net_benefit_held_at_32_steps", "passed": held, "observed": final_net},
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    report = {
        "schema": "taiji-joint-fragment-09-7-v1",
        "step": "09.7",
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
            "cpu_per_gate_reference": CPU_PER_GATE,
            "parts_4_must_not_be_substituted": True,
        },
        "arms": {"replay_only": replay, "consensus_birth": birth},
        "net_by_checkpoint": net_by_checkpoint,
        "extra_cpu_seconds": extra_cpu,
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "One deterministic subject, synthetic Chinese careers, single direction rule. "
            "Holding the net benefit across 32 steps is not a capability, not a long-horizon "
            "solvency proof and not training admission."
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
                "scripts/training/verify_taiji_fragment_09_7.py",
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
                "net_by_checkpoint": [
                    {
                        "steps": point["steps"],
                        "net": {k: round(v, 9) for k, v in point["net_benefit"].items()},
                    }
                    for point in net_by_checkpoint
                ],
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
