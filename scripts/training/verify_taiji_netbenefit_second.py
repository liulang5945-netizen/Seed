"""09.6.9 verification: does the birth net benefit survive a second evaluation career?

Same two arms as 09.6.8, evaluated on HOLDOUT and on a newly declared second
career. RETENTION stays the retention probe and is never an evaluation face here.
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
from taiji.state_migration import CareerJournal  # noqa: E402

SEED = 20261101
PARTS = 5
UNITS_PER_GROUP = 3
STEPS = 8
LR = 1e-5
NET_GATE = 1e-4
RETENTION_GATE = 0.030
REFERENCE_NET = 1.73924e-4
FACE_TOLERANCE = 0.20
BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200

SECOND_HOLDOUT = CareerJournal(
    subject_id="second-holdout",
    prompt=("另一组材料：黄色雨伞放在门厅。随后更正：黄色雨伞被带到车里。" * 6).encode(),
    target="黄色雨伞现在在车里。".encode() * 2,
    goal=pricing.CHINESE_GOAL,
    records=pricing.RECORDS,
    raw_nu=pricing.NU,
    known_nu=pricing.KNOWN,
    now=10,
)
EVALUATION_CAREERS = {"holdout": pricing.HOLDOUT, "second_holdout": SECOND_HOLDOUT}


class Budget(Exception):
    pass


def arm(model, provider, guard) -> dict:
    pre = {
        name: development.evaluate_career_nll(model, journal, provider)
        for name, journal in EVALUATION_CAREERS.items()
    }
    retention_pre = development.evaluate_career_nll(model, pricing.RETENTION, provider)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    guard()
    losses, cpu, wall = pricing.timed(
        lambda: development.declared_feedback_updates(
            model, pricing.LONG, optimizer, provider, steps=STEPS
        )
    )
    post = {
        name: development.evaluate_career_nll(model, journal, provider)
        for name, journal in EVALUATION_CAREERS.items()
    }
    retention_post = development.evaluate_career_nll(model, pricing.RETENTION, provider)
    return {
        "pre": pre,
        "post": post,
        "delta": {name: post[name] - pre[name] for name in EVALUATION_CAREERS},
        "retention_drift": abs(retention_post - retention_pre),
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
        raise FileExistsError("09.6.9 report is write-once")

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
        replay = arm(deepcopy(parent), None, guard)
        guard()

        def build_child():
            directions = bd.consensus_directions(
                deepcopy(parent), pricing.LONG, parts=PARTS, count=UNITS_PER_GROUP
            )
            return gc.birth(deepcopy(parent), directions, rule._credits(parent))[0]

        child, birth_cpu, birth_wall = pricing.timed(build_child)
        guard()
        birth = arm(child, rule.open_gates(bd.RULE_NAME), guard)
        birth["birth_cpu_seconds"] = birth_cpu
        birth["birth_wall_seconds"] = birth_wall
        birth["birth_plus_train_cpu_seconds"] = birth_cpu + birth["train_cpu_seconds"]
    except Budget as exc:
        budget_ok = False
        replay = {"status": "budget_stop", "reason": str(exc)}
        birth = {"status": "budget_stop", "reason": str(exc)}

    net = (
        {name: replay["delta"][name] - birth["delta"][name] for name in EVALUATION_CAREERS}
        if "delta" in replay and "delta" in birth
        else {}
    )
    both_pass = bool(net) and all(value >= NET_GATE for value in net.values())
    only_first = bool(net) and net.get("holdout", 0.0) >= NET_GATE and not both_pass
    face_ok = bool(net and abs(net["holdout"] - REFERENCE_NET) <= FACE_TOLERANCE * REFERENCE_NET)
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
    elif both_pass:
        status = "replicated_on_second_holdout"
    elif only_first:
        status = "fixture_specific"
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
        {"id": "arm_matches_09_6_8", "passed": face_ok, "observed": net.get("holdout")},
        {"id": "net_benefit_on_both_careers", "passed": both_pass, "observed": net},
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    report = {
        "schema": "taiji-netbenefit-second-v1",
        "step": "09.6.9",
        "date": "2026-11-01",
        "status": status,
        "config": asdict(pricing.PRINCIPLE),
        "parent_model_digest": digest,
        "second_holdout": {
            "subject_id": SECOND_HOLDOUT.subject_id,
            "prompt_bytes": len(SECOND_HOLDOUT.prompt),
            "target_bytes": len(SECOND_HOLDOUT.target),
            "events": SECOND_HOLDOUT.event_count,
            "first_declared_here": True,
        },
        "declared": {
            "parts": PARTS,
            "units_per_group": UNITS_PER_GROUP,
            "steps": STEPS,
            "lr": LR,
            "net_gate": NET_GATE,
            "retention_gate": RETENTION_GATE,
            "reference_net": REFERENCE_NET,
            "face_tolerance": FACE_TOLERANCE,
            "retention_journal_is_not_an_evaluation_face": True,
        },
        "arms": {"replay_only": replay, "consensus_birth": birth},
        "net_benefit": net,
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "Two synthetic Chinese evaluation careers on one deterministic subject. The second "
            "career is a fixture, not an unseen task. No capability is shown and 09.7 is not admitted."
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
                "scripts/training/verify_taiji_netbenefit_second.py",
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
                "net_benefit": {name: round(value, 12) for name, value in net.items()},
                "replay_delta": (
                    None
                    if "delta" not in replay
                    else {k: round(v, 9) for k, v in replay["delta"].items()}
                ),
                "birth_delta": (
                    None
                    if "delta" not in birth
                    else {k: round(v, 9) for k, v in birth["delta"].items()}
                ),
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
