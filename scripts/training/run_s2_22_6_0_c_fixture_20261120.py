"""22.6.0 C-task fixture contract runner (zero training).

Freezes and validates the ``deadline inquiry`` fixture on the frozen physical
world: session determinism, the three pre-registered fixture controls (C1 must
be rejected as ``fixture_invalid``; C2/C3 must have unique scripted optima) and
the package price.  Zero optimizer steps, zero check/final reads, no capability
signing.  Modes: ``--freeze`` -> ``--smoke`` -> ``--run`` -> ``--closeout``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import NAMES, RGB_COLORS  # noqa: E402
from scripts.training.validate_22_6_c_fixture_20261120 import (  # noqa: E402
    DEADLINE,
    OBSERVE_COST,
    TICKS,
    WAIT_COST,
    evaluate_policies,
    session,
)
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_6_0_c_task_20261120.json"
SMOKE = ROOT / "reports/taiji_u2_22_6_0_c_fixture_smoke_20261120.json"
RUN = ROOT / "reports/taiji_u2_22_6_0_c_fixture_run_20261120.json"
CAP_CPU = 40.0
FIT_WORLDS = 32
SESSIONS = 3
DECISIONS = 16
C1_SEED = 20261300
C2_SEED = 20261205
C2_TARGET = "青色圆片"
C2_FIRST_CORRECT = 14
C3_FIRST_CORRECT = 0
PINNED_FILES = (
    "scripts/training/run_s2_22_6_0_c_fixture_20261120.py",
    "scripts/training/validate_22_6_c_fixture_20261120.py",
    "taiji/u2_task_world.py",
    "scripts/training/first_a_support_22_5_v2.py",
)


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def criteria() -> dict[str, Any]:
    return {
        "card": "22.6.0 C-task fixture (deadline inquiry)",
        "task": {
            "ticks": TICKS,
            "deadline": DEADLINE,
            "decisions_per_session": DECISIONS,
            "sessions_per_world": SESSIONS,
            "development_fit_worlds": FIT_WORLDS,
            "observe_cost": OBSERVE_COST,
            "wait_cost": WAIT_COST,
            "actions": ["OBSERVE", "WAIT", "REST", "VERIFY", "RECALL", "ANSWER"],
            "answer_reference": "target position word at deadline",
            "utility": "1[answer==truth] - sum(action costs)",
        },
        "fixtures": {
            "C1_no_task_opportunity": {"seed": C1_SEED, "expect": "fixture_invalid"},
            "C2_wait_unique_optimal": {
                "seed": C2_SEED,
                "target": C2_TARGET,
                "expect_first_correct_tick": C2_FIRST_CORRECT,
                "expect_winner": f"wait{C2_FIRST_CORRECT}_observe_answer",
            },
            "C3_act_now_unique_optimal": {
                "seed": C2_SEED,
                "expect_winner": f"wait{C3_FIRST_CORRECT}_observe_answer",
            },
        },
        "contract_requirements": [
            "session(seed) deterministic across two builds (identical observations and truth)",
            "C1 must be rejected as fixture_invalid (constant policy uniquely optimal)",
            "C2 must have the unique scripted optimum that waits until the late evidence",
            "C3 must have the unique scripted optimum that acts at the first decision",
            "zero optimizer steps and zero check/final reads",
        ],
        "exits_order": [
            "engineering_failure",
            "replay_mismatch",
            "fixture_controls_fail",
            "budget_stop",
            "c_task_ready",
        ],
        "check_reads": 0,
        "final_reads": 0,
        "training_authority": "not_required_zero_training",
    }


def guards(contract: dict[str, Any], mode: str) -> None:
    if contract.get("schema") != "taiji-u2-22-6-0-c-task-v1":
        raise ValueError("unsupported_c_task_contract")
    if contract.get("criteria_sha256") != value_digest(contract["criteria"]):
        raise ValueError("criteria_digest_mismatch")
    pins = contract.get("source_pins") or {}
    for name in PINNED_FILES:
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"source_pin_missing:{name}")
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode in {"smoke", "run", "verify"} and contract.get("status") != "frozen":
        raise ValueError("contract_not_frozen")


def build_fixtures() -> dict[str, Any]:
    probe = PhysicalWorld.reset(C1_SEED, {"body_count": 3})
    probe.elapse("c1-probe")
    present = {NAMES[RGB_COLORS.index(tuple(body["color"]))] for body in probe.snapshot()["bodies"]}
    missing = [name for name in NAMES if name not in present][0]
    c1 = session(C1_SEED, missing)
    c1["truth"] = "未知"
    c1["target"] = missing
    c2 = session(C2_SEED, C2_TARGET)
    c3 = dict(c2)
    c3["truth"] = c2["observed"][C3_FIRST_CORRECT]
    return {"c1": c1, "c2": c2, "c3": c3}


def check_controls(fixtures: dict[str, Any]) -> dict[str, Any]:
    c1_eval = evaluate_policies(fixtures["c1"])
    c2_eval = evaluate_policies(fixtures["c2"])
    c3_eval = evaluate_policies(fixtures["c3"])
    c1_invalid = fixtures["c1"]["truth"] == "未知" and c1_eval["winners"] == [
        "answer_without_observe"
    ]
    first_correct = next(
        (
            index
            for index, word in enumerate(fixtures["c2"]["observed"])
            if word == fixtures["c2"]["truth"]
        ),
        None,
    )
    c2_winner = f"wait{C2_FIRST_CORRECT}_observe_answer"
    c2_ok = first_correct == C2_FIRST_CORRECT and c2_eval["winners"] == [c2_winner]
    c3_ok = c3_eval["winners"] == [f"wait{C3_FIRST_CORRECT}_observe_answer"]
    return {
        "c1_invalid": bool(c1_invalid),
        "c1_winners": c1_eval["winners"],
        "c2_first_correct_tick": first_correct,
        "c2_winners": c2_eval["winners"],
        "c2_ok": bool(c2_ok),
        "c3_winners": c3_eval["winners"],
        "c3_ok": bool(c3_ok),
        "all_controls_pass": bool(c1_invalid and c2_ok and c3_ok),
    }


def replay_check(seeds: list[int]) -> dict[str, Any]:
    mismatches = []
    for seed in seeds:
        first = session(seed)
        second = session(seed)
        if first != second:
            mismatches.append(seed)
    return {"seeds": seeds, "mismatches": mismatches, "deterministic": not mismatches}


def freeze(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taiji-u2-22-6-0-c-task-v1",
        "card": "DESIGN-TAIJI_U2_22_6_0_C_TASK_20261120.md",
        "status": "frozen",
        "criteria": criteria(),
        "criteria_sha256": None,
        "plan": {
            "price_cpu": 6.5,
            "price_source": "reports/taiji_u2_22_6_c_fixture_validation_20261120.json",
            "optimizer_steps": 0,
        },
        "cap_cpu": CAP_CPU,
        "check_exposure": {"reads": 0},
        "source_pins": {name: sha_of(ROOT / name) for name in PINNED_FILES},
        "result": {"run": False, "closed": False, "verdict": None},
    }
    payload["criteria_sha256"] = value_digest(payload["criteria"])
    out.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"frozen": str(out), "criteria_sha256": payload["criteria_sha256"]}))
    return 0


def smoke(contract: dict[str, Any]) -> int:
    started = time.process_time()
    guards(contract, "smoke")
    fixtures = build_fixtures()
    controls = check_controls(fixtures)
    replay = replay_check([C1_SEED, C2_SEED])
    checks = {
        "c1_fixture_invalid": controls["c1_invalid"],
        "c2_wait_unique": controls["c2_ok"],
        "c3_act_now_unique": controls["c3_ok"],
        "session_replay_deterministic": replay["deterministic"],
        "zero_optimizer_steps": True,
    }
    passed = all(checks.values())
    payload = {
        "schema": "taiji-u2-22-6-0-c-task-smoke-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "checks": checks,
        "controls": controls,
        "replay": replay,
        "verdict": "smoke_passed" if passed else "smoke_failed",
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(SMOKE, payload)
    print(json.dumps({"verdict": payload["verdict"], "cpu": payload["cost"]["cpu"]}))
    return 0 if passed else 1


def run(contract: dict[str, Any]) -> int:
    started = time.process_time()
    guards(contract, "run")
    fixtures = build_fixtures()
    controls = check_controls(fixtures)
    seeds = list(range(20261200, 20261200 + 8))
    replay = replay_check(seeds)
    budget_ok = time.process_time() - started <= CAP_CPU
    if not replay["deterministic"]:
        verdict = "replay_mismatch"
    elif not controls["all_controls_pass"]:
        verdict = "fixture_controls_fail"
    elif not budget_ok:
        verdict = "budget_stop"
    else:
        verdict = "c_task_ready"
    payload = {
        "schema": "taiji-u2-22-6-0-c-task-run-v1",
        "criteria_sha256": contract["criteria_sha256"],
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "controls": controls,
        "replay": replay,
        "verdict": verdict,
        "budget": {"cap_cpu": CAP_CPU, "cpu": round(time.process_time() - started, 2)},
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(RUN, payload)
    print(json.dumps({"verdict": verdict, "cost": payload["cost"]}))
    return 0 if verdict == "c_task_ready" else 1


def closeout(contract_path: Path) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    report = json.loads(RUN.read_text(encoding="utf-8"))
    contract["result"] = {
        "run": True,
        "closed": True,
        "verdict": report["verdict"],
        "cost_cpu": report["cost"]["cpu"],
        "report": str(RUN.relative_to(ROOT)),
    }
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"closed": report["verdict"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--closeout", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        return freeze(CONTRACT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.smoke:
        return smoke(contract)
    if args.closeout:
        return closeout(CONTRACT)
    return run(contract)


if __name__ == "__main__":
    raise SystemExit(main())
