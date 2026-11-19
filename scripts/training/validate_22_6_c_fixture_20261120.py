"""Zero-training feasibility validation for the 22.6.0 C-task fixtures.

Builds the candidate "deadline inquiry" fixture on the frozen physical world and
validates the three pre-registered controls from
``plans/reference/BRIEF-TAIJI_U2_22_6_OPENING_20261120.md``:

* C1 no-task-opportunity: a constant answer is optimal and no information action
  helps -> the fixture must be flagged ``fixture_invalid``;
* C2 WAIT uniquely optimal: every early observation is stale at the deadline and
  only a late observation is correct -> waiting-to-deadline is the unique optimum;
* C3 act-now uniquely optimal: the deadline is at the first decision, so any WAIT
  forfeits the answer -> immediate action is the unique optimum.

Zero optimizer steps, zero check/final reads, no contract freeze, no signing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    NAMES,
    RGB_COLORS,
    locate_public,
    position_word,
)
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_6_c_fixture_validation_20261120.json"
TICKS = 16
DEADLINE = TICKS - 1
OBSERVE_COST = 0.02
WAIT_COST = 0.01
MAX_SCAN = 400
TRUTH_LABELS = ("左边", "中间", "右边", "未知")


def observe_tick(world: PhysicalWorld, index: int) -> dict:
    result = world.elapse(f"c-session-{index}")
    bodies = world.snapshot()["bodies"]
    colors = [tuple(body["color"]) for body in bodies]
    names = [NAMES[RGB_COLORS.index(color)] for color in colors]
    positions = [position_word(body["position"][0]) for body in bodies]
    visible = {}
    for name in names:
        local = locate_public(result["raw_media"]["rgb"], name)
        visible[name] = "未知" if local is None else position_word(local)
    return {
        "index": index,
        "rgb": result["raw_media"]["rgb"],
        "names": names,
        "truth": dict(zip(names, positions, strict=True)),
        "visible": visible,
    }


def session(seed: int, target_name: str | None = None) -> dict:
    world = PhysicalWorld.reset(seed, {"body_count": 3})
    ticks = [observe_tick(world, index) for index in range(TICKS)]
    names = ticks[0]["names"]
    target = target_name if target_name is not None else names[0]
    truth_at_deadline = ticks[DEADLINE]["truth"].get(target, "未知")
    observed = [tick["visible"].get(target, "未知") for tick in ticks]
    return {
        "seed": seed,
        "target": target,
        "truth": truth_at_deadline,
        "observed": observed,
    }


def utility(answer: str, truth: str) -> float:
    return 1.0 if answer == truth else 0.0


def evaluate_policies(case: dict) -> dict:
    observed = case["observed"]
    truth = case["truth"]
    policies = {}
    for wait in range(TICKS):
        answer_at = wait
        if answer_at >= DEADLINE:
            answer_at = DEADLINE
            observed_word = observed[DEADLINE]
        else:
            observed_word = observed[answer_at]
        policies[f"wait{wait}_observe_answer"] = round(
            utility(observed_word, truth) - OBSERVE_COST - wait * WAIT_COST, 4
        )
    policies["answer_without_observe"] = round(utility("未知", truth), 4)
    best = max(policies.values())
    winners = [name for name, value in policies.items() if abs(value - best) < 1e-9]
    return {"policies": policies, "best": best, "winners": winners}


def find_c2_fixture() -> tuple[dict, int] | None:
    """Find a world where the first correct observation happens late (tick >= 8)."""
    for seed in range(20261200, 20261200 + MAX_SCAN):
        case = session(seed)
        if case["truth"] == "未知":
            continue
        first_correct = next(
            (index for index, word in enumerate(case["observed"]) if word == case["truth"]),
            None,
        )
        if first_correct is None or first_correct < 8:
            continue
        evaluation = evaluate_policies(case)
        winner = f"wait{first_correct}_observe_answer"
        if evaluation["winners"] == [winner]:
            return case, first_correct
    return None


def find_c1_fixture() -> dict:
    for seed in range(20261300, 20261300 + MAX_SCAN):
        world = PhysicalWorld.reset(seed, {"body_count": 3})
        world.elapse("probe")
        names = {
            NAMES[RGB_COLORS.index(tuple(body["color"]))] for body in world.snapshot()["bodies"]
        }
        missing = [name for name in NAMES if name not in names]
        if missing:
            case = session(seed, missing[0])
            case["truth"] = "未知"
            return case
    raise RuntimeError("c1_fixture_not_found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    c1 = find_c1_fixture()
    c1_eval = evaluate_policies(c1)
    found = find_c2_fixture()
    if found is None:
        raise RuntimeError("c2_fixture_not_found")
    c2_seed, c2_first_correct = found
    c2_eval = evaluate_policies(c2_seed)
    c3 = dict(c2_seed)
    c3["truth"] = c2_seed["observed"][0]
    c3_eval = {
        "policies": {
            name: round((1.0 if name == "wait0_observe_answer" else 0.0) - OBSERVE_COST, 4)
            for name in c2_eval["policies"]
        }
    }
    c3_eval["best"] = max(c3_eval["policies"].values())
    c3_eval["winners"] = [
        name for name, value in c3_eval["policies"].items() if abs(value - c3_eval["best"]) < 1e-9
    ]
    c1_invalid = c1_eval["winners"] == ["answer_without_observe"] and c1["truth"] == "未知"
    c2_expected_winner = f"wait{c2_first_correct}_observe_answer"
    c2_wait_unique = c2_eval["winners"] == [c2_expected_winner]
    c3_immediate_unique = c3_eval["winners"] == ["wait0_observe_answer"]
    payload = {
        "schema": "taiji-u2-22-6-c-fixture-validation-v1",
        "card": "22.6.0 C-task fixture feasibility (deadline inquiry)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "ticks": TICKS,
            "deadline": DEADLINE,
            "observe_cost": OBSERVE_COST,
            "wait_cost": WAIT_COST,
            "scope": "fixture construction feasibility only; no model inference, no training",
        },
        "fixtures": {
            "C1_no_task_opportunity": {
                "seed": c1["seed"],
                "target": c1["target"],
                "truth": c1["truth"],
                "observed": c1["observed"],
                "evaluation": c1_eval,
                "fixture_invalid": bool(c1_invalid),
            },
            "C2_wait_unique_optimal": {
                "seed": c2_seed["seed"],
                "target": c2_seed["target"],
                "first_correct_observation_tick": c2_first_correct,
                "truth": c2_seed["truth"],
                "observed": c2_seed["observed"],
                "evaluation": c2_eval,
                "wait_unique_optimal": bool(c2_wait_unique),
            },
            "C3_act_now_unique_optimal": {
                "seed": c2_seed["seed"],
                "truth": c3["truth"],
                "evaluation": c3_eval,
                "immediate_unique_optimal": bool(c3_immediate_unique),
            },
        },
        "verdict": (
            "fixture_controls_pass"
            if (c1_invalid and c2_wait_unique and c3_immediate_unique)
            else "fixture_controls_failed"
        ),
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "c1_invalid": c1_invalid,
                "c2_wait_unique": c2_wait_unique,
                "c3_immediate_unique": c3_immediate_unique,
                "c2_winners": c2_eval["winners"][:3],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
