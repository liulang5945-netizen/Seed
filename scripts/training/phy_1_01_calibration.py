"""PHY.1.01 (zero training): calibration of the synthetic multimodal + physics environment.

Design: plans/reference/DESIGN-PHY-1-01_20261126.md (D-20261126-621).
Checks oracle (control) vs random prediction accuracy, correctable predictor improvement, both
dynamics families covered, modality shapes, outcome labels and absence of task_id.

Usage: python scripts/training/phy_1_01_calibration.py
Output: reports/phy_1_01_calibration.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.multimodal_physics_env import (  # noqa: E402
    ACTIONS,
    FAMILIES,
    HORIZON,
    OUTCOMES,
    CorrectablePredictor,
    MultimodalPhysicsEnv,
    OraclePredictor,
    RandomPredictor,
)

CARD = "PHY.1.01"
SEEDS = list(range(200, 260))
CAP_CPU = 5.0
OUT = ROOT / "reports" / "phy_1_01_calibration.json"


def _oracle_accuracy(seeds: list[int]) -> float:
    correct = total = 0
    pred = OraclePredictor()
    for seed in seeds:
        env = MultimodalPhysicsEnv(seed)
        env.reset()
        while True:
            for action in ACTIONS:
                truth = env.true_outcome(action)
                total += 1
                correct += int(pred.predict(env, action) == truth)
            done = env.step(ACTIONS[env.state.t % len(ACTIONS)])[3]
            if done:
                break
    return round(correct / max(1, total), 4)


def _random_accuracy(seeds: list[int]) -> float:
    correct = total = 0
    pred = RandomPredictor(0)
    for seed in seeds:
        env = MultimodalPhysicsEnv(seed)
        env.reset()
        while True:
            for action in ACTIONS:
                truth = env.true_outcome(action)
                total += 1
                correct += int(pred.predict(env, action) == truth)
            done = env.step(ACTIONS[env.state.t % len(ACTIONS)])[3]
            if done:
                break
    return round(correct / max(1, total), 4)


def _correctable(seeds: list[int]) -> dict:
    pred = CorrectablePredictor()
    cold_hits = cold_total = 0
    early = None
    families = set()
    for i, seed in enumerate(seeds):
        env = MultimodalPhysicsEnv(seed)
        env.reset()
        assert env.state is not None
        families.add(env.state.family)
        first_turn = True
        while True:
            for action in ACTIONS:
                truth = env.true_outcome(action)
                predicted = pred.predict(env, action)
                if i == 0 and first_turn:
                    cold_total += 1
                    cold_hits += int(predicted == truth)
                pred.observe(env, action, truth, predicted)
            first_turn = False
            done = env.step(ACTIONS[env.state.t % len(ACTIONS)])[3]
            if done:
                break
        if i == 1:
            early = pred.accuracy()
    cold = cold_hits / max(1, cold_total)
    return {
        "cold": round(cold, 4),
        "early": early,
        "late": pred.accuracy(),
        "families": sorted(families),
    }


def main() -> int:
    started = time.process_time()
    oracle = _oracle_accuracy(SEEDS)
    rnd = _random_accuracy(SEEDS)
    corr = _correctable(SEEDS)

    env = MultimodalPhysicsEnv(7)
    obs = env.reset()
    shapes = {k: len(v) for k, v in obs.items() if isinstance(v, list)}
    outcomes_ok = True
    for seed in SEEDS[:5]:
        e = MultimodalPhysicsEnv(seed)
        e.reset()
        for action in ACTIONS:
            if e.true_outcome(action) not in OUTCOMES:
                outcomes_ok = False
        for _ in range(HORIZON):
            if e.step("push_soft")[3]:
                break

    checks = {
        "oracle_high": oracle >= 0.99,
        "random_low": rnd <= 0.35,
        "correctable_late_high": corr["late"] >= 0.70,
        "correctable_improves": corr["late"] >= corr["cold"] + 0.20,
        "both_families": set(corr["families"]) == set(FAMILIES),
        "modality_shapes": shapes.get("vision") == 6
        and shapes.get("proprio") == 3
        and shapes.get("audio") == 2,
        "no_task_id": "task_id" not in obs,
        "outcome_labels": outcomes_ok,
    }
    exit_ = "physics_ready" if all(checks.values()) else "physics_failure"
    payload = {
        "schema": "taiji-phy-1-01-v1",
        "card": CARD,
        "design": "DESIGN-PHY-1-01_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": [SEEDS[0], SEEDS[-1], len(SEEDS)],
        "oracle_accuracy": oracle,
        "random_accuracy": rnd,
        "correctable": corr,
        "modalities": shapes,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0, "trained_models": 0},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "checks": checks,
                "oracle": oracle,
                "random": rnd,
                "correctable": corr,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "physics_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
