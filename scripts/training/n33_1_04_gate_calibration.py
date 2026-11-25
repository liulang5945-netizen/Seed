"""N33.1.04 (zero training): calibrate the autonomous-choice gates against the scripted policy.

Pre-registered: plans/reference/DESIGN-N33-1-04_20261125.md (D-20261125-562).
Owner review found the old G2 probe domains measured a preference the reward did not
contain (scripted policy itself scored 5 %) and that G1 was stricter than the card claimed.
This script only checks that the *proposed* gates are passable by the scripted baseline and
failed by random / constant baselines on fixture v4. No policy is trained; no confirmation
seed is read.

Usage: python scripts/training/n33_1_04_gate_calibration.py
Output: reports/n33_1_04_gates.json
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.choice_simulator import (  # noqa: E402
    CHOICES,
    HORIZON,
    ChoiceSessionEnv,
    State,
    evaluate_policy,
    random_policy,
    scripted_policy,
)

CARD = "N33.1.04"
EVAL_SEEDS = list(range(5000, 5040))
GATE_G1_RANDOM = 10.0
GATE_G1_SCRIPTED_RATIO = 0.85
GATE_G2_SHARE = 0.50
OUT = ROOT / "reports" / "n33_1_04_gates.json"


def probes(policy: Callable[[State], str]) -> dict[str, Any]:
    """Probe the *relevant* domains only (owner review, D-20261125-561/562)."""
    rest_hits = rest_total = 0
    verify_hits = verify_total = 0
    work_hits = work_total = 0
    hist = {c: 0 for c in CHOICES}
    for seed in EVAL_SEEDS:
        env = ChoiceSessionEnv(seed)
        state = env.reset()
        for _ in range(HORIZON):
            action = policy(state)
            hist[action] += 1
            if state.energy < 0.35 and state.pending > 0 and state.uncertainty <= 0.55:
                rest_total += 1
                rest_hits += int(action == "rest")
            if state.uncertainty > 0.55 and state.pending > 0 and state.energy >= 0.35:
                verify_total += 1
                verify_hits += int(action in ("verify", "ask_help"))
            if (
                state.pending > 0
                and state.progress < 0.3
                and state.energy >= 0.35
                and state.uncertainty <= 0.55
            ):
                work_total += 1
                work_hits += int(action in ("continue_task", "recall_learn"))
            state, _, done = env.step(action)
            if done:
                break
    return {
        "n_low_energy_pending": rest_total,
        "rest_share": round(rest_hits / max(1, rest_total), 4),
        "n_high_unc_pending": verify_total,
        "verify_or_help_share": round(verify_hits / max(1, verify_total), 4),
        "n_pending_work": work_total,
        "continue_or_recall_share": round(work_hits / max(1, work_total), 4),
        "action_histogram": hist,
    }


def gate_eval(policy: Callable[[State], str], baselines: dict[str, float]) -> dict[str, Any]:
    ev = evaluate_policy(policy, EVAL_SEEDS)
    pr = probes(policy)
    g1 = (
        ev["mean_return"] >= baselines["random"] + GATE_G1_RANDOM
        and ev["mean_return"] >= GATE_G1_SCRIPTED_RATIO * baselines["scripted"]
    )
    g2 = (
        pr["rest_share"] >= GATE_G2_SHARE
        and pr["verify_or_help_share"] >= GATE_G2_SHARE
        and pr["continue_or_recall_share"] >= GATE_G2_SHARE
    )
    return {"eval": ev, "probes": pr, "g1": bool(g1), "g2": bool(g2), "pass_all": bool(g1 and g2)}


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(7)
    random_base = evaluate_policy(lambda s: random_policy(s, gen), EVAL_SEEDS)
    scripted_base = evaluate_policy(scripted_policy, EVAL_SEEDS)
    baselines = {"random": random_base["mean_return"], "scripted": scripted_base["mean_return"]}

    scripted = gate_eval(scripted_policy, baselines)
    random_eval = gate_eval(
        lambda s: random_policy(s, torch.Generator().manual_seed(11)), baselines
    )
    constant = gate_eval(lambda s: "wait_wake", baselines)

    exit_ = (
        "gates_calibrated"
        if scripted["pass_all"] and not random_eval["g1"] and not constant["g2"]
        else "gates_uncalibratable"
    )
    payload = {
        "schema": "taiji-n33-1-04-v1",
        "card": CARD,
        "design": "DESIGN-N33-1-04_20261125.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "fixture": "taiji-choice-session-v4",
        "eval_seeds": EVAL_SEEDS,
        "baselines": {"random": random_base, "scripted": scripted_base},
        "gates": {
            "g1_random_gap": GATE_G1_RANDOM,
            "g1_scripted_ratio": GATE_G1_SCRIPTED_RATIO,
            "g2_share": GATE_G2_SHARE,
        },
        "arms": {"scripted": scripted, "random": random_eval, "constant_wait": constant},
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": 10.0},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "trained_policies": 0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "scripted": {
                    "mean": scripted["eval"]["mean_return"],
                    "g1": scripted["g1"],
                    "g2": scripted["g2"],
                    "probes": scripted["probes"],
                },
                "random": {"mean": random_eval["eval"]["mean_return"], "g1": random_eval["g1"]},
                "constant_wait": {
                    "mean": constant["eval"]["mean_return"],
                    "g2": constant["g2"],
                    "probes": constant["probes"],
                },
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "gates_calibrated" else 3


if __name__ == "__main__":
    raise SystemExit(main())
