"""N33.1.07 (exploration): re-measure N33.1.06 PPO policies on a fixed, pre-registered probe set.

Pre-registered: plans/reference/DESIGN-N33-1-07_20261125.md (D-20261125-567).
Measurement fix only: the G2 probe set no longer comes from each learned policy's own
trajectories. No training variable changes; N33.1.06 is retrained in the original order and
must reproduce bit-for-bit before any new number is read.

Usage: python scripts/training/n33_1_07_fixed_probe.py
Output: reports/n33_1_07_fixed_probe.json
"""

from __future__ import annotations

import dataclasses
import json
import random
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

from scripts.training.n33_1_04_gate_calibration import (  # noqa: E402
    EVAL_SEEDS,
    GATE_G2_SHARE,
    gate_eval,
)
from scripts.training.n33_1_06_ppo_more_episodes import (  # noqa: E402
    EPISODES,
    GATE_G3_DROP,
    TRAIN_SEEDS,
    greedy,
    train_policy,
)
from taiji.choice_simulator import (  # noqa: E402
    CHOICES,
    HORIZON,
    SIM_VERSION,
    ChoiceSessionEnv,
    State,
    evaluate_policy,
    random_policy,
    scripted_policy,
    seed_from,
)

CARD = "N33.1.07"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
DESIGN = f"DESIGN-{CARD_SLUG}_20261125.md"
PROBE_ENV_SEEDS = list(range(7000, 7200))
K_PER_DOMAIN = 150
CAP_CPU = 50.0
REPRO_TOL = 1e-6
PRIOR_REPORT = ROOT / "reports" / "n33_1_06_ppo.json"
OUT = ROOT / "reports" / "n33_1_07_fixed_probe.json"

DOMAINS: dict[str, tuple[Callable[[State], bool], tuple[str, ...]]] = {
    "rest": (
        lambda s: s.energy < 0.35 and s.pending > 0 and s.uncertainty <= 0.55,
        ("rest",),
    ),
    "verify": (
        lambda s: s.uncertainty > 0.55 and s.pending > 0 and s.energy >= 0.35,
        ("verify", "ask_help"),
    ),
    "work": (
        lambda s: s.pending > 0 and s.progress < 0.3 and s.energy >= 0.35 and s.uncertainty <= 0.55,
        ("continue_task", "recall_learn"),
    ),
}


def make_random_policy(tag: str) -> Callable[[State], str]:
    """True uniform-random policy with ONE persistent generator (fixes the N33.1.04 bug)."""
    gen = torch.Generator().manual_seed(seed_from(CARD, tag))

    def act(_state: State) -> str:
        return CHOICES[int(torch.randint(0, len(CHOICES), (), generator=gen))]

    return act


def build_probe_set() -> dict[str, Any]:
    pools: dict[str, list[State]] = {d: [] for d in DOMAINS}
    source_counts: dict[str, dict[str, int]] = {d: {"random": 0, "scripted": 0} for d in DOMAINS}
    rand_pol = make_random_policy("probe-random")
    for src, pol in (("random", rand_pol), ("scripted", scripted_policy)):
        for seed in PROBE_ENV_SEEDS:
            env = ChoiceSessionEnv(seed)
            state = env.reset()
            for _ in range(HORIZON):
                snap = dataclasses.replace(state)
                for d, (pred, _) in DOMAINS.items():
                    if pred(snap):
                        pools[d].append(snap)
                        source_counts[d][src] += 1
                state, _, done = env.step(pol(state))
                if done:
                    break
    rng = random.Random(seed_from(CARD, "probe-sample"))
    pool_sizes = {d: len(v) for d, v in pools.items()}
    sets = {
        d: rng.sample(v, K_PER_DOMAIN) if len(v) >= K_PER_DOMAIN else [] for d, v in pools.items()
    }
    return {"sets": sets, "pool_sizes": pool_sizes, "source_counts": source_counts}


def fixed_g2(policy: Callable[[State], str], sets: dict[str, list[State]]) -> dict[str, Any]:
    shares: dict[str, float] = {}
    for d, (_, wanted) in DOMAINS.items():
        states = sets[d]
        hits = sum(int(policy(dataclasses.replace(s)) in wanted) for s in states)
        shares[d] = round(hits / max(1, len(states)), 4)
    return {"shares": shares, "g2": bool(all(v >= GATE_G2_SHARE for v in shares.values()))}


def main() -> int:
    torch.set_num_threads(1)
    started = time.process_time()

    probe = build_probe_set()
    sets = probe["sets"]
    pools_ok = all(n >= K_PER_DOMAIN for n in probe["pool_sizes"].values())

    calibration: dict[str, Any] = {
        "scripted": fixed_g2(scripted_policy, sets),
        "random_true": fixed_g2(make_random_policy("calib-random"), sets),
    }
    for c in CHOICES:
        calibration[f"constant_{c}"] = fixed_g2(lambda _s, c=c: c, sets)
    calibrated = (
        pools_ok
        and calibration["scripted"]["g2"]
        and not any(v["g2"] for k, v in calibration.items() if k != "scripted")
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "card": CARD,
        "design": DESIGN,
        "created_utc": datetime.now(UTC).isoformat(),
        "fixture": SIM_VERSION,
        "probe": {
            "env_seeds": [PROBE_ENV_SEEDS[0], PROBE_ENV_SEEDS[-1]],
            "k_per_domain": K_PER_DOMAIN,
            "pool_sizes": probe["pool_sizes"],
            "source_counts": probe["source_counts"],
            "g2_share": GATE_G2_SHARE,
        },
        "calibration": calibration,
        "calibrated": bool(calibrated),
    }

    if not calibrated:
        payload["exit"] = "probe_uncalibratable"
    else:
        prior = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
        gen = torch.Generator().manual_seed(7)
        random_base = evaluate_policy(lambda s: random_policy(s, gen), EVAL_SEEDS)
        scripted_base = evaluate_policy(scripted_policy, EVAL_SEEDS)
        baselines = {"random": random_base["mean_return"], "scripted": scripted_base["mean_return"]}
        deadline = started + CAP_CPU
        per_seed: dict[str, Any] = {}
        repro_ok = True
        for seed in TRAIN_SEEDS:
            policy = train_policy(seed, EPISODES, deadline)
            ev = gate_eval(greedy(policy), baselines)
            zero = evaluate_policy(greedy(policy, zero_state=True), EVAL_SEEDS)
            drop = ev["eval"]["mean_return"] - zero["mean_return"]
            p = prior["per_seed"][str(seed)]
            match = (
                abs(ev["eval"]["mean_return"] - p["return_normal"]) <= REPRO_TOL
                and abs(round(drop, 4) - p["state_drop"]) <= REPRO_TOL
            )
            repro_ok = repro_ok and match
            fg = fixed_g2(greedy(policy), sets)
            g3 = drop >= GATE_G3_DROP
            per_seed[str(seed)] = {
                "return_normal": ev["eval"]["mean_return"],
                "state_drop": round(drop, 4),
                "repro_match_n33_1_06": bool(match),
                "g1": ev["g1"],
                "g3": bool(g3),
                "g2_fixed": fg["g2"],
                "fixed_shares": fg["shares"],
                "trajectory_probes_for_reference": ev["probes"],
                "trajectory_g2_for_reference": ev["g2"],
                "positive": bool(ev["g1"] and fg["g2"] and g3),
            }
        payload["baselines"] = {"random": random_base, "scripted": scripted_base}
        payload["per_seed"] = per_seed
        payload["repro_ok"] = bool(repro_ok)
        if not repro_ok:
            payload["exit"] = "repro_mismatch"
        elif all(v["positive"] for v in per_seed.values()):
            payload["exit"] = "fixed_probe_positive"
        elif all(v["g1"] and v["g3"] for v in per_seed.values()):
            payload["exit"] = "fixed_probe_partial"
        else:
            payload["exit"] = "fixed_probe_negative"

    payload["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    payload["discipline"] = {
        "check_reads": 0,
        "final_reads": 0,
        "confirm_reads": 0,
        "scripted_used_in_training": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: payload.get(k) for k in ("exit", "calibrated", "repro_ok", "cost")}
    summary["pool_sizes"] = probe["pool_sizes"]
    summary["calibration"] = {k: v["shares"] for k, v in calibration.items()}
    if "per_seed" in payload:
        summary["per_seed"] = {
            s: {
                k: v[k]
                for k in ("return_normal", "repro_match_n33_1_06", "fixed_shares", "positive")
            }
            for s, v in payload["per_seed"].items()
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if payload["exit"] != "repro_mismatch" else 3


if __name__ == "__main__":
    raise SystemExit(main())
