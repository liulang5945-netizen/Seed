"""N33.2.03 (exploration): single-variable 8,000-episode replication of N33.2.02.

Pre-registered: plans/reference/DESIGN-N33-2-03_20261125.md (D-20261125-582).
Single variable vs N33.2.02: EPISODES 4,000 -> 8,000. Reuses N33.2.02's deterministic
trainer and sampler directly so only training horizon changes.

Usage:
  python scripts/training/n33_2_03_cross_session_8k.py --smoke
  python scripts/training/n33_2_03_cross_session_8k.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training.n31_guard import preflight  # noqa: E402
from scripts.training.n33_2_02_cross_session_use import (  # noqa: E402
    EVAL_SEEDS,
    GATE_COUNTERFACTUAL_DROP,
    GATE_RESOLVE_RATE,
    GATE_SCRIPTED_RATIO,
    GATE_STATELESS_GAP,
    SMOKE_EPISODES,
    SPOT_CHECK_SEED,
    TRAIN_SEEDS,
    calibrate_environment,
    greedy,
    train_policy_det,
)
from taiji.cross_session_env import (  # noqa: E402
    CROSS_DIM,
    ENV_VERSION,
    OBS_S2_DIM,
    STATE_DIM,
    evaluate_policy_with_store,
)

CARD = "N33.2.03"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
EPISODES = 8000
CAP_CPU = 80.0
OUT = ROOT / "reports" / "n33_2_03_cross_session_8k.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    design_file = preflight(CARD, list(TRAIN_SEEDS), record=False).name
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    episodes = SMOKE_EPISODES if args.smoke else EPISODES

    calib = calibrate_environment()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "card": CARD,
        "design": design_file,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "env_version": ENV_VERSION,
        "dims": {"obs_s2": OBS_S2_DIM, "cross": CROSS_DIM, "total": STATE_DIM},
        "episodes": episodes,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS)],
        "gates": {
            "stateless_gap": GATE_STATELESS_GAP,
            "scripted_ratio": GATE_SCRIPTED_RATIO,
            "resolve_rate": GATE_RESOLVE_RATE,
            "counterfactual_drop": GATE_COUNTERFACTUAL_DROP,
        },
        "calibration": calib,
    }

    if not calib["c0_pass"]:
        payload["exit"] = "engineering_failure"
        payload["reason"] = "C0 environment calibration failed"
    else:
        scripted_mean = calib["scripted_normal"]["mean_return"]
        per_seed: dict[str, Any] = {}
        repro_exact = False
        store_ok = True
        try:
            for seed in TRAIN_SEEDS:
                s_t0 = time.process_time()
                pol_pers = train_policy_det(
                    seed, stateless=False, episodes=episodes, deadline=deadline
                )
                pol_stat = train_policy_det(
                    seed, stateless=True, episodes=episodes, deadline=deadline
                )

                if seed == SPOT_CHECK_SEED:
                    pol_pers2 = train_policy_det(
                        seed, stateless=False, episodes=episodes, deadline=deadline
                    )
                    repro_exact = all(
                        torch.equal(a, b)
                        for a, b in zip(pol_pers.parameters(), pol_pers2.parameters(), strict=True)
                    )

                ev_norm = evaluate_policy_with_store(greedy(pol_pers), EVAL_SEEDS, mode="normal")
                ev_zero = evaluate_policy_with_store(
                    greedy(pol_pers), EVAL_SEEDS, mode="zero_cross"
                )
                ev_swap = evaluate_policy_with_store(
                    greedy(pol_pers), EVAL_SEEDS, mode="swap_cross"
                )
                ev_stat = evaluate_policy_with_store(
                    greedy(pol_stat), EVAL_SEEDS, mode="zero_cross"
                )

                store_ok = (
                    store_ok
                    and ev_norm["store_roundtrip_exact"]
                    and ev_norm["post_s2_persist_exact"]
                    and ev_zero["store_roundtrip_exact"]
                    and ev_swap["store_roundtrip_exact"]
                    and ev_stat["store_roundtrip_exact"]
                )

                gap_stateless = ev_norm["mean_return"] - ev_stat["mean_return"]
                drop_zero = ev_norm["mean_return"] - ev_zero["mean_return"]
                drop_swap = ev_norm["mean_return"] - ev_swap["mean_return"]
                g1 = (
                    gap_stateless >= GATE_STATELESS_GAP
                    and ev_norm["mean_return"] >= GATE_SCRIPTED_RATIO * scripted_mean
                    and ev_norm["resolve_rate"] >= GATE_RESOLVE_RATE
                )
                g2 = drop_zero >= GATE_COUNTERFACTUAL_DROP and drop_swap >= GATE_COUNTERFACTUAL_DROP
                per_seed[str(seed)] = {
                    "persistent_normal": ev_norm,
                    "stateless_mean": ev_stat["mean_return"],
                    "stateless_resolve_rate": ev_stat["resolve_rate"],
                    "zero_cross_mean": ev_zero["mean_return"],
                    "swap_cross_mean": ev_swap["mean_return"],
                    "gap_vs_stateless": round(gap_stateless, 4),
                    "drop_zero_cross": round(drop_zero, 4),
                    "drop_swap_cross": round(drop_swap, 4),
                    "g1": bool(g1),
                    "g2": bool(g2),
                    "positive": bool(g1 and g2),
                    "cpu": round(time.process_time() - s_t0, 2),
                }

            payload["per_seed"] = per_seed
            payload["repro_exact"] = bool(repro_exact)
            payload["store_checks_exact"] = bool(store_ok)

            if not repro_exact or not store_ok:
                payload["exit"] = "engineering_failure"
                payload["reason"] = f"repro_exact={repro_exact}, store_ok={store_ok}"
            elif args.smoke:
                payload["exit"] = "smoke"
            elif all(v["positive"] for v in per_seed.values()):
                payload["exit"] = "cross_session_positive"
            elif any(v["positive"] or v["g1"] or v["g2"] for v in per_seed.values()):
                payload["exit"] = "cross_session_partial"
            else:
                payload["exit"] = "cross_session_negative"
        except (TimeoutError, FloatingPointError) as exc:
            payload["per_seed"] = per_seed
            payload["exit"] = "engineering_failure"
            payload["reason"] = str(exc)

    payload["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    payload["discipline"] = {
        "check_reads": 0,
        "final_reads": 0,
        "confirm_reads": 0,
        "scripted_used_in_training": False,
        "continual_plasticity_or_growth": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        k: payload.get(k)
        for k in ("exit", "repro_exact", "store_checks_exact", "cost")
        if payload.get(k) is not None
    }
    if "per_seed" in payload:
        summary["per_seed"] = {
            s: {
                "mean_return": v["persistent_normal"]["mean_return"],
                "resolve_rate": v["persistent_normal"]["resolve_rate"],
                "stateless_mean": v["stateless_mean"],
                "gap_vs_stateless": v["gap_vs_stateless"],
                "drop_zero_cross": v["drop_zero_cross"],
                "drop_swap_cross": v["drop_swap_cross"],
                "g1": v["g1"],
                "g2": v["g2"],
                "positive": v["positive"],
                "cpu": v["cpu"],
            }
            for s, v in payload["per_seed"].items()
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if payload["exit"] != "engineering_failure" else 3


if __name__ == "__main__":
    raise SystemExit(main())
