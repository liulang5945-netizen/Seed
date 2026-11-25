"""PHY.1.02 (exploration): learned multimodal consequence predictor on the physics facility.

Design: plans/reference/DESIGN-PHY-1-02_20261126.md (D-20261126-643).
Trains an MLP consequence predictor on one dynamics family and evaluates in-family and
cross-family generalisation, against the correctable count baseline (control). Exploration seeds
6401-6403; no confirmation seed, check or final is read.

Usage: python scripts/training/phy_1_02_multimodal_learning.py
Output: reports/phy_1_02_learning.json
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

import torch  # noqa: E402

from taiji.multimodal_learning import (  # noqa: E402
    accuracy,
    collect,
    count_baseline_accuracy,
    train_net,
)

CARD = "PHY.1.02"
TRAIN_SEEDS = (6401, 6402, 6403)
SPOT_CHECK_SEED = 6401
TRAIN_EPISODES = 1000
EVAL_EPISODES = 150
CAP_CPU = 30.0
OUT = ROOT / "reports" / "phy_1_02_learning.json"


def main() -> int:
    torch.set_num_threads(1)
    started = time.process_time()
    per_seed: dict[str, dict] = {}
    for seed in TRAIN_SEEDS:
        x_tr, y_tr = collect(seed, "incline", TRAIN_EPISODES)
        x_in, y_in = collect(seed + 100, "incline", EVAL_EPISODES)
        x_cross, y_cross = collect(seed + 200, "collision", EVAL_EPISODES)
        t0 = time.process_time()
        net = train_net(x_tr, y_tr, seed=seed, steps=1200)
        in_acc = accuracy(net, x_in, y_in)
        cross_acc = accuracy(net, x_cross, y_cross)
        baseline = count_baseline_accuracy(seed + 100, "incline", EVAL_EPISODES)
        if seed == SPOT_CHECK_SEED:
            again = train_net(x_tr, y_tr, seed=seed, steps=1200)
            exact = all(
                torch.equal(a, b) for a, b in zip(net.parameters(), again.parameters(), strict=True)
            )
        else:
            exact = None
        per_seed[str(seed)] = {
            "in_family": in_acc,
            "cross_family": cross_acc,
            "count_baseline": baseline,
            "weights_exact": exact,
            "cpu": round(time.process_time() - t0, 2),
            "n_train": int(x_tr.shape[0]),
            "n_eval": int(x_in.shape[0]),
        }
    checks = {
        "g1_in_family": all(v["in_family"] >= 0.85 for v in per_seed.values()),
        "g2_cross_family": all(v["cross_family"] >= 0.50 for v in per_seed.values()),
        "g3_beats_count": all(
            v["in_family"] >= v["count_baseline"] + 0.05 for v in per_seed.values()
        ),
        "g4_deterministic": all(
            v["weights_exact"] is True for v in per_seed.values() if v["weights_exact"] is not None
        ),
    }
    exit_ = "multimodal_learned_positive" if all(checks.values()) else "multimodal_learned_partial"
    payload = {
        "schema": "taiji-phy-1-02-v1",
        "card": CARD,
        "design": "DESIGN-PHY-1-02_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "train_seeds": list(TRAIN_SEEDS),
        "train_family": "incline",
        "eval_families": ["incline (held-out)", "collision (cross-family)"],
        "per_seed": per_seed,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "oracle_used_by_subject": False,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit": exit_, "checks": checks, "per_seed": per_seed, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "multimodal_learned_positive" else 3


if __name__ == "__main__":
    raise SystemExit(main())
