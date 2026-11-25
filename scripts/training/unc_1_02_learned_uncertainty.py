"""UNC.1.02 (exploration): learned uncertainty estimation vs engineering calibration prototypes.

Design: plans/reference/DESIGN-UNC-1-02_20261126.md (D-20261126-641).
Compares constant 0.5, the temperature-calibrated miscalibrated base score (UNC.1.01 prototype),
a learned logistic confidence model (in-subject, trained on verifiable feedback), and the true
generative probability (oracle, control only), on a held-out split.

Usage: python scripts/training/unc_1_02_learned_uncertainty.py
Output: reports/unc_1_02_learned.json
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

from taiji.uncertainty_learning import (  # noqa: E402
    N_EVAL,
    N_TRAIN,
    BaseScore,
    LearnedConfidence,
    TemperatureCalibrator,
    evaluate,
    make_data,
)

CARD = "UNC.1.02"
SEEDS = (6301, 6302, 6303)
CAP_CPU = 15.0
OUT = ROOT / "reports" / "unc_1_02_learned.json"


def main() -> int:
    torch.set_num_threads(1)
    started = time.process_time()
    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        x_tr, y_tr, _ = make_data(seed, N_TRAIN)
        x_ev, y_ev, true_p = make_data(seed + 100, N_EVAL)
        learned = LearnedConfidence.train(x_tr, y_tr, seed=seed, steps=800)
        static = torch.full_like(y_ev, 0.5)
        base = BaseScore()
        cal = TemperatureCalibrator()
        cal.fit(x_tr, y_tr)
        metrics = {
            "static": evaluate(static, y_ev),
            "base": evaluate(base.predict(x_ev), y_ev),
            "temperature": evaluate(cal.predict(x_ev), y_ev),
            "learned": evaluate(learned.predict(x_ev), y_ev),
            "oracle": evaluate(true_p, y_ev),
        }
        per_seed[str(seed)] = {
            "metrics": metrics,
            "learned_vs_static_brier": round(
                metrics["learned"]["brier"] - metrics["static"]["brier"], 4
            ),
            "learned_vs_temperature_brier": round(
                metrics["learned"]["brier"] - metrics["temperature"]["brier"], 4
            ),
            "temperature_vs_base_ece": round(
                metrics["temperature"]["ece"] - metrics["base"]["ece"], 4
            ),
            "temperature": cal.temperature,
        }
    checks = {
        "g1_learned_beats_static_brier": all(
            v["learned_vs_static_brier"] <= -0.05 for v in per_seed.values()
        ),
        "g2_learned_ece_low": all(
            v["metrics"]["learned"]["ece"] <= 0.10 for v in per_seed.values()
        ),
        "g3_temperature_improves_base": all(
            v["temperature_vs_base_ece"] <= 0.0 for v in per_seed.values()
        ),
        "g4_oracle_control": all(v["metrics"]["oracle"]["ece"] <= 0.05 for v in per_seed.values()),
    }
    exit_ = (
        "uncertainty_learned_positive" if all(checks.values()) else "uncertainty_learned_partial"
    )
    payload = {
        "schema": "taiji-unc-1-02-v1",
        "card": CARD,
        "design": "DESIGN-UNC-1-02_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "train_seeds": list(SEEDS),
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
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
            {
                "exit": exit_,
                "checks": checks,
                "per_seed": {
                    k: {
                        "learned_ece": v["metrics"]["learned"]["ece"],
                        "temperature_ece": v["metrics"]["temperature"]["ece"],
                        "brier_gain": v["learned_vs_static_brier"],
                    }
                    for k, v in per_seed.items()
                },
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "uncertainty_learned_positive" else 3


if __name__ == "__main__":
    raise SystemExit(main())
