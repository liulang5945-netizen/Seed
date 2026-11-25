"""UNC.1.01 (zero training): calibration of the uncertainty-calibration facility.

Design: plans/reference/DESIGN-UNC-1-01_20261126.md (D-20261126-618).
Synthetic scenarios check ECE/Brier correctness, detection of over/under-confidence, post-hoc
temperature reduction of ECE, Chinese verbalisation levels, and determinism. No model is trained.

Usage: python scripts/training/unc_1_01_calibration.py
Output: reports/unc_1_01_calibration.json
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

from taiji.uncertainty_calibration import (  # noqa: E402
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_power_temperature,
    verbalize_confidence,
)

CARD = "UNC.1.01"
CAP_CPU = 5.0
OUT = ROOT / "reports" / "unc_1_01_calibration.json"


def _perfect() -> tuple[list[float], list[int]]:
    conf: list[float] = []
    correct: list[int] = []
    for q in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        n = 1000
        conf.extend([q] * n)
        correct.extend([1] * int(q * n) + [0] * (n - int(q * n)))
    return conf, correct


def main() -> int:
    started = time.process_time()
    scenarios: dict[str, dict] = {}

    conf, correct = _perfect()
    scenarios["perfect"] = {
        "ece": expected_calibration_error(conf, correct)["ece"],
        "brier": brier_score(conf, correct),
    }

    over_conf, over_correct = [0.9] * 1000, [1] * 500 + [0] * 500
    over_ece = expected_calibration_error(over_conf, over_correct)["ece"]
    scenarios["overconfident"] = {"ece": over_ece, "brier": brier_score(over_conf, over_correct)}

    under_conf, under_correct = [0.3] * 1000, [1] * 700 + [0] * 300
    scenarios["underconfident"] = {
        "ece": expected_calibration_error(under_conf, under_correct)["ece"]
    }

    flat_conf, flat_correct = [0.5] * 1000, [1] * 500 + [0] * 500
    scenarios["flat_half"] = {"ece": expected_calibration_error(flat_conf, flat_correct)["ece"]}

    fit = fit_power_temperature(over_conf, over_correct)
    scenarios["temperature_fit"] = fit
    cal = apply_temperature(over_conf, fit["temperature"])
    scenarios["temperature_fit"]["calibrated_ece_check"] = expected_calibration_error(
        cal, over_correct
    )["ece"]

    verbal = {}
    for q, phrase in ((0.95, "我很确定"), (0.70, "比较确定"), (0.50, "不太确定"), (0.10, "不确定")):
        verbal[str(q)] = verbalize_confidence(q)
        assert verbal[str(q)] == phrase

    deterministic = (
        expected_calibration_error(conf, correct)["ece"] == scenarios["perfect"]["ece"]
        and fit_power_temperature(over_conf, over_correct)["temperature"] == fit["temperature"]
    )
    checks = {
        "perfect_ece_near_zero": scenarios["perfect"]["ece"] <= 0.01,
        "overconfident_detected": over_ece >= 0.30,
        "underconfident_detected": scenarios["underconfident"]["ece"] >= 0.30,
        "flat_half_ece_zero": scenarios["flat_half"]["ece"] <= 0.01,
        "temperature_reduces_ece": fit["calibrated_ece"] <= over_ece,
        "verbalisation_four_levels": len(set(verbal.values())) == 4,
        "deterministic": deterministic,
    }
    exit_ = "calibration_ready" if all(checks.values()) else "calibration_failure"
    payload = {
        "schema": "taiji-unc-1-01-v1",
        "card": CARD,
        "design": "DESIGN-UNC-1-01_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "scenarios": scenarios,
        "verbalisation": verbal,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0, "trained_models": 0},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit": exit_, "checks": checks, "cpu": payload["cost"]["cpu"]}, ensure_ascii=False
        )
    )
    return 0 if exit_ == "calibration_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
