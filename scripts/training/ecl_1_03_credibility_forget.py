"""ECL.1.03 (exploration): credibility learning with exponential forgetting for shift tracking.

Design: plans/reference/DESIGN-ECL-1-03_20261126.md (D-20261126-640).
Compares the static engineering priors against an online Beta-credibility estimator (learned
inside the subject from verifiable feedback) on a claim stream with a mid-stream reliability
shift; oracle accuracy is a control only. Exploration seeds 6201-6203; no confirmation seed,
check or final is read.

Usage: python scripts/training/ecl_1_03_credibility_forget.py
Output: reports/ecl_1_03_credibility.json
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

from taiji.credibility_learning import (  # noqa: E402
    N_ITEMS,
    SHIFT_INDEX,
    SOURCES,
    TRUE_ACC,
    BetaCredibility,
    ClaimStream,
    OracleAccuracy,
    StaticPrior,
    evaluate,
)

CARD = "ECL.1.03"
SEEDS = (6201, 6202, 6203)
CAP_CPU = 15.0
OUT = ROOT / "reports" / "ecl_1_03_credibility.json"


def main() -> int:
    started = time.process_time()
    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        stream = ClaimStream(seed)
        items = stream.generate(N_ITEMS)
        static = evaluate(StaticPrior(), items)
        beta = evaluate(BetaCredibility(forgetting=0.995), items)
        oracle = evaluate(OracleAccuracy(stream), items)
        # no-shift convergence check on a separate stream
        no_shift_items = ClaimStream(seed + 100, shift=False).generate(4000)
        beta_conv = BetaCredibility()
        for _i, (src, correct) in enumerate(no_shift_items):
            beta_conv.update(src, correct)
        conv_err = max(abs(beta_conv.predict(src, 0) - TRUE_ACC[src]) for src in SOURCES)
        deterministic = evaluate(BetaCredibility(forgetting=0.995), items)["brier"] == beta["brier"]
        per_seed[str(seed)] = {
            "static": static,
            "beta": beta,
            "oracle": oracle,
            "beta_vs_static_brier": round(beta["brier"] - static["brier"], 4),
            "beta_vs_static_post": round(beta["post_abs_error"] - static["post_abs_error"], 4),
            "no_shift_convergence_err": round(conv_err, 4),
            "deterministic": deterministic,
        }
    checks = {
        "g1_brier_better": all(v["beta_vs_static_brier"] <= -0.02 for v in per_seed.values()),
        "g2_post_shift_better": all(v["beta_vs_static_post"] <= -0.05 for v in per_seed.values()),
        "g3_converges": all(v["no_shift_convergence_err"] <= 0.10 for v in per_seed.values()),
        "g4_deterministic": all(v["deterministic"] for v in per_seed.values()),
    }
    exit_ = (
        "credibility_learned_positive" if all(checks.values()) else "credibility_learned_partial"
    )
    payload = {
        "schema": "taiji-ecl-1-03-v1",
        "card": CARD,
        "design": "DESIGN-ECL-1-03_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "train_seeds": list(SEEDS),
        "items": N_ITEMS,
        "shift_index": SHIFT_INDEX,
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
    return 0 if exit_ == "credibility_learned_positive" else 3


if __name__ == "__main__":
    raise SystemExit(main())
