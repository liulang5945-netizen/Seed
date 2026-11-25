"""ECL.1.01 (zero training): calibration of the evidence-credibility ledger.

Design: plans/reference/DESIGN-ECL-1-01_20261126.md (D-20261126-617).
Checks, over 20 deterministic scenarios, that low-credibility contradiction does not flip an
accepted claim, a credible contradiction is recorded as an explicit correction, ties stay
pending, provenance order is preserved and two identical runs digest identically.

Usage: python scripts/training/ecl_1_01_calibration.py
Output: reports/ecl_1_01_ledger.json
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

from taiji.evidence_ledger import ACCEPT_THRESHOLD, CRED_PRIOR, CredibilityLedger  # noqa: E402

CARD = "ECL.1.01"
SEEDS = list(range(4600, 4620))
CAP_CPU = 5.0
OUT = ROOT / "reports" / "ecl_1_01_ledger.json"


def _build(seed: int) -> CredibilityLedger:
    led = CredibilityLedger()
    claim = f"c-{seed}"
    # A: strong accept, then weak contradiction.
    led.add(f"{claim}-A", +1, "owner")
    led.add(f"{claim}-A", -1, "rumor")
    # B: credible contradiction after acceptance -> explicit correction.
    led.add(f"{claim}-B", +1, "forum")
    led.add(f"{claim}-B", +1, "sensor")
    led.add(f"{claim}-B", -1, "owner")
    led.add(f"{claim}-B", -1, "owner")
    # C: tie stays pending.
    led.add(f"{claim}-C", +1, "sensor")
    led.add(f"{claim}-C", -1, "sensor")
    return led


def main() -> int:
    started = time.process_time()
    a_ok = b_ok = c_ok = prov_ok = 0
    for seed in SEEDS:
        led = _build(seed)
        claim = f"c-{seed}"
        sa = led.get(f"{claim}-A")
        a_ok += int(sa.status == "accepted" and sa.polarity == 1)
        sb = led.get(f"{claim}-B")
        b_ok += int(sb.status == "accepted" and sb.polarity == -1 and len(sb.corrections) == 1)
        sc = led.get(f"{claim}-C")
        c_ok += int(sc.status == "pending" and abs(sc.posterior - 0.5) < 1e-9)
        prov_ok += int([e.source for e in led.evidence][-2:] == ["sensor", "sensor"])

    digests_equal = _build(SEEDS[0]).digest() == _build(SEEDS[0]).digest()
    checks = {
        "low_credibility_never_flips": a_ok == len(SEEDS),
        "credible_contradiction_corrected": b_ok == len(SEEDS),
        "ties_pending": c_ok == len(SEEDS),
        "provenance_order_preserved": prov_ok == len(SEEDS),
        "deterministic_digest": digests_equal,
        "credibility_prior_ordered": CRED_PRIOR["owner"]
        > CRED_PRIOR["manual"]
        > CRED_PRIOR["sensor"]
        > CRED_PRIOR["forum"]
        > CRED_PRIOR["rumor"],
        "threshold_in_range": 0.0 < ACCEPT_THRESHOLD < 1.0,
    }
    exit_ = "ledger_ready" if all(checks.values()) else "ledger_failure"
    payload = {
        "schema": "taiji-ecl-1-01-v1",
        "card": CARD,
        "design": "DESIGN-ECL-1-01_20261126.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "scenarios": len(SEEDS),
        "checks": checks,
        "cred_prior": CRED_PRIOR,
        "accept_threshold": ACCEPT_THRESHOLD,
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
    return 0 if exit_ == "ledger_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
