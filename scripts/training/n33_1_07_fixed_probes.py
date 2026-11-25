"""N33.1.07 (zero training): freeze the neutral fixed probe set (K=150) for G2.

Pre-registered: plans/reference/DESIGN-N33-1-07_20261125.md (D-20261125-567).
Only the measurement location changes: the G2 threshold stays at 0.50 per domain. No policy
is trained and no confirmation seed is read.

Usage: python scripts/training/n33_1_07_fixed_probes.py
Output: reports/n33_1_07_probes.json (+ plans/manifests/n33_1_probe_states_20261125.json)
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

from taiji.choice_probes import (
    DOMAINS,
    build_probe_set,
    load_probe_set,
    probe_shares,
    save_probe_set,
)  # noqa: E402
from taiji.choice_simulator import scripted_policy  # noqa: E402

CARD = "N33.1.07"
OUT = ROOT / "reports" / "n33_1_07_probes.json"


def main() -> int:
    started = time.process_time()
    first = build_probe_set()
    second = build_probe_set()
    save_probe_set(first)
    manifest = load_probe_set()
    deterministic = first["sha256"] == second["sha256"] == manifest["sha256"]
    per_domain = {d: manifest["state_domains"].count(d) for d in DOMAINS}
    scripted = probe_shares(scripted_policy)
    constant = probe_shares(lambda s: "wait_wake")
    scripted_pass = (
        min(
            scripted["rest_share"],
            scripted["verify_or_help_share"],
            scripted["continue_or_recall_share"],
        )
        >= 0.50
    )
    constant_fail = (
        max(
            constant["rest_share"],
            constant["verify_or_help_share"],
            constant["continue_or_recall_share"],
        )
        == 0.0
    )
    exit_ = (
        "probes_frozen"
        if deterministic
        and scripted_pass
        and constant_fail
        and all(v == 50 for v in per_domain.values())
        else "probes_invalid"
    )
    payload = {
        "schema": "taiji-n33-1-07-v1",
        "card": CARD,
        "design": "DESIGN-N33-1-07_20261125.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest": "plans/manifests/n33_1_probe_states_20261125.json",
        "sha256": manifest["sha256"],
        "fixture": manifest["fixture"],
        "k_total": manifest["k_total"],
        "per_domain": per_domain,
        "deterministic": deterministic,
        "scripted_probes": scripted,
        "constant_probes": constant,
        "gates": {"share_min": 0.50},
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": 5.0},
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
                "sha256": manifest["sha256"][:16],
                "k_total": manifest["k_total"],
                "per_domain": per_domain,
                "deterministic": deterministic,
                "scripted_probes": scripted,
                "constant_probes": constant,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ == "probes_frozen" else 3


if __name__ == "__main__":
    raise SystemExit(main())
