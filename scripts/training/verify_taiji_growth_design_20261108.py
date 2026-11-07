"""Constructive pre-review of the growth decision rule (0 training, numpy only).

Goal, exactly as the owner asked: show that the trigger fires on a known
key-space collapse and does *not* fire when capacity is sufficient, and that
"capacity short" and "structurally inseparable" are told apart.  Fixtures are
synthetic and seeded; the only real-world reading replayed is the recorded
20.3a aggregate (top cosine 0.99766, margin 0.00835).

No torch, no model, no checkpoint, no data set, no optimizer.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_design as gd  # noqa: E402

REPORT = ROOT / "reports/taiji_growth_design_verify_20261108.json"
SEED = 20261108


def collapsed_keys(dimension: int, count: int = 32, offset: float = 0.999, noise: float = 0.03):
    """Mean-pooled-context stand-in: one shared direction plus a little noise."""
    rng = np.random.default_rng(SEED)
    shared = gd.unit_rows(rng.normal(size=(1, dimension)))[0]
    directions = gd.unit_rows(rng.normal(size=(count, dimension)))
    return gd.unit_rows(offset * shared[None, :] + noise * directions)


def separable_keys(dimension: int, count: int = 32):
    rng = np.random.default_rng(SEED + 1)
    return gd.unit_rows(rng.normal(size=(count, dimension)))


def capacity_case(*, rho_target: float = 6.0, inconsistent: float = 0.0):
    """Rank-deficient key set (m > rank) so an inconsistent target is possible."""
    rng = np.random.default_rng(SEED + 2)
    K = rng.normal(size=(16, 4)) @ rng.normal(size=(4, 8))
    X0 = rng.normal(size=(16, 4))
    pinv = np.linalg.pinv(K)
    X0 *= rho_target / np.linalg.norm(X0.T @ K @ pinv)
    V = X0.T @ K
    if inconsistent:
        junk = rng.normal(size=V.shape)
        junk -= junk @ pinv @ K
        junk *= inconsistent / np.linalg.norm(junk)
        V = V + junk
    return K, V


def case(name: str, purpose: str, decision: dict, expected: dict) -> dict:
    checks = {key: decision.get(key) == value for key, value in expected.items()}
    return {
        "name": name,
        "purpose": purpose,
        "decision": decision["decision"],
        "diagnosis": decision["diagnosis"],
        "expected": expected,
        "checks": checks,
        "passed": all(checks.values()),
        "readings": decision.get("evidence", {}),
    }


def main() -> int:
    started, started_cpu = time.time(), time.process_time()
    cases = []

    spectrum_a = gd.key_spectrum(collapsed_keys(128))
    margins_a = gd.read_margin_stats(collapsed_keys(128))
    decision_a = gd.store_decision(spectrum_a, margins_a, calibrated=True)
    cases.append(
        case(
            "A_shared_offset_collapse_d128",
            "known collapse: one shared direction dominates the key space",
            decision_a,
            {"collapse_detected": True, "decision": "representation_repair"},
        )
    )

    spectrum_b = gd.key_spectrum(separable_keys(128))
    margins_b = gd.read_margin_stats(separable_keys(128))
    decision_b = gd.store_decision(spectrum_b, margins_b, calibrated=True)
    cases.append(
        case(
            "B_separable_d128",
            "known sufficient case: random near-orthogonal keys",
            decision_b,
            {"collapse_detected": False, "decision": "no_growth"},
        )
    )

    spectrum_f = gd.key_spectrum(collapsed_keys(512))
    margins_f = gd.read_margin_stats(collapsed_keys(512))
    decision_f = gd.store_decision(spectrum_f, margins_f, calibrated=True)
    cases.append(
        case(
            "F_same_collapse_d512",
            "dimension growth does not change an angular collapse",
            decision_f,
            {"collapse_detected": True, "decision": decision_a["decision"]},
        )
    )

    replay = {
        "p95_top_cos": 0.9976629252533259,
        "median_margin": 0.008352757617859585,
    }
    decision_g = gd.store_decision(None, replay, calibrated=True)
    cases.append(
        case(
            "G_replay_20_3a_recorded_reading",
            "recorded 20.3a aggregate: collapse detected, attribution withheld",
            decision_g,
            {"collapse_detected": True, "decision": "no_growth"},
        )
    )

    K, V = capacity_case()
    cert_c = gd.capacity_certificate(K, V, 4.0)
    decision_c = gd.operator_decision(cert_c)
    cases.append(
        case(
            "C_capacity_short_consistent",
            "consistent target, ball too small",
            decision_c,
            {"decision": "capacity_growth"},
        )
    )

    K_d, V_d = capacity_case(inconsistent=0.5)
    cert_d = gd.capacity_certificate(K_d, V_d, 4.0)
    decision_d = gd.operator_decision(cert_d)
    cases.append(
        case(
            "D_structurally_inseparable",
            "target outside the key row space: growth cannot help",
            decision_d,
            {"decision": "structural_repair"},
        )
    )

    cert_e = gd.capacity_certificate(K, V, 10.0)
    decision_e = gd.operator_decision(cert_e)
    cases.append(
        case(
            "E_capacity_sufficient",
            "same target, ball large enough",
            decision_e,
            {"decision": "no_growth"},
        )
    )

    allocation = gd.allocate(
        {"object": [0.0, 0.50, 0.52, 0.53], "record": [0.0, 0.20, 0.25, 0.26]},
        {"object": 3, "record": 3},
        r_max=4,
    )
    cases.append(
        {
            "name": "H_allocation_not_even",
            "purpose": "max-min allocation never splits units evenly by default",
            "decision": json.dumps(allocation, sort_keys=True),
            "diagnosis": None,
            "expected": {},
            "checks": {},
            "passed": allocation.get("object") != allocation.get("record"),
            "readings": {"allocation": allocation},
        }
    )

    payload = {
        "format": "taiji-growth-design-verify-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thresholds": gd.THRESHOLDS,
        "cases": cases,
        "all_expected_met": all(item["passed"] for item in cases),
        "slot_growth_bytes_32_to_64": gd.slot_growth_bytes(32),
        "cpu_seconds": round(time.process_time() - started_cpu, 3),
        "wall_seconds": round(time.time() - started, 3),
        "boundary": (
            "synthetic fixtures only; the 20.3a replay uses its recorded aggregate "
            "cosine/margin, not a replayed key tensor, so attribution stays withheld"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(REPORT)
    for item in cases:
        print(
            f"{item['name']:<34} passed={item['passed']!s:<5} "
            f"decision={item['decision']:<22} diagnosis={item['diagnosis']}"
        )
    print(
        json.dumps(
            {
                "all_expected_met": payload["all_expected_met"],
                "cpu_seconds": payload["cpu_seconds"],
                "wall_seconds": payload["wall_seconds"],
                "report": REPORT.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["all_expected_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
