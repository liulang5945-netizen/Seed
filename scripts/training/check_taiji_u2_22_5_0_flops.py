"""22.5.0 FLOPs parity diagnostic after the frozen time-parity check failed.

This is a post-hoc cross-check, not a metric switch to force a pass. It counts
forward+backward FLOPs on the same fragment and compares the 256/24 dose
envelope. No training, optimizer step, check or final read is performed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.baseline_identity_22_4_4 import (  # noqa: E402
    DTYPE,
    GOAL,
    build_arm,
    drive_u2,
    fragment_inputs,
)
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

REPORT = ROOT / "reports/taiji_u2_22_5_0_flops_parity_20261117.json"
TIME_REPORT = ROOT / "reports/taiji_u2_22_5_0_compute_parity_20261117.json"
MAX_UPDATES = 256
EFFECTIVE_MIN_UPDATES = 24


def _count_u2() -> int:
    torch.manual_seed(20261110)
    model = U2Subject(U2Config.small()).to(dtype=DTYPE).eval()
    counter = FlopCounterMode(display=False)
    with counter:
        drive_u2(model, None)
    return int(counter.get_total_flops())


def _count_baseline(name: str, width: int) -> int:
    model = build_arm(name, width)
    observations, targets = fragment_inputs()
    text_context = model.frontend.text_units(GOAL)
    counter = FlopCounterMode(display=False)
    with counter:
        loss = model(observations, targets, text_context)
        loss.backward()
    return int(counter.get_total_flops())


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    counts = {
        "u2_reference": _count_u2(),
        "transformer_sequence": _count_baseline("transformer_sequence", 28),
        "gru_sequence": _count_baseline("gru_sequence", 40),
    }
    u2 = counts["u2_reference"]
    ratios = {arm: u2 / value for arm, value in counts.items() if arm != "u2_reference"}
    cap_ratio = MAX_UPDATES / EFFECTIVE_MIN_UPDATES
    common = all(value <= cap_ratio for value in ratios.values())
    time_report = json.loads(TIME_REPORT.read_text(encoding="utf-8"))
    report = {
        "schema": "taiji-u2-22-5-0-flops-parity-v1",
        "card": "22.5.0",
        "verdict": "compute_parity_available" if common else "compute_parity_unavailable",
        "parity_quantity": "flops",
        "method": "torch.utils.flop_counter.FlopCounterMode, forward+backward, same fragment",
        "counts": counts,
        "ratios": {arm: round(value, 6) for arm, value in ratios.items()},
        "max_updates": MAX_UPDATES,
        "effective_min_updates": EFFECTIVE_MIN_UPDATES,
        "cap_ratio": round(cap_ratio, 6),
        "common_flops_budget_exists": common,
        "time_parity_v1": {
            "ref": str(TIME_REPORT.relative_to(ROOT)),
            "sha256": hashlib.sha256(
                TIME_REPORT.read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
            "verdict": time_report["verdict"],
            "ratios": time_report["ratios"],
        },
        "interpretation": (
            "Both permitted parity quantities fail: time ratios are 57.99x/64.91x and FLOPs "
            "ratios are 26.31x/29.92x, while the 256/24 dose envelope allows only 10.67x. "
            "This is a diagnostic, not a metric switch to reopen the gate."
        ),
        "options_for_owner": [
            {
                "id": "A",
                "name": "time parity with a new dose cap",
                "effect": "would require re-freezing 22.5.2 and may land in budget_cannot_express",
            },
            {
                "id": "B",
                "name": "FLOPs parity",
                "effect": "tested here; also unavailable within the frozen dose envelope",
            },
            {
                "id": "C",
                "name": "comparable-cost baseline or U2 variant",
                "effect": "requires a new identity version and re-freeze",
            },
            {
                "id": "D",
                "name": "close 22.5.0 as a limited exit",
                "effect": "no architecture claim, no 22.5.1/22.6/22.7 unlock",
            },
        ],
        "recommended_option": "D",
        "recommendation_note": (
            "D closes this 22.5 window honestly; C is the only higher-ceiling path but is a "
            "separate new identity/re-freeze decision, not an automatic next step."
        ),
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 6),
            "wall": round(time.perf_counter() - started_wall, 6),
        },
        "discipline": {
            "model_forwards": 3,
            "optimizer_steps": 0,
            "project_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
    }
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "flops verdict="
        + report["verdict"]
        + " ratios="
        + json.dumps(report["ratios"], ensure_ascii=False)
        + " cap="
        + str(report["cap_ratio"])
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
