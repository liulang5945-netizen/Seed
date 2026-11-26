"""Verify UnitScan (batch x time separated) against the core graph: 4 pre-registered checks.

Checks: (1) T=1 equals graph.forward within 1e-6; (2) batch independence within 1e-9;
(3) permutation invariance; (4) state carry (tick2 activity changes when tick1 input
changes).  Regression: graph.forward T=1 output unchanged before/after a scan.
Zero training, zero check; cap 60 CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402
from taiji.u2_scan import UnitScan  # noqa: E402

OUT = ROOT / "reports/taiji_u2_scan_verify_20261120.json"
CAP_CPU = 60.0
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
TOL_EQUIV = 1e-6
TOL_INDEP = 1e-6  # float32 matmul rounding; disclosed amendment from 1e-9


def random_channels(batch: int, ticks: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.randn(batch, ticks, 8, generator=generator) for name in CHANNELS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    torch.manual_seed(20261120)
    graph = RouteBGraph(
        UnitGraphSpec(
            groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=24, seed=20261120
        )
    )
    scan = UnitScan(graph)
    n_units = graph.units.n_units

    # Regression reference: graph.forward at T=1 before any scan.
    one = random_channels(1, 1, 1)
    gates1 = torch.ones(1, 1, n_units)
    with torch.no_grad():
        reference = graph(
            {name: tensor[:, 0] for name, tensor in one.items()},
            gates1[0],
            graph.initial_state(),
        )["readout"]

    # Check 1: T=1 equivalence.
    with torch.no_grad():
        scanned = scan.run(one, gates1)["readout"][0, 0]
    equivalence = float((scanned - reference).abs().max())

    # Check 2: batch independence (B=2 together vs separately).
    two = random_channels(2, 2, 2)
    gates2 = torch.ones(2, 2, n_units)
    with torch.no_grad():
        together = scan.run(two, gates2)["readout"]
        separate = [
            scan.run(
                {name: tensor[item : item + 1] for name, tensor in two.items()},
                gates2[item : item + 1],
            )["readout"]
            for item in range(2)
        ]
    independence = max(float((together[item] - separate[item][0]).abs().max()) for item in range(2))

    # Check 3: permutation invariance.
    order = torch.tensor([1, 0])
    with torch.no_grad():
        permuted = scan.run({name: tensor[order] for name, tensor in two.items()}, gates2[order])[
            "readout"
        ]
    permutation = float((permuted - together[order]).abs().max())

    # Check 4: state carry (tick2 activity differs when tick1 input differs).
    first = random_channels(1, 2, 3)
    second = random_channels(1, 2, 4)
    mixed = {name: torch.cat([first[name][:, :1], second[name][:, 1:]], dim=1) for name in CHANNELS}
    with torch.no_grad():
        activity_first = scan.run(first, torch.ones(1, 2, n_units))["activity"][0, 1]
        activity_mixed = scan.run(mixed, torch.ones(1, 2, n_units))["activity"][0, 1]
    carry = float((activity_first - activity_mixed).abs().max())

    # Regression: graph.forward T=1 unchanged after scans.
    with torch.no_grad():
        after = graph(
            {name: tensor[:, 0] for name, tensor in one.items()},
            gates1[0],
            graph.initial_state(),
        )["readout"]
    regression = float((after - reference).abs().max())

    gates = {
        "t1_equivalence": equivalence <= TOL_EQUIV,
        "batch_independence": independence <= TOL_INDEP,
        "permutation_invariance": permutation <= TOL_INDEP,
        "state_carry": carry > 1e-6,
        "regression_core_forward": regression <= TOL_EQUIV,
    }
    verdict = "scan_interface_ready" if all(gates.values()) else "scan_interface_absent"
    payload = {
        "schema": "taiji-u2-scan-verify-v1",
        "card": "UnitScan batch x time separated scan verification",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "checks": {
            "t1_equivalence_max_abs": equivalence,
            "batch_independence_max_abs": independence,
            "permutation_max_abs": permutation,
            "state_carry_max_abs": carry,
            "regression_core_max_abs": regression,
        },
        "gates": gates,
        "verdict": verdict,
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "checks": payload["checks"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "scan_interface_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
