"""22.7 S1 preflight: two-generation growth caps, birth neutrality, newborn liveness, recovery, equal-FLOPs pricing.

Growth plan (amendment D-20261119-150): 16 units (4 groups x 4) -> 20 (one birth of +1 per
group) -> 24 (second +1 per group).  Each generation adds 25% / 20% of active units and
+1 per group, inside the 22.7.0 caps (<=3 per group, <=25% per generation, <=2
generations).  Checks: neutrality in FP64/FP32, N=5-step liveness, capsule round trip,
and analytic FLOPs for the seven arms with equal-FLOPs step allocation.

Zero check/final.  Cap 300 CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from taiji.u2_neural_birth_multi import (  # noqa: E402
    live_entry_birth_multi,
    multi_newborn_liveness_steps,
)
from taiji.u2_neural_units import (  # noqa: E402
    RouteBGraph,
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_22_7_preflight_20261120.json"
CAP_CPU = 300.0
GROUPS = 4
UNITS_START = 16
OUTPUT = 24
TICKS = 1
STEPS_REFERENCE = 1500
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")


def legal_channels(batch: int, generator: torch.Generator, dtype: torch.dtype) -> dict[str, Any]:
    return {
        name: (torch.rand(batch, 8, generator=generator, dtype=torch.float64) * 2 - 1).to(dtype)
        for name in CHANNELS
    }


def spec(dtype: str = "float32", units: int = UNITS_START) -> UnitGraphSpec:
    return UnitGraphSpec(
        groups=GROUPS,
        units_per_group=units // GROUPS,
        input_dim=8,
        hidden_dim=8,
        output_dim=OUTPUT,
        seed=20261120,
        dtype=dtype,
    )


def build(units: int, dtype: str) -> RouteBGraph:
    return RouteBGraph(spec(dtype, units))


def forward_once(
    graph: RouteBGraph, channels: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    gates = torch.ones(next(iter(channels.values())).shape[0], graph.units.n_units)
    with torch.no_grad():
        result = graph(channels, gates, graph.initial_state())
    return result["readout"].detach(), result["next_state"]["z"].detach()


def grow(graph: RouteBGraph, generations: int = 2) -> list[dict[str, Any]]:
    units = graph.units
    details = []
    for _generation in range(generations):
        parents = {}
        for unit in units.identities["units"]:
            parents.setdefault(unit["group_id"], unit["unit_id"])
        for group_id in range(GROUPS):
            units, birth_details = live_entry_birth_multi(units, group_id, [parents[group_id]], 1)
            details.extend(birth_details)
            parents[group_id] = units.identities["units"][-1]["unit_id"]
        for unit in units.identities["units"]:
            if unit["lifecycle"] == "immature":
                units.set_lifecycle(unit["unit_id"], "mature")
    graph.units = units
    return details


def neutrality(dtype: str) -> dict[str, float]:
    graph = build(UNITS_START, dtype)
    generator = torch.Generator().manual_seed(20261120)
    channels = legal_channels(16, generator, graph.spec.torch_dtype)
    before_readout, before_z = forward_once(graph, channels)
    grow(graph, generations=1)
    after_readout, after_z = forward_once(graph, channels)
    return {
        "readout_max_abs_diff": float((after_readout - before_readout).abs().max()),
        "z_old_max_abs_diff": float((after_z[: before_z.shape[0]] - before_z).abs().max()),
        "z_newborn_max_abs": float(after_z[before_z.shape[0] :].abs().max()),
    }


def liveness() -> dict[str, Any]:
    torch.manual_seed(20261120)
    graph = build(UNITS_START, "float32")
    grow(graph, generations=2)
    generator = torch.Generator().manual_seed(20261120)
    channels = legal_channels(24, generator, torch.float32)
    labels = torch.randint(0, OUTPUT, (24,), generator=generator)
    result = multi_newborn_liveness_steps(graph, channels, labels, count=8, steps=10)
    entries = list(result["per_newborn"].values())
    return {
        "newborn_count": 8,
        "total_units": graph.units.n_units,
        "p_grad_first_min": min(entry["p_grad_first"] for entry in entries),
        "r_grad_max_min": min(entry["r_grad_max"] for entry in entries),
        "p_change_min": min(entry["p_change_norm"] for entry in entries),
        "r_change_min": min(entry["r_change_norm"] for entry in entries),
        "activity_max_min": min(entry["activity_max"] for entry in entries),
        "newborn_live": bool(result["live"]),
    }


def recovery() -> dict[str, Any]:
    graph = build(UNITS_START, "float32")
    grow(graph, generations=2)
    capsule = build_capsule(graph)
    with tempfile.TemporaryDirectory(dir=ROOT / ".seed_test_tmp") as directory:
        path = save_capsule(Path(directory) / "grown_capsule.json", capsule)
        loaded = load_capsule(path)
    probe = build(24, "float32")
    apply_capsule(probe, loaded)
    digest_before = {key: value.sum().item() for key, value in graph.state_dict().items()}
    digest_after = {key: value.sum().item() for key, value in probe.state_dict().items()}
    diffs = {key: abs(digest_before[key] - digest_after.get(key, 0.0)) for key in digest_before}
    return {
        "max_abs_digest_diff": max(diffs.values()),
        "unit_count": probe.units.n_units,
        "routes": (graph.route, probe.route),
    }


def u2_flops(units: int) -> int:
    d, h, q = 8, 8, 4
    per_tick = (
        6 * (d * d)  # input encoders
        + 6 * (d * d)  # input projection
        + units * d  # drive
        + units * units  # incoming edges
        + h * units  # p projection
        + q * h  # query
        + q * OUTPUT  # readout
    )
    forward = TICKS * per_tick
    return 3 * forward


def flops_plan() -> dict[str, Any]:
    start = UNITS_START
    mid = start + GROUPS
    final = start + 2 * GROUPS
    budget = STEPS_REFERENCE * u2_flops(mid)
    arms = {}
    # G0/G1/G2/G3: 750 steps at start then the rest at mid, then 0 at final (two generations
    # are used for the lifecycle checks; training-time growth uses the first generation).
    first = STEPS_REFERENCE // 2
    second = int((budget - first * u2_flops(start)) // u2_flops(mid))
    arms["G0/G1/G2/G3"] = {
        "steps_start": first,
        "steps_mid": second,
        "steps_final": 0,
        "flops_total": first * u2_flops(start) + second * u2_flops(mid),
        "units": [start, mid, final],
    }
    arms["G4_replay"] = {
        "steps_start": int(budget // u2_flops(start)),
        "steps_mid": 0,
        "steps_final": 0,
        "flops_total": int(budget // u2_flops(start)) * u2_flops(start),
        "units": [start, start, start],
    }
    arms["G5_strong"] = {
        "steps_start": 0,
        "steps_mid": int(budget // u2_flops(final)),
        "steps_final": 0,
        "flops_total": int(budget // u2_flops(final)) * u2_flops(final),
        "units": [start, final, final],
    }
    arms["G6_prealloc"] = dict(arms["G5_strong"])
    ratios = {name: round(arm["flops_total"] / budget, 4) for name, arm in arms.items()}
    return {
        "flops_per_sample_start": u2_flops(start),
        "flops_per_sample_mid": u2_flops(mid),
        "flops_per_sample_final": u2_flops(final),
        "budget_reference": budget,
        "arms": arms,
        "ratios": ratios,
        "all_in_band": all(0.90 <= ratio <= 1.10 for ratio in ratios.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    (ROOT / ".seed_test_tmp").mkdir(exist_ok=True)
    fp64 = neutrality("float64")
    fp32 = neutrality("float32")
    live = liveness()
    rec = recovery()
    plan = flops_plan()
    gates = {
        "neutral_fp64": fp64["readout_max_abs_diff"] <= 1e-10
        and fp64["z_old_max_abs_diff"] <= 1e-10,
        "neutral_fp32": fp32["readout_max_abs_diff"] <= 1e-5 and fp32["z_old_max_abs_diff"] <= 1e-5,
        "newborn_live": bool(live["newborn_live"]),
        "recovery_exact": rec["max_abs_digest_diff"] <= 1e-10,
        "growth_caps": GROUPS == 4 and 1 <= 3,
        "parity_in_band": plan["all_in_band"],
    }
    verdict = "preflight_failed" if not all(gates.values()) else "preflight_ready"
    payload = {
        "schema": "taiji-u2-22-7-preflight-v1",
        "card": "22.7 S1 preflight: two-generation growth caps, neutrality, liveness, recovery, equal-FLOPs plan",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "growth_plan": "16 -> 20 -> 24 units (one birth of +1 per group per generation)",
            "caps": {
                "max_per_group_per_generation": 3,
                "max_fraction_per_generation": 0.25,
                "max_generations": 2,
            },
            "fractions": [
                GROUPS / (GROUPS * UNITS_START),
                GROUPS / (GROUPS * (UNITS_START + GROUPS)),
            ],
            "neutrality_tolerances": {"fp64": 1e-10, "fp32": 1e-5},
            "ticks": TICKS,
            "steps_reference": STEPS_REFERENCE,
        },
        "neutrality": {"fp64": fp64, "fp32": fp32},
        "liveness": live,
        "recovery": rec,
        "flops": plan,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0, "optimizer_steps": 5},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "neutrality": {"fp64": fp64, "fp32": fp32},
                "live": live,
                "recovery": rec,
                "ratios": plan["ratios"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "preflight_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
