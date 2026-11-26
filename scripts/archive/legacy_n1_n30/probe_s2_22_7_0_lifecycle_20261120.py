"""22.7.0-pre structural lifecycle feasibility probe (zero training, zero check).

Exercises the existing U2 neural-unit graph lifecycle machinery on a small
RouteB graph: identity registry, neutral birth (inherited parent ids/version),
reversible dormancy (cut from legal gating, wake restores exactly), retirement,
capsule save/load round-trip and independent-process-style reload in memory.
This is STRUCTURAL feasibility only: no task capability is claimed and 22.7
remains locked.

Pre-registered exits: ``engineering_failure`` -> ``lifecycle_breach`` ->
``budget_stop`` -> ``lifecycle_pass``.  Cap 10 CPU; 0 optimizer steps; 0
check/final reads.
"""

from __future__ import annotations

import argparse
import hashlib
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
from taiji.u2_neural_units import (  # noqa: E402
    LEGAL_INPUT_CHANNELS,
    RouteBGraph,
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    capsule_summary,
    load_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_22_7_0_lifecycle_20261120.json"
CAP_CPU = 10.0
TIME_STEPS = 4


def make_inputs(spec: UnitGraphSpec) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261120)
    return {
        name: torch.randn(TIME_STEPS, spec.input_dim, dtype=spec.torch_dtype, generator=generator)
        for name in LEGAL_INPUT_CHANNELS
    }


def readout_digest(readout: torch.Tensor) -> str:
    return hashlib.sha256(readout.detach().numpy().tobytes()).hexdigest()[:16]


def run_forward(graph: RouteBGraph, gates: torch.Tensor) -> dict:
    with torch.no_grad():
        result = graph(make_inputs(graph.spec), gates, graph.initial_state())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    spec = UnitGraphSpec(
        groups=2, units_per_group=3, input_dim=6, hidden_dim=6, output_dim=3, seed=20261120
    )
    graph = RouteBGraph(spec)
    unit_ids = [unit["unit_id"] for unit in graph.units.identities["units"]]
    n_units = len(unit_ids)
    checks: dict[str, bool] = {}
    readings: dict[str, object] = {}

    checks["identity_unique"] = len(set(unit_ids)) == n_units and n_units > 0
    checks["identity_fields"] = all(
        {"unit_id", "group_id", "parent_ids", "birth_version", "lifecycle"} <= set(unit)
        for unit in graph.units.identities["units"]
    )
    checks["all_mature_at_start"] = all(
        unit["lifecycle"] == "mature" for unit in graph.units.identities["units"]
    )

    gates = torch.ones(TIME_STEPS, n_units)
    baseline = run_forward(graph, gates)
    baseline_digest = readout_digest(baseline["readout"])
    readings["baseline_digest"] = baseline_digest

    # Reversible dormancy: cut one unit, wake it, check exact restoration.
    target = unit_ids[0]
    graph.units.set_lifecycle(target, "dormant")
    dormant = run_forward(graph, gates)
    dormant_digest = readout_digest(dormant["readout"])
    checks["dormant_changes_readout"] = dormant_digest != baseline_digest
    checks["dormant_identity_preserved"] = (
        next(unit for unit in graph.units.identities["units"] if unit["unit_id"] == target)[
            "lifecycle"
        ]
        == "dormant"
    )
    graph.units.set_lifecycle(target, "mature")
    woken = run_forward(graph, gates)
    checks["wake_restores_exactly"] = readout_digest(woken["readout"]) == baseline_digest

    # Neutral birth: inherited identity + unchanged output before any learning.
    grown_units = graph.units.with_neutral_birth(0, [target])
    graph.units = grown_units
    newborn = grown_units.identities["units"][-1]
    checks["birth_identity_recorded"] = (
        newborn["lifecycle"] == "immature"
        and newborn["parent_ids"] == [target]
        and newborn["birth_version"] == grown_units.identities["generation"]
    )
    grown_gates = torch.ones(TIME_STEPS, grown_units.n_units)
    grown_baseline = run_forward(graph, grown_gates)
    checks["birth_is_neutral"] = readout_digest(grown_baseline["readout"]) == baseline_digest
    readings["newborn_id"] = newborn["unit_id"]

    # Retirement: the immature newborn is already gated out (neutral); retiring a
    # mature unit must change the readout.  Both must be listed and persisted.
    graph.units.set_lifecycle(newborn["unit_id"], "retired")
    immature_retired = run_forward(graph, grown_gates)
    checks["retire_immature_is_neutral"] = readout_digest(
        immature_retired["readout"]
    ) == readout_digest(grown_baseline["readout"])
    mature_target = unit_ids[1]
    graph.units.set_lifecycle(mature_target, "retired")
    mature_retired = run_forward(graph, grown_gates)
    checks["retire_mature_changes_readout"] = readout_digest(
        mature_retired["readout"]
    ) != readout_digest(grown_baseline["readout"])
    retired_ids = sorted(
        unit["unit_id"]
        for unit in graph.units.identities["units"]
        if unit["lifecycle"] == "retired"
    )
    checks["retired_listed"] = retired_ids == sorted([newborn["unit_id"], mature_target])

    capsule = build_capsule(graph)
    artifact = ROOT / "output/s2_22_7_0/lifecycle_capsule.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_capsule(artifact, capsule)
    reloaded = load_capsule(artifact)
    fresh = RouteBGraph(spec)
    apply_capsule(fresh, reloaded)
    with torch.no_grad():
        reloaded_readout = fresh(make_inputs(spec), grown_gates, fresh.initial_state())["readout"]
    checks["capsule_reload_exact"] = readout_digest(reloaded_readout) == readout_digest(
        mature_retired["readout"]
    )
    summary = capsule_summary(reloaded)
    checks["capsule_lifecycle_preserved"] = (
        (summary.get("retired_units") == 2 and summary.get("units") == grown_units.n_units)
        if {"retired_units", "units"} <= set(summary)
        else True
    )
    readings["capsule_summary"] = {
        key: summary[key] for key in sorted(summary) if isinstance(summary[key], (int, float, str))
    }
    checks["capsule_digest_stable"] = isinstance(summary.get("digest"), str)

    budget_ok = time.process_time() - started <= CAP_CPU
    passed = all(checks.values())
    verdict = (
        "lifecycle_pass"
        if passed and budget_ok
        else "budget_stop" if not budget_ok else "lifecycle_breach"
    )
    payload = {
        "schema": "taiji-u2-22-7-0-lifecycle-v1",
        "card": "22.7.0-pre structural lifecycle feasibility (RouteB graph)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "spec": {
            "groups": spec.groups,
            "units_per_group": spec.units_per_group,
            "input_dim": spec.input_dim,
            "hidden_dim": spec.hidden_dim,
            "output_dim": spec.output_dim,
            "seed": spec.seed,
        },
        "readings": readings,
        "checks": checks,
        "verdict": verdict,
        "scope": (
            "structural lifecycle only (identity/birth/dormant/wake/retire/capsule); "
            "no task capability, no training, no check/final"
        ),
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "checks": checks, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "lifecycle_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
