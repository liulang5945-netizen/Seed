"""22.7.0-pre lifecycle extension: cross-process reload, reference cleanup, state growth.

Builds on ``probe_s2_22_7_0_lifecycle_20261120`` (structural 13/13 pass) and adds
the three gaps listed in the memo:

* independent-process capsule reload: a fresh Python process rebuilds the graph,
  applies the capsule strictly and must reproduce the in-process readout bitwise;
* retired-reference cleanup: no dangling parent ids, no active edge touching a
  retired unit, active gating zero for dormant/retired, registry/capsule summary
  consistent;
* state growth: ``extend_state`` preserves the old z/a prefix bitwise, pads the
  newborn entries with zeros and the post-forward newborn state stays zero while
  it is immature.

Zero training, zero check/final; exits engineering_failure / lifecycle_breach /
budget_stop / lifecycle_ext_pass; cap 10 CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
    build_capsule,
    capsule_summary,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_22_7_0_lifecycle_ext_20261120.json"
HELPER = ROOT / "scripts/training/s2_22_7_0_reload_20261120.py"
CAP_CPU = 10.0
TIME_STEPS = 4


def digest(readout: torch.Tensor) -> str:
    return hashlib.sha256(readout.detach().numpy().tobytes()).hexdigest()


def inputs(spec: UnitGraphSpec) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261120)
    return {
        name: torch.randn(TIME_STEPS, spec.input_dim, dtype=spec.torch_dtype, generator=generator)
        for name in LEGAL_INPUT_CHANNELS
    }


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
    checks: dict[str, bool] = {}

    # ---- state growth: newborn zero-padded, old prefix preserved. -------------
    state = graph.initial_state()
    grown_state = graph.extend_state(state, extra=1)
    checks["state_old_prefix_preserved"] = torch.equal(
        grown_state["z"][: state["z"].numel()], state["z"]
    ) and torch.equal(grown_state["a"][: state["a"].numel()], state["a"])
    checks["state_newborn_zero"] = (
        float(grown_state["z"][state["z"].numel() :].abs().sum()) == 0.0
        and float(grown_state["a"][state["a"].numel() :].abs().sum()) == 0.0
    )
    grown = graph.units.with_neutral_birth(0, [unit_ids[0]])
    graph.units = grown
    grown_gates = torch.ones(TIME_STEPS, grown.n_units)
    state = graph.initial_state()
    with torch.no_grad():
        result = graph(inputs(spec), grown_gates, state)
    checks["newborn_state_neutral_after_forward"] = (
        float(result["next_state"]["z"][-1].abs().sum()) == 0.0
    )
    checks["newborn_is_immature"] = grown.identities["units"][-1]["lifecycle"] == "immature"

    # ---- retirement + reference cleanup. --------------------------------------
    lifecycles = {}
    for unit_id, phase in ((unit_ids[1], "retired"), (unit_ids[2], "dormant")):
        graph.units.set_lifecycle(unit_id, phase)
        lifecycles[unit_id] = phase
    registry = graph.units.identities["units"]
    known = {unit["unit_id"] for unit in registry}
    checks["no_dangling_parent_ids"] = all(
        parent in known for unit in registry for parent in unit["parent_ids"]
    )
    retired_ids = {uid for uid, phase in lifecycles.items() if phase == "retired"}
    dormant_ids = {uid for uid, phase in lifecycles.items() if phase == "dormant"}
    active_edges = [
        edge for edge in graph.units.identities["edges"] if edge.get("retire_version") is None
    ]
    checks["no_active_edge_touches_retired"] = all(
        edge.get("source_unit_id") not in retired_ids
        and edge.get("target_unit_id") not in retired_ids
        for edge in active_edges
    )
    with torch.no_grad():
        gates = torch.ones(TIME_STEPS, graph.units.n_units)
        active = graph.units.active_gate(gates)
    index = {unit["unit_id"]: position for position, unit in enumerate(registry)}
    checks["gating_cut_for_non_mature"] = all(
        float(active[0, index[unit_id]]) == 0.0 for unit_id in retired_ids | dormant_ids
    )
    summary = capsule_summary(build_capsule(graph))
    checks["capsule_summary_consistent"] = (
        summary.get("units") == graph.units.n_units
        and isinstance(summary.get("digest"), str)
        and summary.get("route") == graph.route
    )

    # ---- independent-process reload. -----------------------------------------
    capsule = build_capsule(graph)
    artifact = ROOT / "output/s2_22_7_0/lifecycle_ext_capsule.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_capsule(artifact, capsule)
    with torch.no_grad():
        in_process = digest(graph(inputs(spec), gates, graph.initial_state())["readout"])
    out_path = ROOT / "output/s2_22_7_0/lifecycle_ext_reload.json"
    completed = subprocess.run(
        [sys.executable, str(HELPER), "--capsule", str(artifact), "--out", str(out_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    checks["reload_subprocess_ok"] = completed.returncode == 0
    reloaded = json.loads(out_path.read_text(encoding="utf-8")) if completed.returncode == 0 else {}
    checks["reload_bitwise_exact"] = reloaded.get("digest") == in_process
    checks["reload_counts_preserved"] = reloaded.get(
        "units"
    ) == graph.units.n_units and reloaded.get("retired") == len(retired_ids)

    budget_ok = time.process_time() - started <= CAP_CPU
    passed = all(checks.values())
    verdict = (
        "lifecycle_ext_pass"
        if passed and budget_ok
        else "budget_stop" if not budget_ok else "lifecycle_breach"
    )
    payload = {
        "schema": "taiji-u2-22-7-0-lifecycle-ext-v1",
        "card": "22.7.0-pre lifecycle extension (cross-process / refs / state growth)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "checks": checks,
        "readings": {
            "in_process_digest": in_process,
            "reload_digest": reloaded.get("digest"),
            "retired_units": len(retired_ids),
            "dormant_units": len(dormant_ids),
            "units": graph.units.n_units,
        },
        "verdict": verdict,
        "scope": "structural lifecycle extension only; no task capability, no training",
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
    return 0 if verdict == "lifecycle_ext_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
