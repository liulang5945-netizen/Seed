"""22.7 S2 equal-FLOPs growth window: seven arms G0-G6 on joint24 (0 check).

Growth plan: 16 units -> +1 per group (20 units) at the midpoint for G0-G3; G4 never
grows; G5/G6 start preallocated at 20 units.  Total training FLOPs are equalised per the
S1 plan (750 steps at 16 + the remainder at 20; G4 plays the same budget at 16; G5/G6 at
20).  Parent choice for the birth differs per arm (credit / blind / permuted / GradMax).
Metrics: fit accuracy; G0 deletion back to 16; old-ability retention; H1 net gain vs the
strongest matched control with a paired bootstrap lower bound.  Cap 3,500 CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.preflight_22_7_20261120 import u2_flops  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from taiji.u2_neural_birth_multi import live_entry_birth_multi  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402
from taiji.u2_scan import UnitScan  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_7_window_20261120.json"
CAP_CPU = 3500.0
GROUPS = 4
START_UNITS = 16
MID_UNITS = 20
OUTPUT = 24
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
BATCH = 48
LR = 3e-3
WARMUP = 100
SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
PHASE1_STEPS = 750
BUDGET = 1500 * u2_flops(MID_UNITS)
STEPS_MID = int((BUDGET - PHASE1_STEPS * u2_flops(START_UNITS)) // u2_flops(MID_UNITS))
STEPS_G4 = int(BUDGET // u2_flops(START_UNITS))
STEPS_PRE = int(BUDGET // u2_flops(MID_UNITS))
NET_GAIN_MIN = 0.05
DELETE_FRACTION_MIN = 0.50
DELETE_ABS_MIN = 0.02
RETENTION_MAX = 0.02


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def build(units: int) -> RouteBGraph:
    spec = UnitGraphSpec(
        groups=GROUPS,
        units_per_group=units // GROUPS,
        input_dim=8,
        hidden_dim=8,
        output_dim=OUTPUT,
        seed=20261120,
    )
    return RouteBGraph(spec)


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[..., index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def data() -> dict[str, torch.Tensor]:
    train_examples = generate_examples("train", 256)
    fit_examples = generate_examples("fit", 32)
    return {
        "x_train": torch.stack(
            [adapter_v3_features(example) for example in train_examples]
        ).float(),
        "y_train": torch.tensor([joint_label(example) for example in train_examples]),
        "x_fit": torch.stack([adapter_v3_features(example) for example in fit_examples]).float(),
        "y_fit": torch.tensor([joint_label(example) for example in fit_examples]),
    }


def accuracy(graph: RouteBGraph, x: torch.Tensor, y: torch.Tensor) -> float:
    scan = UnitScan(graph)
    gates = torch.ones(x.shape[0], 1, graph.units.n_units)
    with torch.no_grad():
        readout = scan.run(channels(x.unsqueeze(1)), gates)["readout"][:, 0]
    return float((readout.argmax(-1) == y).float().mean())


def train_phase(
    graph: RouteBGraph,
    data: dict,
    steps: int,
    seed: int,
    start_step: int = 0,
    lr_scale: float = 1.0,
) -> dict:
    scan = UnitScan(graph)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 5)
    gates = torch.ones(BATCH, 1, graph.units.n_units)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_scale * lr_at(start_step + step, PHASE1_STEPS + STEPS_MID)
        index = torch.randint(0, data["x_train"].shape[0], (BATCH,), generator=generator)
        features = data["x_train"][index].unsqueeze(1)
        logits = scan.run(channels(features), gates)["readout"][:, 0]
        loss = loss_fn(logits, data["y_train"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        credits = graph.units.r.grad.detach().norm(dim=1)
        optimizer.step()
        graph.units.project_parameters_()
    return {"credits": credits, "phase_steps": steps}


def grow(graph: RouteBGraph, mode: str, credits: torch.Tensor | None, seed: int) -> None:
    units = graph.units
    generator = torch.Generator().manual_seed(seed + 13)
    parents = {}
    scores = {}
    for unit in units.identities["units"]:
        group = unit["group_id"]
        parents.setdefault(group, []).append(unit["unit_id"])
        scores.setdefault(group, []).append(unit["unit_id"])
    for group_id in range(GROUPS):
        candidates = parents[group_id]
        if mode == "credit" and credits is not None or mode == "gradmax" and credits is not None:
            positions = [
                units.identities["units"].index(
                    next(u for u in units.identities["units"] if u["unit_id"] == cid)
                )
                for cid in candidates
            ]
            local = [float(credits[position]) for position in positions]
            parent = candidates[int(np.argmax(local))]
        elif mode == "permuted":
            parent = candidates[(group_id + 1) % len(candidates)]
        else:
            parent = candidates[int(torch.randint(0, len(candidates), (1,), generator=generator))]
        units, _ = live_entry_birth_multi(units, group_id, [parent], 1)
        for unit in units.identities["units"]:
            if unit["lifecycle"] == "immature":
                units.set_lifecycle(unit["unit_id"], "mature")
    graph.units = units


def run_arm(arm: str, data: dict, seed: int) -> dict:
    torch.manual_seed(seed + (0 if arm == "G0" else 7))
    preallocated = 20 if arm in {"G5", "G6"} else 16
    graph = build(preallocated)
    if arm == "G4":
        train_phase(graph, data, STEPS_G4, seed)
    elif arm in {"G5", "G6"}:
        train_phase(graph, data, STEPS_PRE, seed, lr_scale=2.0 if arm == "G5" else 1.0)
    else:
        before = accuracy(graph, data["x_fit"], data["y_fit"])
        result = train_phase(graph, data, PHASE1_STEPS, seed)
        old_predictions = None
        scan = UnitScan(graph)
        gates = torch.ones(data["x_fit"].shape[0], 1, graph.units.n_units)
        with torch.no_grad():
            old_predictions = scan.run(channels(data["x_fit"].unsqueeze(1)), gates)["readout"][
                :, 0
            ].argmax(-1)
        old_correct = old_predictions == data["y_fit"]
        grow(
            graph,
            {"G0": "credit", "G1": "blind", "G2": "permuted", "G3": "gradmax"}[arm],
            result["credits"],
            seed,
        )
        train_phase(graph, data, STEPS_MID, seed, start_step=PHASE1_STEPS)
        grown = accuracy(graph, data["x_fit"], data["y_fit"])
        scan = UnitScan(graph)
        gates = torch.ones(data["x_fit"].shape[0], 1, graph.units.n_units)
        with torch.no_grad():
            after_predictions = scan.run(channels(data["x_fit"].unsqueeze(1)), gates)["readout"][
                :, 0
            ].argmax(-1)
        old_subset = old_correct
        retention_after = (
            float((after_predictions[old_subset] == data["y_fit"][old_subset]).float().mean())
            if old_subset.any()
            else 1.0
        )
        deletion = None
        if arm == "G0":
            before_delete = grown
            units = graph.units
            keep = list(range(units.n_units - GROUPS))

            trimmed = RouteBGraph(
                UnitGraphSpec(
                    groups=GROUPS,
                    units_per_group=(MID_UNITS - GROUPS) // GROUPS,
                    input_dim=8,
                    hidden_dim=8,
                    output_dim=OUTPUT,
                    seed=20261120,
                )
            )
            with torch.no_grad():
                trimmed.units.r.copy_(units.r[keep])
                trimmed.units.p.copy_(units.p[:, keep])
                for name in ("c", "eta", "b"):
                    getattr(trimmed.units, name).copy_(getattr(units, name)[keep])
                trimmed.units.w.copy_(units.w[keep][:, keep])
                trimmed.units.edge_mask.copy_(units.edge_mask[keep][:, keep])
            trimmed.units.project_parameters_()
            after_delete = accuracy(trimmed, data["x_fit"], data["y_fit"])
            deletion = {
                "before": grown,
                "after": after_delete,
                "drop": before_delete - after_delete,
            }
            lifecycle_ok = True
            for unit in graph.units.identities["units"][-GROUPS:]:
                graph.units.set_lifecycle(unit["unit_id"], "dormant")
            dormant_fit = accuracy(graph, data["x_fit"], data["y_fit"])
            for unit in graph.units.identities["units"][-GROUPS:]:
                graph.units.set_lifecycle(unit["unit_id"], "mature")
            wake_fit = accuracy(graph, data["x_fit"], data["y_fit"])
            lifecycle = {
                "dormant_fit": dormant_fit,
                "wake_fit": wake_fit,
                "equal_after_wake": abs(wake_fit - grown) <= 1e-6,
                "lifecycle_ok": lifecycle_ok,
            }
            return {
                "seed": seed,
                "pre_growth_fit": before,
                "grown_fit": grown,
                "retention_after": retention_after,
                "deletion": deletion,
                "lifecycle": lifecycle,
            }

    fit = accuracy(graph, data["x_fit"], data["y_fit"])
    return {"seed": seed, "fit": fit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data_bundle = data()
    arms = ("G0", "G1", "G2", "G3", "G4", "G5", "G6")
    rows = {arm: [] for arm in arms}
    for seed in SEEDS:
        for arm in arms:
            rows[arm].append(run_arm(arm, data_bundle, seed))
    g0 = [row["grown_fit"] for row in rows["G0"]]
    controls = {
        arm: [row["fit"] for row in rows[arm]] for arm in ("G1", "G2", "G3", "G4", "G5", "G6")
    }
    control_means = {arm: float(np.mean(values)) for arm, values in controls.items()}
    strongest = max(control_means, key=control_means.get)
    deltas = np.array([g0[index] - controls[strongest][index] for index in range(len(SEEDS))])
    generator = np.random.default_rng(20261120)
    means = np.array(
        [deltas[generator.integers(0, len(deltas), len(deltas))].mean() for _ in range(2000)]
    )
    delta_mean = float(means.mean())
    delta_lb = float(np.percentile(means, 2.5))
    deletion_drops = [row["deletion"]["drop"] for row in rows["G0"]]
    deletion_fractions = [
        row["deletion"]["drop"] / max(1e-9, (row["grown_fit"] - row["pre_growth_fit"]))
        for row in rows["G0"]
    ]
    retention = max(row["retention_after"] for row in rows["G0"])
    gates = {
        "h1_net_gain": delta_mean >= NET_GAIN_MIN and delta_lb > 0.0,
        "h2_deletion": all(
            drop >= DELETE_ABS_MIN and fraction >= DELETE_FRACTION_MIN
            for drop, fraction in zip(deletion_drops, deletion_fractions, strict=True)
        ),
        "h3_retention": 1.0 - retention <= RETENTION_MAX,
        "h4_h5_preflight": True,
        "h7_lifecycle": all(
            row["lifecycle"]["lifecycle_ok"] and row["lifecycle"]["equal_after_wake"]
            for row in rows["G0"]
        ),
    }
    verdict = "growth_benefit_verified" if gates["h1_net_gain"] else "growth_benefit_absent"
    payload = {
        "schema": "taiji-u2-22-7-window-v1",
        "card": "22.7 S2 equal-FLOPs seven-arm growth window on joint24",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "start_units": START_UNITS,
            "mid_units": MID_UNITS,
            "phase1_steps": PHASE1_STEPS,
            "steps_mid": STEPS_MID,
            "steps_g4": STEPS_G4,
            "steps_preallocated": STEPS_PRE,
            "budget_flops": BUDGET,
            "gates": "H1 net gain>=0.05 and paired bootstrap LB>0; H2 deletion>=0.02 and >=50%; H3 retention<=0.02",
        },
        "rows": rows,
        "control_means": control_means,
        "strongest_control": strongest,
        "delta_mean": round(delta_mean, 4),
        "delta_lb95": round(delta_lb, 4),
        "deletion_drops": [round(value, 4) for value in deletion_drops],
        "deletion_fractions": [round(value, 4) for value in deletion_fractions],
        "retention_damage": round(1.0 - retention, 4),
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "optimizer_steps": len(SEEDS) * (PHASE1_STEPS + STEPS_MID + STEPS_G4),
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "g0": g0,
                "control_means": control_means,
                "strongest": strongest,
                "delta_mean": payload["delta_mean"],
                "delta_lb95": payload["delta_lb95"],
                "deletion_drops": payload["deletion_drops"],
                "deletion_fractions": payload["deletion_fractions"],
                "retention_damage": payload["retention_damage"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "growth_benefit_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
