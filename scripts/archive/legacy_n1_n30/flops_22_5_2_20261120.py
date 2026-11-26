"""22.5.2 S1: analytic FLOPs estimator, equal-FLOPs parity search and smoke (0 check).

Arms: U2 Route-A (H embedding), U2 Route-B (direct unit query), and a GRU baseline.
The estimator counts multiply-adds per sample per tick for the forward pass and
multiplies by 3 (forward + backward at 2x) for training FLOPs; every term is listed
in the report so it can be recomputed by hand.  The GRU hidden width and step count
are chosen so the total training FLOPs ratio against Route-B falls in [0.90, 1.10];
measured CPU is recorded for accounting only, never used as the parity criterion.

Cap: 300 CPU.  Zero check/final reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from taiji.u2_neural_units import RouteAGraph, RouteBGraph, UnitGraphSpec  # noqa: E402
from taiji.u2_scan import UnitScan  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_2_parity_20261120.json"
CAP_CPU = 300.0
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
TICKS = 3
BATCH = 24
SMOKE_STEPS = 50
TARGET_STEPS = 1200
HIDDEN_GRID = (16, 24, 32, 48)
PARITY_BAND = (0.90, 1.10)
SEED = 20261120
OUTPUT = 24


def u2_breakdown(route: str) -> dict[str, int]:
    groups, d, h, out = 6, 8, 8, OUTPUT
    units = groups * 5
    per_tick: dict[str, int] = {
        "input_encode (6 channels x 8x8 + projection 8x8)": 6 * (d * d) + 6 * (d * d),
        "unit_drive x*r (N x D)": units * d,
        "unit_incoming (N x N)": units * units,
    }
    once: dict[str, int] = {}
    if route == "A":
        per_tick["unit_contribution p (H x N)"] = h * units
        per_tick["w_g drive (G x 8 x 8)"] = groups * d * d
        per_tick["H update (G x 8)"] = groups * h
        once["H mean over ticks"] = TICKS * groups * h
        once["h_readout (H x out)"] = h * out
    else:
        per_tick["p projection (H x N)"] = h * units
        per_tick["query (q x H)"] = 4 * h
        per_tick["readout (q x out)"] = 4 * out
    return {"per_tick": per_tick, "per_sample": once}


def u2_flops_per_sample(route: str) -> int:
    breakdown = u2_breakdown(route)
    forward = TICKS * sum(breakdown["per_tick"].values()) + sum(breakdown["per_sample"].values())
    return 3 * forward


def gru_breakdown(hidden: int) -> dict[str, int]:
    input_dim = 48
    return {
        "gate matmuls 3 x hidden x (input+hidden)": 3 * hidden * (input_dim + hidden),
        "gate elementwise (3 x hidden)": 3 * hidden,
        "output head hidden x out": hidden * OUTPUT,
    }


def gru_flops_per_sample(hidden: int, steps: int) -> int:
    per_tick = 3 * (hidden * (48 + hidden) + hidden)
    forward = TICKS * per_tick + hidden * OUTPUT
    return 3 * forward * steps


def gru_forward_flops(hidden: int) -> int:
    per_tick = 3 * (hidden * (48 + hidden) + hidden)
    return TICKS * per_tick + hidden * OUTPUT


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[..., index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


class GruArm(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(48, hidden, batch_first=True)
        self.head = nn.Linear(hidden, OUTPUT)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(features)
        return self.head(output[:, -1])


def build_u2(route: str) -> tuple[Any, UnitScan]:
    torch.manual_seed(SEED)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=OUTPUT, seed=SEED
    )
    graph = RouteAGraph(spec) if route == "A" else RouteBGraph(spec)
    return graph, UnitScan(graph)


def run_route_a(
    graph: RouteAGraph,
    features: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """Per-sample Route-A forward over [B,T,8] channel tensors (H on the readout path)."""

    units = graph.units
    batch = features.shape[0]
    channels_t = channels(features.unsqueeze(1)) if features.dim() == 2 else channels(features)
    z = torch.zeros(batch, units.n_units)
    a = torch.zeros(batch, units.n_units)
    hidden = torch.zeros(batch, graph.spec.groups, graph.spec.hidden_dim)
    readouts = []
    for tick in range(features.shape[1]):
        tick_channels = {name: channels_t[name][:, tick] for name in CHANNELS}
        inputs = torch.stack(
            [
                graph.input_encoders[name](tick_channels[name]) @ graph.input_projection
                for name in CHANNELS
            ],
            dim=1,
        )
        x = inputs[:, units.group_index, :]
        drive = (x * units.r.unsqueeze(0)).sum(-1)
        gate = gates[:, tick]
        incoming = (gate * z) @ (units.w * units.edge_mask).T
        v = drive + incoming - units.b.unsqueeze(0) * a
        z = units.c.unsqueeze(0) * z + (1.0 - units.c.unsqueeze(0)) * torch.tanh(v)
        a = (1.0 - units.eta.unsqueeze(0)) * a + units.eta.unsqueeze(0) * z
        activity = gate * z
        contribution = torch.einsum("hn,gn,bn->bgh", units.p, units.group_mask, activity)
        drive_g = torch.einsum("ghd,bgd->bgh", graph.w_g, inputs) + contribution
        hidden = (1.0 - graph.ell.view(1, -1, 1)) * hidden + graph.ell.view(1, -1, 1) * torch.tanh(
            drive_g
        )
        readouts.append(graph.h_readout(hidden.mean(dim=1)))
    return torch.stack(readouts, dim=1)


def smoke(route: str) -> dict[str, Any]:
    graph, scan = build_u2(route)
    optimizer = torch.optim.Adam(graph.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED + 7)
    torch.manual_seed(SEED + 3)
    features = torch.rand(BATCH, TICKS, 48, generator=generator)
    labels = torch.randint(0, OUTPUT, (BATCH,), generator=generator)
    gates = torch.ones(BATCH, TICKS, graph.units.n_units)
    started = time.process_time()
    last_loss = 0.0
    for _ in range(SMOKE_STEPS):
        if route == "A":
            logits = run_route_a(graph, features, gates)[:, -1]
        else:
            logits = scan.run(channels(features), gates)["readout"][:, -1]
        loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()
        last_loss = float(loss.detach())
    cpu = time.process_time() - started
    params = sum(parameter.numel() for parameter in graph.parameters())
    return {
        "route": f"U2-{route}",
        "params": params,
        "smoke_steps": SMOKE_STEPS,
        "last_loss": round(last_loss, 4),
        "cpu": round(cpu, 2),
    }


def smoke_gru(hidden: int) -> dict[str, Any]:
    torch.manual_seed(SEED + 11)
    model = GruArm(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED + 7)
    features = torch.rand(BATCH, TICKS, 48, generator=generator)
    labels = torch.randint(0, OUTPUT, (BATCH,), generator=generator)
    started = time.process_time()
    last_loss = 0.0
    for _ in range(SMOKE_STEPS):
        logits = model(features)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    cpu = time.process_time() - started
    params = sum(parameter.numel() for parameter in model.parameters())
    return {
        "route": f"GRU-{hidden}",
        "params": params,
        "smoke_steps": SMOKE_STEPS,
        "last_loss": round(last_loss, 4),
        "cpu": round(cpu, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    breakdown_a = u2_breakdown("A")
    breakdown_b = u2_breakdown("B")
    per_sample_a = u2_flops_per_sample("A")
    per_sample_b = u2_flops_per_sample("B")
    total_u2_b = per_sample_b * TARGET_STEPS
    candidates = []
    for hidden in HIDDEN_GRID:
        per_step = gru_flops_per_sample(hidden, 1)
        steps = max(200, round(total_u2_b / per_step))
        total = per_step * steps
        ratio = total / total_u2_b
        candidates.append(
            {
                "hidden": hidden,
                "steps": steps,
                "flops_per_sample_per_step": per_step,
                "train_flops_total": total,
                "ratio_vs_B": round(ratio, 4),
                "in_band": PARITY_BAND[0] <= ratio <= PARITY_BAND[1],
            }
        )
    in_band = [candidate for candidate in candidates if candidate["in_band"]]
    if not in_band:
        verdict = "comparison_not_admitted"
        chosen = min(candidates, key=lambda item: abs(item["ratio_vs_B"] - 1.0))
    else:
        chosen = min(in_band, key=lambda item: abs(item["ratio_vs_B"] - 1.0))
        verdict = "parity_admissible"
    arms = [smoke("A"), smoke("B"), smoke_gru(chosen["hidden"])]
    payload = {
        "schema": "taiji-u2-22-5-2-parity-v1",
        "card": "22.5.2 S1 analytic FLOPs parity search and smoke",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "ticks": TICKS,
            "batch": BATCH,
            "steps_reference": TARGET_STEPS,
            "parity_band": list(PARITY_BAND),
            "criterion": "analytic training FLOPs ratio in band; measured CPU for accounting only",
            "backward_factor": 3,
        },
        "flops": {
            "u2_a_per_sample_training": per_sample_a,
            "u2_b_per_sample_training": per_sample_b,
            "u2_a_breakdown": breakdown_a,
            "u2_b_breakdown": breakdown_b,
            "gru_breakdown": {hidden: gru_breakdown(hidden) for hidden in HIDDEN_GRID},
            "u2_b_total_training": total_u2_b,
        },
        "candidates": candidates,
        "chosen": chosen,
        "smoke": arms,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0, "optimizer_steps": SMOKE_STEPS * 3},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "chosen": chosen,
                "candidates": candidates,
                "smoke": arms,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "parity_admissible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
