"""22.7b follow-up: birth-neutrality engineering diagnostic (zero check).

Root cause found post-hoc (D-20261119-42 attribution refined): the 22.7b training
loop never called ``graph.units.project_parameters_()``, so c/eta/b/w drifted out
of the parameterized space during training while the forward pass kept using them;
``with_neutral_birth`` copies then projects the child, clamping the copied old
parameters and therefore changing the old readout.  With per-step projection the
birth is expected to be bitwise neutral.

This probe measures, on the joint24 adapter-v3 task: (a) constraint violations
after unprojected vs projected training, (b) birth logit difference and argmax
flips for both, 3 seeds.  Pre-registered birth-neutrality gate for future growth
cards: max |delta logit| == 0 and 0 argmax flips on identical inputs.
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
from scripts.training.run_s2_22_7b_growth_joint24_20261120 import (  # noqa: E402
    BASE,
    channels,
    learning_rate,
    make_graph,
    predict,
    split,
)
from taiji.u2_neural_units import C_MAX, ETA_MAX, ETA_MIN, W_MAX  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_7b_birth_neutrality_20261120.json"
CAP_CPU = 80.0
STEPS = 500
SEEDS = (0, 1, 2)


def train(seed: int, data: tuple, projected: bool):
    graph = make_graph(seed, 32)
    optimizer = torch.optim.Adam(graph.parameters(), lr=BASE["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(20261120 + seed)
    indices = torch.arange(BASE["train_worlds"])
    x_train, y_train = data[0], data[1]
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = indices[torch.randperm(BASE["train_worlds"], generator=generator)[: BASE["batch"]]]
        gates = torch.ones(BASE["batch"], graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(x_train[batch]), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, y_train[batch])
        loss.backward()
        optimizer.step()
        if projected:
            graph.units.project_parameters_()
    return graph


def violations(graph) -> dict[str, float]:
    units = graph.units
    return {
        "c_above_max": float((units.c - C_MAX).clamp_min(0.0).max()),
        "c_below_min": float((0.0 - units.c).clamp_min(0.0).max()),
        "eta_above_max": float((units.eta - ETA_MAX).clamp_min(0.0).max()),
        "eta_below_min": float((ETA_MIN - units.eta).clamp_min(0.0).max()),
        "b_above_max": float((units.b - 1.0).clamp_min(0.0).max()),
        "b_below_zero": float((0.0 - units.b).clamp_min(0.0).max()),
        "w_unmasked": float(((units.w * (1.0 - units.edge_mask)).abs()).max()),
        "w_above_max": float((units.w.abs() - W_MAX).clamp_min(0.0).max()),
    }


def birth_effect(graph, x_fit: torch.Tensor) -> dict[str, float]:
    before = predict(graph, x_fit)[0]
    first = graph.units.identities["units"][0]["unit_id"]
    grown = graph.units.with_neutral_birth(0, [first])
    graph.units = grown
    after = predict(graph, x_fit)[0]
    return {
        "max_abs_logit_diff": float((before - after).abs().max()),
        "argmax_flips": int((before.argmax(-1) != after.argmax(-1)).sum()),
        "items": int(x_fit.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = split()
    x_fit = data[2]
    results = []
    for seed in SEEDS:
        unprojected = train(seed, data, projected=False)
        projected = train(seed, data, projected=True)
        results.append(
            {
                "seed": seed,
                "unprojected": {
                    "violations": violations(unprojected),
                    "birth": birth_effect(unprojected, x_fit),
                },
                "projected": {
                    "violations": violations(projected),
                    "birth": birth_effect(projected, x_fit),
                },
            }
        )
    neutral = all(
        entry["projected"]["birth"]["max_abs_logit_diff"] == 0.0
        and entry["projected"]["birth"]["argmax_flips"] == 0
        for entry in results
    )
    unprojected_nonneutral = all(
        entry["unprojected"]["birth"]["max_abs_logit_diff"] > 0.0 for entry in results
    )
    verdict = (
        "birth_neutral_with_projection"
        if neutral and unprojected_nonneutral
        else "birth_neutrality_unresolved"
    )
    payload = {
        "schema": "taiji-u2-22-7b-birth-neutrality-v1",
        "card": "22.7b follow-up birth-neutrality engineering diagnostic",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "task": "joint24 adapter-v3",
            "steps": STEPS,
            "seeds": list(SEEDS),
            "projection_rule": "project_parameters_() after every optimizer step",
            "gate": "max |delta logit| == 0 and 0 argmax flips on identical inputs",
        },
        "results": results,
        "verdict": verdict,
        "root_cause": (
            "22.7b's training loop omitted project_parameters_(); c/eta/b/w drifted "
            "out of the parameterized space and with_neutral_birth's child projection "
            "clamped copied old parameters, changing the old readout"
        ),
        "discipline": {
            "optimizer_steps": STEPS * len(SEEDS) * 2,
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "cpu": payload["cost"]["cpu"],
                "projected_birth": [entry["projected"]["birth"] for entry in results],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "birth_neutral_with_projection" else 1


if __name__ == "__main__":
    raise SystemExit(main())
