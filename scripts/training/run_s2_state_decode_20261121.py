"""Per-sample sequential state probe: does z/a/H retain the t0 target across ticks? (0 check)

The unit graph's persistent state z/a/H is a single shared vector, so batched calls conflate
batch with time and cannot test per-sample retention.  This probe runs each sample as its own
single-step sequence (batch=1), records z/a/H/activity at every tick, and trains linear
probes to decode the t0 target position from each tick's state.  Shuffled-label control is
included.  Zero check/final.
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
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import channels  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402

OUT = ROOT / "reports/taiji_u2_state_decode_20261121.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
TRAIN = 96
FIT = 48
TICKS = 10
FIT_OFFSET = 10_000_000
DIM = 48
BODY = 3
BODY_STRIDE = 12
TICK_OFFSET = 36
TICK_SLOTS = 12
PROBE_STEPS = 400
LR = 1e-2


def episode(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target_colour = int(torch.randint(0, 6, (1,), generator=generator))
    target_position = int(torch.randint(0, 3, (1,), generator=generator))
    ticks = []
    for tick_index in range(TICKS):
        features = torch.zeros(DIM)
        for body in range(BODY):
            if tick_index == 0 and body == 0:
                colour, position = target_colour, target_position
            else:
                colour = int(torch.randint(0, 6, (1,), generator=generator))
                position = int(torch.randint(0, 3, (1,), generator=generator))
            base = body * BODY_STRIDE
            features[base + colour] = 1.0
            features[base + 6 + position] = 1.0
        features[TICK_OFFSET + (tick_index % TICK_SLOTS)] = 1.0
        ticks.append(features)
    return {"ticks": torch.stack(ticks), "label": target_position}


def build(count: int, seed: int) -> list[dict]:
    return [episode(seed + index) for index in range(count)]


def record_states(graph, rows: list[dict]) -> dict[str, torch.Tensor]:
    """Run every sample as a 1-step sequence and record per-tick state."""
    labels = []
    seq = {"z": [], "a": [], "H": [], "activity": []}
    for row in rows:
        state = graph.initial_state()
        z_list, a_list, h_list, act_list = [], [], [], []
        for tick_index in range(TICKS):
            gates = torch.ones(1, graph.units.n_units)
            with torch.no_grad():
                result = graph(channels(row["ticks"][tick_index : tick_index + 1]), gates, state)
            z_list.append(result["unit_result"]["z"].clone())
            a_list.append(result["unit_result"]["a"].clone())
            h_list.append(result["next_state"]["H"].reshape(-1).clone())
            act_list.append(result["unit_result"]["activity"].squeeze(0).clone())
            state = result["next_state"]
        seq["z"].append(torch.stack(z_list))
        seq["a"].append(torch.stack(a_list))
        seq["H"].append(torch.stack(h_list))
        seq["activity"].append(torch.stack(act_list))
        labels.append(row["label"])
    return {
        "z": torch.stack(seq["z"]),
        "a": torch.stack(seq["a"]),
        "H": torch.stack(seq["H"]),
        "activity": torch.stack(seq["activity"]),
        "labels": torch.tensor(labels),
    }


class Probe(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def train_probe(probe: nn.Module, features: torch.Tensor, labels: torch.Tensor, seed: int) -> None:
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(PROBE_STEPS):
        index = torch.randint(0, features.shape[0], (32,), generator=generator)
        loss = loss_fn(probe(features[index]), labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(probe: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        logits = probe(features)
    return float((logits.argmax(-1) == labels).float().mean())


def run_seed(seed: int) -> dict:
    graph, _ = train_joint_parent(seed)
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    train = record_states(graph, build(TRAIN, seed))
    fit = record_states(graph, build(FIT, seed + FIT_OFFSET))
    results = {}
    generator = torch.Generator().manual_seed(seed + 5)
    shuffled = train["labels"][torch.randperm(train["labels"].shape[0], generator=generator)]
    for variable in ("z", "a", "H", "activity"):
        per_tick = {}
        for tick_index in range(TICKS):
            probe = Probe(train[variable].shape[-1])
            train_probe(probe, train[variable][:, tick_index], train["labels"], seed)
            per_tick[str(tick_index)] = evaluate(probe, fit[variable][:, tick_index], fit["labels"])
        # shuffled-label control on the first tick only
        probe = Probe(train[variable].shape[-1])
        train_probe(probe, train[variable][:, 0], shuffled, seed)
        per_tick["shuffled_t0"] = evaluate(probe, fit[variable][:, 0], fit["labels"])
        results[variable] = per_tick
    return {"seed": seed, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {
        variable: {
            tick: sum(row["results"][variable][tick] for row in rows) / len(rows)
            for tick in rows[0]["results"][variable]
        }
        for variable in ("z", "a", "H", "activity")
    }
    payload = {
        "schema": "taiji-u2-state-decode-v1",
        "card": "per-sample sequential state decoding of the t0 target",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "train": TRAIN,
            "fit": FIT,
            "fit_offset": FIT_OFFSET,
            "probe_steps": PROBE_STEPS,
        },
        "rows": rows,
        "means": means,
        "verdict": "state_decode_diagnostic",
        "disclosure": "batch=1 per sample so z/a/H evolve per sample; chance = 1/3",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": payload["verdict"], "means": means, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
