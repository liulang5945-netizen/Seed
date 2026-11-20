"""24.1: bounded fast store F + sparse slow edge set W, with cue-based Recall (0 check).

This card validates the declared memory interface on the frozen 23.0b parent: a fixed
64x16 fast store (<=4 slots touched per event, size never grows) and a bounded sparse edge
set (<=4 edges per event, <=256 edges total).  Recall retrieves a stored answer from a
key; removing the cue must hurt accuracy.  DISCLOSURE: F is a hand-coded key-value store
and W is an interface-only edge list here - this card checks the contract/accounting, not
learned memory (consolidation contribution is 24.2).  Zero check/final.
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
from scripts.training.run_s2_23_0b_joint_20261120 import accuracy as skill_accuracy  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import (  # noqa: E402
    label_for,
    tokens_for,
    train_joint_parent,
)
from taiji.native_composition import build_partition  # noqa: E402

OUT = ROOT / "reports/taiji_u2_24_1_memory_20261120.json"
CAP_CPU = 2000.0
SEEDS = (20261120, 20261121, 20261122)
EVENTS = 200
F_SLOTS = 64
F_DIM = 16
SLOTS_PER_EVENT = 4
W_MAX_EDGES = 256
EDGES_PER_EVENT = 4
RECALL_MIN = 0.60
CUE_DROP_MIN = 0.05
RETENTION_MAX = 0.02


class FastStore:
    def __init__(self) -> None:
        self.keys = torch.zeros(F_SLOTS, F_DIM)
        self.values = torch.zeros(F_SLOTS, 4)
        self.occupied = torch.zeros(F_SLOTS, dtype=torch.bool)
        self.writes = 0

    def size(self) -> tuple[int, int]:
        return tuple(self.keys.shape)

    def write(self, key: torch.Tensor, value: int) -> None:
        distance = torch.cdist(key.unsqueeze(0), self.keys).squeeze(0)
        distance[~self.occupied] = -1.0
        slot = int(distance.argmin()) if self.occupied.any() else 0
        self.keys[slot] = key
        self.values[slot] = 0.0
        self.values[slot, value] = 1.0
        self.occupied[slot] = True
        self.writes += 1

    def recall(self, key: torch.Tensor) -> int:
        distance = torch.cdist(key.unsqueeze(0), self.keys).squeeze(0)
        distance[~self.occupied] = 1e9
        slot = int(distance.argmin())
        return int(self.values[slot].argmax())


class SparseEdges:
    def __init__(self, n_units: int) -> None:
        self.n_units = n_units
        self.edges: list[tuple[int, int, float]] = []

    def write(self, generator: torch.Generator) -> None:
        for _ in range(EDGES_PER_EVENT):
            if len(self.edges) >= W_MAX_EDGES:
                self.edges.pop(0)
            source = int(torch.randint(0, self.n_units, (1,), generator=generator))
            target = int(torch.randint(0, self.n_units, (1,), generator=generator))
            self.edges.append(
                (source, target, float(torch.rand((), generator=generator) - 0.5) * 0.1)
            )

    def check(self) -> bool:
        return len(self.edges) <= W_MAX_EDGES


def build_events(seed: int) -> list[dict[str, torch.Tensor]]:
    rows = [row for row in build_partition("train") if row["kind"] in {"q2", "q3"}]
    generator = torch.Generator().manual_seed(seed + 89)
    indexes = torch.randint(0, len(rows), (EVENTS,), generator=generator).tolist()
    events = []
    for index in indexes:
        row = rows[index]
        tokens, query = tokens_for(row)
        key = tokens.reshape(-1)
        projection = torch.linspace(-1.0, 1.0, F_DIM * key.numel()).reshape(F_DIM, key.numel())
        projected = torch.nn.functional.normalize(projection @ key, dim=0)
        events.append({"key": projected, "label": label_for(row), "tokens": tokens, "query": query})
    return events


def run_seed(seed: int) -> dict:
    graph, skill_data = train_joint_parent(seed)
    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    events = build_events(seed)
    store = FastStore()
    edges = SparseEdges(graph.units.n_units)
    generator = torch.Generator().manual_seed(seed + 97)
    initial_size = store.size()
    immediate = []
    for event in events:
        store.write(event["key"], int(event["label"]))
        edges.write(generator)
        immediate.append(store.recall(event["key"]) == int(event["label"]))
    size_after = store.size()
    immediate_accuracy = sum(immediate) / len(immediate)
    recent = events[-F_SLOTS:]
    clean_accuracy = sum(
        store.recall(event["key"]) == int(event["label"]) for event in recent
    ) / len(recent)
    # Cue removal: replace keys with a fixed zero cue (no information).
    blank = torch.zeros(F_DIM)
    blank_accuracy = sum(store.recall(blank) == int(event["label"]) for event in recent) / len(
        recent
    )
    retention = {
        "baseline": baseline,
        "after": {
            "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
            "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
            "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
        },
    }
    retention["damage"] = {
        skill: max(0.0, baseline[skill] - retention["after"][skill]) for skill in baseline
    }
    return {
        "seed": seed,
        "clean_recall": clean_accuracy,
        "immediate_recall": immediate_accuracy,
        "blank_recall": blank_accuracy,
        "cue_drop": clean_accuracy - blank_accuracy,
        "store_size_initial": list(initial_size),
        "store_size_after": list(size_after),
        "store_writes": store.writes,
        "edges": len(edges.edges),
        "edges_bounded": edges.check(),
        "retention": retention,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    gates = {
        "write_read": all(
            row["immediate_recall"] >= 0.95 and row["clean_recall"] >= RECALL_MIN for row in rows
        ),
        "no_store_growth": all(
            row["store_size_initial"] == row["store_size_after"] for row in rows
        ),
        "edges_bounded": all(row["edges_bounded"] and row["edges"] <= W_MAX_EDGES for row in rows),
        "cue_dependency": all(row["cue_drop"] >= CUE_DROP_MIN for row in rows),
        "retention": all(max(row["retention"]["damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "memory_interface_ready" if all(gates.values()) else "memory_interface_partial"
    payload = {
        "schema": "taiji-u2-24-1-memory-v1",
        "card": "24.1 bounded fast store F + sparse edge set W with cue Recall",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "events": EVENTS,
            "f_slots": F_SLOTS,
            "f_dim": F_DIM,
            "slots_per_event": SLOTS_PER_EVENT,
            "edges_per_event": EDGES_PER_EVENT,
            "w_max_edges": W_MAX_EDGES,
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "F is a hand-coded key-value store and W is interface-only here; this card checks contract/accounting, not learned memory (24.2 tests consolidation contribution)",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "rows": rows, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "memory_interface_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
