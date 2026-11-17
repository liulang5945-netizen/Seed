"""22.5.W.1 structural memory probe: does a fixed write/hold/read slot carry DMS?

Runs the same DMS dataset with MemoryRouteBGraph (fixed sample write, hold over
delay, read at query) and a diagnostic linear head on [m, present, m*present].
Normal should reach 0.95+, lesion/test-only should stay at chance. Zero graph
optimizer steps; no sealed reads.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import probe_u2_w_dms_20261117 as wprobe
from taiji import u2_unit_input_dms as dms
from taiji.u2_history import value_digest
from taiji.u2_neural_units import UnitGraphSpec
from taiji.u2_unit_memory_readout import MemoryRouteBGraph

REPORT = ROOT / "reports/taiji_u2_w1_memory_20261117.json"
CAP_CPU = 120.0
CAP_WALL = 240.0
GRAPH_SEEDS = (2026111701, 2026111702, 2026111703)
NORMAL_GATE = 0.95
LESION_MAX = 0.60
TEST_ONLY_MAX = 0.60
POSITIVE_MIN = 0.80
DERANGED_MAX = 0.50


def run_memory_episode(
    graph: MemoryRouteBGraph,
    episode: dict[str, Any],
    lesion: bool = False,
    test_only: bool = False,
    delay: int = dms.DELAY_STEPS,
) -> dict[str, Any]:
    gates = torch.ones(dms.COLUMNS, graph.spec.n_units, dtype=torch.float64)
    stages = dms.encode_episode(episode, delay=delay)
    if lesion:
        stages[0] = wprobe.neutral_stage(episode)
    if test_only:
        stages = stages[-2:]
    state = graph.initial_state()
    out: dict[str, Any] | None = None
    with torch.no_grad():
        for stage in stages:
            out = graph(stage, gates, state)
            state = out["next_state"]
    assert out is not None
    memory = state["m"].to(torch.float64)
    present = torch.zeros(4, dtype=torch.float64)
    for attribute in episode["test"]:
        if attribute is not None:
            present[attribute] = 1.0
    features = torch.cat([memory, present, memory * present])
    sample = torch.zeros(4, dtype=torch.float64)
    sample[episode["sample"]] = 1.0
    return {
        "features": features,
        "memory": memory,
        "memory_exact": (
            bool(torch.equal(memory, sample))
            if not (lesion or test_only)
            else bool(float(memory.abs().max()) == 0.0)
        ),
    }


def features(
    graph: MemoryRouteBGraph,
    dataset: dict[str, list[dict[str, Any]]],
    lesion: bool,
    test_only: bool,
    delay: int = dms.DELAY_STEPS,
) -> dict[str, Any]:
    rows: dict[str, list[torch.Tensor]] = {"train": [], "fit": []}
    labels: dict[str, list[int]] = {"train": [], "fit": []}
    exact = 0
    for split in ("train", "fit"):
        for episode in dataset[split]:
            result = run_memory_episode(graph, episode, lesion, test_only, delay)
            rows[split].append(result["features"])
            labels[split].append(dms.label(episode))
            exact += int(result["memory_exact"])
    return {
        "train_x": torch.stack(rows["train"]),
        "fit_x": torch.stack(rows["fit"]),
        "train_y": labels["train"],
        "fit_y": labels["fit"],
        "memory_exact": exact,
        "memory_denominator": len(rows["train"]) + len(rows["fit"]),
    }


def main() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    dataset = wprobe.make_dataset()
    total = sum(len(items) for items in dataset.values())
    oracle = sum(
        int(dms.label(item) == int(any(a == item["sample"] for a in item["test"])))
        for split in dataset
        for item in dataset[split]
    )
    seeds: list[dict[str, Any]] = []
    budget_stopped = False
    for seed in GRAPH_SEEDS:
        if time.process_time() - started_cpu > CAP_CPU:
            budget_stopped = True
            break
        graph = MemoryRouteBGraph(UnitGraphSpec(dtype="float64", seed=seed))
        before = value_digest(
            {name: tensor.detach().clone() for name, tensor in graph.state_dict().items()}
        )
        normal = features(graph, dataset, False, False)
        lesion = features(graph, dataset, True, False)
        test_only = features(graph, dataset, False, True)
        normal_head = wprobe.fit_head(
            normal["train_x"], normal["train_y"], normal["fit_x"], normal["fit_y"]
        )
        lesion_head = wprobe.fit_head(
            lesion["train_x"], lesion["train_y"], lesion["fit_x"], lesion["fit_y"]
        )
        test_head = wprobe.fit_head(
            test_only["train_x"], test_only["train_y"], test_only["fit_x"], test_only["fit_y"]
        )
        positive = wprobe.positive_control(dataset)
        deranged = wprobe.fit_head(
            normal["train_x"], list(reversed(normal["train_y"])), normal["fit_x"], normal["fit_y"]
        )
        sweep = []
        for delay in (0, 1, 2, 4, 8):
            normal_d = features(graph, dataset, False, False, delay)
            lesion_d = features(graph, dataset, True, False, delay)
            sweep.append(
                {
                    "delay": delay,
                    "normal": wprobe.fit_head(
                        normal_d["train_x"],
                        normal_d["train_y"],
                        normal_d["fit_x"],
                        normal_d["fit_y"],
                    ),
                    "lesion": wprobe.fit_head(
                        lesion_d["train_x"],
                        lesion_d["train_y"],
                        lesion_d["fit_x"],
                        lesion_d["fit_y"],
                    ),
                }
            )
        after = value_digest(
            {name: tensor.detach().clone() for name, tensor in graph.state_dict().items()}
        )
        checks = {
            "encoding_oracle": oracle == total,
            "partition_disjoint": not (
                {(item["sample"], item["position"]) for item in dataset["train"]}
                & {(item["sample"], item["position"]) for item in dataset["fit"]}
            ),
            "params_unchanged": before == after,
            "positive_control_reaches_gate": positive["fit_accuracy"] >= POSITIVE_MIN,
            "deranged_below_band": deranged["fit_accuracy"] <= DERANGED_MAX,
            "normal_memory_exact": normal["memory_exact"] == normal["memory_denominator"],
            "lesion_memory_zero": lesion["memory_exact"] == lesion["memory_denominator"],
            "normal_fit_reaches_gate": normal_head["fit_accuracy"] >= NORMAL_GATE,
            "lesion_fit_below_band": lesion_head["fit_accuracy"] <= LESION_MAX,
            "test_only_fit_below_band": test_head["fit_accuracy"] <= TEST_ONLY_MAX,
        }
        seeds.append(
            {
                "seed": seed,
                "normal": normal_head,
                "lesion": lesion_head,
                "test_only": test_head,
                "positive_control": positive,
                "deranged": deranged,
                "delay_sweep": sweep,
                "checks": checks,
                "uncovered": [key for key, value in checks.items() if not value],
            }
        )
    uncovered = [f"seed{row['seed']}:{key}" for row in seeds for key in row["uncovered"]]
    viable = [
        row["seed"]
        for row in seeds
        if row["normal"]["fit_accuracy"] >= NORMAL_GATE
        and row["lesion"]["fit_accuracy"] <= LESION_MAX
        and row["test_only"]["fit_accuracy"] <= TEST_ONLY_MAX
    ]
    failed = [row["seed"] for row in seeds if row["normal"]["fit_accuracy"] < 0.60]
    if budget_stopped:
        verdict = "budget_stop"
    elif uncovered:
        verdict = "instrument_invalid"
    elif len(viable) == len(seeds):
        verdict = "structural_memory_viable"
    elif len(failed) >= 2:
        verdict = "memory_path_failed"
    else:
        verdict = "mixed"
    report = {
        "schema": "taiji-u2-w1-memory-v1",
        "card": "22.5.W.1",
        "phase": "structural_memory_zero_training",
        "verdict": verdict,
        "encoding_contract": dms.encoding_contract(),
        "data_digest": wprobe.dataset_digest(dataset),
        "dataset_sizes": {split: len(items) for split, items in dataset.items()},
        "seeds": seeds,
        "caps": {"cap_cpu": CAP_CPU, "cap_wall": CAP_WALL},
        "discipline": {
            "optimizer_steps": 0,
            "weight_writes": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "sealed_partitions_read": [],
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        "MEMORY verdict="
        + verdict
        + " normal="
        + str([round(row["normal"]["fit_accuracy"], 4) for row in seeds])
        + " lesion="
        + str([round(row["lesion"]["fit_accuracy"], 4) for row in seeds])
        + " test_only="
        + str([round(row["test_only"]["fit_accuracy"], 4) for row in seeds])
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0 if (not uncovered and not budget_stopped) else 1


if __name__ == "__main__":
    sys.exit(main())
