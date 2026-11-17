"""22.5.W zero-training feasibility probe: delayed match-to-sample on unit-core B.

Builds a deterministic DMS dataset (train/fit split by attribute x position), runs
the birth unit-core B graph over the sample -> delay -> test -> query sequence,
and asks whether a converged linear head on the graph's own z/a/direct features
can decode same/different on unseen combinations. Lesion and test-only controls
check that the sample, not the test row, carries the label.

Zero optimizer steps on the graph; the linear heads are diagnostic. No sealed reads.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from taiji import u2_unit_input_dms as dms
from taiji.u2_history import value_digest
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec

REPORT = ROOT / "reports/taiji_u2_w_dms_feasibility_20261117.json"
CAP_CPU = 120.0
CAP_WALL = 240.0
ATTRIBUTES = dms.ATTRIBUTES
POSITIONS = 8
TRAIN_COMBOS = 24
FIT_COMBOS = 8
GRAPH_SEEDS = (2026111701, 2026111702, 2026111703)
FIT_GATE = 0.80
LESION_MAX = 0.60
TEST_ONLY_MAX = 0.60
DERANGED_MAX = 0.50
LBFGS_ITER = 300


def make_episode(rng: random.Random, sample: int, position: int, same: bool) -> dict[str, Any]:
    test: list[int | None] = [None] * POSITIONS
    if same:
        test[position] = sample
    else:
        test[position] = (sample + 1 + rng.randrange(ATTRIBUTES - 1)) % ATTRIBUTES
        if test[position] == sample:
            test[position] = (sample + 1) % ATTRIBUTES
    for slot in range(POSITIONS):
        if slot == position:
            continue
        roll = rng.random()
        if roll < 0.45:
            continue
        attribute = rng.randrange(ATTRIBUTES)
        if attribute == sample:
            attribute = (attribute + 1) % ATTRIBUTES
        test[slot] = attribute
    if any(attribute == sample for attribute in test) != same:
        raise ValueError("episode_label_inconsistent")
    return {"sample": sample, "position": position, "test": test}


def make_dataset() -> dict[str, list[dict[str, Any]]]:
    combos = [
        (attribute, position) for attribute in range(ATTRIBUTES) for position in range(POSITIONS)
    ]
    rng = random.Random(20261117)
    rng.shuffle(combos)
    train_combos = combos[:TRAIN_COMBOS]
    fit_combos = combos[TRAIN_COMBOS : TRAIN_COMBOS + FIT_COMBOS]
    if set(train_combos) & set(fit_combos):
        raise ValueError("partition_overlap")
    train: list[dict[str, Any]] = []
    fit: list[dict[str, Any]] = []
    for index, (attribute, position) in enumerate(train_combos):
        train.append(make_episode(rng, attribute, position, True))
        train.append(make_episode(rng, attribute, position, False))
        if index < 8:
            train.append(make_episode(rng, attribute, position, True))
            train.append(make_episode(rng, attribute, position, False))
    for attribute, position in fit_combos:
        for same in (True, False, True, False):
            fit.append(make_episode(rng, attribute, position, same))
    return {"train": train, "fit": fit}


def episode_digest(episode: dict[str, Any]) -> str:
    return value_digest(
        {"sample": episode["sample"], "position": episode["position"], "test": episode["test"]}
    )


def dataset_digest(dataset: dict[str, list[dict[str, Any]]]) -> str:
    return value_digest(
        {split: [episode_digest(item) for item in items] for split, items in dataset.items()}
    )


def neutral_stage(episode: dict[str, Any]) -> dict[str, torch.Tensor]:
    return dms.encode_stage("delay", None, episode["test"])


def run_episode(
    graph: RouteBGraph,
    episode: dict[str, Any],
    lesion: bool = False,
    test_only: bool = False,
    delay: int = dms.DELAY_STEPS,
) -> torch.Tensor:
    gates = torch.ones(dms.COLUMNS, graph.spec.n_units, dtype=torch.float64)
    stages = dms.encode_episode(episode, delay=delay)
    if lesion:
        stages[0] = neutral_stage(episode)
    if test_only:
        stages = stages[-2:]
    state = graph.initial_state()
    out: dict[str, Any] | None = None
    with torch.no_grad():
        for stage in stages:
            out = graph(stage, gates, state)
            state = out["next_state"]
    assert out is not None
    direct = out["direct_query"].mean(dim=0)
    return torch.cat([state["z"], state["a"], direct]).to(torch.float64)


def fit_head(
    train_x: torch.Tensor, train_y: list[int], fit_x: torch.Tensor, fit_y: list[int]
) -> dict[str, Any]:
    weights = torch.zeros(train_x.shape[1], 2, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(train_y, dtype=torch.long)
    optimizer = torch.optim.LBFGS(
        [weights],
        max_iter=LBFGS_ITER,
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(train_x @ weights, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        train_pred = (train_x @ weights).argmax(-1)
        fit_pred = (fit_x @ weights).argmax(-1)
    return {
        "train_accuracy": float((train_pred == target).to(torch.float64).mean()),
        "fit_accuracy": float((fit_pred == torch.tensor(fit_y)).to(torch.float64).mean()),
        "fit_distinct": len(set(fit_pred.tolist())),
    }


def features(
    graph: RouteBGraph,
    dataset: dict[str, list[dict[str, Any]]],
    lesion: bool,
    test_only: bool,
    delay: int = dms.DELAY_STEPS,
) -> dict[str, Any]:
    rows: dict[str, list[torch.Tensor]] = {"train": [], "fit": []}
    labels: dict[str, list[int]] = {"train": [], "fit": []}
    for split in ("train", "fit"):
        for episode in dataset[split]:
            rows[split].append(run_episode(graph, episode, lesion, test_only, delay))
            labels[split].append(dms.label(episode))
    return {
        "train_x": torch.stack(rows["train"]),
        "fit_x": torch.stack(rows["fit"]),
        "train_y": labels["train"],
        "fit_y": labels["fit"],
    }


def positive_control(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows: dict[str, list[torch.Tensor]] = {"train": [], "fit": []}
    labels: dict[str, list[int]] = {"train": [], "fit": []}
    for split in ("train", "fit"):
        for episode in dataset[split]:
            sample = torch.zeros(ATTRIBUTES, dtype=torch.float64)
            sample[episode["sample"]] = 1.0
            present = torch.zeros(ATTRIBUTES, dtype=torch.float64)
            for attribute in episode["test"]:
                if attribute is not None:
                    present[attribute] = 1.0
            rows[split].append(torch.cat([sample, present, sample * present]))
            labels[split].append(dms.label(episode))
    return fit_head(
        torch.stack(rows["train"]), labels["train"], torch.stack(rows["fit"]), labels["fit"]
    )


def main() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    dataset = make_dataset()
    oracle = sum(
        int(dms.label(item) == int(any(a == item["sample"] for a in item["test"])))
        for split in dataset
        for item in dataset[split]
    )
    total = sum(len(items) for items in dataset.values())
    partition = {
        "train_combinations": sorted(
            {(item["sample"], item["position"]) for item in dataset["train"]}
        ),
        "fit_combinations": sorted({(item["sample"], item["position"]) for item in dataset["fit"]}),
        "disjoint": not (
            {(item["sample"], item["position"]) for item in dataset["train"]}
            & {(item["sample"], item["position"]) for item in dataset["fit"]}
        ),
        "train_labels": {
            str(label): sum(dms.label(item) == label for item in dataset["train"])
            for label in (0, 1)
        },
        "fit_labels": {
            str(label): sum(dms.label(item) == label for item in dataset["fit"]) for label in (0, 1)
        },
    }
    seeds: list[dict[str, Any]] = []
    budget_stopped = False
    for seed in GRAPH_SEEDS:
        if time.process_time() - started_cpu > CAP_CPU:
            budget_stopped = True
            break
        graph = RouteBGraph(UnitGraphSpec(dtype="float64", seed=seed))
        before = value_digest(
            {name: tensor.detach().clone() for name, tensor in graph.state_dict().items()}
        )
        normal = features(graph, dataset, lesion=False, test_only=False)
        lesion = features(graph, dataset, lesion=True, test_only=False)
        test_only = features(graph, dataset, lesion=False, test_only=True)
        normal_head = fit_head(
            normal["train_x"], normal["train_y"], normal["fit_x"], normal["fit_y"]
        )
        lesion_head = fit_head(
            lesion["train_x"], lesion["train_y"], lesion["fit_x"], lesion["fit_y"]
        )
        test_head = fit_head(
            test_only["train_x"], test_only["train_y"], test_only["fit_x"], test_only["fit_y"]
        )
        positive = positive_control(dataset)
        y = list(normal["train_y"])
        deranged = list(reversed(y))
        deranged_head = fit_head(normal["train_x"], deranged, normal["fit_x"], normal["fit_y"])
        sweep = []
        for delay in (0, 1, 2, 4, 8):
            normal_d = features(graph, dataset, False, False, delay)
            lesion_d = features(graph, dataset, True, False, delay)
            test_d = features(graph, dataset, False, True, delay)
            sweep.append(
                {
                    "delay": delay,
                    "normal": fit_head(
                        normal_d["train_x"],
                        normal_d["train_y"],
                        normal_d["fit_x"],
                        normal_d["fit_y"],
                    ),
                    "lesion": fit_head(
                        lesion_d["train_x"],
                        lesion_d["train_y"],
                        lesion_d["fit_x"],
                        lesion_d["fit_y"],
                    ),
                    "test_only": fit_head(
                        test_d["train_x"], test_d["train_y"], test_d["fit_x"], test_d["fit_y"]
                    ),
                }
            )
        after = value_digest(
            {name: tensor.detach().clone() for name, tensor in graph.state_dict().items()}
        )
        checks = {
            "encoding_oracle": oracle == total,
            "partition_disjoint": partition["disjoint"],
            "params_unchanged": before == after,
            "positive_control_reaches_gate": positive["fit_accuracy"] >= FIT_GATE,
            "deranged_below_band": deranged_head["fit_accuracy"] <= DERANGED_MAX,
            "label_balance_train": partition["train_labels"]["0"] == partition["train_labels"]["1"],
            "label_balance_fit": partition["fit_labels"]["0"] == partition["fit_labels"]["1"],
        }
        normal_ok = normal_head["fit_accuracy"] >= FIT_GATE
        lesion_ok = lesion_head["fit_accuracy"] <= LESION_MAX
        test_ok = test_head["fit_accuracy"] <= TEST_ONLY_MAX
        seeds.append(
            {
                "seed": seed,
                "normal": normal_head,
                "lesion": lesion_head,
                "test_only": test_head,
                "positive_control": positive,
                "deranged": deranged_head,
                "delay_sweep": sweep,
                "normal_ok": normal_ok,
                "lesion_ok": lesion_ok,
                "test_only_ok": test_ok,
                "checks": checks,
                "uncovered": [key for key, value in checks.items() if not value],
            }
        )
    uncovered = [f"seed{row['seed']}:{key}" for row in seeds for key in row["uncovered"]]
    learnable = [
        row["seed"]
        for row in seeds
        if row["normal_ok"] and row["lesion_ok"] and row["test_only_ok"]
    ]
    not_learnable = [row["seed"] for row in seeds if row["normal"]["fit_accuracy"] < 0.60]
    if budget_stopped:
        verdict = "budget_stop"
    elif uncovered:
        verdict = "instrument_invalid"
    elif len(learnable) >= 2:
        verdict = "parent_candidate_learnable"
    elif len(not_learnable) >= 2:
        verdict = "task_not_learnable"
    else:
        verdict = "mixed"
    report = {
        "schema": "taiji-u2-w-dms-feasibility-v1",
        "card": "22.5.W",
        "phase": "feasibility_zero_training",
        "verdict": verdict,
        "encoding_contract": dms.encoding_contract(),
        "data_digest": dataset_digest(dataset),
        "partition": partition,
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
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        "DMS verdict="
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
