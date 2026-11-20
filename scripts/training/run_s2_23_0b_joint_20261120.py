"""23.0b: jointly trained three-skill parent on one RouteB subject (0 check).

Follow-up to 23.0 (which showed catastrophic forgetting under sequential training).
Here the three skills (appearance joint24, dynamics final bin, occlusion non-degenerate
position) are trained jointly with interleaved batches, so the parent carries all three
abilities at the end.  This card does NOT claim sequential retention; it only establishes
that a three-skill parent exists for the 23 main window.  Caps 600 CPU, zero check/final.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from scripts.training.probe_s2_action_consequence_20261120 import build_partition  # noqa: E402
from taiji.u2_multihead import MultiHeadGraph  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_23_0b_joint_20261120.json"
CAP_CPU = 600.0
SEEDS = (20261120, 20261121, 20261122)
OUTPUT = 24
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
STEPS = {"s1": 1152, "s2": 2000, "s3": 1500}
BATCH = 32
LR = 3e-3
WARMUP = 50
TARGET = 0.80


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def build_graph(seed: int) -> MultiHeadGraph:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=OUTPUT, seed=seed
    )
    return MultiHeadGraph(spec, heads=3)


def head_logits(graph: MultiHeadGraph, features: torch.Tensor, head: int) -> torch.Tensor:
    gates = torch.ones(features.shape[0], graph.units.n_units)
    return graph(channels(features), gates, graph.initial_state(), head=head)["readout"]


def build_data(seed: int) -> dict[str, torch.Tensor]:
    train = generate_examples("train", 512)
    fit = generate_examples("fit", 128)
    x1 = torch.stack([adapter_v3_features(item) for item in train]).float()
    y1 = torch.tensor([joint_label(item) for item in train])
    x1f = torch.stack([adapter_v3_features(item) for item in fit]).float()
    y1f = torch.tensor([joint_label(item) for item in fit])
    action_train = build_partition("train", 512, seed)
    action_fit = build_partition("fit", 128, seed + 1)
    x2 = torch.stack([row["features"] for row in action_train]).float()
    y2 = torch.tensor([row["final_bin"] for row in action_train])
    x2f = torch.stack([row["features"] for row in action_fit]).float()
    y2f = torch.tensor([row["final_bin"] for row in action_fit])

    def occlude(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        result = features.clone()
        for index in range(result.shape[0]):
            colour = int(labels[index]) // 4
            masked = (colour + 1) % 6
            result[index, masked * 3 : (masked + 1) * 3] = 0.0
        return result

    return {
        "x1": x1,
        "y1": y1,
        "x1f": x1f,
        "y1f": y1f,
        "x2": x2,
        "y2": y2,
        "x2f": x2f,
        "y2f": y2f,
        "x3": occlude(x1, y1),
        "y3": y1 % 4,
        "x3f": occlude(x1f, y1f),
        "y3f": y1f % 4,
    }


def accuracy(
    graph: MultiHeadGraph, features: torch.Tensor, labels: torch.Tensor, head: int
) -> float:
    with torch.no_grad():
        logits = head_logits(graph, features, head)
    k = int(labels.max()) + 1
    return float((logits[:, :k].argmax(-1) == labels).float().mean())


def run_seed(seed: int) -> dict:
    data = build_data(seed)
    graph = build_graph(seed)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 7)
    schedule = ["s1"] * STEPS["s1"] + ["s2"] * STEPS["s2"] + ["s3"] * STEPS["s3"]
    shuffle = torch.randperm(len(schedule), generator=generator).tolist()
    total = len(schedule)
    features_by_skill = {"s1": data["x1"], "s2": data["x2"], "s3": data["x3"]}
    labels_by_skill = {"s1": data["y1"], "s2": data["y2"], "s3": data["y3"]}
    head_by_skill = {"s1": 0, "s2": 1, "s3": 2}
    for step, position in enumerate(shuffle):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, total)
        skill = schedule[position]
        features = features_by_skill[skill]
        labels = labels_by_skill[skill]
        head = head_by_skill[skill]
        index = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        logits = head_logits(graph, features[index], head)
        k = int(labels.max()) + 1
        loss = loss_fn(logits[:, :k], labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()
    s1 = accuracy(graph, data["x1f"], data["y1f"], 0)
    s2 = accuracy(graph, data["x2f"], data["y2f"], 1)
    s3 = accuracy(graph, data["x3f"], data["y3f"], 2)
    capsule = build_capsule(graph)
    with tempfile.TemporaryDirectory(dir=ROOT / ".seed_test_tmp") as directory:
        path = save_capsule(Path(directory) / "joint.json", capsule)
        loaded = load_capsule(path)
    fresh = build_graph(seed)
    apply_capsule(fresh, loaded)
    digest_a = {key: value.sum().item() for key, value in graph.state_dict().items()}
    digest_b = {key: value.sum().item() for key, value in fresh.state_dict().items()}
    recovery_diff = max(abs(digest_a[key] - digest_b[key]) for key in digest_a)
    return {"seed": seed, "s1": s1, "s2": s2, "s3": s3, "recovery_diff": recovery_diff}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    (ROOT / ".seed_test_tmp").mkdir(exist_ok=True)
    rows = [run_seed(seed) for seed in SEEDS]
    k1 = all(min(row["s1"], row["s2"], row["s3"]) >= TARGET for row in rows)
    k3 = all(row["recovery_diff"] <= 1e-10 for row in rows)
    gates = {"k1_three_skills": k1, "k3_identity": k3}
    verdict = "joint_parent_ready" if k1 and k3 else "joint_parent_absent"
    payload = {
        "schema": "taiji-u2-23-0b-joint-v1",
        "card": "23.0b jointly trained three-skill parent",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "batch": BATCH,
            "target": TARGET,
            "joint": True,
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "optimizer_steps": len(SEEDS) * sum(STEPS.values()),
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "rows": rows, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "joint_parent_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
