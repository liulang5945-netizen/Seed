"""Cross-session per-task-pathway (option 2) probe: old abilities exact, new task learnable."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (
    adapter_v3_features,  # noqa: E402
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from scripts.training.probe_s2_action_consequence_20261120 import build_partition  # noqa: E402
from taiji.u2_multihead import freeze_module  # noqa: E402
from taiji.u2_multihead2 import MultiHeadGraph2, task_parameters  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_multihead7_20261120.json"
WORKER = ROOT / "scripts/training/s2_multihead7_worker_20261120.py"
OUTPUT = ROOT / "output/s2_multihead7"
CAP_CPU = 200.0
SEEDS = (0, 1, 2)
SESSIONS = (1, 2, 3, 4, 5, 6, 7, 8)
STEPS = 700
BASE_STEPS = 1200
BATCH = 64
LR = 3e-3
WARMUP = 100
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
OLD_WORLDS = 192
OLD_FIT = 32
NEW_WORLDS = 256
NEW_FIT = 32
HEADS = 8
HEAD_HIDDEN = 16
RETENTION_MAX = 0.02
THIRD_TASK_MIN = 0.60
FOURTH_TASK_MIN = 0.45
FIFTH_TASK_MIN = 0.55
SIXTH_TASK_MIN = 0.45
SEVENTH_TASK_MIN = 0.60
EIGHTH_TASK_MIN = 0.50


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def digest(tensors: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(tensors):
        hasher.update(name.encode())
        hasher.update(tensors[name].detach().cpu().to(torch.float64).numpy().tobytes())
    return hasher.hexdigest()


def prediction(graph, features, head, classes):
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state(), head=head)["readout"]
    return logits[:, :classes]


def accuracy(graph, features, labels, head, classes):
    return float((prediction(graph, features, head, classes).argmax(-1) == labels).float().mean())


def data_bundle():
    old_examples = generate_examples("train", OLD_WORLDS)
    old_fit_examples = generate_examples("fit", OLD_FIT)
    rows = build_partition("train", NEW_WORLDS + NEW_FIT, 20261120)
    old_train = torch.stack([adapter_v3_features(ex) for ex in old_examples]).float()
    old_labels = torch.tensor([joint_label(ex) for ex in old_examples])
    old_fit = torch.stack([adapter_v3_features(ex) for ex in old_fit_examples]).float()
    old_fit_labels = torch.tensor([joint_label(ex) for ex in old_fit_examples])
    new_train = torch.stack([row["features"] for row in rows[:NEW_WORLDS]])
    new_labels = torch.tensor([row["final_bin"] for row in rows[:NEW_WORLDS]])
    new_fit = torch.stack([row["features"] for row in rows[NEW_WORLDS:]])
    new_fit_labels = torch.tensor([row["final_bin"] for row in rows[NEW_WORLDS:]])
    direction_train = (new_train[:, 37] > 0).long()
    direction_fit = (new_fit[:, 37] > 0).long()
    return {
        "old_train": (old_train, old_labels),
        "old_fit": (old_fit, old_fit_labels),
        "new_train": (new_train, new_labels),
        "new_fit": (new_fit, new_fit_labels),
        "direction_train": (new_train, direction_train),
        "direction_fit": (new_fit, direction_fit),
        "position_train": (new_train, (new_labels % 4).clone()),
        "position_fit": (new_fit, (new_fit_labels % 4).clone()),
        "color_train": (old_train, (old_labels // 4).clone()),
        "color_fit": (old_fit, (old_fit_labels // 4).clone()),
        "quadrant_train": (
            old_train,
            (((old_labels // 4) >= 3).long() * 2 + ((old_labels % 4) >= 2).long()),
        ),
        "quadrant_fit": (
            old_fit,
            (((old_fit_labels // 4) >= 3).long() * 2 + ((old_fit_labels % 4) >= 2).long()),
        ),
        "cross_train": (
            old_train,
            (((old_labels // 4) >= 3) & ((old_labels % 4) >= 2)).long(),
        ),
        "cross_fit": (
            old_fit,
            (((old_fit_labels // 4) >= 3) & ((old_fit_labels % 4) >= 2)).long(),
        ),
        "colormod_train": (old_train, ((old_labels // 4) % 3).clone()),
        "colormod_fit": (old_fit, ((old_fit_labels // 4) % 3).clone()),
    }


def train_head(graph: MultiHeadGraph2, head: int, features, labels, steps, seed):
    freeze_module(graph.units)
    parameters = [
        parameter for parameter in task_parameters(graph, head) if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    classes = int(labels.max().item()) + 1
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        batch = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        logits = graph(channels(features[batch]), gates, graph.initial_state(), head=head)[
            "readout"
        ]
        loss = loss_fn(logits[:, :classes], labels[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def build_graph(seed: int) -> MultiHeadGraph2:
    torch.manual_seed(20261120 + seed)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=24, seed=20261120 + seed
    )
    return MultiHeadGraph2(spec, heads=HEADS, head_hidden=HEAD_HIDDEN)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--in-capsule", default="")
    parser.add_argument("--out-capsule", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = data_bundle()

    if args.in_capsule:
        capsule = load_capsule(args.in_capsule)
        load_digest = digest(capsule["runtime"]["state_dict"])
        graph = build_graph(args.seed)
        apply_capsule(graph, capsule)
    else:
        graph = build_graph(args.seed)
        load_digest = ""

    if args.session == 1:
        optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
        loss_fn = torch.nn.CrossEntropyLoss()
        features, labels = data["old_train"]
        generator = torch.Generator().manual_seed(args.seed)
        for step in range(BASE_STEPS):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, BASE_STEPS)
            batch = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
            gates = torch.ones(BATCH, graph.units.n_units)
            optimizer.zero_grad(set_to_none=True)
            logits = graph(channels(features[batch]), gates, graph.initial_state(), head=0)[
                "readout"
            ]
            loss = loss_fn(logits, labels[batch])
            loss.backward()
            optimizer.step()
            graph.units.project_parameters_()
    elif args.session == 2:
        freeze_module(graph.input_encoders)
        train_head(graph, 1, *data["new_train"], STEPS, args.seed + 10)
    elif args.session == 3:
        freeze_module(graph.input_encoders)
        train_head(graph, 2, *data["direction_train"], STEPS, args.seed + 20)
    elif args.session == 4:
        freeze_module(graph.input_encoders)
        train_head(graph, 3, *data["position_train"], STEPS, args.seed + 30)
    elif args.session == 5:
        freeze_module(graph.input_encoders)
        train_head(graph, 4, *data["color_train"], STEPS, args.seed + 40)
    elif args.session == 6:
        freeze_module(graph.input_encoders)
        train_head(graph, 5, *data["quadrant_train"], STEPS, args.seed + 50)
    elif args.session == 7:
        freeze_module(graph.input_encoders)
        train_head(graph, 6, *data["cross_train"], STEPS, args.seed + 60)
    else:
        freeze_module(graph.input_encoders)
        train_head(graph, 7, *data["colormod_train"], STEPS, args.seed + 70)

    schedule = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 2],
        4: [0, 1, 2, 3],
        5: [0, 1, 2, 3, 4],
        6: [0, 1, 2, 3, 4, 5],
        7: [0, 1, 2, 3, 4, 5, 6],
        8: [0, 1, 2, 3, 4, 5, 6, 7],
    }[args.session]
    accuracies = {
        "old": (
            round(accuracy(graph, *data["old_fit"], head=0, classes=24), 4)
            if 0 in schedule
            else None
        ),
        "new": (
            round(accuracy(graph, *data["new_fit"], head=1, classes=3), 4)
            if 1 in schedule
            else None
        ),
        "direction": (
            round(accuracy(graph, *data["direction_fit"], head=2, classes=2), 4)
            if 2 in schedule
            else None
        ),
        "position": (
            round(accuracy(graph, *data["position_fit"], head=3, classes=4), 4)
            if 3 in schedule
            else None
        ),
        "color": (
            round(accuracy(graph, *data["color_fit"], head=4, classes=6), 4)
            if 4 in schedule
            else None
        ),
        "quadrant": (
            round(accuracy(graph, *data["quadrant_fit"], head=5, classes=4), 4)
            if 5 in schedule
            else None
        ),
        "cross": (
            round(accuracy(graph, *data["cross_fit"], head=6, classes=2), 4)
            if 6 in schedule
            else None
        ),
        "colormod": (
            round(accuracy(graph, *data["colormod_fit"], head=7, classes=3), 4)
            if 7 in schedule
            else None
        ),
    }
    save_digest = digest(graph.state_dict())
    save_capsule(Path(args.out_capsule), build_capsule(graph))
    receipt = {
        "session": args.session,
        "seed": args.seed,
        "load_digest": load_digest,
        "save_digest": save_digest,
        "accuracies": accuracies,
        "units": graph.units.n_units,
        "cpu": round(time.process_time() - started, 2),
    }
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"session": args.session, "accuracies": accuracies}))
    return 0


def coordinator() -> int:
    started = time.process_time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    child_cpu = 0.0
    for seed in SEEDS:
        receipts = []
        previous = ""
        for session in SESSIONS:
            capsule_path = OUTPUT / f"seed{seed}_session{session}.pt"
            receipt_path = OUTPUT / f"seed{seed}_session{session}.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--seed",
                    str(seed),
                    "--session",
                    str(session),
                    "--in-capsule",
                    previous,
                    "--out-capsule",
                    str(capsule_path),
                    "--receipt",
                    str(receipt_path),
                    "--child",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr[-500:])
            receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
            previous = str(capsule_path)
        child_cpu += sum(receipt["cpu"] for receipt in receipts)
        receipts_head = receipts[:-1]
        first, second, third, fourth, fifth, sixth, seventh = receipts_head
        eighth = receipts[-1]
        inheritance = all(
            receipts[index]["load_digest"] == receipts[index - 1]["save_digest"]
            for index in (1, 2, 3, 4, 5, 6, 7)
        )
        results.append(
            {
                "seed": seed,
                "final": eighth["accuracies"],
                "old_retention_drop": round(
                    first["accuracies"]["old"] - fourth["accuracies"]["old"], 6
                ),
                "new_retention_drop": round(
                    second["accuracies"]["new"] - fourth["accuracies"]["new"], 6
                ),
                "direction_retention_drop": round(
                    third["accuracies"]["direction"] - fifth["accuracies"]["direction"], 6
                ),
                "position_retention_drop": round(
                    fourth["accuracies"]["position"] - sixth["accuracies"]["position"], 6
                ),
                "color_retention_drop": round(
                    fifth["accuracies"]["color"] - seventh["accuracies"]["color"], 6
                ),
                "quadrant_retention_drop": round(
                    sixth["accuracies"]["quadrant"] - eighth["accuracies"]["quadrant"], 6
                ),
                "cross_retention_drop": round(
                    seventh["accuracies"]["cross"] - eighth["accuracies"]["cross"], 6
                ),
                "inheritance_exact": inheritance,
            }
        )
    gates = {
        "inheritance_exact": all(result["inheritance_exact"] for result in results),
        "old_retention": all(result["old_retention_drop"] <= RETENTION_MAX for result in results),
        "new_retention": all(result["new_retention_drop"] <= RETENTION_MAX for result in results),
        "third_task_learnable": all(
            result["final"]["direction"] >= THIRD_TASK_MIN for result in results
        ),
        "fourth_task_learnable": all(
            result["final"]["position"] >= FOURTH_TASK_MIN for result in results
        ),
        "fifth_task_learnable": all(
            result["final"]["color"] >= FIFTH_TASK_MIN for result in results
        ),
        "sixth_task_learnable": all(
            result["final"]["quadrant"] >= SIXTH_TASK_MIN for result in results
        ),
        "seventh_task_learnable": all(
            result["final"]["cross"] >= SEVENTH_TASK_MIN for result in results
        ),
        "eighth_task_learnable": all(
            result["final"]["colormod"] >= EIGHTH_TASK_MIN for result in results
        ),
    }
    gates["all_retentions_exact"] = all(
        max(
            result["old_retention_drop"],
            result["new_retention_drop"],
            result["direction_retention_drop"],
            result["position_retention_drop"],
            result["color_retention_drop"],
            result["quadrant_retention_drop"],
            result["cross_retention_drop"],
        )
        <= RETENTION_MAX
        for result in results
    )
    verdict = "multihead7_ready" if all(gates.values()) else "multihead7_absent"
    payload = {
        "schema": "taiji-u2-multihead7-v1",
        "card": "eight cross-session per-task pathways limit probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "sessions": len(SESSIONS),
            "heads": HEADS,
            "head_hidden": HEAD_HIDDEN,
            "steps": STEPS,
            "base_steps": BASE_STEPS,
            "retention_max": RETENTION_MAX,
            "third_task_min": THIRD_TASK_MIN,
            "fourth_task_min": FOURTH_TASK_MIN,
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (BASE_STEPS + STEPS * 2) * len(SEEDS),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started, 2),
            "children_cpu": round(child_cpu, 2),
        },
    }
    write_report(Path(OUT), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cost": payload["cost"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "multihead2_ready" else 1


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.argv = [argument for argument in sys.argv if argument != "--child"]
        raise SystemExit(main())
    raise SystemExit(coordinator())
