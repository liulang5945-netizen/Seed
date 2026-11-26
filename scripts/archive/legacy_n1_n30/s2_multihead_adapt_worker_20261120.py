"""One session of the limited-shared-adaptation multi-head probe (3 heads, 3 tasks).

Session 1 trains head A on joint24; sessions 2/3 freeze shared units and earlier
heads and train one new MLP head each (action bins, push direction).  Receipts
carry per-head accuracies and parameter digests for cross-session inheritance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (
    adapter_v3_features,  # noqa: E402
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from scripts.training.probe_s2_action_consequence_20261120 import build_partition  # noqa: E402
from taiji.u2_multihead import MultiHeadGraph, freeze_module  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

STEPS = 600
BASE_STEPS = 1200
BATCH = 64
LR = 3e-3
WARMUP = 100
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
OLD_WORLDS = 192
OLD_FIT = 32
NEW_WORLDS = 256
NEW_FIT = 32
HEADS = 3
HEAD_HIDDEN = 16


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


def train_head(graph, head, features, labels, steps, seed, freeze_units=True, allow_shared=False):
    if freeze_units and not allow_shared:
        freeze_module(graph.units)
    for index in range(HEADS):
        if index != head:
            freeze_module(graph.heads[index])
    if freeze_units and not allow_shared:
        parameters = [
            parameter for parameter in graph.heads[head].parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.Adam(parameters, lr=LR)
    elif allow_shared:
        head_parameters = [
            parameter for parameter in graph.heads[head].parameters() if parameter.requires_grad
        ]
        shared_parameters = [
            parameter
            for parameter in graph.parameters()
            if parameter.requires_grad and parameter not in set(head_parameters)
        ]
        optimizer = torch.optim.Adam(
            [
                {"params": head_parameters, "lr": LR},
                {"params": shared_parameters, "lr": LR * 0.1},
            ],
            lr=LR,
        )
    else:
        parameters = [parameter for parameter in graph.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        batch = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        logits = graph(channels(features[batch]), gates, graph.initial_state(), head=head)[
            "readout"
        ]
        loss = loss_fn(logits[:, : labels.max().item() + 1], labels[batch])
        loss.backward()
        optimizer.step()


def build_data():
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
    }


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
    data = build_data()

    if args.in_capsule:
        capsule = load_capsule(args.in_capsule)
        load_digest = digest(capsule["runtime"]["state_dict"])
        spec = UnitGraphSpec(
            groups=6,
            units_per_group=5,
            input_dim=8,
            hidden_dim=8,
            output_dim=24,
            seed=20261120 + args.seed,
        )
        graph = MultiHeadGraph(spec, heads=HEADS, head_hidden=HEAD_HIDDEN)
        apply_capsule(graph, capsule)
    else:
        torch.manual_seed(20261120 + args.seed)
        spec = UnitGraphSpec(
            groups=6,
            units_per_group=5,
            input_dim=8,
            hidden_dim=8,
            output_dim=24,
            seed=20261120 + args.seed,
        )
        graph = MultiHeadGraph(spec, heads=HEADS, head_hidden=HEAD_HIDDEN)
        load_digest = ""

    if args.session == 1:
        train_head(graph, 0, *data["old_train"], BASE_STEPS, args.seed, freeze_units=False)
    elif args.session == 2:
        train_head(graph, 1, *data["new_train"], STEPS, args.seed + 10, allow_shared=True)
    else:
        train_head(graph, 2, *data["direction_train"], STEPS, args.seed + 20, allow_shared=True)

    old_accuracy = accuracy(graph, *data["old_fit"], head=0, classes=24)
    new_accuracy = accuracy(graph, *data["new_fit"], head=1, classes=3)
    direction_accuracy = accuracy(graph, *data["direction_fit"], head=2, classes=2)
    schedule = {1: [0], 2: [0, 1], 3: [0, 1, 2]}[args.session]
    available = {
        "old": round(old_accuracy if 0 in schedule else float("nan"), 4),
        "new": round(new_accuracy if 1 in schedule else float("nan"), 4),
        "direction": round(direction_accuracy if 2 in schedule else float("nan"), 4),
    }
    save_digest = digest(graph.state_dict())
    save_capsule(Path(args.out_capsule), build_capsule(graph))
    receipt = {
        "session": args.session,
        "seed": args.seed,
        "load_digest": load_digest,
        "save_digest": save_digest,
        "accuracies": available,
        "heads": HEADS,
        "units": graph.units.n_units,
        "cpu": round(time.process_time() - started, 2),
    }
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"session": args.session, "accuracies": available}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
