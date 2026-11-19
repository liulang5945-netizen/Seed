"""One session of the multi-session continuity probe (run in its own process).

Session 1 starts from a fresh unit graph; later sessions load the previous
capsule into a FRESH graph instance (cross-process parameter inheritance),
continue training on the same joint24 task, then save the updated capsule and a
receipt with the parameter/identity digests plus the fit accuracy.
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
from taiji.u2_neural_units import (  # noqa: E402
    RouteBGraph,
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

STEPS = 400
BATCH = 64
LR = 3e-3
WARMUP = 50
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
TRAIN_WORLDS = 256
FIT_WORLDS = 32


def learning_rate(step: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def digest(tensor_by_name: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(tensor_by_name):
        hasher.update(name.encode())
        hasher.update(tensor_by_name[name].detach().cpu().to(torch.float64).numpy().tobytes())
    return hasher.hexdigest()


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

    train_examples = generate_examples("train", TRAIN_WORLDS)
    fit_examples = generate_examples("fit", FIT_WORLDS)
    train_features = torch.stack([adapter_v3_features(ex) for ex in train_examples]).float()
    train_labels = torch.tensor([joint_label(ex) for ex in train_examples])
    fit_features = torch.stack([adapter_v3_features(ex) for ex in fit_examples]).float()
    fit_labels = torch.tensor([joint_label(ex) for ex in fit_examples])

    torch.manual_seed(20261120 + args.seed)
    spec = UnitGraphSpec(
        groups=6,
        units_per_group=5,
        input_dim=8,
        hidden_dim=8,
        output_dim=24,
        seed=20261120 + args.seed,
    )
    graph = RouteBGraph(spec)
    load_digest = ""
    if args.in_capsule:
        capsule = load_capsule(args.in_capsule)
        load_digest = digest(
            {name: value for name, value in capsule["runtime"]["state_dict"].items()}
        )
        apply_capsule(graph, capsule)

    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(20261120 + args.seed * 10 + args.session)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = torch.randint(0, train_features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(train_features[batch]), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, train_labels[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()

    with torch.no_grad():
        gates = torch.ones(fit_features.shape[0], graph.units.n_units)
        logits = graph(channels(fit_features), gates, graph.initial_state())["readout"]
    accuracy = float((logits.argmax(-1) == fit_labels).float().mean())
    parameters = {name: value.detach().clone() for name, value in graph.state_dict().items()}
    save_digest = digest(parameters)
    capsule = build_capsule(graph)
    save_capsule(Path(args.out_capsule), capsule)
    units = graph.units.identities["units"]
    receipt = {
        "session": args.session,
        "seed": args.seed,
        "accuracy": round(accuracy, 4),
        "load_digest": load_digest,
        "save_digest": save_digest,
        "units": len(units),
        "unit_ids": [unit.get("unit_id") for unit in units],
        "groups": len({unit.get("group_id") for unit in units}),
        "n_units_graph": graph.units.n_units,
        "identity_digest": hashlib.sha256(
            json.dumps(capsule.get("identity", {}), sort_keys=True).encode()
        ).hexdigest(),
        "cpu": round(time.process_time() - started, 2),
    }
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "session": args.session,
                "seed": args.seed,
                "accuracy": receipt["accuracy"],
                "save_digest": save_digest[:12],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
