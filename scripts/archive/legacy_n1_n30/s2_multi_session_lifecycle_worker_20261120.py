"""Cross-session lifecycle worker: birth -> dormant/retire in session 2 -> wake in 3.

Records capsule-persisted lifecycles, dormant-unit activity, the logits change
from waking a unit, parameter digests around wake, and retained accuracy.
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
from taiji.u2_neural_birth_multi import live_entry_birth_multi  # noqa: E402
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
BIRTH_SESSION = 2
WAKE_SESSION = 3
BIRTH_COUNT = 3


def learning_rate(step: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def digest(tensors: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(tensors):
        hasher.update(name.encode())
        hasher.update(tensors[name].detach().cpu().to(torch.float64).numpy().tobytes())
    return hasher.hexdigest()


def spec_from_capsule(capsule: dict) -> UnitGraphSpec:
    spec = capsule["runtime"]["spec"]
    return UnitGraphSpec(
        groups=int(spec["groups"]),
        units_per_group=int(spec["units_per_group"]),
        input_dim=int(spec["input_dim"]),
        hidden_dim=int(spec["hidden_dim"]),
        output_dim=int(spec["output_dim"]),
        seed=int(spec["seed"]),
        dtype=str(spec.get("dtype", "float32")),
    )


def logits(graph: RouteBGraph, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        return graph(channels(features), gates, graph.initial_state())["readout"]


def accuracy(graph: RouteBGraph, features: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits(graph, features).argmax(-1) == labels).float().mean())


def unit_activity(graph: RouteBGraph, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        return (
            graph(channels(features), gates, graph.initial_state())["unit_result"]["activity"]
            .abs()
            .amax(dim=0)
        )


def lifecycle_map(graph: RouteBGraph) -> dict[str, str]:
    return {unit["unit_id"]: unit["lifecycle"] for unit in graph.units.identities["units"]}


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

    load_digest = ""
    birth = None
    before_wake_digest = ""
    after_wake_digest = ""
    wake = None
    if args.in_capsule:
        capsule = load_capsule(args.in_capsule)
        spec = spec_from_capsule(capsule)
        load_digest = digest(capsule["runtime"]["state_dict"])
        graph = RouteBGraph(spec)
        apply_capsule(graph, capsule)
    else:
        torch.manual_seed(20261120 + args.seed)
        graph = RouteBGraph(
            UnitGraphSpec(
                groups=6,
                units_per_group=5,
                input_dim=8,
                hidden_dim=8,
                output_dim=24,
                seed=20261120 + args.seed,
            )
        )

    if args.session == BIRTH_SESSION:
        before = accuracy(graph, fit_features, fit_labels)
        parents = [unit["unit_id"] for unit in graph.units.identities["units"][:BIRTH_COUNT]]
        graph.units, details = live_entry_birth_multi(graph.units, 0, parents, count=BIRTH_COUNT)
        newborns = [unit["unit_id"] for unit in graph.units.identities["units"][-BIRTH_COUNT:]]
        for unit in graph.units.identities["units"][-BIRTH_COUNT:]:
            graph.units.set_lifecycle(unit["unit_id"], "mature")
        after = accuracy(graph, fit_features, fit_labels)
        birth = {
            "newborns": newborns,
            "neutral_drop": round(before - after, 6),
            "lifecycles": {},
            "details": details,
        }
    if args.session == WAKE_SESSION:
        states = lifecycle_map(graph)
        dormant = [unit for unit, life in states.items() if life == "dormant"]
        retired = [unit for unit, life in states.items() if life == "retired"]
        activity = unit_activity(graph, fit_features)
        index = [unit["unit_id"] for unit in graph.units.identities["units"]]
        dormant_activity_before = {
            unit: round(float(activity[index.index(unit)]), 8) for unit in dormant
        }
        target = dormant[0]
        before_logits = logits(graph, fit_features)
        before_wake_digest = digest(graph.state_dict())
        graph.units.set_lifecycle(target, "mature")
        after_logits = logits(graph, fit_features)
        after_wake_digest = digest(graph.state_dict())
        after_activity = unit_activity(graph, fit_features)
        wake = {
            "woken": target,
            "dormant_before": dormant_activity_before,
            "dormant_after_wake": round(float(after_activity[index.index(target)]), 8),
            "logits_change": round(float((before_logits - after_logits).abs().max()), 8),
            "retired": retired,
            "retired_activity": {
                unit: round(float(after_activity[index.index(unit)]), 8) for unit in retired
            },
        }
        graph.units.set_lifecycle(target, "dormant")

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

    if args.session == BIRTH_SESSION:
        graph.units.set_lifecycle(birth["newborns"][0], "dormant")
        graph.units.set_lifecycle(birth["newborns"][1], "dormant")
        graph.units.set_lifecycle(birth["newborns"][2], "retired")
        birth["lifecycles"] = {unit: lifecycle_map(graph)[unit] for unit in birth["newborns"]}

    acc = accuracy(graph, fit_features, fit_labels)
    parameters = {name: value.detach().clone() for name, value in graph.state_dict().items()}
    save_digest = digest(parameters)
    save_capsule(Path(args.out_capsule), build_capsule(graph))
    receipt = {
        "session": args.session,
        "seed": args.seed,
        "accuracy": round(acc, 4),
        "load_digest": load_digest,
        "save_digest": save_digest,
        "units": graph.units.n_units,
        "lifecycles": lifecycle_map(graph),
        "birth": birth,
        "before_wake_digest": before_wake_digest,
        "after_wake_digest": after_wake_digest,
        "wake": wake,
        "cpu": round(time.process_time() - started, 2),
    }
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"session": args.session, "acc": receipt["accuracy"], "units": receipt["units"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
