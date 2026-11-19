"""22.6.3 recursive-development feasibility probe: two dependent corrections.

Task family (joint24, adapter v3): a colour-conditional position rule is applied
to labels while features keep showing the true scene.

* rule 1 (dense, group-level): for colours {0,1,2} the reported position is
  shifted by +1 (left->middle->right->left);
* rule 2 (sparse, refinement): for colours {1,2} one more +1 is applied (+2);
  only 24 feedback samples are given for this step.

Dependent path: base (clean labels) -> correction A (dense rule 1) -> correction B
(sparse rule 2). Fixed-parent counterfactual: base -> correction B directly.
If A is necessary for B, the dependent path should reach clearly higher accuracy
on the rule-2 held-out items. This probe only checks constructibility (zero
check); the real card, thresholds and budget come later.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
    entity_from_request,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_6_3_recursive_feasibility_20261120.json"
CAP_CPU = 80.0
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
GROUP = {0, 1, 2}
REFINE = {1, 2}
BASE_STEPS = 600
A_STEPS = 600
B_STEPS = 120
LR = 3e-3
BATCH = 64
WARMUP = 100
TRAIN_WORLDS = 256
FIT_WORLDS = 32
SPARSE_SAMPLES = 12


def shift(label: int, amount: int) -> int:
    colour, position = divmod(label, 4)
    if position in (0, 1, 2):
        position = (position + amount) % 3
    return colour * 4 + position


def labels_for(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    colours = []
    clean = []
    for example in examples:
        entity = entity_from_request(example)
        from taiji.u2_task_episode import NAMES  # noqa: PLC0415

        colours.append(NAMES.index(entity) if entity in NAMES else 0)
        clean.append(joint_label(example))
    return torch.tensor(colours), torch.tensor(clean)


def make_features(examples: list[dict]) -> torch.Tensor:
    return torch.stack([adapter_v3_features(example) for example in examples]).float()


def rule_labels(colours: torch.Tensor, clean: torch.Tensor, amount: int) -> torch.Tensor:
    out = clean.clone()
    for index, colour in enumerate(colours.tolist()):
        if colour in GROUP:
            out[index] = shift(int(clean[index]), amount if colour in REFINE else 1)
    return out


def learning_rate(step: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(min(step, 500)) / 500.0))


def train(
    graph: RouteBGraph, features: torch.Tensor, labels: torch.Tensor, steps: int, seed: int
) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    count = features.shape[0]
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = torch.randint(0, count, (min(BATCH, count),), generator=generator)
        gates = torch.ones(batch.numel(), graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(features[batch]), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, labels[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def evaluate(graph: RouteBGraph, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        readout = graph(channels(features), gates, graph.initial_state())["readout"]
    return readout.argmax(-1) == labels


def make_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=4, units_per_group=8, input_dim=8, hidden_dim=8, output_dim=24, seed=20261120 + seed
    )
    return RouteBGraph(spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train_examples = generate_examples("train", TRAIN_WORLDS)
    fit_examples = generate_examples("fit", FIT_WORLDS)
    train_features = make_features(train_examples)
    train_colours, train_clean = labels_for(train_examples)
    fit_features = make_features(fit_examples)
    fit_colours, fit_clean = labels_for(fit_examples)

    rule1_train = rule_labels(train_colours, train_clean, amount=1)
    rule2_train = rule_labels(train_colours, train_clean, amount=2)
    rule2_fit = rule_labels(fit_colours, fit_clean, amount=2)

    group_mask = torch.tensor([colour in GROUP for colour in train_colours.tolist()])
    refine_mask = torch.tensor([colour in REFINE for colour in train_colours.tolist()])
    sparse_index = torch.nonzero(refine_mask).flatten()[:SPARSE_SAMPLES]
    fit_refine = torch.tensor([colour in REFINE for colour in fit_colours.tolist()])

    seed = 0
    base = make_graph(seed)
    train(base, train_features, train_clean, BASE_STEPS, seed)
    base_refine = float(
        evaluate(base, fit_features[fit_refine], rule2_fit[fit_refine]).float().mean()
    )

    dependent = make_graph(seed)
    train(dependent, train_features, train_clean, BASE_STEPS, seed)
    train(
        dependent,
        train_features[group_mask],
        rule1_train[group_mask],
        A_STEPS,
        seed + 1,
    )
    after_a_refine = float(
        evaluate(dependent, fit_features[fit_refine], rule2_fit[fit_refine]).float().mean()
    )
    train(
        dependent,
        train_features[sparse_index],
        rule2_train[sparse_index],
        B_STEPS,
        seed + 2,
    )
    dependent_refine = float(
        evaluate(dependent, fit_features[fit_refine], rule2_fit[fit_refine]).float().mean()
    )

    fixed_parent = make_graph(seed)
    train(fixed_parent, train_features, train_clean, BASE_STEPS, seed)
    train(
        fixed_parent,
        train_features[sparse_index],
        rule2_train[sparse_index],
        B_STEPS,
        seed + 2,
    )
    fixed_refine = float(
        evaluate(fixed_parent, fit_features[fit_refine], rule2_fit[fit_refine]).float().mean()
    )

    verdict = (
        "recursive_feasibility_ready"
        if base_refine < 0.40 and after_a_refine < 0.60 and dependent_refine >= fixed_refine + 0.10
        else "recursive_feasibility_absent"
    )
    payload = {
        "schema": "taiji-u2-22-6-3-recursive-feasibility-v1",
        "card": "22.6.3 two dependent corrections feasibility probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "base_steps": BASE_STEPS,
            "correction_a_steps": A_STEPS,
            "correction_b_steps": B_STEPS,
            "correction_b_samples": SPARSE_SAMPLES,
            "group_colours": sorted(GROUP),
            "refine_colours": sorted(REFINE),
            "train_worlds": TRAIN_WORLDS,
            "fit_worlds": FIT_WORLDS,
            "seeds": [seed],
        },
        "readings": {
            "base_refine_accuracy": round(base_refine, 4),
            "after_a_refine_accuracy": round(after_a_refine, 4),
            "dependent_refine_accuracy": round(dependent_refine, 4),
            "fixed_parent_refine_accuracy": round(fixed_refine, 4),
            "dependent_minus_fixed": round(dependent_refine - fixed_refine, 4),
        },
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": BASE_STEPS * 3 + A_STEPS + B_STEPS * 2,
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "readings": payload["readings"], "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "recursive_feasibility_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
