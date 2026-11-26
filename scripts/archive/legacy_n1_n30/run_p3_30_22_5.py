"""P3.30 validation card: read the answer off the body's own column tokens.

P3.28 reached fit 0.969 on raw pixels, but that readout had its own observation
path. P3.11-P3.16 had already shown the body's pooled 8-dim encoding cannot carry
the answer. The question here is narrower: if the body stops pooling patches and
instead encodes *columns* with the same weights (``column-observation-v1``), do
its tokens carry enough for the same readout?

The feasibility probe said the regrouping nearly doubles colour decodability per
position (patch tokens 0.385/0.387, column tokens 0.690/0.674, raw 1.000), so the
answer is not obvious and this card treats "tokens too lossy" as a live outcome.

Design, single difference against P3.28: the feature space is the body's [16, 8]
column tokens instead of raw 3-channel pixels. The six prototypes are not learned
here; they are what the same encoder produces for a column filled with each
palette colour, so no free index alignment can go wrong. The only learnable part
is the 54-parameter query classifier plus four scalars.

Verdicts (frozen before any reading):
  fit >= gate -> column_tokens_carry_the_task
  fit <= low  -> column_tokens_too_lossy
  otherwise   -> inconclusive_column_tokens

Modes: ``price`` and ``run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import RGB_COLORS, make_model
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_observation_columns import column_tokens
from taiji.u2_task_episode import NAMES, locate_public

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_30_column_readout_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_30_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_30_column_readout_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
CENTRES = [-2 / 3, 0.0, 2 / 3]
QUERY = 8
SIDE = 16
COLOURS = 6
SUFFIX = "现在在哪里？"


class BudgetStop(RuntimeError):
    pass


def palette_tensor() -> torch.Tensor:
    return torch.tensor([[value / 255 for value in row] for row in RGB_COLORS])


def column_positions() -> torch.Tensor:
    """One entry per column, not per pixel: the tokens are already columns."""
    return 2 * (torch.arange(SIDE) + 0.5) / SIDE - 1


def prototype_tokens(subject) -> torch.Tensor:
    """What the body's encoder produces for a column filled with each palette colour."""
    palette = palette_tensor()
    values = palette[:, None, :].repeat(1, SIDE, 1).reshape(COLOURS, SIDE * 3)
    frame = torch.zeros(COLOURS, 8)
    frame[:, 1] = 0
    frame[:, 5] = 1
    frame[:, 7] = 1
    tokens = torch.tanh(
        subject.encoder["rgb"](values) + subject.metadata(frame) + subject.modality[0]
    )
    return tokens.detach()


class TokenDetector(nn.Module):
    def __init__(self, tau: float, prototypes: torch.Tensor) -> None:
        super().__init__()
        self.classifier = nn.Linear(QUERY, COLOURS)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.log_mix = nn.Parameter(torch.zeros(()))
        self.presence = nn.Parameter(torch.tensor(2.0))
        self.abstain = nn.Parameter(torch.tensor(-1.0))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer("prototypes", prototypes)

    def forward(self, tokens: torch.Tensor, query: torch.Tensor):
        squared = ((tokens[None, :, :] - self.prototypes[:, None, :]) ** 2).sum(-1)
        temperature = self.log_tau.exp().clamp(1e-3, 1.0)
        weights = (-squared / temperature).softmax(-1)
        centroids = (weights * column_positions()).sum(-1)
        nearest = squared.min(-1).values
        logits = self.classifier(query)
        colour = logits.softmax(-1)
        mix = (logits / self.log_mix.exp().clamp(0.05, 20.0)).softmax(-1)
        centre = (mix * centroids).sum()
        null = self.abstain + self.presence * (mix * nearest).sum()
        scale = self.log_scale.exp().clamp(1e-2, 1e2)
        shape = torch.stack([-scale * (centre - c).abs() for c in CENTRES] + [scale * null])
        return shape, colour, mix, logits, centroids, null


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    if sha_of(Path(__file__)) != contract["source_pins"]["runner"]:
        raise ValueError("runner_source_drift")
    if (
        sha_of(ROOT / "scripts/training/first_a_support_22_5.py")
        != contract["source_pins"]["support"]
    ):
        raise ValueError("support_source_drift")
    if (
        sha_of(ROOT / "taiji/u2_observation_columns.py")
        != contract["source_pins"]["column_pathway"]
    ):
        raise ValueError("column_pathway_source_drift")
    if mode == "run":
        if contract["status"] != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("training_authority_absent")
        if not contract.get("cpu_cap"):
            raise ValueError("cpu_cap_absent")


def build(contract: dict):
    seed = int(contract["candidate"]["seed"])
    torch.manual_seed(seed)
    model = make_model("native", seed)
    module = TokenDetector(contract["candidate"]["tau"], prototype_tokens(model))
    spec = contract["optimizer"]
    optimizer = torch.optim.Adam(
        list(module.parameters()),
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    return model, module, optimizer


def query_of(model, example: dict) -> torch.Tensor:
    return model.text_units(example["public"]["request"]).mean(0)


def tokens_of(model, example: dict) -> torch.Tensor:
    return column_tokens(model, example["public"]["observations"], example["public"]["coordinates"])


def colour_name(example: dict) -> str:
    return example["public"]["request"].decode()[: -len(SUFFIX)]


def colour_index(example: dict) -> int:
    return NAMES.index(colour_name(example))


_TRUTH: dict[str, int | None] = {}


def oracle_column(example: dict) -> int | None:
    key = example["id"]
    if key in _TRUTH:
        return _TRUTH[key]
    image = (
        example["public"]["observations"]["rgb"]
        .reshape(4, 4, 4, 4, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(SIDE, SIDE, 3)
        * 255.0
    )
    position = locate_public(image, colour_name(example))
    column = None if position is None else min(15, max(0, int((position + 1) / 2 * 16)))
    _TRUTH[key] = column
    return column


def objective(model, module, batch: list[dict], lam: float):
    shapes, colours = [], []
    for example in batch:
        shape, _, _, logits, _, _ = module(tokens_of(model, example), query_of(model, example))
        shapes.append(shape)
        colours.append(logits)
    labels = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    loss = functional.cross_entropy(torch.stack(shapes), labels)
    if lam:
        targets = torch.tensor([colour_index(example) for example in batch])
        loss = loss + lam * functional.cross_entropy(torch.stack(colours), targets)
    return loss


def accuracy(model, module, examples: list[dict]) -> dict:
    with torch.no_grad():
        correct = 0
        colour_correct = 0
        hits = total = 0
        abstain_unknown = abstain_other = 0
        for example in examples:
            shape, colour, mix, _, centroids, _ = module(
                tokens_of(model, example), query_of(model, example)
            )
            label = example["target"].decode()
            correct += int(int(shape.argmax()) == LABELS.index(label))
            colour_correct += int(int(colour.argmax()) == colour_index(example))
            if int(shape.argmax()) == 3:
                abstain_unknown += int(label == "未知")
                abstain_other += int(label != "未知")
            if label != "未知":
                truth = oracle_column(example)
                if truth is not None:
                    total += 1
                    chosen = float((mix * centroids).sum())
                    column = min(15, max(0, int((chosen + 1) / 2 * 16)))
                    hits += int(column == truth)
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "colour_accuracy": colour_correct / len(examples),
        "centroid_column_hits": hits,
        "centroid_column_total": total,
        "abstain_on_unknown": abstain_unknown,
        "abstain_on_other": abstain_other,
    }


def batch_at(data, order, batch_size, update):
    return [
        data["train"][order[((update - 1) * batch_size + offset) % len(order)]]
        for offset in range(batch_size)
    ]


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, module, optimizer = build(contract)
    per_step = []
    for update in range(1, 4):
        batch = batch_at(data, contract["order"], contract["batch"], update)
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, module, batch, float(contract["aux_weight"]))
        loss.backward()
        optimizer.step()
        per_step.append(time.process_time() - started)
        if time.process_time() - cpu_start > contract["price_cap_cpu"]:
            raise BudgetStop("price_probe_budget_stop")
    started = time.process_time()
    scored = accuracy(model, module, data["fit"])
    eval_cpu = time.process_time() - started
    step_cpu = statistics.median(per_step)
    curve_cpu = max(contract["dose_points"]) * step_cpu + 8 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], max(curve_cpu * 1.1, 60.0)), 3)
    report = {
        "schema": "taiji-u2-p3-30-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.30",
        "phase": "price_probe",
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "measurements": {
            "per_step_cpu": [round(value, 6) for value in per_step],
            "per_step_cpu_median": round(step_cpu, 6),
            "eval_cpu": round(eval_cpu, 6),
            "eval_accuracy_at_zero": scored["accuracy"],
            "eval_colour_accuracy_at_zero": scored["colour_accuracy"],
        },
        "derived": {
            "curve_updates": max(contract["dose_points"]),
            "curve_cpu_estimate": round(curve_cpu, 3),
            "cap_cpu": cap,
            "cap_wall": round(cap * 2, 3),
        },
        "verdict": "price_measured" if cap <= contract["design_cap_cpu"] else "pricing_over_budget",
        "discipline": {
            "checkpoints_written": 0,
            "optimizer_calls": 3,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
        "scope": "pricing only; three in-memory steps are discarded, no capability claim",
    }
    write_report(PRICE_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cap_cpu": cap,
                "curve_cpu": round(curve_cpu, 3),
                "trainable": report["trainable_parameters"],
            }
        )
    )
    return 0 if report["verdict"] == "price_measured" else 4


def decide(final: dict, criteria: dict) -> str:
    if final["fit_accuracy"] >= criteria["fit_gate"]:
        return "column_tokens_carry_the_task"
    if final["fit_accuracy"] <= criteria["fit_low"]:
        return "column_tokens_too_lossy"
    return "inconclusive_column_tokens"


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, module, optimizer = build(contract)
    doses = contract["dose_points"]
    root = Path(contract["run_root"])
    if root.exists() or RUN_REPORT.exists():
        raise FileExistsError("p3_30_run_already_started")
    root.mkdir(parents=True)
    results = []
    for update in range(1, max(doses) + 1):
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_30_budget_stop")
        batch = batch_at(data, contract["order"], contract["batch"], update)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, module, batch, float(contract["aux_weight"]))
        loss.backward()
        optimizer.step()
        if update not in doses:
            continue
        started = time.process_time()
        train = accuracy(model, module, data["train"])
        fit = accuracy(model, module, data["fit"])
        torch.save(module.state_dict(), root / f"detector-{update:06d}.pt")
        results.append(
            {
                "update": update,
                "train_accuracy": train["accuracy"],
                "fit_accuracy": fit["accuracy"],
                "train_correct": train["correct"],
                "fit_correct": fit["correct"],
                "colour_accuracy_train": train["colour_accuracy"],
                "colour_accuracy_fit": fit["colour_accuracy"],
                "centroid_column_hits_fit": fit["centroid_column_hits"],
                "centroid_column_total_fit": fit["centroid_column_total"],
                "abstain_on_unknown_fit": fit["abstain_on_unknown"],
                "abstain_on_other_fit": fit["abstain_on_other"],
                "cpu": round(time.process_time() - started, 6),
            }
        )
    verdict = decide(results[-1], contract["criteria"])
    report = {
        "schema": "taiji-u2-p3-30-column-readout-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.30",
        "phase": "column_token_readout",
        "verdict": verdict,
        "doses": results,
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "prototype_tokens": [
            [round(float(value), 4) for value in row] for row in module.prototypes.detach()
        ],
        "batch": contract["batch"],
        "optimizer": contract["optimizer"],
        "control_arms": contract["control_arms"],
        "training_authority": contract["training_authority"],
        "cost": {
            "total_cpu": round(time.process_time() - cpu_start, 6),
            "total_wall": round(time.perf_counter() - wall_start, 6),
            "cap_cpu": contract["cpu_cap"],
            "cap_wall": contract["wall_cap"],
        },
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "scope": (
            "validation card for column-observation-v1; features are the body's own [16, 8] "
            "column tokens, prototypes are encoder outputs for palette-filled columns, only "
            "the 54-parameter classifier and four scalars are learned; no capability claim"
        ),
    }
    write_report(RUN_REPORT, report)
    print(json.dumps({"verdict": verdict, "last": results[-1], "cost": report["cost"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("price", "run"), default="price")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.mode == "price":
        return price(contract)
    try:
        guards(contract, "run")
    except ValueError as error:
        print(json.dumps({"verdict": "refused", "reason": str(error)}))
        return 2
    try:
        return run(contract)
    except BudgetStop as error:
        if not RUN_REPORT.exists():
            write_report(
                RUN_REPORT,
                {
                    "schema": "taiji-u2-p3-30-column-readout-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.30",
                    "verdict": "budget_stop",
                    "reason": str(error),
                },
            )
        return 3
    except Exception as error:  # noqa: BLE001
        if not RUN_REPORT.exists():
            write_report(
                RUN_REPORT,
                {
                    "schema": "taiji-u2-p3-30-column-readout-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.30",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
