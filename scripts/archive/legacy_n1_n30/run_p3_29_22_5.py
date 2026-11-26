"""P3.29 reproduction card: second seed and a paraphrased request.

P3.28 reached fit 31/32 with the palette anchored and the colour named in the
request. Before anything is built on that, two adjacent settings have to hold:

  seed_b      the same recipe with seed 20261111 instead of 20261110
  paraphrase  the same recipe with a different request template,
              "请说明{colour}此刻的位置。" instead of "{colour}现在在哪里？"

The paraphrase cell is the sharper test: the request is rewritten on the fly and
the frozen text encoder sees bytes it never saw, so a cell that passes has
routed on the colour rather than on the template. The reference cell is P3.28's
palette_colour arm (fit 0.9688) and is not re-run.

Verdicts (frozen before any reading):
  both cells fit >= gate           -> reproduced_on_both_axes
  seed_b fails, paraphrase passes  -> seed_unstable
  paraphrase fails, seed_b passes  -> template_bound_routing
  both fail                        -> not_reproduced
  otherwise                        -> inconclusive_reproduction

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
from taiji.u2_task_episode import NAMES, locate_public

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_29_reproduction_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_29_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_29_reproduction_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
CENTRES = [-2 / 3, 0.0, 2 / 3]
QUERY = 8
SIDE = 16
PIXELS = SIDE * SIDE
COLOURS = 6
SUFFIX = "现在在哪里？"


class BudgetStop(RuntimeError):
    pass


def unpatched(patches: torch.Tensor) -> torch.Tensor:
    """[16, 48] patches -> [16, 16, 3] image."""
    return patches.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 3)


def palette_tensor() -> torch.Tensor:
    return torch.tensor([[value / 255 for value in row] for row in RGB_COLORS])


def pixel_positions() -> torch.Tensor:
    # Index i in the flattened image is row*SIDE + column, so the x coordinate must
    # repeat as a whole row. repeat_interleave would give every pixel its row.
    x = 2 * (torch.arange(SIDE) + 0.5) / SIDE - 1
    return x.repeat(SIDE)


def colour_name(example: dict) -> str:
    return example["public"]["request"].decode()[: -len(SUFFIX)]


def request_bytes(example: dict, style: str) -> bytes:
    if style == "original":
        return example["public"]["request"]
    if style == "paraphrase":
        return f"请说明{colour_name(example)}此刻的位置。".encode()
    raise ValueError(f"unknown_request_style:{style}")


class PaletteDetector(nn.Module):
    def __init__(self, tau: float, seed: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(QUERY, COLOURS)
        generator = torch.Generator().manual_seed(seed)
        anchor = palette_tensor() + 0.02 * torch.randn(COLOURS, 3, generator=generator)
        self.prototype = nn.Parameter(torch.logit(anchor.clamp(0.01, 0.99)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.log_mix = nn.Parameter(torch.zeros(()))
        self.presence = nn.Parameter(torch.tensor(2.0))
        self.abstain = nn.Parameter(torch.tensor(-1.0))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer("positions", pixel_positions(), persistent=False)

    def forward(self, patches: torch.Tensor, query: torch.Tensor):
        pixels = unpatched(patches).reshape(PIXELS, 3)
        proto = torch.sigmoid(self.prototype)
        squared = ((pixels[None, :, :] - proto[:, None, :]) ** 2).sum(-1)
        temperature = self.log_tau.exp().clamp(1e-3, 1.0)
        weights = (-squared / temperature).softmax(-1)
        centroids = (weights * self.positions).sum(-1)
        nearest = squared.min(-1).values
        colour_logits = self.classifier(query)
        colour = colour_logits.softmax(-1)
        mix = (colour_logits / self.log_mix.exp().clamp(0.05, 20.0)).softmax(-1)
        centre = (mix * centroids).sum()
        null = self.abstain + self.presence * (mix * nearest).sum()
        scale = self.log_scale.exp().clamp(1e-2, 1e2)
        shape = torch.stack([-scale * (centre - c).abs() for c in CENTRES] + [scale * null])
        return shape, colour, mix, colour_logits, centroids, null


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
    if mode == "run":
        if contract["status"] != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("training_authority_absent")
        if not contract.get("cpu_cap"):
            raise ValueError("cpu_cap_absent")


def build_cell(contract: dict, seed: int):
    torch.manual_seed(seed)
    model = make_model("native", seed)
    module = PaletteDetector(contract["candidate"]["tau"], seed + 7)
    spec = contract["optimizer"]
    optimizer = torch.optim.Adam(
        list(module.parameters()),
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    return model, module, optimizer


def query_of(model, example: dict, style: str) -> torch.Tensor:
    return model.text_units(request_bytes(example, style)).mean(0)


def patches_of(example: dict) -> torch.Tensor:
    return example["public"]["observations"]["rgb"]


def colour_index(example: dict) -> int:
    return NAMES.index(colour_name(example))


_TRUTH: dict[str, int | None] = {}


def oracle_column(example: dict) -> int | None:
    key = example["id"]
    if key in _TRUTH:
        return _TRUTH[key]
    image = unpatched(example["public"]["observations"]["rgb"]) * 255.0
    position = locate_public(image, colour_name(example))
    column = None if position is None else min(15, max(0, int((position + 1) / 2 * 16)))
    _TRUTH[key] = column
    return column


def objective(model, module, batch: list[dict], style: str, lam: float):
    shapes, colours = [], []
    for example in batch:
        shape, _, _, colour_logits, _, _ = module(
            patches_of(example), query_of(model, example, style)
        )
        shapes.append(shape)
        colours.append(colour_logits)
    labels = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    loss = functional.cross_entropy(torch.stack(shapes), labels)
    if lam:
        targets = torch.tensor([colour_index(example) for example in batch])
        loss = loss + lam * functional.cross_entropy(torch.stack(colours), targets)
    return loss


def accuracy(model, module, examples: list[dict], style: str) -> dict:
    with torch.no_grad():
        correct = 0
        colour_correct = 0
        hits = total = 0
        abstain_unknown = abstain_other = 0
        for example in examples:
            shape, colour, mix, _, centroids, _ = module(
                patches_of(example), query_of(model, example, style)
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
    seed = int(contract["cells"]["seed_b"]["seed"])
    model, module, optimizer = build_cell(contract, seed)
    per_step = []
    for update in range(1, 4):
        batch = batch_at(data, contract["order"], contract["batch"], update)
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, module, batch, "original", contract["aux_weight"])
        loss.backward()
        optimizer.step()
        per_step.append(time.process_time() - started)
        if time.process_time() - cpu_start > contract["price_cap_cpu"]:
            raise BudgetStop("price_probe_budget_stop")
    started = time.process_time()
    scored = accuracy(model, module, data["fit"], "original")
    eval_cpu = time.process_time() - started
    step_cpu = statistics.median(per_step)
    curve_cpu = 2 * max(contract["dose_points"]) * step_cpu + 12 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], max(curve_cpu * 1.1, 100.0)), 3)
    report = {
        "schema": "taiji-u2-p3-29-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.29",
        "phase": "price_probe",
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "measurements": {
            "per_step_cpu": [round(value, 6) for value in per_step],
            "per_step_cpu_median": round(step_cpu, 6),
            "eval_cpu": round(eval_cpu, 6),
            "eval_accuracy_at_zero": scored["accuracy"],
        },
        "derived": {
            "curve_updates": 2 * max(contract["dose_points"]),
            "cells": 2,
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


def decide(cells: dict, criteria: dict) -> str:
    gate = criteria["fit_gate"]
    seed_ok = cells["seed_b"]["doses"][-1]["fit_accuracy"] >= gate
    para_ok = cells["paraphrase"]["doses"][-1]["fit_accuracy"] >= gate
    if seed_ok and para_ok:
        return "reproduced_on_both_axes"
    if seed_ok and not para_ok:
        return "template_bound_routing"
    if para_ok and not seed_ok:
        return "seed_unstable"
    if not seed_ok and not para_ok:
        return "not_reproduced"
    return "inconclusive_reproduction"


def run_cell(contract: dict, data, name: str, spec: dict) -> dict:
    seed = int(spec["seed"])
    style = spec["style"]
    model, module, optimizer = build_cell(contract, seed)
    doses = contract["dose_points"]
    root = Path(contract["run_root"]) / name
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for update in range(1, max(doses) + 1):
        batch = batch_at(data, contract["order"], contract["batch"], update)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, module, batch, style, float(contract["aux_weight"]))
        loss.backward()
        optimizer.step()
        if update not in doses:
            continue
        started = time.process_time()
        train = accuracy(model, module, data["train"], style)
        fit = accuracy(model, module, data["fit"], style)
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
    return {
        "cell": name,
        "seed": seed,
        "style": style,
        "parameters": sum(p.numel() for p in module.parameters()),
        "doses": results,
    }


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    root = Path(contract["run_root"])
    if root.exists() or RUN_REPORT.exists():
        raise FileExistsError("p3_29_run_already_started")
    root.mkdir(parents=True)
    cells = {}
    for name in ("seed_b", "paraphrase"):
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_29_budget_stop")
        cells[name] = run_cell(contract, data, name, contract["cells"][name])
    verdict = decide(cells, contract["criteria"])
    report = {
        "schema": "taiji-u2-p3-29-reproduction-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.29",
        "phase": "reproduction",
        "verdict": verdict,
        "cells": cells,
        "reference_cell": contract["cells"]["reference"],
        "aux_weight": contract["aux_weight"],
        "batch": contract["batch"],
        "optimizer": contract["optimizer"],
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
            "detector-only reproduction card; palette-anchored readout with the colour named "
            "in the request; one cell changes the seed, one rewrites the request template; "
            "U2 body frozen; no capability claim"
        ),
    }
    write_report(RUN_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "cells": {name: cell["doses"][-1] for name, cell in cells.items()},
                "cost": report["cost"],
            }
        )
    )
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
                    "schema": "taiji-u2-p3-29-reproduction-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.29",
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
                    "schema": "taiji-u2-p3-29-reproduction-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.29",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
