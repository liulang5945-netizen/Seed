"""P3.17 query-conditioned patch detector on the P3.10 task.

Trains only the rgb encoder and a small attention module (about 600 parameters)
to answer the four-way position question from a query-conditioned readout over
the 16 patches. The byte path is not wired and not evaluated: this card asks
whether the structure can express the colour-by-position conjunction at all.

Modes: ``price`` and ``run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_17_detection_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_17_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_17_detection_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
GATE = 0.80
LOW = 0.35


class BudgetStop(RuntimeError):
    pass


class PatchAttention(nn.Module):
    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.patch = nn.Linear(width, width, bias=False)
        self.readout = nn.Linear(width * 2, classes)

    def forward(
        self, patches: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = (self.patch(patches) @ self.query(query)) / (patches.shape[-1] ** 0.5)
        weights = scores.softmax(-1)
        attended = (weights[..., None] * patches).sum(0)
        logits = self.readout(torch.cat([attended, query]))
        return logits, weights


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


def build(contract: dict):
    torch.manual_seed(int(contract["candidate"]["seed"]))
    model = make_model("native", int(contract["candidate"]["seed"]))
    module = PatchAttention(contract["candidate"]["encoding_width"], len(LABELS))
    trainable = [
        parameter for name, parameter in model.named_parameters() if name.startswith("encoder.rgb")
    ]
    trainable += list(module.parameters())
    spec = contract["optimizer"]
    optimizer = torch.optim.Adam(
        trainable,
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    return model, module, optimizer, trainable


def query_of(model, example: dict) -> torch.Tensor:
    units = model.text_units(example["public"]["request"])
    return units.mean(0)


def patches_of(model, example: dict) -> torch.Tensor:
    public = example["public"]
    tokens = model.encode(public["observations"], public["coordinates"], public["outcome"])
    return tokens["rgb"]


def objective(model, module, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    logits, weights = [], []
    for example in batch:
        out, attention = module(patches_of(model, example), query_of(model, example))
        logits.append(out)
        weights.append(attention)
    targets = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    return functional.cross_entropy(torch.stack(logits), targets), torch.stack(weights)


def accuracy(model, module, examples: list[dict]) -> dict:
    with torch.no_grad():
        correct = 0
        entropies = []
        for example in examples:
            logits, weights = module(patches_of(model, example), query_of(model, example))
            correct += int(int(logits.argmax()) == LABELS.index(example["target"].decode()))
            entropies.append(float(-(weights * weights.clamp_min(1e-12).log()).sum()))
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "mean_attention_entropy": sum(entropies) / len(entropies),
    }


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, module, optimizer, trainable = build(contract)
    order = contract["order"]
    batch_size = contract["batch"]
    per_step = []
    for update in range(3):
        batch = [
            data["train"][order[(update * batch_size + offset) % 64]]
            for offset in range(batch_size)
        ]
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(model, module, batch)
        loss.backward()
        optimizer.step()
        per_step.append(time.process_time() - started)
        if time.process_time() - cpu_start > contract["price_cap_cpu"]:
            raise BudgetStop("price_probe_budget_stop")
    started = time.process_time()
    scored = accuracy(model, module, data["fit"])
    eval_cpu = time.process_time() - started
    step_cpu = statistics.median(per_step)
    curve_cpu = 1844 * step_cpu + 5 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], max(curve_cpu * 1.1, 30.0)), 3)
    report = {
        "schema": "taiji-u2-p3-17-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.17",
        "phase": "price_probe",
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "measurements": {
            "per_step_cpu": [round(value, 6) for value in per_step],
            "per_step_cpu_median": round(step_cpu, 6),
            "eval_cpu": round(eval_cpu, 6),
            "eval_accuracy_at_zero": scored["accuracy"],
        },
        "derived": {
            "curve_updates": 1844,
            "curve_evaluations": 5,
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


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, module, optimizer, trainable = build(contract)
    order = contract["order"]
    batch_size = contract["batch"]
    doses = contract["dose_points"]
    root = Path(contract["run_root"])
    if root.exists() or RUN_REPORT.exists():
        raise FileExistsError("p3_17_run_already_started")
    root.mkdir(parents=True)
    results = []
    for update in range(1, max(doses) + 1):
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_17_budget_stop")
        batch = [
            data["train"][order[((update - 1) * batch_size + offset) % 64]]
            for offset in range(batch_size)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(model, module, batch)
        loss.backward()
        optimizer.step()
        if update not in doses:
            continue
        started = time.process_time()
        train = accuracy(model, module, data["train"])
        fit = accuracy(model, module, data["fit"])
        torch.save(
            {
                "module": module.state_dict(),
                "encoder": {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                    if name.startswith("encoder.rgb")
                },
            },
            root / f"detector-{update:04d}.pt",
        )
        results.append(
            {
                "update": update,
                "train_accuracy": train["accuracy"],
                "fit_accuracy": fit["accuracy"],
                "train_correct": train["correct"],
                "fit_correct": fit["correct"],
                "attention_entropy_train": round(train["mean_attention_entropy"], 6),
                "attention_entropy_fit": round(fit["mean_attention_entropy"], 6),
                "cpu": round(time.process_time() - started, 6),
            }
        )
    final = results[-1]
    if final["train_accuracy"] >= GATE and final["fit_accuracy"] >= GATE:
        verdict = "conjunction_expressed_by_detector"
    elif final["train_accuracy"] >= GATE:
        verdict = "detector_learns_task_split_blocks"
    elif final["train_accuracy"] <= LOW:
        verdict = "structure_insufficient"
    else:
        verdict = "inconclusive_patch_attention"
    report = {
        "schema": "taiji-u2-p3-17-detection-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.17",
        "phase": "detector",
        "verdict": verdict,
        "doses": results,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "module_parameters": sum(parameter.numel() for parameter in module.parameters()),
        "batch": batch_size,
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
        "scope": "isolated detector only; byte path not wired and not evaluated; no capability claim",
    }
    write_report(RUN_REPORT, report)
    print(json.dumps({"verdict": verdict, "doses": results, "cost": report["cost"]}))
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
                    "schema": "taiji-u2-p3-17-detection-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.17",
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
                    "schema": "taiji-u2-p3-17-detection-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.17",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
