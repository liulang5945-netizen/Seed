"""P3.24 soft-centroid card: read the answer as a differentiable centroid.

P3.22 and P3.23 both collapsed, and each collapse traced to a defect in the
readout rather than to the task:
  * a free null slot let the model abstain on everything (24/24 non-未知 fit);
  * the log-mass readout gives every column inside a class the same gradient,
    so attention flattened to exactly log(16) and no column could be selected.

The centroid readout removes both. Attention runs over the 256 pixels; the
position is their attention-weighted mean, and the class logits are the
(negative) squared distance from that mean to the three class centres, with a
learned temperature. Abstention stays coupled to match strength. At uniform
attention the centroid gradient is proportional to each pixel's offset from the
scene centre, so every pixel receives a distinct signal from the first step.

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
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_task_episode import locate_public

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_24_centroid_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_24_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_24_centroid_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
CENTRES = [-2 / 3, 0.0, 2 / 3]
PIXELS = 48
QUERY = 8
SIDE = 16
SCORE_LOG = math.log(SIDE * SIDE)


class BudgetStop(RuntimeError):
    pass


def unpatched(patches: torch.Tensor) -> torch.Tensor:
    """[16, 48] patches -> [16, 16, 3] image."""
    return patches.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 3)


def pixel_positions() -> torch.Tensor:
    x = 2 * (torch.arange(SIDE) + 0.5) / SIDE - 1
    return x.repeat_interleave(SIDE)


class CentroidDetector(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.colour = nn.Linear(3, width)
        self.query = nn.Linear(QUERY, width, bias=False)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.abstain = nn.Parameter(torch.zeros(()))
        self.register_buffer("positions", pixel_positions(), persistent=False)

    def forward(self, patches: torch.Tensor, query: torch.Tensor):
        pixels = unpatched(patches).reshape(SIDE * SIDE, 3)
        scores = (torch.tanh(self.colour(pixels)) @ self.query(query)) / (pixels.shape[0] ** 0.5)
        weights = scores.softmax(-1)
        centre = (weights * self.positions).sum()
        scale = self.log_scale.exp().clamp(1e-2, 1e2)
        # -|centre - c| has the same argmax as position_word(centre); the only
        # learned part of the answer is the colour-conditioned attention.
        shape = torch.stack(
            [-scale * (centre - c).abs() for c in CENTRES]
            + [
                scale
                * (self.abstain - (torch.logsumexp(scores, dim=0) - scores.new_tensor(SCORE_LOG)))
            ]
        )
        return shape, weights


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
    seed = int(contract["candidate"]["seed"])
    torch.manual_seed(seed)
    model = make_model("native", seed)
    module = CentroidDetector(contract["candidate"]["width"])
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


def patches_of(example: dict) -> torch.Tensor:
    return example["public"]["observations"]["rgb"]


def objective(model, module, batch: list[dict]):
    logits, weights = [], []
    for example in batch:
        out, attention = module(patches_of(example), query_of(model, example))
        logits.append(out)
        weights.append(attention)
    targets = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    return functional.cross_entropy(torch.stack(logits), targets), torch.stack(weights)


_TRUTH: dict[str, int | None] = {}


def oracle_column(example: dict) -> int | None:
    key = example["id"]
    if key in _TRUTH:
        return _TRUTH[key]
    image = unpatched(example["public"]["observations"]["rgb"]) * 255.0
    entity = example["public"]["request"].decode()[: -len("现在在哪里？")]
    position = locate_public(image, entity)
    column = None if position is None else min(15, max(0, int((position + 1) / 2 * 16)))
    _TRUTH[key] = column
    return column


def accuracy(model, module, examples: list[dict]) -> dict:
    with torch.no_grad():
        correct = 0
        entropies = []
        hits = total = 0
        abstain_unknown = abstain_other = 0
        for example in examples:
            logits, weights = module(patches_of(example), query_of(model, example))
            label = example["target"].decode()
            correct += int(int(logits.argmax()) == LABELS.index(label))
            entropies.append(float(-(weights * weights.clamp_min(1e-12).log()).sum()))
            if int(logits.argmax()) == 3:
                abstain_unknown += int(label == "未知")
                abstain_other += int(label != "未知")
            if label != "未知":
                truth = oracle_column(example)
                if truth is not None:
                    total += 1
                    centre = float((weights * module.positions).sum())
                    column = min(15, max(0, int((centre + 1) / 2 * 16)))
                    hits += int(column == truth)
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "mean_attention_entropy": sum(entropies) / len(entropies),
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
    curve_cpu = max(contract["dose_points"]) * step_cpu + 5 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], max(curve_cpu * 1.1, 60.0)), 3)
    report = {
        "schema": "taiji-u2-p3-24-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.24",
        "phase": "price_probe",
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "measurements": {
            "per_step_cpu": [round(value, 6) for value in per_step],
            "per_step_cpu_median": round(step_cpu, 6),
            "eval_cpu": round(eval_cpu, 6),
            "eval_accuracy_at_zero": scored["accuracy"],
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
        return "centroid_readout_generalises"
    if final["fit_accuracy"] <= criteria["fit_low"]:
        if final["train_accuracy"] < criteria["train_gate"]:
            return "centroid_cannot_fit"
        return "centroid_readout_insufficient"
    return "inconclusive_centroid"


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, module, optimizer = build(contract)
    order = contract["order"]
    batch_size = contract["batch"]
    doses = contract["dose_points"]
    root = Path(contract["run_root"])
    if root.exists() or RUN_REPORT.exists():
        raise FileExistsError("p3_24_run_already_started")
    root.mkdir(parents=True)
    results = []
    for update in range(1, max(doses) + 1):
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_24_budget_stop")
        batch = batch_at(data, order, batch_size, update)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(model, module, batch)
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
                "attention_entropy_train": round(train["mean_attention_entropy"], 6),
                "attention_entropy_fit": round(fit["mean_attention_entropy"], 6),
                "centroid_column_hits_fit": fit["centroid_column_hits"],
                "centroid_column_total_fit": fit["centroid_column_total"],
                "abstain_on_unknown_fit": fit["abstain_on_unknown"],
                "abstain_on_other_fit": fit["abstain_on_other"],
                "cpu": round(time.process_time() - started, 6),
            }
        )
    verdict = decide(results[-1], contract["criteria"])
    report = {
        "schema": "taiji-u2-p3-24-centroid-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.24",
        "phase": "soft_centroid",
        "verdict": verdict,
        "doses": results,
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "scale": round(float(module.log_scale.detach().exp()), 6),
        "abstain_bias": round(float(module.abstain.detach()), 6),
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
        "scope": (
            "detector-only card; position is the attention-weighted centroid of 256 pixels, "
            "classes are squared distances to the three centres, abstention is coupled to "
            "match strength; U2 body frozen; no capability claim"
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
                    "schema": "taiji-u2-p3-24-centroid-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.24",
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
                    "schema": "taiji-u2-p3-24-centroid-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.24",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
