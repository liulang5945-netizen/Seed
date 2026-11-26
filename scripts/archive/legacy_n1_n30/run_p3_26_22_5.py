"""P3.26 supervision-density card: does a denser target unlock the matching?

Every readout so far collapsed, and each collapse was traced to a specific defect.
The one lever never varied is how dense the supervision is. This card holds the
readout fixed and varies only that.

Both arms share:
  * the P3.24 soft-centroid readout over 256 pixels,
  * a decoded-colour score ``-||rgb - Wq||^2 / tau`` with ``tau`` initialised at
    0.02 (declared in advance), so attention is sharp from the first step and the
    loss cannot buy a softer fit the way it did in P3.25,
  * the same data, order, optimiser and 7376 steps.

Arm A sees the 4-way answer only. Arm B adds a cross-entropy on a 17-way
distribution (16 columns plus null) whose target is the public locator's column,
i.e. the intermediate quantity the sparse loss has to discover by itself.

Verdicts:
  B generalises, A does not  -> supervision_density_is_the_wall
  both generalise            -> both_learn
  neither generalises        -> line_closed_matching_not_learnable
  anything else              -> inconclusive_density

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
from taiji.u2_task_episode import locate_public

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_26_density_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_26_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_26_density_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
CENTRES = [-2 / 3, 0.0, 2 / 3]
QUERY = 8
SIDE = 16
PIXELS = SIDE * SIDE
COLUMN_INDEX = torch.arange(PIXELS) % SIDE


class BudgetStop(RuntimeError):
    pass


def unpatched(patches: torch.Tensor) -> torch.Tensor:
    """[16, 48] patches -> [16, 16, 3] image."""
    return patches.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 3)


def pixel_positions() -> torch.Tensor:
    x = 2 * (torch.arange(SIDE) + 0.5) / SIDE - 1
    return x.repeat_interleave(SIDE)


class DensityDetector(nn.Module):
    def __init__(self, tau: float) -> None:
        super().__init__()
        self.decode = nn.Linear(QUERY, 3)
        self.presence_scale = 4.0
        self.abstain = nn.Parameter(torch.full((), -2.0))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.tau = float(tau)
        self.register_buffer("positions", pixel_positions(), persistent=False)
        members = []
        for column in range(SIDE):
            members.append((column == COLUMN_INDEX).nonzero().flatten())
        self.columns = members

    def forward(self, patches: torch.Tensor, query: torch.Tensor):
        pixels = unpatched(patches).reshape(PIXELS, 3)
        # The target must live in the pixel gamut: an unconstrained linear readout
        # drifted to a negative green channel, where no pixel can ever match, and
        # the loss escaped through the abstention constant instead.
        target = torch.sigmoid(self.decode(query))
        squared = ((pixels - target) ** 2).sum(-1)
        scores = -squared / self.tau
        weights = (scores - scores.max()).softmax(-1)
        centre = (weights * self.positions).sum()
        # Column evidence is the nearest pixel's distance in that column, on a
        # bounded scale. Abstention is a constant, so it cannot dominate without
        # earning it: for a present colour the best column sits at zero while an
        # absent colour leaves every column far away. A free presence
        # coefficient was tried first and drifted to "always abstain".
        column_min = torch.stack([squared[members].min() for members in self.columns])
        column_logits = -self.presence_scale * column_min
        null = self.abstain
        scale = self.log_scale.exp().clamp(1e-2, 1e2)
        shape = torch.stack([-scale * (centre - c).abs() for c in CENTRES] + [scale * null])
        dense = torch.cat([column_logits, null[None]])
        return shape, weights, dense


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


def build_arm(contract: dict, lam: float):
    seed = int(contract["candidate"]["seed"])
    torch.manual_seed(seed)
    model = make_model("native", seed)
    module = DensityDetector(contract["candidate"]["tau"])
    spec = contract["optimizer"]
    optimizer = torch.optim.Adam(
        list(module.parameters()),
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    return model, module, optimizer, lam


def query_of(model, example: dict) -> torch.Tensor:
    return model.text_units(example["public"]["request"]).mean(0)


def patches_of(example: dict) -> torch.Tensor:
    return example["public"]["observations"]["rgb"]


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


def dense_target(example: dict) -> int:
    column = oracle_column(example)
    return SIDE if column is None else column


def objective(model, module, batch: list[dict], lam: float):
    logits, dense, weights = [], [], []
    for example in batch:
        shape, attention, column_scores = module(patches_of(example), query_of(model, example))
        logits.append(shape)
        dense.append(column_scores)
        weights.append(attention)
    labels = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    loss = functional.cross_entropy(torch.stack(logits), labels)
    if lam:
        targets = torch.tensor([dense_target(example) for example in batch])
        loss = loss + lam * functional.cross_entropy(torch.stack(dense), targets)
    return loss, torch.stack(weights)


def accuracy(model, module, examples: list[dict]) -> dict:
    with torch.no_grad():
        correct = 0
        dense_correct = 0
        dense_total = 0
        entropies = []
        hits = total = 0
        abstain_unknown = abstain_other = 0
        for example in examples:
            shape, weights, column_scores = module(patches_of(example), query_of(model, example))
            label = example["target"].decode()
            correct += int(int(shape.argmax()) == LABELS.index(label))
            entropies.append(float(-(weights * weights.clamp_min(1e-12).log()).sum()))
            dense_total += 1
            dense_correct += int(int(column_scores.argmax()) == dense_target(example))
            if int(shape.argmax()) == 3:
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
        "dense_correct": dense_correct,
        "dense_accuracy": dense_correct / dense_total,
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
    model, module, optimizer, lam = build_arm(contract, 0.0)
    per_step = []
    for update in range(1, 4):
        batch = batch_at(data, contract["order"], contract["batch"], update)
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(model, module, batch, lam)
        loss.backward()
        optimizer.step()
        per_step.append(time.process_time() - started)
        if time.process_time() - cpu_start > contract["price_cap_cpu"]:
            raise BudgetStop("price_probe_budget_stop")
    started = time.process_time()
    scored = accuracy(model, module, data["fit"])
    eval_cpu = time.process_time() - started
    step_cpu = statistics.median(per_step)
    curve_cpu = 2 * max(contract["dose_points"]) * step_cpu + 10 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], max(curve_cpu * 1.1, 100.0)), 3)
    report = {
        "schema": "taiji-u2-p3-26-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.26",
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
            "arms": 2,
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


def decide(arms: dict, criteria: dict) -> str:
    sparse = arms["sparse"]["doses"][-1]["fit_accuracy"]
    dense = arms["dense"]["doses"][-1]["fit_accuracy"]
    gate, low = criteria["fit_gate"], criteria["fit_low"]
    if dense >= gate and sparse <= low:
        return "supervision_density_is_the_wall"
    if dense >= gate and sparse >= gate:
        return "both_learn"
    if dense <= low and sparse <= low:
        return "line_closed_matching_not_learnable"
    return "inconclusive_density"


def run_arm(contract: dict, data, arm: str, lam: float) -> dict:
    model, module, optimizer, lam = build_arm(contract, lam)
    doses = contract["dose_points"]
    root = Path(contract["run_root"]) / arm
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for update in range(1, max(doses) + 1):
        batch = batch_at(data, contract["order"], contract["batch"], update)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(model, module, batch, lam)
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
                "dense_accuracy_train": train["dense_accuracy"],
                "dense_accuracy_fit": fit["dense_accuracy"],
                "attention_entropy_fit": round(fit["mean_attention_entropy"], 6),
                "centroid_column_hits_fit": fit["centroid_column_hits"],
                "centroid_column_total_fit": fit["centroid_column_total"],
                "abstain_on_unknown_fit": fit["abstain_on_unknown"],
                "abstain_on_other_fit": fit["abstain_on_other"],
                "cpu": round(time.process_time() - started, 6),
            }
        )
    return {
        "arm": arm,
        "lambda": lam,
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
        raise FileExistsError("p3_26_run_already_started")
    root.mkdir(parents=True)
    arms = {}
    for arm, lam in (("sparse", 0.0), ("dense", float(contract["aux_weight"]))):
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_26_budget_stop")
        arms[arm] = run_arm(contract, data, arm, lam)
    verdict = decide(arms, contract["criteria"])
    report = {
        "schema": "taiji-u2-p3-26-density-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.26",
        "phase": "supervision_density",
        "verdict": verdict,
        "arms": arms,
        "tau_initial": contract["candidate"]["tau"],
        "aux_weight": contract["aux_weight"],
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
            "detector-only card; both arms use the P3.24 soft-centroid readout with tau "
            "initialised at 0.02; the dense arm adds a 17-way column cross-entropy against "
            "the public locator; U2 body frozen; no capability claim"
        ),
    }
    write_report(RUN_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "arms": {name: arm["doses"][-1] for name, arm in arms.items()},
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
                    "schema": "taiji-u2-p3-26-density-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.26",
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
                    "schema": "taiji-u2-p3-26-density-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.26",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
