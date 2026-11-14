"""P3.27 prototype-mixture card: make the selection a differentiable parameter.

P3.17-P3.26 closed with a structural diagnosis: the answer depends on a discrete
selection, every hand-built readout either saturated or took a cheaper escape,
and the centroid is piecewise constant in the selection variable.

This card replaces the discrete switch with a parameterised one:

    p          = softmax(W q + b)                       six colours
    w_c,i      = softmax_i(-||pixel_i - proto_c||^2/t)  attention of colour c
    centroid_c = sum_i w_c,i x_i
    centre     = sum_c p_c centroid_c

and the answer is the class of ``centre``. Selection is a soft mixture of learned
prototypes rather than an argmax; abstention reads the mixture's nearest-prototype
distance, so it is earned rather than free.

Two arms, because a first trial showed the six-way classifier staying at chance
while the query provably decodes colour (ridge probe 1.000/1.000):

  outcome  the 4-way answer only
  colour   the 4-way answer plus the colour the request names, which is a public
           intermediate label inside the request string itself

The colour arm answers whether the mapping is representable and learnable at all;
the outcome arm answers whether outcome supervision alone reaches it.

Verdicts (frozen before any reading):
  outcome fit >= fit_gate                           -> prototype_mixture_generalises
  colour fit >= fit_gate and outcome fit <= fit_low -> colour_supervision_unlocks_mixture
  colour colour-acc >= colour_gate                  -> colour_learned_centroid_path_broken
  colour colour-acc <= colour_low                   -> colour_head_cannot_learn
  otherwise                                         -> inconclusive_prototype

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
from taiji.u2_task_episode import NAMES, locate_public

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_27_prototype_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_27_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_27_prototype_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
CENTRES = [-2 / 3, 0.0, 2 / 3]
QUERY = 8
SIDE = 16
PIXELS = SIDE * SIDE
COLOURS = 6


class BudgetStop(RuntimeError):
    pass


def unpatched(patches: torch.Tensor) -> torch.Tensor:
    """[16, 48] patches -> [16, 16, 3] image."""
    return patches.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 3)


def pixel_positions() -> torch.Tensor:
    x = 2 * (torch.arange(SIDE) + 0.5) / SIDE - 1
    return x.repeat_interleave(SIDE)


class PrototypeDetector(nn.Module):
    def __init__(self, tau: float, seed: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(QUERY, COLOURS)
        generator = torch.Generator().manual_seed(seed)
        self.prototype = nn.Parameter(torch.rand(COLOURS, 3, generator=generator))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
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
        centre = (colour * centroids).sum()
        null = self.abstain + self.presence * (colour * nearest).sum()
        scale = self.log_scale.exp().clamp(1e-2, 1e2)
        shape = torch.stack([-scale * (centre - c).abs() for c in CENTRES] + [scale * null])
        return shape, colour, colour_logits, centroids, null


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
    module = PrototypeDetector(contract["candidate"]["tau"], seed + 7)
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


def colour_index(example: dict) -> int:
    return NAMES.index(example["public"]["request"].decode()[: -len("现在在哪里？")])


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


def objective(model, module, batch: list[dict], lam: float):
    shapes, colours = [], []
    for example in batch:
        shape, _, colour_logits, _, _ = module(patches_of(example), query_of(model, example))
        shapes.append(shape)
        colours.append(colour_logits)
    labels = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    loss = functional.cross_entropy(torch.stack(shapes), labels)
    if lam:
        colour_targets = torch.tensor([colour_index(example) for example in batch])
        loss = loss + lam * functional.cross_entropy(torch.stack(colours), colour_targets)
    return loss


def accuracy(model, module, examples: list[dict]) -> dict:
    with torch.no_grad():
        correct = 0
        colour_correct = 0
        hits = total = 0
        abstain_unknown = abstain_other = 0
        for example in examples:
            shape, colour, _, centroids, _ = module(patches_of(example), query_of(model, example))
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
                    chosen = float((colour * centroids).sum())
                    column = min(15, max(0, int((chosen + 1) / 2 * 16)))
                    hits += int(column == truth)
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "colour_correct": colour_correct,
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
    model, module, optimizer, lam = build_arm(contract, 0.0)
    per_step = []
    for update in range(1, 4):
        batch = batch_at(data, contract["order"], contract["batch"], update)
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, module, batch, lam)
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
        "schema": "taiji-u2-p3-27-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.27",
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
    outcome = arms["outcome"]["doses"][-1]
    colour = arms["colour"]["doses"][-1]
    gate, low = criteria["fit_gate"], criteria["fit_low"]
    if outcome["fit_accuracy"] >= gate:
        return "prototype_mixture_generalises"
    if colour["fit_accuracy"] >= gate and outcome["fit_accuracy"] <= low:
        return "colour_supervision_unlocks_mixture"
    if colour["colour_accuracy_fit"] >= criteria["colour_gate"]:
        return "colour_learned_centroid_path_broken"
    if colour["colour_accuracy_fit"] <= criteria["colour_low"]:
        return "colour_head_cannot_learn"
    return "inconclusive_prototype"


def run_arm(contract: dict, data, arm: str, lam: float) -> dict:
    model, module, optimizer, lam = build_arm(contract, lam)
    doses = contract["dose_points"]
    root = Path(contract["run_root"]) / arm
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for update in range(1, max(doses) + 1):
        batch = batch_at(data, contract["order"], contract["batch"], update)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, module, batch, lam)
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
    return {
        "arm": arm,
        "lambda_colour": lam,
        "parameters": sum(p.numel() for p in module.parameters()),
        "prototypes": [
            [round(float(value), 4) for value in row]
            for row in torch.sigmoid(module.prototype).detach()
        ],
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
        raise FileExistsError("p3_27_run_already_started")
    root.mkdir(parents=True)
    arms = {}
    for arm, lam in (("outcome", 0.0), ("colour", float(contract["aux_weight"]))):
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_27_budget_stop")
        arms[arm] = run_arm(contract, data, arm, lam)
    verdict = decide(arms, contract["criteria"])
    report = {
        "schema": "taiji-u2-p3-27-prototype-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.27",
        "phase": "prototype_mixture",
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
            "detector-only card; six learned colour prototypes plus a six-way query "
            "classifier; the answer is the class of the mixture centroid; arm colour also "
            "sees the colour named in the request; U2 body frozen; no capability claim"
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
                    "schema": "taiji-u2-p3-27-prototype-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.27",
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
                    "schema": "taiji-u2-p3-27-prototype-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.27",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
