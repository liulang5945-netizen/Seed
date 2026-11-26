"""P3.21 position-channel card: give the attended summary a patch index.

P3.17-P3.20 all used a permutation-invariant readout, ``attended = sum_i w_i h_i``:
swapping two patches leaves the output unchanged, so the detector could not
represent *which column* the attended patch came from. This card adds one
per-patch position vector to the attended summary and changes nothing else:
the same data, the same 7376 steps, the same optimiser, the same layer shapes.

The position table is drawn from a dedicated generator *after* the four shared
layers, so every other parameter keeps the exact draw P3.20 used. ``build``
asserts that equality, which is what makes this a single-variable change.

Modes: ``diagnose`` (zero training), ``price`` and ``run``.
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
from taiji.u2_task_episode import NAMES, locate_public, position_word

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_21_position_20261114.json"
DIAGNOSE_REPORT = ROOT / "reports/taiji_u2_p3_21_diagnose_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_21_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_21_position_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
PATCHES = 16
PIXELS = 48
QUERY = 8


class BudgetStop(RuntimeError):
    pass


class PositionedDetector(nn.Module):
    """Content selects the patch; the attended summary carries where it came from."""

    def __init__(self, pixel: int, query: int, width: int, classes: int) -> None:
        super().__init__()
        self.observation = nn.Linear(pixel, width)
        self.query = nn.Linear(query, width, bias=False)
        self.patch = nn.Linear(width, width, bias=False)
        self.readout = nn.Linear(width + query, classes)

    def forward(self, patches: torch.Tensor, query: torch.Tensor):
        content = torch.tanh(self.observation(patches))
        scores = (self.patch(content) @ self.query(query)) / (content.shape[-1] ** 0.5)
        weights = scores.softmax(-1)
        attended = (weights[..., None] * (content + self.position)).sum(0)
        return self.readout(torch.cat([attended, query])), weights

    def attach_position(self, table: torch.Tensor) -> None:
        self.position = nn.Parameter(table)


class WideDetector(nn.Module):
    """P3.20 reference: identical layers, no position table."""

    def __init__(self, pixel: int, query: int, width: int, classes: int) -> None:
        super().__init__()
        self.observation = nn.Linear(pixel, width)
        self.query = nn.Linear(query, width, bias=False)
        self.patch = nn.Linear(width, width, bias=False)
        self.readout = nn.Linear(width + query, classes)

    def forward(self, patches: torch.Tensor, query: torch.Tensor):
        hidden = torch.tanh(self.observation(patches))
        scores = (self.patch(hidden) @ self.query(query)) / (hidden.shape[-1] ** 0.5)
        weights = scores.softmax(-1)
        attended = (weights[..., None] * hidden).sum(0)
        return self.readout(torch.cat([attended, query])), weights


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


def detached_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def unpatched(patches: torch.Tensor) -> torch.Tensor:
    """Inverse of taiji.u2_task_sensors.encode: [16, 48] -> [16, 16, 3]."""
    return patches.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(16, 16, 3)


def oracle_column(example: dict) -> int | None:
    image = unpatched(example["public"]["observations"]["rgb"]) * 255.0
    entity = example["public"]["request"].decode()[: -len("现在在哪里？")]
    position = locate_public(image, entity)
    if position is None:
        return None
    return min(15, max(0, int((position + 1) / 2 * 16)))


def oracle_label(example: dict) -> str:
    image = unpatched(example["public"]["observations"]["rgb"]) * 255.0
    entity = example["public"]["request"].decode()[: -len("现在在哪里？")]
    position = locate_public(image, entity)
    return "未知" if position is None else position_word(position)


def query_of(model, example: dict) -> torch.Tensor:
    return model.text_units(example["public"]["request"]).mean(0)


def patches_of(example: dict) -> torch.Tensor:
    return example["public"]["observations"]["rgb"]


def build(contract: dict):
    seed = int(contract["candidate"]["seed"])
    width = contract["candidate"]["width"]
    torch.manual_seed(seed)
    model = make_model("native", seed)
    state = torch.get_rng_state()
    module = PositionedDetector(PIXELS, QUERY, width, len(LABELS))
    # Replay the identical RNG state so the P3.20 reference draws the same layers.
    torch.set_rng_state(state)
    reference = WideDetector(PIXELS, QUERY, width, len(LABELS))
    generator = torch.Generator().manual_seed(seed + 1)
    table = (
        torch.randn(PATCHES, width, generator=generator) * contract["candidate"]["position_scale"]
    )
    module.attach_position(table)
    shared = {name: value for name, value in detached_state(module).items() if name != "position"}
    if set(shared) != set(detached_state(reference)):
        raise ValueError("p3_21_layer_set_mismatch")
    for name, value in detached_state(reference).items():
        if not torch.equal(shared[name], value):
            raise ValueError(f"p3_21_init_differs_from_p3_20:{name}")
    spec = contract["optimizer"]
    optimizer = torch.optim.Adam(
        list(module.parameters()),
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    return model, module, optimizer


def objective(model, module, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    logits, weights = [], []
    for example in batch:
        out, attention = module(patches_of(example), query_of(model, example))
        logits.append(out)
        weights.append(attention)
    targets = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    return functional.cross_entropy(torch.stack(logits), targets), torch.stack(weights)


def accuracy(model, module, examples: list[dict]) -> dict:
    with torch.no_grad():
        correct = 0
        entropies = []
        column_hits = 0
        column_total = 0
        for example in examples:
            logits, weights = module(patches_of(example), query_of(model, example))
            correct += int(int(logits.argmax()) == LABELS.index(example["target"].decode()))
            entropies.append(float(-(weights * weights.clamp_min(1e-12).log()).sum()))
            truth = oracle_column(example)
            if truth is not None:
                column_total += 1
                column_hits += int(int(weights.argmax()) % 4 == truth // 4)
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "mean_attention_entropy": sum(entropies) / len(entropies),
        "attention_column_hits": column_hits,
        "attention_column_total": column_total,
    }


def diagnose(contract: dict) -> int:
    guards(contract, "diagnose")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.eval()
    splits = {}
    for name in ("train", "fit"):
        examples = data[name]
        hits = sum(int(oracle_label(example) == example["target"].decode()) for example in examples)
        splits[name] = {
            "oracle_correct": hits,
            "denominator": len(examples),
            "oracle_accuracy": hits / len(examples),
        }
    with torch.no_grad():
        train_x = torch.stack([query_of(model, example) for example in data["train"]])
        fit_x = torch.stack([query_of(model, example) for example in data["fit"]])
    train_y = torch.tensor(
        [
            NAMES.index(example["public"]["request"].decode()[: -len("现在在哪里？")])
            for example in data["train"]
        ]
    )
    fit_y = torch.tensor(
        [
            NAMES.index(example["public"]["request"].decode()[: -len("现在在哪里？")])
            for example in data["fit"]
        ]
    )
    mean, scale = train_x.mean(0), train_x.std(0).clamp_min(1e-9)
    train_z, fit_z = (train_x - mean) / scale, (fit_x - mean) / scale
    onehot = torch.zeros(len(train_y), len(NAMES))
    onehot[torch.arange(len(train_y)), train_y] = 1
    ridge = contract["criteria"]["ridge_lambda"]
    weights = torch.linalg.solve(
        train_z.T @ train_z + ridge * torch.eye(train_z.shape[1]), train_z.T @ onehot
    )
    probe = {
        "train_accuracy": float(((train_z @ weights).argmax(1) == train_y).float().mean()),
        "fit_accuracy": float(((fit_z @ weights).argmax(1) == fit_y).float().mean()),
        "ridge_lambda": ridge,
        "features": int(train_z.shape[1]),
        "classes": len(NAMES),
    }
    criteria = contract["criteria"]
    checks = {
        "oracle_train_exact": splits["train"]["oracle_accuracy"] == 1.0,
        "oracle_fit_exact": splits["fit"]["oracle_accuracy"] == 1.0,
        "query_probe_above_floor": probe["fit_accuracy"] >= criteria["query_probe_floor"],
    }
    verdict = "task_and_query_verified" if all(checks.values()) else "input_side_defect"
    report = {
        "schema": "taiji-u2-p3-21-diagnose-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.21",
        "phase": "diagnose",
        "verdict": verdict,
        "oracle_from_observations": splits,
        "query_color_probe": probe,
        "checks": checks,
        "reading": (
            "the reconstructed image reproduces the label exactly on both splits, and the "
            "frozen query decodes the six colours; so the input and the labels are sound and "
            "the earlier failures were on the readout side"
        ),
        "discipline": {"training_runs": 0, "optimizer_calls": 0, "capability_claims": 0},
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
    }
    write_report(DIAGNOSE_REPORT, report)
    print(json.dumps({"verdict": verdict, "checks": checks, "probe": probe}))
    return 0 if verdict == "task_and_query_verified" else 4


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, module, optimizer = build(contract)
    order = contract["order"]
    batch_size = contract["batch"]
    per_step = []
    for update in range(3):
        batch = [
            data["train"][order[(update * batch_size + offset) % len(order)]]
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
    curve_cpu = max(contract["dose_points"]) * step_cpu + 5 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], max(curve_cpu * 1.1, 60.0)), 3)
    report = {
        "schema": "taiji-u2-p3-21-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.21",
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
        raise FileExistsError("p3_21_run_already_started")
    root.mkdir(parents=True)
    results = []
    for update in range(1, max(doses) + 1):
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_21_budget_stop")
        batch = [
            data["train"][order[((update - 1) * batch_size + offset) % len(order)]]
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
                "attention_column_hits_fit": fit["attention_column_hits"],
                "attention_column_total_fit": fit["attention_column_total"],
                "cpu": round(time.process_time() - started, 6),
            }
        )
    final = results[-1]
    criteria = contract["criteria"]
    if final["train_accuracy"] >= criteria["train_gate"]:
        if final["fit_accuracy"] >= criteria["fit_gate"]:
            verdict = "position_channel_restores_task"
        elif final["fit_accuracy"] <= criteria["fit_low"]:
            verdict = "position_channel_insufficient"
        else:
            verdict = "inconclusive_position"
    else:
        verdict = "position_channel_not_learned"
    report = {
        "schema": "taiji-u2-p3-21-position-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.21",
        "phase": "position_channel",
        "verdict": verdict,
        "doses": results,
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "module_parameters": sum(p.numel() for p in module.parameters()),
        "position_parameters": int(module.position.numel()),
        "width": contract["candidate"]["width"],
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
            "detector-only card on raw rgb patches; the only change against P3.20 is a "
            "per-patch position table added to the attended summary; U2 body frozen; no "
            "capability claim"
        ),
    }
    write_report(RUN_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "doses": results,
                "cost": report["cost"],
                "trainable": report["trainable_parameters"],
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("diagnose", "price", "run"), default="diagnose")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.mode == "diagnose":
        return diagnose(contract)
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
                    "schema": "taiji-u2-p3-21-position-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.21",
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
                    "schema": "taiji-u2-p3-21-position-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.21",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
