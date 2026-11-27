"""P3.22 selection-as-output card: the answer is the column the query attends to.

P3.21 showed the readout could bypass selection: ``sum_i w_i (content_i + pos_i)``
keeps the whole scene, so the 4-way loss was minimised by memorising scene
fingerprints. On the final checkpoint the attention column matched the queried
object in 7/192 train and 0/24 fit cases.

This card removes the bypass. Attention runs over the sixteen *pixel columns*
(each column is the 48 values of that column across all rows), and the class
logits are the log of the attention mass falling in each class's columns, plus
a null slot for 未知. There is no other path to the logits, so being right
requires looking at the right column.

Arm B restores a scene-summary bypass (mean column feature -> logits) and is
otherwise identical. If A generalises and B memorises, the bypass is the cause.

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
from taiji.u2_task_episode import locate_public, position_word

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_22_selection_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_22_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_22_selection_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
NAMES_CLASS = {name: index for index, name in enumerate(LABELS)}
COLUMN_CLASS = [NAMES_CLASS[position_word(2 * (k + 0.5) / 16 - 1)] for k in range(16)]
COLUMN_MEMBERS = {c: [k for k in range(16) if COLUMN_CLASS[k] == c] for c in range(3)}
COLUMNS = 16
PIXELS = 48
QUERY = 8


class BudgetStop(RuntimeError):
    pass


def columns_of(patches: torch.Tensor) -> torch.Tensor:
    """[16, 48] patches -> [16 columns, 48 values]; column k is (block c=k//4, slot j=k%4)."""
    five = patches.reshape(4, 4, 4, 4, 3)
    return torch.stack([five[:, k // 4, :, k % 4, :].reshape(-1) for k in range(COLUMNS)])


class SelectionDetector(nn.Module):
    def __init__(self, pixel: int, query: int, width: int, bypass: bool) -> None:
        super().__init__()
        self.column = nn.Linear(pixel, width)
        self.query = nn.Linear(query, width, bias=False)
        self.null = nn.Linear(query, 1)
        self.bypass = nn.Linear(width, len(LABELS)) if bypass else None

    def forward(self, patches: torch.Tensor, query: torch.Tensor):
        hidden = torch.tanh(self.column(columns_of(patches)))
        scores = (hidden @ self.query(query)) / (hidden.shape[-1] ** 0.5)
        null = self.null(query).squeeze(-1)
        weights = torch.cat([scores, null[None]]).softmax(-1)
        mass = weights[:COLUMNS]
        logits = torch.stack(
            [torch.log(mass[[k for k in COLUMN_MEMBERS[c]]].sum() + 1e-9) for c in range(3)]
            + [torch.log(weights[COLUMNS] + 1e-9)]
        )
        if self.bypass is not None:
            logits = logits + self.bypass(hidden.mean(0))
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


def shared_init_digest(module: nn.Module) -> str:
    """Hash of every initial tensor except the bypass, so the arms can be compared."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if name.startswith("bypass"):
            continue
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def build_arm(contract: dict, bypass: bool):
    seed = int(contract["candidate"]["seed"])
    width = contract["candidate"]["width"]
    torch.manual_seed(seed)
    model = make_model("native", seed)
    module = SelectionDetector(PIXELS, QUERY, width, bypass)
    spec = contract["optimizer"]
    optimizer = torch.optim.Adam(
        list(module.parameters()),
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    return model, module, optimizer


def unpatched(patches: torch.Tensor) -> torch.Tensor:
    """Inverse of taiji.u2_task_sensors.encode: [16, 48] patches -> [16, 16, 3] image."""
    return patches.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(16, 16, 3)


_COLUMN_TRUTH: dict[str, int | None] = {}


def oracle_column(example: dict) -> int | None:
    key = example["id"]
    if key in _COLUMN_TRUTH:
        return _COLUMN_TRUTH[key]
    image = unpatched(example["public"]["observations"]["rgb"]) * 255.0
    entity = example["public"]["request"].decode()[: -len("现在在哪里？")]
    position = locate_public(image, entity)
    column = None if position is None else min(15, max(0, int((position + 1) / 2 * 16)))
    _COLUMN_TRUTH[key] = column
    return column


def patches_of(example: dict) -> torch.Tensor:
    return example["public"]["observations"]["rgb"]


def query_of(model, example: dict) -> torch.Tensor:
    return model.text_units(example["public"]["request"]).mean(0)


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
        null_on_unknown = 0
        null_total = 0
        false_abstain = 0
        for example in examples:
            logits, weights = module(patches_of(example), query_of(model, example))
            label = example["target"].decode()
            correct += int(int(logits.argmax()) == LABELS.index(label))
            entropies.append(float(-(weights * weights.clamp_min(1e-12).log()).sum()))
            null_total += int(int(weights.argmax()) == COLUMNS)
            null_on_unknown += int(label == "未知" and int(weights.argmax()) == COLUMNS)
            false_abstain += int(label != "未知" and int(weights.argmax()) == COLUMNS)
            if label != "未知":
                truth = oracle_column(example)
                if truth is not None:
                    column_total += 1
                    column_hits += int(int(weights[:COLUMNS].argmax()) == truth)
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "mean_attention_entropy": sum(entropies) / len(entropies),
        "attention_column_hits": column_hits,
        "attention_column_total": column_total,
        "null_on_unknown": null_on_unknown,
        "null_total": null_total,
        "false_abstain": false_abstain,
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
    model, module, optimizer = build_arm(contract, bypass=False)
    batch_size = contract["batch"]
    per_step = []
    for update in range(1, 4):
        batch = batch_at(data, contract["order"], batch_size, update)
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
    total_steps = 2 * max(contract["dose_points"])
    curve_cpu = total_steps * step_cpu + 10 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], max(curve_cpu * 1.1, 100.0)), 3)
    report = {
        "schema": "taiji-u2-p3-22-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.22",
        "phase": "price_probe",
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "measurements": {
            "per_step_cpu": [round(value, 6) for value in per_step],
            "per_step_cpu_median": round(step_cpu, 6),
            "eval_cpu": round(eval_cpu, 6),
            "eval_accuracy_at_zero": scored["accuracy"],
        },
        "derived": {
            "curve_updates": total_steps,
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


def run_arm(contract: dict, data, arm: str, bypass: bool) -> dict:
    model, module, optimizer = build_arm(contract, bypass)
    init_digest = shared_init_digest(module)
    order = contract["order"]
    batch_size = contract["batch"]
    doses = contract["dose_points"]
    root = Path(contract["run_root"]) / arm
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for update in range(1, max(doses) + 1):
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
                "attention_column_hits_fit": fit["attention_column_hits"],
                "attention_column_total_fit": fit["attention_column_total"],
                "null_on_unknown_fit": fit["null_on_unknown"],
                "false_abstain_fit": fit["false_abstain"],
                "cpu": round(time.process_time() - started, 6),
            }
        )
    return {
        "arm": arm,
        "bypass": bypass,
        "parameters": sum(p.numel() for p in module.parameters()),
        "shared_init_digest": init_digest,
        "doses": results,
    }


def decide(arms: dict, criteria: dict) -> str:
    selection = arms["selection"]["doses"][-1]
    bypass = arms["bypass"]["doses"][-1]
    a_train, a_fit, b_fit = (
        selection["train_accuracy"],
        selection["fit_accuracy"],
        bypass["fit_accuracy"],
    )
    if a_fit >= criteria["fit_gate"]:
        return "bypass_is_the_cause" if b_fit <= criteria["fit_low"] else "both_generalise"
    if a_fit <= criteria["fit_low"]:
        if a_train < criteria["train_gate"]:
            return "selection_cannot_fit"
        return "selection_readout_insufficient"
    return "inconclusive_selection"


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    root = Path(contract["run_root"])
    if root.exists() or RUN_REPORT.exists():
        raise FileExistsError("p3_22_run_already_started")
    root.mkdir(parents=True)
    arms = {}
    for arm, bypass in (("selection", False), ("bypass", True)):
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_22_budget_stop")
        arms[arm] = run_arm(contract, data, arm, bypass)
    if arms["selection"]["shared_init_digest"] != arms["bypass"]["shared_init_digest"]:
        raise ValueError("p3_22_arms_do_not_share_initial_weights")
    verdict = decide(arms, contract["criteria"])
    report = {
        "schema": "taiji-u2-p3-22-selection-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.22",
        "phase": "selection_as_output",
        "verdict": verdict,
        "arms": arms,
        "column_members": {str(c): COLUMN_MEMBERS[c] for c in range(3)},
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
            "detector-only card; class logits are the attention mass per column class plus a "
            "null slot; arm B adds a scene-summary bypass; U2 body frozen; no capability claim"
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
                    "schema": "taiji-u2-p3-22-selection-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.22",
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
                    "schema": "taiji-u2-p3-22-selection-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.22",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
