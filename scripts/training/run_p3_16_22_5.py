"""P3.16 encoder-level position supervision on the P3.10 task.

Same 194-tensor initialization as P3.10, plus a zero-initialized linear head on
the pooled observation encoding (the encoder output, before H binding and before
the readout context). The training loss is the byte NLL plus lambda times a
four-way cross-entropy on that head.

The single change relative to P3.14 is the attachment point: encoder output
instead of readout context. Modes: ``price`` and ``run``.
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
from scripts.training.first_a_support_22_5 import evaluate, make_model, save_learner, task_loss
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_task_executor import writer_lease

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_16_encoder_supervision_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_16_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_16_encoder_supervision_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
ORDER = ("rgb", "audio", "proprio", "outcome")
GATE = 0.80
LOW = 0.35


class BudgetStop(RuntimeError):
    pass


class EncoderHead(nn.Module):
    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, classes)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        return self.projection(encoding)


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
    model = make_model("native", int(contract["candidate"]["seed"]))
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == contract["candidate"]["parameters"]
    )
    assert len(list(model.named_parameters())) == contract["candidate"]["tensors"]
    head = EncoderHead(contract["candidate"]["encoding_width"], len(LABELS))
    spec = contract["optimizer"]

    def make_optimizer(parameters):
        return torch.optim.Adam(
            parameters,
            lr=spec["lr"],
            betas=tuple(spec["betas"]),
            eps=spec["eps"],
            weight_decay=spec["weight_decay"],
        )

    return model, head, make_optimizer(model.parameters()), make_optimizer(head.parameters())


def batch_at(data: dict, order: list[int], cursor: int) -> list[dict]:
    return [data["train"][order[(cursor + offset) % 64]] for offset in range(2)]


def encoding_of(model, example: dict) -> torch.Tensor:
    public = example["public"]
    tokens = model.encode(public["observations"], public["coordinates"], public["outcome"])
    return torch.cat([tokens[name].mean(0) for name in ORDER if name in tokens])


def objective(model, head, batch: list[dict], lam: float) -> torch.Tensor:
    byte_loss = task_loss(model, batch)
    logits = torch.stack([head(encoding_of(model, example)) for example in batch])
    targets = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    return byte_loss + lam * functional.cross_entropy(logits, targets)


def head_accuracy(model, head, examples: list[dict]) -> float:
    with torch.no_grad():
        hits = 0
        for example in examples:
            logits = head(encoding_of(model, example))
            hits += int(int(logits.argmax()) == LABELS.index(example["target"].decode()))
    return hits / len(examples)


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, head, body_optimizer, head_optimizer = build(contract)
    order = contract["order"]
    lam = contract["criteria"]["lambda"]
    per_step, losses = [], []
    for update in range(3):
        batch = batch_at(data, order, 2 * update)
        started = time.process_time()
        body_optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        loss = objective(model, head, batch, lam)
        loss.backward()
        body_optimizer.step()
        head_optimizer.step()
        per_step.append(time.process_time() - started)
        losses.append(round(float(loss.detach()), 6))
        if time.process_time() - cpu_start > contract["price_cap_cpu"]:
            raise BudgetStop("price_probe_budget_stop")
    eval_started = time.process_time()
    initial = evaluate(model, data["fit"])
    eval_cpu = time.process_time() - eval_started
    step_cpu = statistics.median(per_step)
    curve_cpu = 1844 * step_cpu + 5 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], curve_cpu * 1.1), 3)
    report = {
        "schema": "taiji-u2-p3-16-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.16",
        "phase": "price_probe",
        "candidate": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "tensors": len(list(model.named_parameters())),
            "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        },
        "measurements": {
            "steps": 3,
            "per_step_cpu": [round(value, 6) for value in per_step],
            "per_step_cpu_median": round(step_cpu, 6),
            "losses": losses,
            "eval_cpu": round(eval_cpu, 6),
            "eval_correct_at_zero": initial["correct"],
            "eval_denominator": initial["denominator"],
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
            "ledger_written": False,
            "optimizer_calls": 3,
            "new_parent": False,
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
                "head_parameters": report["candidate"]["head_parameters"],
            }
        )
    )
    return 0 if report["verdict"] == "price_measured" else 4


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    root = Path(contract["run_root"])
    if root.exists() or RUN_REPORT.exists():
        raise FileExistsError("p3_16_run_already_started")
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, head, body_optimizer, head_optimizer = build(contract)
    order = contract["order"]
    doses = contract["dose_points"]
    lam = contract["criteria"]["lambda"]
    root.mkdir(parents=True)
    results = []

    def guard() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_16_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("p3_16_wall_stop")

    with writer_lease(root / "run.lock"):
        guard()
        started = time.process_time()
        zero = evaluate(model, data["fit"])
        with (root / "run.ndjson").open("ab") as stream:
            stream.write(
                (
                    json.dumps(
                        {
                            "event": "evaluate",
                            "update": 0,
                            "correct": zero["correct"],
                            "denominator": zero["denominator"],
                            "cpu": round(time.process_time() - started, 6),
                        }
                    )
                    + "\n"
                ).encode()
            )
        cursor = 0
        for update in range(1, max(doses) + 1):
            guard()
            body_optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)
            loss = objective(model, head, batch_at(data, order, cursor), lam)
            loss.backward()
            body_optimizer.step()
            head_optimizer.step()
            model.parameter_revision += 1
            cursor += 2
            if update not in doses:
                continue
            started = time.process_time()
            bundle = root / f"dose-{update:04d}.bin"
            digest = save_learner(
                bundle,
                model,
                body_optimizer,
                "native",
                update,
                cursor,
                order,
                contract["data_digest"],
            )
            scored = evaluate(model, data["fit"])
            train = evaluate(model, data["train"])
            head_fit = head_accuracy(model, head, data["fit"])
            head_train = head_accuracy(model, head, data["train"])
            torch.save(head.state_dict(), root / f"head-{update:04d}.pt")
            with (root / "run.ndjson").open("ab") as stream:
                stream.write(
                    (
                        json.dumps(
                            {
                                "event": "dose",
                                "update": update,
                                "correct": scored["correct"],
                                "train_correct": train["correct"],
                                "head_fit_accuracy": head_fit,
                                "head_train_accuracy": head_train,
                                "bundle_digest": digest,
                                "elapsed_cpu": round(time.process_time() - cpu_start, 6),
                                "cpu": round(time.process_time() - started, 6),
                            }
                        )
                        + "\n"
                    ).encode()
                )
            results.append(
                {
                    "update": update,
                    "correct": scored["correct"],
                    "accuracy": scored["accuracy"],
                    "train_correct": train["correct"],
                    "train_accuracy": train["accuracy"],
                    "head_fit_accuracy": head_fit,
                    "head_train_accuracy": head_train,
                    "distinct_outputs": len({row["output_hex"] for row in scored["rows"]}),
                    "bundle_digest": digest,
                }
            )
        final = results[-1]
        encoder_learned = final["head_train_accuracy"] >= contract["criteria"]["encoder_gate"]
        if encoder_learned and final["accuracy"] >= GATE:
            verdict = "marginal_solution_broken"
        elif encoder_learned:
            verdict = "encoder_can_carry_position_readout_still_constant"
        elif final["head_train_accuracy"] <= LOW and final["head_fit_accuracy"] <= LOW:
            verdict = "encoder_cannot_carry_position"
        else:
            verdict = "inconclusive_encoder_supervision"
        report = {
            "schema": "taiji-u2-p3-16-encoder-v1",
            "date": "2026-11-14",
            "card": "22.5.P3.16",
            "phase": "encoder_supervision",
            "verdict": verdict,
            "lambda": lam,
            "doses": results,
            "zero": {
                "correct": zero["correct"],
                "denominator": zero["denominator"],
                "accuracy": zero["accuracy"],
            },
            "head": {
                "parameters": sum(parameter.numel() for parameter in head.parameters()),
                "initialised_zero": True,
                "attach_point": "pooled observation encoding (encoder output)",
            },
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
            "scope": "single-seed encoder-supervised arm on synthetic legal inputs; no natural-language, no capability claim",
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
                    "schema": "taiji-u2-p3-16-encoder-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.16",
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
                    "schema": "taiji-u2-p3-16-encoder-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.16",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
