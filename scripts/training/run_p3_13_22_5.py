"""P3.13 accepted-update transplant on the P3.10 task.

Same candidate (22,742 parameters), same seed, same data order and dose as the
P3.10 plain-Adam curve; the only change is that every parameter step goes
through the verified train-objective acceptance search (the recipe that reached
179/180 in the P3.8 calibration).

Modes: ``price`` (three in-memory accepted steps plus one evaluation) and
``run`` (requires a frozen contract with an owner training authority).
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

from scripts.training.check_report_run_conditions import write_report
from scripts.training.credit_update_22_5 import accepted_adam_step
from scripts.training.first_a_support_22_5 import evaluate, make_model, save_learner, task_loss
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_task_executor import writer_lease

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_13_acceptance_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_13_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p3_13_acceptance_20261114.json"
GATE = 0.80
LOW = 0.35


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    if sha_of(Path(__file__)) != contract["source_pins"]["runner"]:
        raise ValueError("runner_source_drift")
    if (
        sha_of(ROOT / "scripts/training/credit_update_22_5.py")
        != contract["source_pins"]["credit_update"]
    ):
        raise ValueError("credit_update_source_drift")
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
    spec = contract["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    return model, optimizer


def batch_at(data: dict, order: list[int], cursor: int) -> list[dict]:
    return [data["train"][order[(cursor + offset) % 64]] for offset in range(2)]


def one_step(model, optimizer, batch, counters, tolerance):
    return accepted_adam_step(
        list(model.parameters()),
        optimizer,
        lambda: task_loss(model, batch),
        counters,
        tolerance=tolerance,
    )


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, optimizer = build(contract)
    order = contract["order"]
    counters = {"optimizer_steps": 0, "accepted_parameter_updates": 0}
    per_step, results = [], []
    for update in range(3):
        started = time.process_time()
        result = one_step(
            model,
            optimizer,
            batch_at(data, order, 2 * update),
            counters,
            contract["criteria"]["tolerance"],
        )
        per_step.append(time.process_time() - started)
        results.append(
            {
                "accepted": result["accepted"],
                "scale": result["scale"],
                "evaluations": result["evaluations"],
            }
        )
        if time.process_time() - cpu_start > contract["price_cap_cpu"]:
            raise BudgetStop("price_probe_budget_stop")
    eval_started = time.process_time()
    initial = evaluate(model, data["fit"])
    eval_cpu = time.process_time() - eval_started
    step_cpu = statistics.median(per_step)
    curve_cpu = 1844 * step_cpu + 5 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], curve_cpu * 1.1), 3)
    report = {
        "schema": "taiji-u2-p3-13-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.13",
        "phase": "price_probe",
        "steps": results,
        "measurements": {
            "per_step_cpu": [round(value, 6) for value in per_step],
            "per_step_cpu_median": round(step_cpu, 6),
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
            "optimizer_calls": counters["optimizer_steps"],
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
        "scope": "pricing only; three in-memory accepted steps are discarded, no capability claim",
    }
    write_report(PRICE_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cap_cpu": cap,
                "curve_cpu": round(curve_cpu, 3),
                "steps": results,
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
        raise FileExistsError("p3_13_run_already_started")
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, optimizer = build(contract)
    order = contract["order"]
    doses = contract["dose_points"]
    root.mkdir(parents=True)
    counters = {"optimizer_steps": 0, "accepted_parameter_updates": 0}
    results = []
    accepted = rejected = 0

    def guard() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_13_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("p3_13_wall_stop")

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
            result = one_step(
                model,
                optimizer,
                batch_at(data, order, cursor),
                counters,
                contract["criteria"]["tolerance"],
            )
            if result["accepted"]:
                accepted += 1
                model.parameter_revision += 1
            else:
                rejected += 1
            cursor += 2
            if update not in doses:
                continue
            started = time.process_time()
            bundle = root / f"dose-{update:04d}.bin"
            digest = save_learner(
                bundle, model, optimizer, "native", update, cursor, order, contract["data_digest"]
            )
            scored = evaluate(model, data["fit"])
            train = evaluate(model, data["train"])
            with (root / "run.ndjson").open("ab") as stream:
                stream.write(
                    (
                        json.dumps(
                            {
                                "event": "dose",
                                "update": update,
                                "correct": scored["correct"],
                                "train_correct": train["correct"],
                                "bundle_digest": digest,
                                "accepted": accepted,
                                "rejected": rejected,
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
                    "distinct_outputs": len({row["output_hex"] for row in scored["rows"]}),
                    "bundle_digest": digest,
                }
            )
        final_accuracy = results[-1]["accuracy"]
        final_train = results[-1]["train_accuracy"]
        if final_accuracy >= GATE:
            verdict = "marginal_solution_broken"
        elif final_accuracy <= LOW and final_train <= LOW:
            verdict = "acceptance_alone_insufficient"
        else:
            verdict = "partial_movement_without_gate"
        report = {
            "schema": "taiji-u2-p3-13-acceptance-v1",
            "date": "2026-11-14",
            "card": "22.5.P3.13",
            "phase": "acceptance_transplant",
            "verdict": verdict,
            "doses": results,
            "zero": {
                "correct": zero["correct"],
                "denominator": zero["denominator"],
                "accuracy": zero["accuracy"],
            },
            "acceptance": {
                "accepted": accepted,
                "rejected": rejected,
                "optimizer_steps": counters["optimizer_steps"],
            },
            "control": contract["control_arm"],
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
            "scope": "single-seed accepted-update arm on synthetic legal inputs; no natural-language, no capability claim",
        }
        write_report(RUN_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "doses": results,
                "acceptance": report["acceptance"],
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
                    "schema": "taiji-u2-p3-13-acceptance-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.13",
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
                    "schema": "taiji-u2-p3-13-acceptance-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.13",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
