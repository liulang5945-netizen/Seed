"""P3.10 gate-dose curve: price probe and the single-seed dose curve.

Two modes share one code path for initialization, updates and evaluation:

* ``--mode price``  three in-memory Adam calls plus one fit32 evaluation; writes
  only the price report, no run root, no checkpoint, no parent.
* ``--mode run``    requires a frozen contract with an owner training authority;
  trains to the frozen dose points and writes one bundle per point.

No check/final/confirm read, no capability claim, no gate value change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import (
    evaluate,
    load_learner,
    make_model,
    save_learner,
    task_loss,
)
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_task_executor import writer_lease

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_10_execution_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p3_10_price_20261114.json"
CURVE_REPORT = ROOT / "reports/taiji_u2_p3_10_curve_20261114.json"
GATE = 0.80
FLAT_DELTA = 0.05


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    # Text-mode hash so CRLF checkout does not change the pinned identity.
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def build(contract: dict) -> tuple[object, torch.optim.Adam]:
    model = make_model("native", int(contract["candidate"]["seed"]))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    tensors = len(list(model.named_parameters()))
    if parameters != contract["candidate"]["parameters"]:
        raise ValueError("candidate_parameter_count_mismatch")
    if tensors != contract["candidate"]["tensors"]:
        raise ValueError("candidate_tensor_count_mismatch")
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
    indices = [order[(cursor + offset) % 64] for offset in range(2)]
    return [data["train"][index] for index in indices]


def with_rgb_zero(example: dict) -> dict:
    public = dict(example["public"])
    observations = dict(public["observations"])
    observations["rgb"] = torch.zeros_like(observations["rgb"])
    public["observations"] = observations
    return {**example, "public": public}


def with_fixed_question(example: dict, question: bytes) -> dict:
    public = dict(example["public"])
    public["request"] = question
    return {**example, "public": public}


def agreement(rows: list[dict], baseline: list[str]) -> float:
    hits = sum(row["output_hex"] == other for row, other in zip(rows, baseline, strict=True))
    return hits / len(rows)


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


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    model, optimizer = build(contract)
    order = contract["order"]
    per_update, losses = [], []
    for update in range(3):
        batch = batch_at(data, order, 2 * update)
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model, batch)
        loss.backward()
        optimizer.step()
        per_update.append(time.process_time() - started)
        losses.append(float(loss.detach()))
        if time.process_time() - cpu_start > contract["price_cap_cpu"]:
            raise BudgetStop("price_probe_budget_stop")
    eval_started = time.process_time()
    initial = evaluate(model, data["fit"])
    eval_cpu = time.process_time() - eval_started
    update_cpu = statistics.median(per_update)
    curve_cpu = 1844 * update_cpu + 7 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], curve_cpu * 1.1), 3)
    payload = {
        "schema": "taiji-u2-p3-10-price-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.10",
        "phase": "price_probe",
        "candidate": {
            "architecture": "native",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "tensors": len(list(model.named_parameters())),
            "seed": contract["candidate"]["seed"],
        },
        "measurements": {
            "updates": 3,
            "per_update_cpu": [round(value, 6) for value in per_update],
            "per_update_cpu_median": round(update_cpu, 6),
            "losses": [round(value, 6) for value in losses],
            "eval_cpu": round(eval_cpu, 6),
            "eval_correct": initial["correct"],
            "eval_denominator": initial["denominator"],
            "eval_accuracy_at_zero": initial["accuracy"],
        },
        "derived": {
            "curve_updates": 1844,
            "curve_evaluations": 7,
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
        "run_conditions": {
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
        },
        "scope": "pricing only; three in-memory Adam calls are discarded, no capability claim",
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
    }
    write_report(PRICE_REPORT, payload)
    print(
        json.dumps(
            {"verdict": payload["verdict"], "cap_cpu": cap, "curve_cpu": round(curve_cpu, 3)}
        )
    )
    return 0 if payload["verdict"] == "price_measured" else 4


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    root = Path(contract["run_root"])
    if root.exists() or CURVE_REPORT.exists():
        raise FileExistsError("p3_10_run_already_started")
    data = atomic_load(Path(contract["data_root"]))
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    model, optimizer = build(contract)
    order = contract["order"]
    doses = contract["dose_points"]
    root.mkdir(parents=True)

    def guard() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p3_10_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("p3_10_wall_stop")

    def ledger(row: dict) -> None:
        with (root / "curve.ndjson").open("ab") as stream:
            stream.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
            stream.flush()

    results = []
    with writer_lease(root / "curve.lock"):
        guard()
        started = time.process_time()
        zero = evaluate(model, data["fit"])
        ledger(
            {
                "event": "evaluate",
                "update": 0,
                "correct": zero["correct"],
                "denominator": zero["denominator"],
                "cpu": round(time.process_time() - started, 6),
            }
        )
        cursor = 0
        losses = []
        for update in range(1, max(doses) + 1):
            guard()
            batch = batch_at(data, order, cursor)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(model, batch)
            loss.backward()
            optimizer.step()
            model.parameter_revision += 1
            cursor += 2
            losses.append(float(loss.detach()))
            if update not in doses:
                continue
            started = time.process_time()
            bundle = root / f"dose-{update:04d}.bin"
            digest = save_learner(
                bundle, model, optimizer, "native", update, cursor, order, contract["data_digest"]
            )
            next_loss = float(task_loss(model, batch_at(data, order, cursor)).detach())
            scored = evaluate(model, data["fit"])
            cpu_here = round(time.process_time() - started, 6)
            ledger(
                {
                    "event": "dose",
                    "update": update,
                    "cursor": cursor,
                    "bundle": str(bundle),
                    "bundle_digest": digest,
                    "next_loss": round(next_loss, 9),
                    "correct": scored["correct"],
                    "denominator": scored["denominator"],
                    "cpu": cpu_here,
                }
            )
            results.append(
                {
                    "update": update,
                    "correct": scored["correct"],
                    "denominator": scored["denominator"],
                    "accuracy": scored["accuracy"],
                    "next_loss": round(next_loss, 9),
                    "bundle_digest": digest,
                    "rows": scored["rows"],
                }
            )
        control_started = time.process_time()
        fixed = contract["controls"]["fixed_question"].encode()
        zero_rgb = evaluate([with_rgb_zero(example) for example in data["fit"]])
        fixed_question = evaluate([with_fixed_question(example, fixed) for example in data["fit"]])
        baseline_outputs = [row["output_hex"] for row in results[-1]["rows"]]
        controls = {
            "rgb_zero_correct": zero_rgb["correct"],
            "rgb_zero_accuracy": zero_rgb["accuracy"],
            "rgb_zero_agreement_with_final": round(
                agreement(zero_rgb["rows"], baseline_outputs), 6
            ),
            "fixed_question_agreement_with_final": round(
                agreement(fixed_question["rows"], baseline_outputs), 6
            ),
            "fixed_question_text": fixed.decode(),
            "control_cpu": round(time.process_time() - control_started, 6),
        }
        verify = subprocess.run(
            [
                sys.executable,
                str(Path(__file__)),
                "--mode",
                "verify",
                "--bundle",
                str(root / f"dose-{max(doses):04d}.bin"),
                "--expect-next-loss",
                str(results[-1]["next_loss"]),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        restored = (
            json.loads(verify.stdout.strip().splitlines()[-1]) if verify.stdout.strip() else {}
        )
        accuracy = {str(item["update"]): item["accuracy"] for item in results}
        zero_accuracy = zero["accuracy"]
        last = accuracy[str(max(doses))]
        if zero_accuracy < GATE <= accuracy.get("256", -1.0):
            verdict = "dose_curve_measured"
        elif accuracy.get("256", -1.0) < GATE <= last:
            verdict = "gate_dose_infeasible"
        elif last < GATE and last - accuracy.get("64", 0.0) < FLAT_DELTA:
            verdict = "recipe_scoped_flat"
        else:
            verdict = "inconclusive_curve"
        payload = {
            "schema": "taiji-u2-p3-10-curve-v1",
            "date": "2026-11-14",
            "card": "22.5.P3.10",
            "phase": "dose_curve",
            "verdict": verdict,
            "gate": GATE,
            "flat_delta": FLAT_DELTA,
            "doses": [
                {key: value for key, value in item.items() if key != "rows"} for item in results
            ],
            "zero": {
                "correct": zero["correct"],
                "denominator": zero["denominator"],
                "accuracy": zero["accuracy"],
            },
            "controls": controls,
            "restore": {
                "returncode": verify.returncode,
                "stdout": verify.stdout.strip()[-400:],
                "stderr": verify.stderr.strip()[-400:],
                "next_loss_equal": bool(restored.get("next_loss_equal")),
                "next_loss": restored.get("next_loss"),
            },
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
                "gate_values_changed": 0,
            },
            "run_conditions": {
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
                "threads": torch.get_num_threads(),
                "torch": torch.__version__,
            },
            "scope": "single-seed dose curve on synthetic legal inputs; no natural-language, no project task, no capability claim",
        }
        write_report(CURVE_REPORT, payload)
    print(
        json.dumps(
            {"verdict": verdict, "accuracy": accuracy, "controls": controls, "restore": restored}
        )
    )
    return 0 if verdict != "engineering_failure" else 5


def verify(bundle: str, expect: float) -> int:
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data_root"]))
    model, _, payload = load_learner(Path(bundle))
    cursor = payload["cursor"]
    batch = batch_at(data, payload["order"], cursor)
    loss = float(task_loss(model, batch).detach())
    print(
        json.dumps(
            {
                "next_loss": round(loss, 9),
                "expected": round(expect, 9),
                "next_loss_equal": abs(loss - expect) <= 1e-9,
                "update": payload["update"],
                "cursor": cursor,
            }
        )
    )
    return 0 if abs(loss - expect) <= 1e-9 else 6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("price", "run", "verify"), default="price")
    parser.add_argument("--bundle", default="")
    parser.add_argument("--expect-next-loss", default="0")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.mode == "price":
        return price(contract)
    if args.mode == "verify":
        return verify(args.bundle, float(args.expect_next_loss))
    try:
        return run(contract)
    except BudgetStop as error:
        if not CURVE_REPORT.exists():
            write_report(
                CURVE_REPORT,
                {
                    "schema": "taiji-u2-p3-10-curve-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.10",
                    "phase": "dose_curve",
                    "verdict": "budget_stop",
                    "reason": str(error),
                    "training_authority": contract.get("training_authority"),
                    "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0},
                },
            )
        return 3
    except Exception as error:  # noqa: BLE001 -- record the failure instead of losing it
        if not CURVE_REPORT.exists():
            write_report(
                CURVE_REPORT,
                {
                    "schema": "taiji-u2-p3-10-curve-v1",
                    "date": "2026-11-14",
                    "card": "22.5.P3.10",
                    "phase": "dose_curve",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                    "training_authority": contract.get("training_authority"),
                    "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0},
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
