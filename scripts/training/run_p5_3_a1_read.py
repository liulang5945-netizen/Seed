"""P5.3 / 22.5.A.1 stage 1: single-query four-class read on the first-A task.

The A task asks one requested colour's position. This runner trains the v4
subject with the P4.1 colour-carrying encoder and a training-only four-class
head, then checks whether the subject reads the scene and the request instead of
emitting a constant. It does not train the byte head and does not claim a parent.
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
from torch import nn
from torch.nn import functional

from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import LABELS, make_model, prepare
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_a1_read_20261115.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_a1_read_price_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_a1_read_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a1_read_guard_20261115.json"


class BudgetStop(RuntimeError):
    pass


def root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


class ReadHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, 16), nn.Tanh(), nn.Linear(16, len(LABELS)))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context.mean(0))


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("subject", ROOT / "taiji/u2_subject.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("p4_2_runner", ROOT / "scripts/training/run_p4_2_22_5.py"),
        ("column_pathway", ROOT / "taiji/u2_observation_columns.py"),
        ("review_manifest", ROOT / "plans/manifests/taiji_u2_aparent_review_20261115.json"),
    ):
        if pins.get(name) != p4_2.sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract.get("cpu_cap") or not contract.get("wall_cap"):
            raise ValueError("missing_measured_cap")


def load_encoder(contract: dict) -> dict:
    candidate = contract["candidate"]
    bundle = Path(candidate["encoder_bundle"])
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if digest != candidate["encoder_bundle_sha256"]:
        raise ValueError("encoder_bundle_mismatch")
    return torch.load(bundle, weights_only=False)["encoder"]


def build(contract: dict, encoder: dict | None = None):
    seed = int(contract["candidate"]["seed"])
    torch.manual_seed(seed)
    model = make_model("native", seed)
    if encoder is not None:
        model.encoder["rgb"].load_state_dict(encoder)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    tensors = len(list(model.named_parameters()))
    if parameters != contract["candidate"]["parameters"]:
        raise ValueError("candidate_parameter_count_mismatch")
    if tensors != contract["candidate"]["tensors"]:
        raise ValueError("candidate_tensor_count_mismatch")
    head = ReadHead()
    optimizer = torch.optim.Adam(
        [*model.parameters(), *head.parameters()],
        lr=contract["optimizer"]["lr"],
        betas=tuple(contract["optimizer"]["betas"]),
        eps=contract["optimizer"]["eps"],
        weight_decay=contract["optimizer"]["weight_decay"],
    )
    return model, head, optimizer


def control_example(example: dict, *, rgb_zero: bool = False, request: bytes | None = None) -> dict:
    public = dict(example["public"])
    observations = dict(public["observations"])
    if rgb_zero:
        observations["rgb"] = torch.zeros_like(observations["rgb"])
    if request is not None:
        public["request"] = request
    public["observations"] = observations
    return {**example, "public": public}


def column_example(example: dict) -> dict:
    return p4_2.column_public(example)


def read_logits(model, head, example: dict) -> torch.Tensor:
    state = prepare(model, column_example(example), example["id"])
    context = p4_2.read_context(model, state)
    return head(context)


def label_of(example: dict) -> int:
    return LABELS.index(example["target"].decode())


def train_loss(model, head, batch: list[dict]) -> torch.Tensor:
    terms = [
        functional.cross_entropy(
            read_logits(model, head, example)[None], torch.tensor([label_of(example)])
        )
        for example in batch
    ]
    return torch.stack(terms).mean()


def evaluate(model, head, examples: list[dict]) -> dict:
    predictions = []
    with torch.no_grad():
        for example in examples:
            predictions.append(int(read_logits(model, head, example).argmax()))
    correct = sum(
        int(LABELS[index] == example["target"].decode())
        for index, example in zip(predictions, examples, strict=True)
    )
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "predictions": predictions,
        "distinct_predictions": len(set(predictions)),
    }


def agreement(left: list[int], right: list[int]) -> float:
    return sum(int(a == b) for a, b in zip(left, right, strict=True)) / len(left)


def digest_predictions(model, head, examples: list[dict]) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for example in examples:
            logits = read_logits(model, head, example)
            digest.update(logits.numpy().tobytes())
            digest.update(example["id"].encode())
    return digest.hexdigest()


def recover(path: Path, contract: dict) -> str:
    torch.set_num_threads(1)
    data = atomic_load(Path(contract["data"]["root"]))
    checkpoint = torch.load(path, weights_only=False)
    model, head, _ = build(contract)
    model.load_state_dict(checkpoint["model"])
    head.load_state_dict(checkpoint["head"])
    return digest_predictions(model, head, data["fit"][:8])


def initial_gate(contract: dict, data: dict, encoder: dict, root: Path) -> dict:
    model, head, optimizer = build(contract, encoder)
    path = root / "initial.pt"
    digest = p4_2.save_checkpoint(
        path,
        {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "head": {key: value.detach().clone() for key, value in head.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "update": 0,
            "cursor": 0,
            "order": contract["order"],
            "data_digest": contract["data"]["digest"],
        },
    )
    in_process = digest_predictions(model, head, data["fit"][:8])
    independent = subprocess.run(
        [sys.executable, str(Path(__file__)), "--recover", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    recovered = json.loads(independent.stdout.strip().splitlines()[-1])["digest"]
    batch = [data["train"][contract["order"][offset]] for offset in range(2)]
    optimizer.zero_grad(set_to_none=True)
    train_loss(model, head, batch).backward()
    optimizer.step()
    step_path = root / "engineering-step.pt"
    step_digest = p4_2.save_checkpoint(
        step_path,
        {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "head": {key: value.detach().clone() for key, value in head.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "update": 1,
            "cursor": 2,
            "order": contract["order"],
            "data_digest": contract["data"]["digest"],
        },
    )
    saved = torch.load(step_path, weights_only=False)
    rebuilt, rebuilt_head, rebuilt_optimizer = build(contract)
    rebuilt.load_state_dict(saved["model"])
    rebuilt_head.load_state_dict(saved["head"])
    rebuilt_optimizer.load_state_dict(saved["optimizer"])
    return {
        "initial_checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "initial_sha256": digest,
        "initial_recovery_identical": in_process == recovered,
        "engineering_step_checkpoint": str(step_path.relative_to(ROOT)).replace("\\", "/"),
        "engineering_step_sha256": step_digest,
        "model_reload_identical": all(
            torch.equal(saved["model"][key], value) for key, value in rebuilt.state_dict().items()
        ),
        "head_reload_identical": all(
            torch.equal(saved["head"][key], value)
            for key, value in rebuilt_head.state_dict().items()
        ),
        "optimizer_state_entries": len(rebuilt_optimizer.state),
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
    }


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data"]["root"]))
    assert value_digest(data) == contract["data"]["digest"]
    encoder = load_encoder(contract)
    model, head, optimizer = build(contract, encoder)
    order = contract["order"]
    samples = []
    for index in range(5):
        batch = [data["train"][order[(2 * index + offset) % 64]] for offset in range(2)]
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        train_loss(model, head, batch).backward()
        optimizer.step()
        samples.append(time.process_time() - started)
    started = time.process_time()
    zero = evaluate(model, head, data["fit"])
    eval_cpu = time.process_time() - started
    update_cpu = statistics.median(samples)
    curve_cpu = 1844 * update_cpu + 7 * eval_cpu
    cap = round(min(contract["cpu_cap"], curve_cpu * 1.1), 3)
    payload = {
        "schema": "taiji-u2-a1-read-price-v1",
        "card": "22.5.A.1",
        "date": "2026-11-15",
        "candidate": contract["candidate"],
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "per_update_cpu_samples": [round(value, 6) for value in samples],
        "per_update_cpu_median": round(update_cpu, 6),
        "eval_cpu": round(eval_cpu, 6),
        "zero_fit": {
            key: zero[key] for key in ("correct", "denominator", "accuracy", "distinct_predictions")
        },
        "curve_cpu_estimate": round(curve_cpu, 3),
        "adopted_cap_cpu": cap,
        "cap_wall": round(cap * 2, 3),
        "pricing_over_budget": cap > contract["cpu_cap"],
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "discipline": {
            "checkpoints_written": 0,
            "optimizer_calls": 5,
            "capability_claims": 0,
        },
    }
    write_report(PRICE_REPORT, payload)
    print(
        json.dumps({"verdict": "price_measured", "cap_cpu": cap, "curve_cpu": round(curve_cpu, 3)})
    )
    return 0 if cap <= contract["cpu_cap"] else 4


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data"]["root"]))
    assert value_digest(data) == contract["data"]["digest"]
    encoder = load_encoder(contract)
    root = root_path(contract["run_root"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("a1_run_already_started")
    root.mkdir(parents=True, exist_ok=True)

    def budget() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("a1_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("a1_wall_stop")

    initial = initial_gate(contract, data, encoder, root)
    model, head, optimizer = build(contract, encoder)
    order = contract["order"]
    cursor = 0
    losses: list[float] = []
    status = "stage1_complete"
    try:
        for _update in range(1, int(contract["dose"]) + 1):
            budget()
            batch = [data["train"][order[(cursor + offset) % 64]] for offset in range(2)]
            cursor += 2
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss(model, head, batch)
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in [*model.parameters(), *head.parameters()]
            ):
                raise ValueError("nonfinite_a1_gradient")
            optimizer.step()
            model.parameter_revision += 1
            losses.append(float(loss.detach()))
    except BudgetStop:
        status = "budget_stop"

    path = root / f"a1-{int(contract['dose']):06d}.pt"
    digest = p4_2.save_checkpoint(
        path,
        {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "head": {key: value.detach().clone() for key, value in head.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "update": int(contract["dose"]),
            "cursor": cursor,
            "order": order,
            "data_digest": contract["data"]["digest"],
        },
    )
    fit = evaluate(model, head, data["fit"])
    train = evaluate(model, head, data["train"])
    fixed = contract["controls"]["fixed_question"].encode()
    zero = evaluate(
        model, head, [control_example(example, rgb_zero=True) for example in data["fit"]]
    )
    fixed_question = evaluate(
        model, head, [control_example(example, request=fixed) for example in data["fit"]]
    )
    controls = {
        "rgb_zero_agreement": agreement(zero["predictions"], fit["predictions"]),
        "fixed_question_agreement": agreement(fixed_question["predictions"], fit["predictions"]),
        "fixed_question_text": fixed.decode(),
        "rgb_zero_distinct": zero["distinct_predictions"],
        "fixed_question_distinct": fixed_question["distinct_predictions"],
    }
    in_process = digest_predictions(model, head, data["fit"][:8])
    independent = subprocess.run(
        [sys.executable, str(Path(__file__)), "--recover", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    recovered = json.loads(independent.stdout.strip().splitlines()[-1])["digest"]
    recovery = {
        "in_process_digest": in_process,
        "independent_digest": recovered,
        "identical": in_process == recovered,
    }
    criteria = contract["criteria"]
    gates = {
        "fit": fit["accuracy"] >= criteria["fit_gate"],
        "train": train["accuracy"] >= criteria["train_gate"],
        "distinct": fit["distinct_predictions"] >= criteria["distinct_min"],
        "rgb_zero": controls["rgb_zero_agreement"] < criteria["rgb_zero_agreement_max"],
        "fixed_question": controls["fixed_question_agreement"]
        < criteria["fixed_question_agreement_max"],
        "recovery": recovery["identical"],
    }
    if status == "budget_stop":
        verdict = "budget_stop"
    elif not all(gates.values()) and fit["accuracy"] < criteria["blind"]:
        verdict = "a_parent_read_blind"
    elif all(gates.values()):
        verdict = "a_parent_read_taught"
    else:
        verdict = "a_parent_read_partial"
    payload = {
        "schema": "taiji-u2-a1-read-v1",
        "card": "22.5.A.1",
        "date": "2026-11-15",
        "status": status,
        "verdict": verdict,
        "candidate": contract["candidate"],
        "initial_gate": initial,
        "fit": {
            key: fit[key]
            for key in ("correct", "denominator", "accuracy", "distinct_predictions", "predictions")
        },
        "train": {
            key: train[key]
            for key in ("correct", "denominator", "accuracy", "distinct_predictions")
        },
        "controls": controls,
        "recovery": recovery,
        "gates": gates,
        "loss_first50": statistics.fmean(losses[:50]) if losses else None,
        "loss_last50": statistics.fmean(losses[-50:]) if losses else None,
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest,
        "data_digest": contract["data"]["digest"],
        "encoder_bundle_sha256": contract["candidate"]["encoder_bundle_sha256"],
        "criteria": criteria,
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "scope": "stage 1 only: single-query four-class read; no byte head, no parent claim",
    }
    write_report(RUN_REPORT, payload)
    print(json.dumps({"verdict": verdict, "fit": round(fit["accuracy"], 6), "status": status}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--recover", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    if args.recover is not None:
        print(json.dumps({"digest": recover(args.recover, contract)}, ensure_ascii=False))
        return 0
    if args.price:
        return price(contract)
    return run(contract)


if __name__ == "__main__":
    sys.exit(main())
