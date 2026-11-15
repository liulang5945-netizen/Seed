"""22.5.A.1b: per-column colour head plus a non-learned query read.

The subject is frozen at the P4.1 B encoder. A seven-class head (six palette
colours plus background) is trained on the first-A train split's visible column
labels. The position read is fixed: select the columns predicted as the
requested colour, average their x, and bin to left/middle/right/unknown.
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

from scripts.training import run_p4_1_22_5 as p4_1
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_task_episode import NAMES

CONTRACT = ROOT / "plans/manifests/taiji_u2_a1b_colour_20261115.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_a1b_colour_price_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_a1b_colour_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a1b_colour_guard_20261115.json"
SUFFIX = "现在在哪里？"
BLIND = 0.30


class BudgetStop(RuntimeError):
    pass


class ColourHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 7))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


def root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("subject", ROOT / "taiji/u2_subject.py"),
        ("p4_1_runner", ROOT / "scripts/training/run_p4_1_22_5.py"),
        ("p4_2_runner", ROOT / "scripts/training/run_p4_2_22_5.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("columns", ROOT / "taiji/u2_observation_columns.py"),
        ("a1_contract", ROOT / "plans/manifests/taiji_u2_a1_read_20261115.json"),
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
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != candidate["encoder_bundle_sha256"]:
        raise ValueError("encoder_bundle_mismatch")
    return torch.load(bundle, weights_only=False)["encoder"]


def build_subject(contract: dict) -> nn.Module:
    torch.manual_seed(int(contract["candidate"]["seed"]))
    model = make_model("native", int(contract["candidate"]["seed"]))
    model.encoder["rgb"].load_state_dict(load_encoder(contract))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != contract["candidate"]["parameters"]:
        raise ValueError("candidate_parameter_count_mismatch")
    if len(list(model.named_parameters())) != contract["candidate"]["tensors"]:
        raise ValueError("candidate_tensor_count_mismatch")
    return model


def control_example(example: dict, *, rgb_zero: bool = False, request: bytes | None = None) -> dict:
    public = dict(example["public"])
    observations = dict(public["observations"])
    if rgb_zero:
        observations["rgb"] = torch.zeros_like(observations["rgb"])
    if request is not None:
        public["request"] = request
    public["observations"] = observations
    return {**example, "public": public}


def column_tokens(model, example: dict) -> torch.Tensor:
    with torch.no_grad():
        return p4_1.tokens_of(model, example)


def colour_of(example: dict) -> int:
    return NAMES.index(example["public"]["request"].decode()[: -len(SUFFIX)])


def position_of(x: float) -> int:
    return 0 if x < -1 / 3 else 2 if x > 1 / 3 else 1


def read_position(model, head, example: dict) -> int:
    tokens = column_tokens(model, example)
    with torch.no_grad():
        predicted = head(tokens).argmax(-1).tolist()
    cols = [index for index, label in enumerate(predicted) if label == colour_of(example)]
    if not cols:
        return 3
    return position_of(2 * (sum(cols) / len(cols) + 0.5) / 16 - 1)


def position_accuracy(model, head, examples: list[dict]) -> dict:
    predictions = [read_position(model, head, example) for example in examples]
    correct = sum(
        int(("左边", "中间", "右边", "未知")[prediction] == example["target"].decode())
        for prediction, example in zip(predictions, examples, strict=True)
    )
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "predictions": predictions,
        "distinct_predictions": len(set(predictions)),
    }


def column_metrics(model, head, examples: list[dict]) -> dict:
    tokens = torch.cat([column_tokens(model, example) for example in examples])
    truth = torch.cat([p4_1.column_labels(example) for example in examples])
    with torch.no_grad():
        predicted = head(tokens).argmax(-1)
    painted = truth < 6
    background = truth == 6
    return {
        "overall": float((predicted == truth).float().mean()),
        "painted": (
            float((predicted[painted] == truth[painted]).float().mean())
            if bool(painted.any())
            else 0.0
        ),
        "background": (
            float((predicted[background] == truth[background]).float().mean())
            if bool(background.any())
            else 0.0
        ),
        "painted_columns": int(painted.sum()),
        "background_columns": int(background.sum()),
        "background_fraction": float(background.float().mean()),
    }


def digest(model, head, examples: list[dict]) -> str:
    sha = hashlib.sha256()
    with torch.no_grad():
        for example in examples:
            logits = head(column_tokens(model, example))
            sha.update(logits.numpy().tobytes())
            sha.update(example["id"].encode())
    return sha.hexdigest()


def train_head(model, contract: dict, data: dict) -> tuple[nn.Module, list[float]]:
    head = ColourHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=contract["optimizer"]["lr"])
    tokens = torch.cat([column_tokens(model, example) for example in data["train"]])
    truth = torch.cat([p4_1.column_labels(example) for example in data["train"]])
    losses = []
    for _step in range(int(contract["steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(head(tokens), truth)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return head, losses


def initial_gate(model, contract: dict, data: dict, root: Path) -> dict:
    head = ColourHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=contract["optimizer"]["lr"])
    path = root / "initial.pt"
    digest_value = p4_2.save_checkpoint(
        path,
        {
            "head": {key: value.detach().clone() for key, value in head.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "steps": 0,
            "data_digest": contract["data"]["digest"],
        },
    )
    before = digest(model, head, data["fit"][:8])
    recovered = subprocess.run(
        [sys.executable, str(Path(__file__)), "--recover", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    independent = json.loads(recovered.stdout.strip().splitlines()[-1])["digest"]
    tokens = torch.cat([column_tokens(model, example) for example in data["train"]])
    truth = torch.cat([p4_1.column_labels(example) for example in data["train"]])
    optimizer.zero_grad(set_to_none=True)
    functional.cross_entropy(head(tokens), truth).backward()
    optimizer.step()
    step_path = root / "engineering-step.pt"
    step_digest = p4_2.save_checkpoint(
        step_path,
        {
            "head": {key: value.detach().clone() for key, value in head.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "steps": 1,
            "data_digest": contract["data"]["digest"],
        },
    )
    saved = torch.load(step_path, weights_only=False)
    rebuilt = ColourHead()
    rebuilt.load_state_dict(saved["head"])
    return {
        "initial_checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "initial_sha256": digest_value,
        "initial_recovery_identical": before == independent,
        "engineering_step_checkpoint": str(step_path.relative_to(ROOT)).replace("\\", "/"),
        "engineering_step_sha256": step_digest,
        "head_reload_identical": all(
            torch.equal(saved["head"][key], value) for key, value in rebuilt.state_dict().items()
        ),
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
    }


def recover(path: Path, contract: dict) -> str:
    torch.set_num_threads(1)
    data = atomic_load(Path(contract["data"]["root"]))
    model = build_subject(contract)
    checkpoint = torch.load(path, weights_only=False)
    head = ColourHead()
    head.load_state_dict(checkpoint["head"])
    return digest(model, head, data["fit"][:8])


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data"]["root"]))
    assert value_digest(data) == contract["data"]["digest"]
    model = build_subject(contract)
    head, losses = train_head(model, {**contract, "steps": 5}, data)
    fit = position_accuracy(model, head, data["fit"])
    columns = column_metrics(model, head, data["fit"])
    capex = time.process_time() - cpu_start
    per_step = capex / 5
    curve = per_step * int(contract["steps"]) + capex
    cap = round(min(contract["cpu_cap"], curve * 1.10), 3)
    payload = {
        "schema": "taiji-u2-a1b-colour-price-v1",
        "card": "22.5.A.1b",
        "date": "2026-11-15",
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "per_step_cpu": round(per_step, 6),
        "steps": int(contract["steps"]),
        "zero_fit": {key: fit[key] for key in ("correct", "denominator", "accuracy")},
        "zero_columns": columns,
        "curve_cpu_estimate": round(curve, 3),
        "adopted_cap_cpu": cap,
        "cap_wall": round(cap * 2, 3),
        "pricing_over_budget": cap > contract["cpu_cap"],
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "discipline": {
            "training": 0,
            "subject_optimizer_steps": 0,
            "head_updates": 5,
            "capability_claims": 0,
        },
    }
    write_report(PRICE_REPORT, payload)
    print(json.dumps({"verdict": "price_measured", "cap_cpu": cap}))
    return 0 if cap <= contract["cpu_cap"] else 4


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data"]["root"]))
    assert value_digest(data) == contract["data"]["digest"]
    root = root_path(contract["run_root"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("a1b_run_already_started")
    root.mkdir(parents=True, exist_ok=True)
    model = build_subject(contract)
    initial = initial_gate(model, contract, data, root)
    head, losses = train_head(model, contract, data)

    path = root / f"a1b-{int(contract['steps']):05d}.pt"
    head_digest = p4_2.save_checkpoint(
        path,
        {
            "head": {key: value.detach().clone() for key, value in head.state_dict().items()},
            "steps": int(contract["steps"]),
            "data_digest": contract["data"]["digest"],
            "encoder_bundle_sha256": contract["candidate"]["encoder_bundle_sha256"],
        },
    )
    in_process = digest(model, head, data["fit"][:8])
    independent = subprocess.run(
        [sys.executable, str(Path(__file__)), "--recover", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    recovered = json.loads(independent.stdout.strip().splitlines()[-1])["digest"]
    fit = position_accuracy(model, head, data["fit"])
    train = position_accuracy(model, head, data["train"])
    zero = position_accuracy(
        model, head, [control_example(example, rgb_zero=True) for example in data["fit"]]
    )
    fixed = position_accuracy(
        model,
        head,
        [
            control_example(example, request=contract["controls"]["fixed_question"].encode())
            for example in data["fit"]
        ],
    )
    base = fit["predictions"]
    controls = {
        "rgb_zero_agreement": sum(
            int(a == b) for a, b in zip(zero["predictions"], base, strict=True)
        )
        / len(base),
        "fixed_question_agreement": sum(
            int(a == b) for a, b in zip(fixed["predictions"], base, strict=True)
        )
        / len(base),
        "rgb_zero_distinct": zero["distinct_predictions"],
        "fixed_question_distinct": fixed["distinct_predictions"],
        "fixed_question_text": contract["controls"]["fixed_question"],
    }
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
    if not all(gates.values()) and fit["accuracy"] < contract["criteria"]["blind"]:
        verdict = "a_parent_read_blind"
    elif all(gates.values()):
        verdict = "a_parent_read_taught"
    else:
        verdict = "a_parent_read_partial"
    payload = {
        "schema": "taiji-u2-a1b-colour-v1",
        "card": "22.5.A.1b",
        "date": "2026-11-15",
        "verdict": verdict,
        "candidate": contract["candidate"],
        "initial_gate": initial,
        "fit": fit,
        "train": {
            key: train[key]
            for key in ("correct", "denominator", "accuracy", "distinct_predictions")
        },
        "columns_fit": column_metrics(model, head, data["fit"]),
        "columns_train": column_metrics(model, head, data["train"]),
        "controls": controls,
        "recovery": recovery,
        "gates": gates,
        "loss_first10": statistics.fmean(losses[:10]),
        "loss_last10": statistics.fmean(losses[-10:]),
        "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": head_digest,
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "data_digest": contract["data"]["digest"],
        "encoder_bundle_sha256": contract["candidate"]["encoder_bundle_sha256"],
        "criteria": criteria,
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
        "discipline": {
            "training": 0,
            "check_reads": 0,
            "final_reads": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "scope": "per-column colour head plus a non-learned query read; no subject training, no byte head, no parent claim",
    }
    write_report(RUN_REPORT, payload)
    print(json.dumps({"verdict": verdict, "fit": round(fit["accuracy"], 6)}))
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
