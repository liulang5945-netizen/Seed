"""22.5.A.2: write the frozen four-way read into natural Chinese bytes.

The A.1b per-column colour head, read through the zero-training train-prior
correction, names one of four position words and reaches 32/32 on fit.  This
card freezes that read and asks one question: can the subject's own byte
emission say the answer it was given?

Interface (``read-code-v1``): the read's four-way class enters the byte path as
a fixed 24-dim one-hot code (``3 * d``) at ``joint_readout``.  No parameter
encodes the read.  The only trainable tensors are ``joint_readout``,
``byte_head`` and ``copy_gate`` (8 tensors, 3,919 scalars).  Stage-2 contexts
are cached exactly as P4.2 does: the arguments the frozen body itself passed to
the emission operator, replayed once before training.

Scope: this is an expression positive control, not a content-conditioned
binding test.  A pass says the byte path can write a known answer; it does not
say the body can read the answer from the scene.

Modes: ``smoke`` (zero updates), ``price`` (5 engineering updates, in memory),
``recover`` (independent-process readback), ``run`` (formal training).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

import taiji.u2_subject as subject_module
from scripts.training import run_p4_1_22_5 as p4_1
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import make_model
from scripts.training.run_p5_4_a1b_colour import ColourHead
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_task_episode import NAMES

CONTRACT = ROOT / "plans/manifests/taiji_u2_a2_byte_20261115.json"
A1B_CONTRACT = ROOT / "plans/manifests/taiji_u2_a1b_colour_20261115.json"
A1B_RUN = ROOT / "scripts/training/run_p5_4_a1b_colour.py"
PRICE_REPORT = ROOT / "reports/taiji_u2_a2_byte_price_20261115.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_a2_byte_smoke_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_a2_byte_20261115.json"
LABELS = ("左边", "中间", "右边", "未知")
SUFFIX = "现在在哪里？"


class BudgetStop(RuntimeError):
    pass


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
        ("a1b_runner", A1B_RUN),
        ("a1b_contract", A1B_CONTRACT),
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


def build_subject(contract: dict) -> torch.nn.Module:
    candidate = contract["candidate"]
    bundle = ROOT / candidate["encoder_bundle"]
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != candidate["encoder_bundle_sha256"]:
        raise ValueError("encoder_bundle_mismatch")
    torch.manual_seed(int(candidate["seed"]))
    model = make_model("native", int(candidate["seed"]))
    model.encoder["rgb"].load_state_dict(torch.load(bundle, weights_only=False)["encoder"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != candidate["parameters"]:
        raise ValueError("candidate_parameter_count_mismatch")
    if len(list(model.named_parameters())) != candidate["tensors"]:
        raise ValueError("candidate_tensor_count_mismatch")
    return model


def load_head(contract: dict) -> ColourHead:
    candidate = contract["candidate"]
    path = ROOT / candidate["a1b_checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["a1b_checkpoint_sha256"]:
        raise ValueError("a1b_checkpoint_mismatch")
    head = ColourHead()
    head.load_state_dict(torch.load(path, weights_only=False)["head"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def prior_of(data: dict) -> torch.Tensor:
    labels = torch.cat([p4_1.column_labels(example) for example in data["train"]])
    counts = torch.bincount(labels, minlength=7).float()
    return counts / counts.sum()


def read_class(model, head, prior, example: dict) -> int:
    with torch.no_grad():
        logits = head(p4_1.tokens_of(model, example))
        predicted = (logits - prior.log()).argmax(-1).tolist()
    colour = NAMES.index(example["public"]["request"].decode()[: -len(SUFFIX)])
    columns = [index for index, label in enumerate(predicted) if label == colour]
    if not columns:
        return 3
    x = 2 * (sum(columns) / len(columns) + 0.5) / 16 - 1
    if x < -1 / 3:
        return 0
    return 2 if x > 1 / 3 else 1


def read_rows(model, head, prior, examples: list[dict]) -> dict:
    predictions = [read_class(model, head, prior, example) for example in examples]
    correct = sum(
        int(LABELS[prediction] == example["target"].decode())
        for prediction, example in zip(predictions, examples, strict=True)
    )
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "predictions": predictions,
    }


def code_of(class_index: int, width: int) -> torch.Tensor:
    code = torch.zeros(width)
    code[class_index] = 1.0
    return code


def code_by_id(model, head, prior, examples: list[dict], contract: dict) -> dict[str, torch.Tensor]:
    width = int(contract["read_code"]["width"])
    if width != 3 * int(model.c.d):
        raise ValueError("read_code_width_mismatch")
    return {
        example["id"]: code_of(read_class(model, head, prior, example), width)
        for example in examples
    }


@contextmanager
def reading_code(model, code: torch.Tensor) -> Iterator[None]:
    """Replace the joint readout's output by a fixed code for every call."""

    def inject(module, inputs, output):  # noqa: ANN001 - torch hook signature
        return module.forward(code[None].expand(inputs[0].shape[0], -1).to(inputs[0].dtype))

    handle = model.joint_readout.register_forward_hook(inject)
    try:
        yield
    finally:
        handle.remove()


def set_trainable(model, contract: dict) -> list[str]:
    declared = set(contract["emission_parameters"])
    found = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
        if name in declared:
            found.append(name)
    if sorted(found) != sorted(declared):
        raise ValueError("trainable_set_mismatch")
    scalars = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if scalars != int(contract["emission_scalar_count"]):
        raise ValueError("trainable_scalar_mismatch")
    return sorted(found)


def optimizer_for(parameters, spec: dict):
    return torch.optim.Adam(
        parameters,
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )


def cache_positions(model, examples: list[dict], codes: dict) -> list[dict]:
    """Cache exactly what the frozen body passed to the emission operator."""
    positions: list[dict] = []
    real_emission = subject_module.emission
    captured: dict[str, Any] = {}

    def wrapper(state, log_alpha, log_beta, lam, gamma, generator, ids, valid, mask):
        captured["args"] = {
            "cursor": {key: value.detach().clone() for key, value in state.items()},
            "alpha": log_alpha.detach().clone(),
            "beta": log_beta.detach().clone(),
            "ids": ids.detach().clone(),
            "valid": valid.detach().clone(),
            "mask": mask.detach().clone(),
        }
        return real_emission(state, log_alpha, log_beta, lam, gamma, generator, ids, valid, mask)

    def capture(module, inputs, result):  # noqa: ANN001 - torch hook signature
        captured.setdefault("x", inputs[0].detach().clone())
        return None

    subject_module.emission = wrapper
    hook = model.joint_readout.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for example in examples:
                state = p4_2.prepare(model, p4_2.column_public(example), example["id"])
                for position, symbol in enumerate([*example["target"], 256]):
                    captured.clear()
                    goal = model.language_goal(state)
                    _, _, details = model.output(state, goal, state["prediction_action"])
                    positions.append(
                        {
                            "example": example["id"],
                            "position": position,
                            "x": captured["x"],
                            "goal": goal.detach().clone(),
                            "previous": int(state["previous_byte"]),
                            "symbol": symbol,
                            "code": codes[example["id"]],
                            **captured["args"],
                        }
                    )
                    state = model.accept_byte(state, symbol, details)
    finally:
        hook.remove()
        subject_module.emission = real_emission
    return positions


def emission_loss(model, items: list[dict], *, use_code: bool) -> torch.Tensor:
    terms = []
    for item in items:
        source = item["code"][None].expand(model.c.branches, -1) if use_code else item["x"]
        context = model.joint_readout(source)
        previous_value = model.previous_embedding(item["previous"])[None].expand(
            context.shape[0], -1
        )
        generator = model.byte_head(torch.cat([context, previous_value], -1)).log_softmax(-1)
        gate = model.copy_gate(
            torch.cat(
                [
                    context,
                    item["goal"][None].expand(context.shape[0], -1),
                    previous_value,
                ],
                -1,
            )
        ).sigmoid()
        lam, gamma = gate.unbind(-1)
        out = subject_module.emission(
            item["cursor"],
            item["alpha"],
            item["beta"],
            lam,
            gamma,
            generator,
            item["ids"],
            item["valid"],
            item["mask"],
        )
        terms.append(-out["log_output"][item["symbol"]].mean())
    return torch.stack(terms).mean()


def gradient_census(model, positions: list[dict], contract: dict) -> dict:
    """One backward on a fixed batch: no optimizer step, no parameter change."""
    stride = max(1, len(positions) // int(contract["batch"]))
    batch = [positions[offset * stride] for offset in range(int(contract["batch"]))]
    model.zero_grad(set_to_none=True)
    emission_loss(model, batch, use_code=True).backward()
    sums = {
        name: round(0.0 if parameter.grad is None else float(parameter.grad.abs().sum()), 12)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    model.zero_grad(set_to_none=True)
    records = [int(item["valid"].sum()) for item in positions]
    return {
        "positions": len(positions),
        "positions_with_records": sum(1 for value in records if value > 0),
        "valid_records_total": sum(records),
        "alpha_last_zero_all": bool(
            torch.stack([item["alpha"][:, -1] for item in positions]).eq(0).all()
        ),
        "gradient_abs_sum": sums,
        "copy_gate_inert": sums.get("copy_gate.weight", 0.0) == 0.0
        and sums.get("copy_gate.bias", 0.0) == 0.0,
        "protocol": "one backward on a fixed batch; optimizer_step_calls = 0",
    }


def replay_checks(model, positions: list[dict], data: dict) -> list[dict]:
    checks = []
    stride = max(1, len(positions) // 3)
    for item in positions[::stride][:3]:
        example = next(e for e in data["train"] if e["id"] == item["example"])
        body = p4_2.body_log_probability(model, example, item["position"], item["symbol"])
        with torch.no_grad():
            replay = -float(emission_loss(model, [item], use_code=False))
        checks.append(
            {
                "example": item["example"],
                "position": item["position"],
                "body_log_probability": body,
                "replayed": replay,
                "equal": abs(body - replay) <= 1e-6,
            }
        )
    if not all(check["equal"] for check in checks):
        raise ValueError("a2_cache_replay_mismatch")
    return checks


def round_rows(model, examples: list[dict], codes: dict) -> list[dict]:
    rows = []
    cap = int(model.c.max_output_bytes) + 1
    with torch.no_grad():
        for example in examples:
            state = p4_2.prepare(model, p4_2.column_public(example), example["id"])
            content: list[int] = []
            eos = False
            with reading_code(model, codes[example["id"]]):
                for _ in range(cap):
                    probability, _, details = model.output(
                        state, model.language_goal(state), state["prediction_action"]
                    )
                    symbol = int(probability.argmax())
                    if symbol == 256:
                        eos = True
                        break
                    content.append(symbol)
                    state = model.accept_byte(state, symbol, details)
            raw = bytes(content)
            try:
                raw.decode("utf-8", errors="strict")
                valid = True
            except UnicodeDecodeError:
                valid = False
            rows.append(
                {
                    "id": example["id"],
                    "target": example["target"].decode(),
                    "output_hex": raw.hex(),
                    "eos": eos,
                    "valid_utf8": valid,
                    "correct": bool(eos and valid and raw == example["target"]),
                }
            )
    return rows


def rows_digest(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def accuracy(rows: list[dict]) -> float:
    return sum(int(row["correct"]) for row in rows) / len(rows)


def byte_loss(model, state, target: bytes, code: torch.Tensor) -> torch.Tensor:
    terms = []
    with reading_code(model, code):
        for symbol in [*target, 256]:
            probability, _, details = model.output(
                state, model.language_goal(state), state["prediction_action"]
            )
            terms.append(-probability[symbol])
            with torch.no_grad():
                state = model.accept_byte(state, symbol, details)
    return torch.stack(terms).mean()


def train_step(model, optimizer, items: list[dict]) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = emission_loss(model, items, use_code=True)
    loss.backward()
    for parameter in model.parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
        ):
            raise ValueError("nonfinite_a2_gradient")
    optimizer.step()
    model.parameter_revision += 1
    return float(loss.detach())


def initial_gate(model, contract: dict, positions: list[dict], root: Path) -> dict:
    path = root / "initial.pt"
    digest_value = p4_2.save_checkpoint(
        path,
        {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "revision": int(model.parameter_revision),
            "data_digest": contract["data"]["digest"],
            "read_code": contract["read_code"]["interface"],
        },
    )
    saved = torch.load(path, weights_only=False)
    rebuilt = build_subject(contract)
    rebuilt.load_state_dict(saved["model"])
    identical = all(
        torch.equal(saved["model"][key], value) for key, value in rebuilt.state_dict().items()
    )
    probe = copy.deepcopy(model)
    set_trainable(probe, contract)
    optimizer = optimizer_for(
        [p for p in probe.parameters() if p.requires_grad], contract["optimizer"]
    )
    train_step(probe, optimizer, positions[: int(contract["batch"])])
    step_path = root / "engineering-step.pt"
    step_digest = p4_2.save_checkpoint(
        step_path,
        {
            "model": {key: value.detach().clone() for key, value in probe.state_dict().items()},
            "revision": int(probe.parameter_revision),
            "data_digest": contract["data"]["digest"],
            "read_code": contract["read_code"]["interface"],
        },
    )
    step_saved = torch.load(step_path, weights_only=False)
    step_rebuilt = build_subject(contract)
    step_rebuilt.load_state_dict(step_saved["model"])
    step_identical = all(
        torch.equal(step_saved["model"][key], value)
        for key, value in step_rebuilt.state_dict().items()
    )
    return {
        "initial_checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
        "initial_sha256": digest_value,
        "initial_reload_identical": identical,
        "engineering_step_checkpoint": str(step_path.relative_to(ROOT)).replace("\\", "/"),
        "engineering_step_sha256": step_digest,
        "engineering_step_reload_identical": step_identical,
        "engineering_probe_updates": 1,
    }


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(ROOT / contract["data"]["root"])
    assert value_digest(data) == contract["data"]["digest"]
    model = build_subject(contract)
    head = load_head(contract)
    prior = prior_of(data)
    expected = torch.tensor(contract["candidate"]["prior"], dtype=torch.float32)
    if not torch.allclose(prior, expected, atol=1e-6, rtol=0):
        raise ValueError("prior_mismatch")
    read_fit = read_rows(model, head, prior, data["fit"])
    read_train = read_rows(model, head, prior, data["train"])
    build_read_cpu = time.process_time() - cpu_start

    codes = code_by_id(model, head, prior, data["train"], contract)
    cache_start = time.process_time()
    positions = cache_positions(model, data["train"], codes)
    cache_cpu = time.process_time() - cache_start
    checks = replay_checks(model, positions, data)

    set_trainable(model, contract)
    optimizer = optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
    )
    updates_start = time.process_time()
    probe_updates = 5
    for update in range(1, probe_updates + 1):
        items = [
            positions[((update - 1) * contract["batch"] + offset) % len(positions)]
            for offset in range(contract["batch"])
        ]
        train_step(model, optimizer, items)
    updates_cpu = time.process_time() - updates_start
    per_update = updates_cpu / probe_updates

    eval_start = time.process_time()
    fit_rows = round_rows(model, data["fit"], code_by_id(model, head, prior, data["fit"], contract))
    train_codes = code_by_id(model, head, prior, data["train"], contract)
    train_rows = round_rows(model, data["train"], train_codes)
    eval_cpu = time.process_time() - eval_start

    feedback_updates_total = int(contract["feedback_errors_allowance"]) * int(
        contract["feedback_updates"]
    )
    feedback_cpu = feedback_updates_total * (per_update / int(contract["batch"]))
    recovery_cpu = build_read_cpu + eval_cpu / 2
    fixed = build_read_cpu + cache_cpu
    estimate = (
        fixed
        + int(contract["steps"]) * per_update
        + len(contract["dose_points"]) * eval_cpu
        + 3 * (eval_cpu / 2)
        + feedback_cpu
        + recovery_cpu
        + per_update
    )
    design = contract["design_cap"]
    cap = round(min(float(design["cpu"]), estimate * 1.10), 3)
    payload = {
        "schema": "taiji-u2-a2-byte-price-v1",
        "card": "22.5.A.2",
        "date": "2026-11-15",
        "prior": [round(float(value), 8) for value in prior],
        "read_fit": {key: read_fit[key] for key in ("correct", "denominator", "accuracy")},
        "read_train": {key: read_train[key] for key in ("correct", "denominator", "accuracy")},
        "read_code_width": int(contract["read_code"]["width"]),
        "trainable_names": sorted(contract["emission_parameters"]),
        "trainable_scalars": int(contract["emission_scalar_count"]),
        "cache_positions": len(positions),
        "cache_replay_checks": checks,
        "per_unit_cpu": {
            "build_and_read": round(build_read_cpu, 6),
            "cache_positions": round(cache_cpu, 6),
            "per_update": round(per_update, 6),
            "fit_and_train_evaluate": round(eval_cpu, 6),
        },
        "probe": {
            "engineering_updates": probe_updates,
            "fit_rows": len(fit_rows),
            "train_rows": len(train_rows),
            "fit_accuracy": accuracy(fit_rows),
            "train_accuracy": accuracy(train_rows),
        },
        "estimate_cpu": round(estimate, 3),
        "adopted_cap_cpu": cap,
        "cap_wall": round(cap * 2, 3),
        "design_cap_cpu": design["cpu"],
        "design_cap_wall": design["wall"],
        "pricing_over_budget": cap > float(design["cpu"]),
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
        "discipline": {
            "formal_training_updates": 0,
            "price_engineering_updates": probe_updates,
            "checkpoints_written": 0,
            "capability_claims": 0,
            "check_final_reads": 0,
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
    }
    write_report(PRICE_REPORT, payload)
    print(json.dumps({"verdict": "price_measured", "cap_cpu": cap}))
    return 0 if cap <= float(design["cpu"]) else 4


def smoke(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(ROOT / contract["data"]["root"])
    assert value_digest(data) == contract["data"]["digest"]
    model = build_subject(contract)
    head = load_head(contract)
    prior = prior_of(data)
    read_fit = read_rows(model, head, prior, data["fit"])
    read_train = read_rows(model, head, prior, data["train"])
    codes = code_by_id(model, head, prior, data["train"], contract)
    positions = cache_positions(model, data["train"], codes)
    checks = replay_checks(model, positions, data)
    set_trainable(model, contract)
    fit_codes = code_by_id(model, head, prior, data["fit"], contract)
    rows = round_rows(model, data["fit"][:8], fit_codes)
    payload = {
        "schema": "taiji-u2-a2-byte-smoke-v1",
        "card": "22.5.A.2",
        "date": "2026-11-15",
        "read_fit": {key: read_fit[key] for key in ("correct", "denominator", "accuracy")},
        "read_train": {key: read_train[key] for key in ("correct", "denominator", "accuracy")},
        "cache_positions": len(positions),
        "cache_replay_checks": checks,
        "untrained_fit_rows": rows,
        "untrained_fit_accuracy": accuracy(rows),
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
        "discipline": {
            "optimizer_updates": 0,
            "checkpoints_written": 0,
            "capability_claims": 0,
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
    }
    write_report(SMOKE_REPORT, payload)
    print(
        json.dumps(
            {
                "verdict": "smoke_passed",
                "read_fit": read_fit["accuracy"],
                "cache": len(positions),
                "untrained_fit": accuracy(rows),
            }
        )
    )
    return 0


def recover(path: Path, contract: dict) -> str:
    torch.set_num_threads(1)
    data = atomic_load(ROOT / contract["data"]["root"])
    model = build_subject(contract)
    saved = torch.load(path, weights_only=False)
    model.load_state_dict(saved["model"])
    head = load_head(contract)
    prior = prior_of(data)
    codes = code_by_id(model, head, prior, data["fit"], contract)
    rows = round_rows(model, data["fit"], codes)
    return json.dumps({"rows": rows, "digest": rows_digest(rows)}, ensure_ascii=False)


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(ROOT / contract["data"]["root"])
    assert value_digest(data) == contract["data"]["digest"]
    root = root_path(contract["run_root"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("a2_run_already_started")
    root.mkdir(parents=True, exist_ok=True)

    def budget() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("a2_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("a2_wall_stop")

    model = build_subject(contract)
    head = load_head(contract)
    prior = prior_of(data)
    read_fit = read_rows(model, head, prior, data["fit"])
    read_train = read_rows(model, head, prior, data["train"])
    codes = code_by_id(model, head, prior, data["train"], contract)
    fit_codes = code_by_id(model, head, prior, data["fit"], contract)
    positions = cache_positions(model, data["train"], codes)
    checks = replay_checks(model, positions, data)
    names = set_trainable(model, contract)
    census = gradient_census(model, positions, contract)
    initial = initial_gate(model, contract, positions, root)
    optimizer = optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
    )

    rows = []
    losses = []
    for update in range(1, int(contract["steps"]) + 1):
        budget()
        items = [
            positions[((update - 1) * contract["batch"] + offset) % len(positions)]
            for offset in range(contract["batch"])
        ]
        losses.append(train_step(model, optimizer, items))
        if update not in contract["dose_points"]:
            continue
        train_rows = round_rows(model, data["train"], codes)
        fit_rows = round_rows(model, data["fit"], fit_codes)
        path = root / f"a2-{update:05d}.pt"
        digest_value = p4_2.save_checkpoint(
            path,
            {
                "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
                "update": update,
                "revision": int(model.parameter_revision),
                "data_digest": contract["data"]["digest"],
                "read_code": contract["read_code"]["interface"],
            },
        )
        rows.append(
            {
                "update": update,
                "train_correct": sum(int(row["correct"]) for row in train_rows),
                "train_accuracy": accuracy(train_rows),
                "fit_correct": sum(int(row["correct"]) for row in fit_rows),
                "fit_accuracy": accuracy(fit_rows),
                "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": digest_value,
            }
        )

    first_rows = round_rows(model, data["fit"], fit_codes)
    errors = [row for row in first_rows if not row["correct"]]
    control = copy.deepcopy(model)
    control_before = value_digest(control.state_dict())
    control_rows = round_rows(control, data["fit"], fit_codes)
    control_after = value_digest(control.state_dict())
    learned = copy.deepcopy(model)
    feedback_optimizer = optimizer_for(
        [p for p in learned.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    feedback_updates = 0
    for row in errors:
        example = next(e for e in data["fit"] if e["id"] == row["id"])
        for _ in range(int(contract["feedback_updates"])):
            budget()
            feedback_optimizer.zero_grad(set_to_none=True)
            state = p4_2.prepare(learned, p4_2.column_public(example), example["id"])
            byte_loss(learned, state, example["target"], fit_codes[example["id"]]).backward()
            feedback_optimizer.step()
            learned.parameter_revision += 1
            feedback_updates += 1
    learned_rows = round_rows(learned, data["fit"], fit_codes)
    learned_hits = {row["id"]: row["correct"] for row in learned_rows}
    corrected = sum(learned_hits[row["id"]] for row in errors)
    correction = corrected / len(errors) if len(errors) >= 4 else None
    gain = accuracy(learned_rows) - accuracy(control_rows) if len(errors) >= 4 else None
    first_correct = sum(int(row["correct"]) for row in first_rows)
    stability_losses = [
        row["id"] for row in first_rows if row["correct"] and not learned_hits[row["id"]]
    ]
    stability_ok = len(stability_losses) <= max(1, 0.10 * first_correct)
    learned_path = root / "feedback-learned.pt"
    learned_digest = p4_2.save_checkpoint(
        learned_path,
        {
            "model": {key: value.detach().clone() for key, value in learned.state_dict().items()},
            "update": "feedback",
            "revision": int(learned.parameter_revision),
        },
    )
    final_path = root / "final.pt"
    final_digest = p4_2.save_checkpoint(
        final_path,
        {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "update": int(contract["steps"]),
            "revision": int(model.parameter_revision),
            "data_digest": contract["data"]["digest"],
            "read_code": contract["read_code"]["interface"],
        },
    )
    independent = json.loads(
        __import__("subprocess")
        .run(
            [sys.executable, str(Path(__file__)), "--recover", str(final_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[-1]
    )
    recovery = {
        "in_process_digest": rows_digest(first_rows),
        "independent_digest": independent["digest"],
        "rows_equal": independent["rows"] == first_rows,
        "final_sha256": final_digest,
    }

    criteria = contract["criteria"]
    main_accuracy = accuracy(first_rows)
    main_gate = main_accuracy >= criteria["main_gate"]
    gate_table = {
        "main_full_answer_fit": "passed" if main_gate else "failed",
        "correction": (
            "passed"
            if correction is not None and correction >= criteria["correction_gate"]
            else "failed" if correction is not None else "not_identifiable_by_design"
        ),
        "feedback_gain": (
            "passed"
            if gain is not None and gain >= criteria["feedback_gain_gate"]
            else "failed" if gain is not None else "not_identifiable_by_design"
        ),
        "stability": "passed" if stability_ok else "failed",
        "recovery": "passed" if recovery["rows_equal"] else "failed",
        "control_zero_updates": "passed",
        "a_retention": "not_applicable_no_A_parent",
        "three_seed": "not_in_scope_single_seed",
        "check2": "not_read",
    }
    if not main_gate:
        verdict = "byte_expression_below_gate"
    elif gate_table["correction"] == "not_identifiable_by_design":
        verdict = (
            "byte_expression_confirmed_feedback_unmeasured"
            if stability_ok and recovery["rows_equal"]
            else "byte_expression_partial"
        )
    elif (
        gate_table["correction"] == "passed"
        and gate_table["feedback_gain"] == "passed"
        and stability_ok
        and recovery["rows_equal"]
    ):
        verdict = "byte_expression_confirmed"
    else:
        verdict = "byte_expression_partial"

    payload = {
        "schema": "taiji-u2-a2-byte-v1",
        "card": "22.5.A.2",
        "date": "2026-11-15",
        "verdict": verdict,
        "candidate": contract["candidate"],
        "data_digest": contract["data"]["digest"],
        "read": {
            "interface": contract["read_code"]["interface"],
            "prior": [round(float(value), 8) for value in prior],
            "fit": {key: read_fit[key] for key in ("correct", "denominator", "accuracy")},
            "train": {key: read_train[key] for key in ("correct", "denominator", "accuracy")},
            "code_width": int(contract["read_code"]["width"]),
        },
        "initial_gate": initial,
        "cache": {"positions": len(positions), "replay_checks": checks},
        "trainable": {"names": sorted(names), "scalars": int(contract["emission_scalar_count"])},
        "gradient_census": census,
        "doses": rows,
        "loss_first10": statistics.fmean(losses[:10]),
        "loss_last10": statistics.fmean(losses[-10:]),
        "feedback": {
            "first_pass_correct": first_correct,
            "first_pass_errors": len(errors),
            "control_second_pass": accuracy(control_rows),
            "learned_second_pass": accuracy(learned_rows),
            "correction_rate": correction,
            "feedback_gain": gain,
            "feedback_updates": feedback_updates,
            "control_updates": 0,
            "control_state_unchanged": control_before == control_after,
            "stability_losses": len(stability_losses),
            "learned_checkpoint": str(learned_path.relative_to(ROOT)).replace("\\", "/"),
            "learned_sha256": learned_digest,
        },
        "final_rows": first_rows,
        "recovery": recovery,
        "gate_table": gate_table,
        "criteria": criteria,
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
            "cap_cpu": contract["cpu_cap"],
            "cap_wall": contract["wall_cap"],
        },
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "scope": (
            "expression positive control only: the read is a frozen four-way class injected "
            "as a fixed code; no content-conditioned binding claim and no parent claim"
        ),
    }
    write_report(RUN_REPORT, payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "fit": main_accuracy,
                "correction": correction,
                "gain": gain,
                "cost": payload["cost"],
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--recover", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    if args.recover is not None:
        print(recover(args.recover, contract))
        return 0
    if args.smoke:
        return smoke(contract)
    if args.price:
        return price(contract)
    return run(contract)


if __name__ == "__main__":
    sys.exit(main())
