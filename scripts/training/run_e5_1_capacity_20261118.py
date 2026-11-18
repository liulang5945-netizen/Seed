"""E5.1: one-variable capacity test at the shared emission bottleneck.

E5 showed that dose buys correction only by destroying the old ability.  B.1
showed that moving updates into a small adapter does not provide a usable
correction substrate.  This card changes exactly one structural variable: the
hidden width of the shared ``joint_readout`` MLP, 16 -> 32.  The model width,
unit count, data, read code, trainable parameter names, A training, feedback
dose and optimizer settings stay fixed.

The replacement is born neutral: inherited hidden rows/columns are copied,
new first-layer rows are seeded, and new last-layer columns are zero.  Thus the
small and wide arms start with byte-identical answers and codes.  New columns
receive non-zero gradient immediately; new rows become reachable after those
columns move.  No product default or subject schema is changed.

Modes: ``smoke`` (zero optimizer steps), ``price`` (discarded-model unit
pricing), ``run`` (requires explicit training authority), ``recover`` and
``verify``.  A positive verdict requires the joint correction/retention gates,
counter-fact following, and a correction drop when the added capacity is
ablated.  A negative capacity result closes this E5.1 scope; it never triggers
an automatic wider/dose revision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import probe_e5_dose_20261118 as e5  # noqa: E402
from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_subject import MLP, U2Subject  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_e5_1_capacity_20261118.json"
A5_CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_e5_1_capacity_run_20261118.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_e5_1_capacity_smoke_20261118.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_e5_1_capacity_price_20261118.json"
RECOVER_REPORT = ROOT / "reports/taiji_u2_e5_1_capacity_recover_20261118.json"
VERIFY_REPORT = ROOT / "reports/taiji_u2_e5_1_capacity_verify_20261118.json"
E5_SMALL_REPORT = ROOT / "reports/taiji_u2_e5_dose_20261118_d8.json"

BASE_HIDDEN = 16
WIDE_HIDDEN = 32
STEPS = 600
CURVE = (400, 600)
DOSE = 8
CORRECTION_GATE = 0.80
DAMAGE_GATE = 0.02
ABLATION_DROP_GATE = 0.05


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    return contract, data


def guards(contract: dict[str, Any], mode: str) -> None:
    paths = {
        "runner": Path(__file__),
        "e5_probe": ROOT / "scripts/training/probe_e5_dose_20261118.py",
        "a5_runner": ROOT / "scripts/training/run_a5_22_5.py",
        "a2_runner": ROOT / "scripts/training/run_a2_byte_20261115.py",
        "support": ROOT / "scripts/training/first_a_support_22_5.py",
        "subject": ROOT / "taiji/u2_subject.py",
        "events": ROOT / "taiji/u2_events.py",
        "columns": ROOT / "taiji/u2_observation_columns.py",
    }
    for name, path in paths.items():
        if contract["source_pins"].get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if contract.get("criteria_sha256") != value_digest(contract["criteria"]):
        raise ValueError("criteria_digest_mismatch")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if contract["plan"].get("option") not in {"paired_full", "wide_only_lean"}:
            raise ValueError("missing_priced_option")
        if not contract["plan"].get("cap_cpu") or not contract["plan"].get("cap_wall"):
            raise ValueError("missing_measured_cap")


def arm_contract(contract: dict[str, Any], arm: str) -> dict[str, Any]:
    result = copy.deepcopy(contract)
    identity = contract["arms"][arm]
    result["trainable_tensor_count"] = identity["trainable_tensors"]
    result["trainable_scalar_count"] = identity["trainable_scalars"]
    result["emission_scalar_count"] = identity["emission_scalars"]
    return result


def widen_joint(model: U2Subject, seed: int) -> None:
    """Replace only joint_readout, preserving its parameter names and neutrality."""
    old = model.joint_readout
    if tuple(old.first.weight.shape) != (BASE_HIDDEN, 24) or tuple(old.last.weight.shape) != (
        8,
        BASE_HIDDEN,
    ):
        raise ValueError("unexpected_joint_readout_shape")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 501)
        replacement = MLP(24, 8, WIDE_HIDDEN // 2).to(
            dtype=old.first.weight.dtype, device=old.first.weight.device
        )
        with torch.no_grad():
            replacement.first.weight[:BASE_HIDDEN].copy_(old.first.weight)
            replacement.first.bias[:BASE_HIDDEN].copy_(old.first.bias)
            replacement.first.weight[BASE_HIDDEN:].normal_(
                0.0, 0.3 / math.sqrt(replacement.first.weight[0].numel())
            )
            replacement.first.bias[BASE_HIDDEN:].fill_(0.01)
            replacement.last.weight.zero_()
            replacement.last.weight[:, :BASE_HIDDEN].copy_(old.last.weight)
            replacement.last.bias.copy_(old.last.bias)
    model.joint_readout = replacement
    model.parameter_revision += 1


def build_pair(contract: dict[str, Any], data: dict[str, Any]) -> tuple[U2Subject, U2Subject, dict]:
    """Build both arms from one base and one code map; widen after code creation."""
    base, codes = e5.assemble(contract, data)
    small = copy.deepcopy(cast(U2Subject, base))
    wide = copy.deepcopy(cast(U2Subject, base))
    widen_joint(wide, int(contract["candidate"]["seed"]))
    a5.set_trainable(wide, arm_contract(contract, "wide"))
    return small, wide, codes


def template_of(contract: dict[str, Any]):
    return e5.template_function(contract)


def b_error_rows(model, items, errors, codes, template, counter: bool = False) -> list[dict]:
    return [
        a5.b_round(
            model,
            items[index],
            codes[items[index]["example"]["id"]],
            template(index),
            counter,
        )
        for index in errors
    ]


def feedback_dose(
    model,
    contract: dict[str, Any],
    items,
    errors,
    codes,
    template,
    dose: int,
    budget,
    counter: bool = False,
) -> int:
    optimizer = a5.optimizer_for(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        contract["feedback_optimizer"],
    )
    updates = 0
    for index in errors:
        item = items[index]
        public = a5.p4_2.column_public(item["example"])
        code = codes[item["example"]["id"]]
        raw = (
            item["counter_records"][template(index)]
            if counter
            else item["records"][template(index)]
        )
        target = item["counter_target"] if counter else item["target"]
        for _ in range(dose):
            budget()
            optimizer.zero_grad(set_to_none=True)
            loss = a5.record_loss(
                model,
                public,
                item["example"]["id"],
                raw,
                target,
                code,
            )
            loss.backward()
            optimizer.step()
            model.parameter_revision += 1
            updates += 1
    return updates


def save_bundle(path: Path, model: torch.nn.Module, hidden: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "taiji-u2-e5-1-capacity-bundle-v1",
            "joint_hidden": hidden,
            "revision": model.parameter_revision,
            "model": e5.state_of(model),
        },
        path,
    )


def purity(
    model: torch.nn.Module, saved: dict[str, torch.Tensor], contract: dict[str, Any]
) -> dict:
    trainable = set(a5.declared_trainable(contract))
    names = sorted(set(saved) - trainable)
    current = model.state_dict()
    drift = {name: float((current[name] - saved[name]).abs().max()) for name in names}
    return {
        "frozen_tensors": len(names),
        "exact": sum(value == 0.0 for value in drift.values()),
        "max_abs": max(drift.values(), default=0.0),
    }


def anchor_matches(result: dict[str, Any], contract: dict[str, Any]) -> bool:
    expected = contract["criteria"]["small_anchor"]
    return bool(
        result.get("old_ability") == expected["old_ability"]
        and result.get("b_first_pass_correct") == expected["b_first_pass_correct"]
        and result.get("corrected") == expected["corrected"]
        and result.get("old_ability_after_correct") == expected["old_ability_after_correct"]
        and result.get("damage") == expected["damage"]
        and result.get("a_curve") == expected["a_curve"]
    )


def run_arm(
    arm: str,
    model: U2Subject,
    contract: dict[str, Any],
    data: dict[str, Any],
    codes: dict,
    root: Path,
    budget,
) -> dict[str, Any]:
    started = time.process_time()
    hidden = BASE_HIDDEN if arm == "small" else WIDE_HIDDEN
    config = arm_contract(contract, arm)
    items, skipped = a5.task_b_examples(data, contract)
    template = template_of(contract)
    training = e5.train_a(model, config, data, codes, STEPS, CURVE, budget)
    pre_state = e5.state_of(model)
    save_bundle(root / f"{arm}-pre-feedback.pt", model, hidden)
    old_rows = e5.a_fit(model, data, codes)
    old_accuracy = a5.accuracy(old_rows)
    result: dict[str, Any] = {
        "arm": arm,
        "joint_hidden": hidden,
        "a_curve": {key: value["fit_accuracy"] for key, value in training["curve"].items()},
        "a_training": training,
        "old_ability": round(old_accuracy, 4),
        "old_rows": old_rows,
        "skipped": len(skipped),
    }
    if old_accuracy < float(contract["criteria"]["old_ability_gate"]):
        result.update(
            verdict="old_ability_not_reached", cpu=round(time.process_time() - started, 4)
        )
        return result

    first = a5.b_pass(model, items, codes, template)
    errors = [index for index, row in enumerate(first) if not row["correct"]]
    result["b_first_pass_correct"] = sum(int(row["correct"]) for row in first)
    result["errors"] = errors
    updates = feedback_dose(model, config, items, errors, codes, template, DOSE, budget)
    corrected_rows = b_error_rows(model, items, errors, codes, template)
    after_rows = e5.a_fit(model, data, codes)
    corrected = sum(int(row["correct"]) for row in corrected_rows)
    after_accuracy = a5.accuracy(after_rows)
    result.update(
        updates=updates,
        corrected=corrected,
        correction_rate=round(corrected / max(len(errors), 1), 4),
        corrected_rows=corrected_rows,
        old_ability_after=round(after_accuracy, 4),
        old_ability_after_correct=sum(int(row["correct"]) for row in after_rows),
        after_rows=after_rows,
        damage=round(old_accuracy - after_accuracy, 4),
        purity=purity(model, pre_state, config),
        a_digest=a5.rows_digest(after_rows),
        b_digest=a5.rows_digest(corrected_rows),
    )
    save_bundle(root / f"{arm}-post-feedback.pt", model, hidden)

    if arm == "wide":
        main_state = e5.state_of(model)
        counter_updates = feedback_dose(
            model, config, items, errors, codes, template, DOSE, budget, counter=True
        )
        counter_rows = b_error_rows(model, items, errors, codes, template, counter=True)
        counter_corrected = sum(int(row["correct"]) for row in counter_rows)
        result["counterfactual"] = {
            "updates": counter_updates,
            "corrected": counter_corrected,
            "rate": round(counter_corrected / max(len(errors), 1), 4),
            "rows": counter_rows,
            "digest": a5.rows_digest(counter_rows),
        }
        save_bundle(root / "wide-counterfactual.pt", model, hidden)

        model.load_state_dict(main_state)
        with torch.no_grad():
            model.joint_readout.last.weight[:, BASE_HIDDEN:].zero_()
        ablated_b_rows = b_error_rows(model, items, errors, codes, template)
        ablated_a_rows = e5.a_fit(model, data, codes)
        ablated_corrected = sum(int(row["correct"]) for row in ablated_b_rows)
        result["capacity_ablation"] = {
            "corrected": ablated_corrected,
            "correction_rate": round(ablated_corrected / max(len(errors), 1), 4),
            "drop": round((corrected - ablated_corrected) / max(len(errors), 1), 4),
            "old_ability": round(a5.accuracy(ablated_a_rows), 4),
            "a_digest": a5.rows_digest(ablated_a_rows),
            "b_digest": a5.rows_digest(ablated_b_rows),
        }
    result["cpu"] = round(time.process_time() - started, 4)
    return result


def classify(contract: dict[str, Any], arms: list[dict[str, Any]], anchor_ok: bool) -> str:
    if not anchor_ok:
        return "anchor_mismatch"
    wide = next(arm for arm in arms if arm["arm"] == "wide")
    if wide.get("verdict") == "old_ability_not_reached":
        return "old_ability_not_reached"
    if wide["purity"]["exact"] != wide["purity"]["frozen_tensors"]:
        return "engineering_failure"
    if wide["damage"] > DAMAGE_GATE:
        return "capacity_tradeoff_persists"
    if wide["correction_rate"] < CORRECTION_GATE:
        return "capacity_correction_absent"
    if wide["counterfactual"]["rate"] < CORRECTION_GATE:
        return "counterfactual_failed"
    if wide["capacity_ablation"]["drop"] < ABLATION_DROP_GATE:
        return "capacity_not_necessary"
    return "capacity_earns_joint_gate"


def smoke(contract: dict[str, Any], data: dict[str, Any]) -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    small, wide, codes = build_pair(contract, data)
    items, _ = a5.task_b_examples(data, contract)
    template = template_of(contract)
    small_a = e5.a_fit(small, data, codes)
    wide_a = e5.a_fit(wide, data, codes)
    small_b = a5.b_pass(small, items, codes, template)
    wide_b = a5.b_pass(wide, items, codes, template)
    wide_codes = a2.code_by_id(
        wide,
        a2.load_head(contract),
        a2.prior_of(data),
        data["train"] + data["fit"],
        contract,
    )
    saved = e5.state_of(wide)
    with torch.no_grad():
        wide.joint_readout.last.weight[:, BASE_HIDDEN:].add_(2.0)
    perturbed_a = e5.a_fit(wide, data, codes)
    liveness = a5.rows_digest(perturbed_a) != a5.rows_digest(wide_a)

    wide.zero_grad(set_to_none=True)
    index = 0
    item = items[index]
    loss = a5.record_loss(
        wide,
        a5.p4_2.column_public(item["example"]),
        item["example"]["id"],
        item["records"][template(index)],
        item["target"],
        codes[item["example"]["id"]],
    )
    loss.backward()
    first_gradient = wide.joint_readout.first.weight.grad
    assert first_gradient is not None
    first_extra_gradient = float(first_gradient[BASE_HIDDEN:].abs().sum())
    wide.load_state_dict(saved)
    wide.zero_grad(set_to_none=True)
    loss = a5.record_loss(
        wide,
        a5.p4_2.column_public(item["example"]),
        item["example"]["id"],
        item["records"][template(index)],
        item["target"],
        codes[item["example"]["id"]],
    )
    loss.backward()
    last_gradient = wide.joint_readout.last.weight.grad
    assert last_gradient is not None
    last_extra_gradient = float(last_gradient[:, BASE_HIDDEN:].abs().sum())
    wide.load_state_dict(saved)
    restored = e5.max_drift(wide, saved)
    wide.zero_grad(set_to_none=True)

    inherited = []
    for name, parameter in small.state_dict().items():
        other = wide.state_dict()[name]
        if name.startswith("joint_readout.first"):
            exact = torch.equal(parameter, other[:BASE_HIDDEN])
        elif name == "joint_readout.last.weight":
            exact = torch.equal(parameter, other[:, :BASE_HIDDEN])
        elif name == "joint_readout.last.bias":
            exact = torch.equal(parameter, other)
        else:
            exact = torch.equal(parameter, other)
        inherited.append(exact)
    checks = {
        "a_birth_neutral": a5.rows_digest(small_a) == a5.rows_digest(wide_a),
        "b_birth_neutral": a5.rows_digest(small_b) == a5.rows_digest(wide_b),
        "codes_neutral": all(torch.equal(codes[key], wide_codes[key]) for key in codes),
        "inherited_slices_exact": all(inherited),
        "wide_parameter_count": sum(p.numel() for p in wide.parameters())
        == contract["arms"]["wide"]["parameters"],
        "wide_trainable_count": sum(p.numel() for p in wide.parameters() if p.requires_grad)
        == contract["arms"]["wide"]["trainable_scalars"],
        "added_capacity_live": liveness,
        "last_added_gradient_nonzero": last_extra_gradient > 0.0,
        "first_added_gradient_after_output_moves": first_extra_gradient > 0.0,
        "snapshot_restore_exact": restored == 0.0,
    }
    payload = {
        "schema": "taiji-u2-e5-1-capacity-smoke-v1",
        "card": "E5.1",
        "criteria_sha256": contract["criteria_sha256"],
        "checks": checks,
        "uncovered": [name for name, passed in checks.items() if not passed],
        "identity": {
            "small": contract["arms"]["small"],
            "wide": contract["arms"]["wide"],
            "first_added_gradient": first_extra_gradient,
            "last_added_gradient": last_extra_gradient,
            "restore_max_abs": restored,
        },
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 4)},
        "verdict": "smoke_passed" if all(checks.values()) else "smoke_failed",
    }
    write_report(SMOKE_REPORT, payload)
    print(json.dumps({"verdict": payload["verdict"], "checks": checks}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


def measure_primitives(
    model: U2Subject, contract: dict[str, Any], data: dict[str, Any], codes: dict
) -> dict[str, float]:
    items, _ = a5.task_b_examples(data, contract)
    template = template_of(contract)
    positions = a2.cache_positions(model, data["train"], codes)
    batch = positions[: int(contract["batch"])]
    optimizer = a5.optimizer_for(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        contract["optimizer"],
    )
    required = set(contract["emission_parameters"])
    marker = time.process_time()
    for _ in range(3):
        a5.train_step(model, optimizer, batch, required)
    a_step = (time.process_time() - marker) / 3
    marker = time.process_time()
    a5.b_pass(model, items[:2], codes, template)
    b_round = (time.process_time() - marker) / 2
    marker = time.process_time()
    for example in data["fit"][:2]:
        a5.a_round(model, example, codes[example["id"]])
    a_round = (time.process_time() - marker) / 2
    marker = time.process_time()
    feedback_dose(model, contract, items, [0], codes, template, 3, lambda: None)
    feedback_step = (time.process_time() - marker) / 3
    return {
        "a_train_step": a_step,
        "feedback_step": feedback_step,
        "a_round": a_round,
        "b_round": b_round,
    }


def price(contract: dict[str, Any], data: dict[str, Any]) -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    small, wide, codes = build_pair(contract, data)
    units = {
        "small": measure_primitives(small, arm_contract(contract, "small"), data, codes),
        "wide": measure_primitives(wide, arm_contract(contract, "wide"), data, codes),
    }
    counts = {
        "small": {"a_training": 600, "feedback": 192, "a_rounds": 128, "b_rounds": 48},
        "wide": {"a_training": 600, "feedback": 384, "a_rounds": 160, "b_rounds": 72},
    }

    def estimate(arm: str) -> float:
        unit = units[arm]
        count = counts[arm]
        return float(
            count["a_training"] * unit["a_train_step"]
            + count["feedback"] * unit["feedback_step"]
            + count["a_rounds"] * unit["a_round"]
            + count["b_rounds"] * unit["b_round"]
        )

    small_estimate = estimate("small")
    wide_estimate = estimate("wide")
    contingency = float(contract["plan"]["contingency"])
    options = {}
    for name, subtotal in (
        ("paired_full", small_estimate + wide_estimate),
        ("wide_only_lean", wide_estimate),
    ):
        cap = subtotal * (1.0 + contingency)
        options[name] = {
            "estimate_cpu": round(subtotal, 4),
            "cap_cpu": round(cap, 3),
            "cap_wall": round(cap * 2.0, 3),
            "over_design_cap": cap > float(contract["plan"]["design_cap_cpu"]),
        }
    payload = {
        "schema": "taiji-u2-e5-1-capacity-price-v1",
        "card": "E5.1",
        "criteria_sha256": contract["criteria_sha256"],
        "unit_price_cpu": {
            arm: {key: round(value, 6) for key, value in unit.items()}
            for arm, unit in units.items()
        },
        "counts": counts,
        "options": options,
        "one_shot_adam_steps": 12,
        "models_discarded": True,
        "instrument_cpu": round(time.process_time() - started, 4),
    }
    write_report(PRICE_REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run(contract: dict[str, Any], data: dict[str, Any]) -> int:
    torch.set_num_threads(1)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    root = ROOT / contract["run_root"]
    root.mkdir(parents=True, exist_ok=True)

    def budget() -> None:
        if time.process_time() - cpu_start > float(contract["plan"]["cap_cpu"]):
            raise BudgetStop("e5_1_cpu_budget_stop")
        if time.perf_counter() - wall_start > float(contract["plan"]["cap_wall"]):
            raise BudgetStop("e5_1_wall_budget_stop")

    small, wide, codes = build_pair(contract, data)
    models = {"small": small, "wide": wide}
    selected = ("small", "wide") if contract["plan"]["option"] == "paired_full" else ("wide",)
    arms = []
    anchor_ok = False
    budget_reason = None
    try:
        for name in selected:
            arms.append(run_arm(name, models[name], contract, data, codes, root, budget))
            if name == "small" and not anchor_matches(arms[-1], contract):
                break
        if contract["plan"]["option"] == "paired_full":
            anchor_ok = bool(
                arms and arms[0]["arm"] == "small" and anchor_matches(arms[0], contract)
            )
        else:
            published = json.loads(E5_SMALL_REPORT.read_text(encoding="utf-8"))
            anchor_ok = (
                published["doses"][0]["corrected"]
                == contract["criteria"]["small_anchor"]["corrected"]
            )
        verdict = classify(contract, arms, anchor_ok)
    except BudgetStop as error:
        verdict = "budget_stop"
        budget_reason = str(error)
    payload = {
        "schema": "taiji-u2-e5-1-capacity-run-v1",
        "card": "E5.1",
        "criteria_sha256": contract["criteria_sha256"],
        "option": contract["plan"]["option"],
        "arms": arms,
        "anchor_ok": anchor_ok,
        "verdict": verdict,
        "budget_reason": budget_reason,
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 4),
            "wall": round(time.perf_counter() - wall_start, 4),
        },
        "discipline": {"check_reads": 0, "final_reads": 0, "formal_seeds": 1},
    }
    write_report(RUN_REPORT, payload)
    print(json.dumps({"verdict": verdict, "cost": payload["cost"]}, ensure_ascii=False))
    return 0


def recover(contract: dict[str, Any], data: dict[str, Any]) -> int:
    started = time.process_time()
    run_report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    small, wide, codes = build_pair(contract, data)
    models = {"small": small, "wide": wide}
    items, _ = a5.task_b_examples(data, contract)
    template = template_of(contract)
    root = ROOT / contract["run_root"]
    checks: dict[str, bool] = {}
    for arm in run_report["arms"]:
        name = arm["arm"]
        if "updates" not in arm:
            continue
        bundle = torch.load(root / f"{name}-post-feedback.pt", weights_only=False)
        model = models[name]
        model.load_state_dict(bundle["model"])
        a_rows = e5.a_fit(model, data, codes)
        b_rows = b_error_rows(model, items, arm["errors"], codes, template)
        checks[f"{name}_a_byte_exact"] = a5.rows_digest(a_rows) == arm["a_digest"]
        checks[f"{name}_b_byte_exact"] = a5.rows_digest(b_rows) == arm["b_digest"]
        if name == "wide":
            counter = torch.load(root / "wide-counterfactual.pt", weights_only=False)
            model.load_state_dict(counter["model"])
            counter_rows = b_error_rows(model, items, arm["errors"], codes, template, counter=True)
            checks["wide_counterfactual_byte_exact"] = (
                a5.rows_digest(counter_rows) == arm["counterfactual"]["digest"]
            )
    payload = {
        "schema": "taiji-u2-e5-1-capacity-recover-v1",
        "card": "E5.1",
        "checks": checks,
        "uncovered": [name for name, passed in checks.items() if not passed],
        "cpu": round(time.process_time() - started, 4),
        "verdict": "recover_passed" if all(checks.values()) else "recover_failed",
    }
    write_report(RECOVER_REPORT, payload)
    print(json.dumps({"verdict": payload["verdict"], "checks": checks}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


def verify_reports(contract: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    run_report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    recover_report = json.loads(RECOVER_REPORT.read_text(encoding="utf-8"))
    root = ROOT / contract["run_root"]
    checks: dict[str, bool] = {
        "recover_passed": recover_report["verdict"] == "recover_passed",
        "anchor_ok": run_report["anchor_ok"],
        "verdict_matches": run_report["verdict"]
        == classify(contract, run_report["arms"], run_report["anchor_ok"]),
    }
    for arm in run_report["arms"]:
        name = arm["arm"]
        if "updates" not in arm:
            continue
        pre = torch.load(root / f"{name}-pre-feedback.pt", weights_only=False)
        post = torch.load(root / f"{name}-post-feedback.pt", weights_only=False)
        trainable = set(a5.declared_trainable(contract))
        frozen = sorted(set(pre["model"]) - trainable)
        checks[f"{name}_purity"] = all(
            torch.equal(pre["model"][key], post["model"][key]) for key in frozen
        )
        checks[f"{name}_correction_recomputed"] = arm["corrected"] == sum(
            int(row["correct"]) for row in arm["corrected_rows"]
        )
        checks[f"{name}_damage_recomputed"] = arm["damage"] == round(
            arm["old_ability"] - a5.accuracy(arm["after_rows"]), 4
        )
    return {
        "checks": checks,
        "uncovered": [name for name, passed in checks.items() if not passed],
        "verdict": "verify_passed" if all(checks.values()) else "verify_failed",
    }


def verify(contract: dict[str, Any], data: dict[str, Any]) -> int:
    started = time.process_time()
    payload = verify_reports(contract, data)
    payload.update(
        schema="taiji-u2-e5-1-capacity-verify-v1",
        card="E5.1",
        criteria_sha256=contract["criteria_sha256"],
        cpu=round(time.process_time() - started, 4),
    )
    write_report(VERIFY_REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["verdict"] == "verify_passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E5.1 joint-readout capacity test")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    contract, data = load_contract()
    mode = "run" if args.run else "verify"
    guards(contract, mode)
    if args.smoke:
        return smoke(contract, data)
    if args.price:
        return price(contract, data)
    if args.run:
        return run(contract, data)
    if args.recover:
        return recover(contract, data)
    if args.verify:
        return verify(contract, data)
    parser.error("select a mode")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
