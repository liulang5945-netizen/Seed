"""22.5.A.6: the correction budget as the only independent variable.

A.5 measured the correction gate at zero and traced it to its own rollback
clause: one update per error, rolled back whenever the re-ask was still wrong,
so every update was discarded. Its attribution arm - eight accumulated updates
per error, no rollback - reached 0.333. That leaves one question, and this card
asks exactly it: does the correction rate move with the per-error budget, or is
0.333 already a ceiling?

Design, frozen in the contract before any arm runs:

* start state is the A.5 pre-feedback bundle, byte-identical, so every arm shares
  the same first pass and the same subject;
* the released record and the four templates are unchanged from A.5;
* the only thing that varies across arms is k, the number of updates an error
  consumes before the next one is released;
* no rollback - rollback would erase the very variable being measured;
* re-ask once per arm, after every error has consumed its budget;
* the A-task fit is measured on every arm, because the B updates touch
  ``byte_head`` and ``joint_readout``, which the A task also uses. Unlike A.5,
  retention here can actually fail.

Modes: ``run`` (formal), ``recover`` (independent readback of one arm).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_a5_22_5 as a5
from taiji.u2_history import atomic_load

CONTRACT = ROOT / "plans/manifests/taiji_u2_a6_budget_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_a6_budget_run_20261116.json"


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("a5_runner", ROOT / "scripts/training/run_a5_22_5.py"),
        ("subject", ROOT / "taiji/u2_subject.py"),
        ("events", ROOT / "taiji/u2_events.py"),
        ("a2_runner", ROOT / "scripts/training/run_a2_byte_20261115.py"),
    ):
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    for key, path in contract["start_bundles"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != contract["start_bundle_sha256"][key]:
            raise ValueError(f"start_bundle_mismatch:{key}")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract["plan"].get("cap_cpu"):
            raise ValueError("missing_measured_cap")


def load_start(contract: dict, seed: int):
    model = a5.build_subject(contract)
    model.load_state_dict(
        torch.load(ROOT / contract["start_bundles"][str(seed)], weights_only=False)["model"]
    )
    model.eval()
    return model


def freeze_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def arm(
    contract: dict, seed: int, budget: int, codes: dict, errors, templates, data, spend
) -> dict:
    """One arm: every error consumes ``budget`` updates, then one re-ask."""
    model = load_start(contract, seed)
    freeze_parameters(model)
    a5.set_trainable(model, contract)
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    updates = 0
    for item, template in zip(errors, templates, strict=True):
        public = a5.p4_2.column_public(item["example"])
        code = codes[item["example"]["id"]]
        for _ in range(budget):
            spend()
            optimizer.zero_grad(set_to_none=True)
            loss = a5.record_loss(
                model,
                public,
                item["example"]["id"],
                item["records"][template],
                item["target"],
                code,
            )
            loss.backward()
            optimizer.step()
            model.parameter_revision += 1
            updates += 1
    rows = a5.b_pass(model, errors, codes, lambda position: templates[position])
    a_rows = [a5.a_round(model, example, codes[example["id"]]) for example in data["fit"]]
    return {
        "budget": budget,
        "updates": updates,
        "second_pass_accuracy": a5.accuracy(rows),
        "a_fit_accuracy": a5.accuracy(a_rows),
        "a_fit_digest": a5.rows_digest(a_rows),
        "correct": sum(int(row["correct"]) for row in rows),
        "parameter_abs_sum": round(
            sum(
                float(parameter.detach().abs().sum())
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            8,
        ),
    }


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(ROOT / contract["data_root"])
    if a5.value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    root = ROOT / contract["run_root"]
    root.mkdir(parents=True, exist_ok=True)
    items, skipped = a5.task_b_examples(data, contract)
    template_of = lambda index: index % len(contract["task_b"]["templates"])  # noqa: E731

    def spend() -> None:
        if time.process_time() - cpu_start > float(contract["plan"]["cap_cpu"]):
            raise BudgetStop("a6_budget_stop")
        if time.perf_counter() - wall_start > float(contract["plan"]["cap_wall"]):
            raise BudgetStop("a6_wall_stop")

    seed = int(contract["plan"]["seeds"][0])
    start = load_start(contract, seed)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(start, head, prior, data["train"] + data["fit"], contract)
    first = a5.b_pass(start, items, codes, template_of)
    baseline_a = a5.accuracy(
        [a5.a_round(start, example, codes[example["id"]]) for example in data["fit"]]
    )
    errors = [items[index] for index, row in enumerate(first) if not row["correct"]]
    error_templates = [template_of(index) for index, row in enumerate(first) if not row["correct"]]

    arms = []
    for budget in [int(value) for value in contract["plan"]["budgets"]]:
        spend()
        entry = arm(contract, seed, budget, codes, errors, error_templates, data, spend)
        entry["damage"] = round(baseline_a - entry["a_fit_accuracy"], 10)
        arms.append(entry)
        (root / f"arm-k{budget}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

    rates = [entry["second_pass_accuracy"] for entry in arms]
    budgets = [entry["budget"] for entry in arms]
    monotone = all(
        later >= earlier - 1e-12 for earlier, later in zip(rates, rates[1:], strict=False)
    )
    endpoint = rates[-1] - rates[0]
    gate = float(contract["criteria"]["correction_gate"])
    zero_band = float(contract["criteria"]["zero_band"])
    retention_gate = float(contract["criteria"]["retention_gate"])
    broken = [entry["budget"] for entry in arms if entry["damage"] > retention_gate]
    if broken:
        verdict = "retention_broken_by_budget"
    elif max(rates) >= gate and monotone and endpoint >= 2 / len(errors):
        verdict = "correction_scales_with_budget"
    elif max(rates) <= zero_band:
        verdict = "correction_zero_at_every_budget"
    else:
        verdict = "correction_plateaus_below_gate"

    payload = {
        "schema": "taiji-u2-a6-budget-run-v1",
        "card": "22.5.A.6",
        "date": "2026-11-16",
        "verdict": verdict,
        "seed": seed,
        "denominator": {
            "task_b_examples": len(items),
            "first_pass_errors": len(errors),
            "skipped_unknown": len(skipped),
        },
        "first_pass_accuracy": a5.accuracy(first),
        "baseline_a_fit": baseline_a,
        "arms": arms,
        "rates": rates,
        "monotone": monotone,
        "endpoint_gain": round(endpoint, 10),
        "retention_broken_at": broken,
        "criteria": contract["criteria"],
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
            "cap_cpu": contract["plan"]["cap_cpu"],
            "cap_wall": contract["plan"]["cap_wall"],
        },
        "discipline": {
            "capability_claims": 0,
            "accepted_parent": False,
            "check_reads": 0,
            "confirm_reads": 0,
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "scope": (
            "dose response of the correction rate in the per-error update budget; one seed; "
            "no parent claim and no capability claim"
        ),
    }
    RUN_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "rates": rates,
                "budgets": budgets,
                "damage": [round(entry["damage"], 4) for entry in arms],
                "cost": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run",), default="run")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    try:
        guards(contract, args.mode)
    except ValueError as error:
        print(json.dumps({"verdict": "refused", "reason": str(error)}))
        return 2
    try:
        return run(contract)
    except BudgetStop as error:
        RUN_REPORT.write_text(
            json.dumps(
                {
                    "schema": "taiji-u2-a6-budget-run-v1",
                    "card": "22.5.A.6",
                    "verdict": "budget_stop",
                    "reason": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return 3
    except Exception as error:  # noqa: BLE001
        RUN_REPORT.write_text(
            json.dumps(
                {
                    "schema": "taiji-u2-a6-budget-run-v1",
                    "card": "22.5.A.6",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
