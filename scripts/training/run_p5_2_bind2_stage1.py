"""P5.2: stage-1 full-dose run for object-centric binding v2 (arm A only).

This is the pre-registered test, not the pilot. It trains the candidate binding
with the same encoder, data, truth-anchored visible labels and 458-parameter read
scaffold as P5.1, evaluates at 1844 and 3688 steps, saves both checkpoints, and
adds the slot-knockout diagnostic. No capability or parent claim is written here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training import run_p4_3_22_5 as p4_3
from scripts.training import run_p4_4_22_5 as p4_4
from scripts.training.check_report_run_conditions import write_report
from scripts.training.run_p5_0_bind2 import (
    build_v2,
    digest_outputs,
    predicted_pairs,
    root_path,
    scaffold_accuracy,
    scaffold_loss,
)
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_bind_2_stage1_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_bind2_stage1_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_bind2_stage1_guard_20261115.json"


class BudgetStop(RuntimeError):
    pass


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("subject", ROOT / "taiji/u2_subject.py"),
        ("pilot_runner", ROOT / "scripts/training/run_p5_0_bind2.py"),
        ("p4_2_runner", ROOT / "scripts/training/run_p4_2_22_5.py"),
        ("p4_3_runner", ROOT / "scripts/training/run_p4_3_22_5.py"),
        ("p4_4_runner", ROOT / "scripts/training/run_p4_4_22_5.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("column_pathway", ROOT / "taiji/u2_observation_columns.py"),
        ("design_manifest", ROOT / "plans/manifests/taiji_u2_bind_2_design_20261115.json"),
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


def knockout(model, scaffold, examples: list[dict], labels: dict) -> dict:
    """Replace one slot's object vector with the scene's slot mean and re-read.

    A scene counts when the unperturbed prediction is exactly the three-pair
    truth and some single-slot knockout drops exactly that slot's pair while
    keeping the other two. The replacement is the scene-level slot mean, the
    single-scene equivalent of the design card's batch mean.
    """
    hits = total = zero_effect = 0
    with torch.no_grad():
        for example in examples:
            truth = labels[example["id"]]
            if len(truth) != 3:
                continue
            state = p4_2.prepared(model, example)
            z = model.object_z(state)
            colour, position = scaffold(z)
            pairs = predicted_pairs(colour, position)
            if sorted(pairs) != sorted(truth):
                continue
            total += 1
            replacement = z.mean(1, keepdim=True).expand_as(z)
            lost_any = False
            any_effect = False
            for slot in range(3):
                perturbed = z.clone()
                perturbed[:, slot] = replacement[:, slot]
                colour2, position2 = scaffold(perturbed)
                pairs2 = set(predicted_pairs(colour2, position2))
                lost = set(pairs) - pairs2
                kept = set(pairs) & pairs2
                if len(lost) == 1 and len(kept) == 2:
                    lost_any = True
                if pairs2 != set(pairs):
                    any_effect = True
            hits += int(lost_any)
            zero_effect += int(not any_effect)
    return {
        "scenes": total,
        "hits": hits,
        "rate": hits / total if total else 0.0,
        "zero_effect_scenes": zero_effect,
        "measurable": total > 0,
    }


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    labels = p4_4.label_cache(data)
    encoder = p4_3.load_p4_1_bundle(contract)
    root = root_path(contract["run_root"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("stage1_run_already_started")
    root.mkdir(parents=True, exist_ok=True)

    def budget() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("stage1_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("stage1_wall_stop")

    model = build_v2(contract, encoder)
    frozen = {name for name, _ in model.named_parameters() if name.startswith("encoder.rgb.")}
    names = p4_2.set_requires_grad(model, {name for name, _ in model.named_parameters()} - frozen)
    model._binding_steps = 2
    model._binding_anchor = True
    torch.manual_seed(int(contract["candidate"]["seed"]))
    scaffold = p4_3.ObjectScaffold()
    optimizer = p4_2.optimizer_for(
        [p for p in model.parameters() if p.requires_grad] + list(scaffold.parameters()),
        contract["optimizer"],
    )
    doses = [int(value) for value in contract["dose_points"]]
    rows = []
    status = "max_dose"
    try:
        for update in range(1, max(doses) + 1):
            budget()
            batch = p4_2.batch_at(data["train"], int(contract["batch"]), update)
            optimizer.zero_grad(set_to_none=True)
            loss = scaffold_loss(model, scaffold, batch, labels)
            loss.backward()
            if not all(
                p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in [*model.parameters(), *scaffold.parameters()]
            ):
                raise ValueError("nonfinite_stage1_gradient")
            optimizer.step()
            model.parameter_revision += 1
            if update not in doses:
                continue
            train = scaffold_accuracy(model, scaffold, data["train"][:32], labels)
            fit = scaffold_accuracy(model, scaffold, data["fit"], labels)
            path = root / f"stage1-{update:06d}.pt"
            digest = p4_2.save_checkpoint(
                path,
                {
                    "model": {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    },
                    "scaffold": {
                        key: value.detach().clone() for key, value in scaffold.state_dict().items()
                    },
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "revision": model.parameter_revision,
                    "arm": {"name": "stage1", "steps": 2, "anchor": True},
                },
            )
            rows.append(
                {
                    "update": update,
                    "train": train,
                    "fit": fit,
                    "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "checkpoint_sha256": digest,
                }
            )
            if (
                update == doses[0]
                and fit["set_accuracy"] < contract["criteria"]["early_stop_set"]
                and (fit["pair_accuracy"] < contract["criteria"]["early_stop_pair"])
            ):
                status = "binding_cannot_form_slots"
                break
    except BudgetStop:
        status = "budget_stop"
    if status == "max_dose":
        status = "stage1_complete"

    if status in {"binding_cannot_form_slots", "budget_stop"}:
        verdict = status
        knockout_result = {}
        recovery = {}
    else:
        knockout_result = knockout(model, scaffold, data["fit"], labels)
        final = rows[-1]
        in_process = digest_outputs(model, scaffold, data["fit"][:3], labels)
        independent = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/training/run_p5_0_bind2.py"),
                "--recover",
                str(ROOT / final["checkpoint"]),
            ],
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
        main_ok = (
            final["fit"]["pair_accuracy"] >= contract["criteria"]["read_pair_gate"]
            and final["fit"]["set_accuracy"] >= contract["criteria"]["read_set_gate"]
            and final["fit"]["duplicate_rate"] <= contract["criteria"]["duplicate_rate_max"]
            and final["fit"]["count_accuracy"] >= contract["criteria"]["count_accuracy_gate"]
            and final["train"]["set_accuracy"] >= contract["criteria"]["train_set_gate"]
        )
        if not recovery["identical"]:
            verdict = "engineering_failure"
        elif knockout_result["measurable"] and knockout_result["rate"] == 0.0:
            verdict = "read_ignores_slots"
        elif main_ok and knockout_result["rate"] >= contract["criteria"]["knockout_gate"]:
            verdict = "binding_forms_object_slots"
        elif main_ok:
            verdict = "binding_partial"
        else:
            verdict = "binding_partial"

    payload = {
        "schema": "taiji-u2-bind2-stage1-v1",
        "card": "22.5.P5.2",
        "date": "2026-11-15",
        "status": status,
        "verdict": verdict,
        "candidate": contract["candidate"],
        "doses": rows,
        "knockout": knockout_result,
        "recovery": recovery,
        "trainable_parameters": len(names),
        "data_digest": contract["data_digest"],
        "parent_bundle_sha256": contract["p4_1_bundle_sha256"],
        "criteria": contract["criteria"],
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
        "scope": "stage 1 only: object-set formation with the read scaffold",
    }
    write_report(RUN_REPORT, payload)
    print(
        json.dumps({"verdict": verdict, "status": status, "doses": len(rows)}, ensure_ascii=False)
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    return run(contract)


if __name__ == "__main__":
    sys.exit(main())
