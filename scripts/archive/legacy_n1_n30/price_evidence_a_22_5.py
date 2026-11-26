"""Price the frozen four-parent graph and bind a development-only episode partition."""

# ruff: noqa: E402 -- include startup in measured costs
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import load_learner, save_learner, source_pins
from scripts.training.first_a_support_22_5 import generate_examples
from scripts.training.prepare_first_a_22_5 import gradients
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_history import assert_values_equal, atomic_load, atomic_save, value_digest
from taiji.u2_task_executor import writer_lease

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_4_2_price_contract_20261112.json"
MANIFEST = ROOT / "plans/manifests/taiji_u2_evidence_A_price_20261112.json"
REPORT = ROOT / "reports/taiji_u2_evidence_A_price_20261112.json"


def price():
    if MANIFEST.exists() or REPORT.exists():
        raise FileExistsError("evidence_A_price_already_frozen")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parents = json.loads(
        (ROOT / "plans/manifests/taiji_u2_evidence_A_parents_20261112.json").read_text(
            encoding="utf-8"
        )
    )
    assert parents["source_pins"] == source_pins()
    root = Path(get_external_path(f"data/experiments/taiji_u2/evidence_A_price/{time.time_ns()}"))
    with StepGuard() as guard, writer_lease(root / "prepare.lock"):
        start, end = contract["fit_recipe_range"]
        data = {
            "train": generate_examples("train", contract["train_count"]),
            "fit": generate_examples("fit", end)[start:end],
            "price": atomic_load(parents["price_fixtures"]),
        }
        assert value_digest(data["price"]) == parents["price_fixture_digest"]
        previous = json.loads(
            (ROOT / "plans/manifests/taiji_u2_conditional_A_execution_20261112_v2.json").read_text(
                encoding="utf-8"
            )
        )
        assert_values_equal(
            data["train"],
            atomic_load(Path(previous["run_root"]) / "data.bin")["train"],
            "same_train_recipe",
        )
        order = torch.randperm(
            len(data["train"]), generator=torch.Generator().manual_seed(contract["order_seed"])
        ).tolist()
        data_digest = value_digest(data)
        atomic_save(data, root / "data.bin")
        rows = {}
        for architecture, parent in parents["parents"].items():
            path = Path(parent["checkpoint"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == parent["checkpoint_sha256"]
            model, optimizer, payload = load_learner(path)
            assert not optimizer.state and payload["update"] == 0
            parameter_digest = value_digest(model.state_dict())
            assert parameter_digest == parent["parameter_digest"]
            directory = root / architecture
            directory.mkdir()
            initial = directory / "initial.bin"
            save_learner(
                initial,
                model,
                optimizer,
                architecture,
                payload["lineage"],
                update=0,
                cursor=0,
                order=order,
                data_digest=data_digest,
            )
            restored, other, bound = load_learner(initial)
            assert not other.state and bound["data_digest"] == data_digest
            first_loss, first_gradients = gradients(model, data["train"][:2])
            second_loss, second_gradients = gradients(restored, data["train"][:2])
            assert first_loss == second_loss
            assert_values_equal(first_gradients, second_gradients, "next_training_batch")
            samples = []
            for block in range(contract["price_blocks"]):
                if (
                    time.process_time() > contract["cpu_cap"]
                    or time.monotonic() - STARTED > contract["wall_cap"]
                ):
                    raise RuntimeError("evidence_A_price_budget_stop")
                started = time.process_time()
                for repeat in range(contract["price_inner_repeats"]):
                    offset = ((block + repeat) % 4) * 2
                    gradients(model, data["price"][offset : offset + 2])
                    with torch.no_grad():
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                shadow = parameter - 0.001 * parameter.grad / (
                                    parameter.grad.abs() + 1e-8
                                )
                                assert bool(torch.isfinite(shadow).all())
                samples.append((time.process_time() - started) / contract["price_inner_repeats"])
            median = statistics.median(samples)
            assert median > 0 and value_digest(model.state_dict()) == parameter_digest
            rows[architecture] = {
                "initial": str(initial),
                "initial_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
                "parameter_digest": parameter_digest,
                "median_step_cpu": median,
                "price_samples": samples,
                "next_training_batch_equal": True,
                "optimizer_state_entries": len(optimizer.state),
            }
    common = min(
        contract["target_training_cpu_per_arm"],
        0.95
        * min(row["median_step_cpu"] for row in rows.values())
        * contract["maximum_updates_per_arm"],
    )
    for row in rows.values():
        row["proposed_updates"] = int(common / row["median_step_cpu"])
        row["proposed_training_cpu"] = row["proposed_updates"] * row["median_step_cpu"]
        assert 1 <= row["proposed_updates"] <= contract["maximum_updates_per_arm"]
    planned = [row["proposed_training_cpu"] for row in rows.values()]
    assert (max(planned) - min(planned)) / max(planned) <= contract["training_cpu_pair_tolerance"]
    cpu = time.process_time()
    assert cpu <= contract["cpu_cap"] and time.monotonic() - STARTED <= contract["wall_cap"]
    pins = source_pins()
    for path in (Path(__file__), ROOT / "scripts/training/prepare_first_a_22_5.py"):
        pins[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(
            path.read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
    manifest = {
        "schema": "taiji-u2-evidence-A-price-v1",
        "status": "priced_pending_runner_admission",
        "arms": rows,
        "data": {
            "run_root": str(root),
            "digest": data_digest,
            "train": 64,
            "fit": 32,
            "fit_recipe_range": [start, end],
            "price": 8,
            "order": order,
            "public_rights": "forward consumes public fields; target is training/development supervision only; no physical private state",
            "model_fit_reads": 0,
        },
        "source_pins": pins,
        "price_scope": "forward/backward plus local first-Adam expression on temporary tensors; no optimizer or parameter update; checkpoint I/O and evaluation are separately budgeted",
        "main_gate": contract["main_gate"],
        "bootstrap": contract["bootstrap"],
        "scope_budget": contract["scope_budget"],
        "optimizer_steps": guard.calls,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": cpu,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        **manifest,
        "schema": "taiji-u2-evidence-A-price-report-v1",
        "verdict": "priced_zero_updates",
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch": torch.__version__,
        },
        "contract_sha256": hashlib.sha256(
            CONTRACT.read_text(encoding="utf-8-sig").encode()
        ).hexdigest(),
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cpu_seconds": cpu,
                "prices": {name: row["median_step_cpu"] for name, row in rows.items()},
                "proposed_updates": {name: row["proposed_updates"] for name, row in rows.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    price()
