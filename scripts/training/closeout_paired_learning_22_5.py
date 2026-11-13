"""Audit common-CPU endpoints, isolated overshoots and inherited real optimizer state."""

# ruff: noqa: E402 -- include imports in audit cost
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import load_learner
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import value_digest


def audit():
    torch.set_num_threads(1)
    output = ROOT / "reports/taiji_u2_paired_learning_closeout_20261113.json"
    if output.exists():
        raise FileExistsError(output)
    contract = json.loads(
        (ROOT / "plans/manifests/taiji_u2_paired_closeout_contract_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    execution_path = ROOT / "plans/manifests/taiji_u2_paired_learning_execution_20261113.json"
    report_path = ROOT / "reports/taiji_u2_paired_learning_20261113.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    result = json.loads(report_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(execution_path.read_bytes()).hexdigest() == result["execution_sha256"]
    admission = json.loads(
        (ROOT / "plans/manifests/taiji_u2_paired_learning_admission_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    root = Path(execution["run_root"])
    ledger_path = root / "ledger.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    accepted = [row for row in ledger if row["status"] == "accepted"]
    quarantined = [row for row in ledger if row["status"] == "quarantined"]
    opened = [row for row in ledger if row["status"] == "opened"]
    audits = {}
    with StepGuard() as blocked:
        for name in admission["learners"]:
            rows = [row for row in accepted if row["learner"] == name]
            q = [row for row in quarantined if row["learner"] == name]
            assert [row["update"] for row in rows] == list(range(1, len(rows) + 1))
            for row in [*rows, *q]:
                if time.process_time() > contract["cpu_cap"]:
                    raise RuntimeError("paired_closeout_budget_stop")
                assert (
                    hashlib.sha256(Path(row["checkpoint"]).read_bytes()).hexdigest()
                    == row["checkpoint_sha256"]
                )
            record: dict[str, Any] = {"accepted": len(rows), "quarantined": len(q)}
            if name in result["results"]:
                measured = result["results"][name]
                assert len(rows) == measured["selected_updates"] and len(q) == 1
                assert (
                    rows[-1]["training_cpu"]
                    == measured["selected_training_cpu"]
                    <= execution["train_cpu"]
                )
                assert (
                    q[0]["training_cpu"] == measured["spent_training_cpu"] > execution["train_cpu"]
                )
                assert q[0]["update"] == len(rows) + 1
                assert q[0]["step_cpu"] <= execution["max_single_update_cpu"]
                assert rows[-1]["checkpoint"] == measured["checkpoint"]
                model, optimizer, payload = load_learner(measured["checkpoint"])
                _, inherited, initial = load_learner(admission["learners"][name]["initial"])
                assert payload["update"] == payload["cursor"] == len(rows)
                assert payload["order"] == admission["order"]
                assert payload["data_digest"] == admission["data"]["data_digest"]
                assert value_digest(model.state_dict()) == measured["parameter_digest"]
                before_max = max(int(state["step"]) for state in inherited.state.values())
                after_max = max(int(state["step"]) for state in optimizer.state.values())
                assert after_max == before_max + len(rows)
                assert initial["update"] == initial["cursor"] == 0
                for split in ("train", "dev"):
                    endpoint = measured["final"][split]
                    assert len(endpoint["rows"]) == endpoint["denominator"] == 360
                    assert endpoint["correct"] == sum(row["correct"] for row in endpoint["rows"])
                record.update(
                    {
                        "selected_cpu": measured["selected_training_cpu"],
                        "spent_cpu": measured["spent_training_cpu"],
                        "initial_adam_max_step": before_max,
                        "selected_adam_max_step": after_max,
                        "adam_parameter_states": len(optimizer.state),
                        "train_accuracy": measured["final"]["train"]["accuracy"],
                        "dev_accuracy": measured["final"]["dev"]["accuracy"],
                        "control_accuracy": {
                            k: v["accuracy"] for k, v in measured["controls"].items()
                        },
                        "partial_input_bound_consistent": all(
                            v["correct"] <= 180 for v in measured["controls"].values()
                        ),
                    }
                )
            audits[name] = record
    complete = all(name in result["results"] for name in admission["learners"])
    if complete:
        assert len(opened) == len(accepted) + len(quarantined) == result["optimizer_steps"]
    cpu = time.process_time()
    assert cpu < contract["cpu_cap"] and time.monotonic() - STARTED < contract["wall_cap"]
    receipt = {
        "schema": "taiji-u2-paired-learning-closeout-v1",
        "verdict": result["verdict"],
        "complete": complete,
        "learners": audits,
        "accepted_files_verified": len(accepted),
        "quarantined_files_verified": len(quarantined),
        "original_model_optimizer_steps": result["optimizer_steps"],
        "optimizer_steps": blocked.calls,
        "new_predictions": 0,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "run_artifact_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "evidence_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (execution_path, report_path, ledger_path)
            if p.is_relative_to(ROOT)
        },
    }
    write_report(output, receipt)
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "complete": complete,
                "accepted": len(accepted),
                "quarantined": len(quarantined),
                "cpu_seconds": cpu,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    audit()
