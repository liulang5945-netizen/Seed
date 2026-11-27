"""Read-only six-arm audit of CPU endpoints, provenance and inherited Adam state."""

# ruff: noqa: E402 -- account for startup in the audit
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
from scripts.training.run_task_goal_learning_22_5 import digest, sources
from scripts.training.task_goal_learner_22_5 import load_learner
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import value_digest
from taiji.u2_task_executor import writer_lease

CONTRACT = ROOT / "plans/manifests/taiji_u2_task_goal_closeout_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_task_goal_learning_closeout_20261113.json"


def verify_score(item):
    rows = item["rows"]
    assert len(rows) == item["denominator"] == 360
    assert item["correct"] == sum(row["correct"] for row in rows)
    assert item["accuracy"] == item["correct"] / 360
    for turn in (0, 1):
        assert len([row for row in rows if row["turn"] == turn]) == 180


def audit():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == contract["auditor_sha256"]
    execution_path = ROOT / "plans/manifests/taiji_u2_task_goal_learning_execution_20261113.json"
    admission_path = ROOT / "plans/manifests/taiji_u2_task_goal_learning_admission_20261113_v2.json"
    result_path = ROOT / "reports/taiji_u2_task_goal_learning_20261113.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert digest(execution_path) == result["execution_sha256"]
    assert digest(admission_path) == execution["admission_sha256"]
    assert sources() == execution["source_pins"]
    root = Path(execution["run_root"])
    ledger_path = root / "ledger.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    with writer_lease(root / "train.lock"):
        pass
    records: dict[str, Any] = {}

    def guard():
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("task_goal_closeout_budget_stop")

    with StepGuard() as blocked:
        for name, recipe in admission["groups"].items():
            opened = [row for row in ledger if row["learner"] == name and row["status"] == "opened"]
            accepted = [
                row for row in ledger if row["learner"] == name and row["status"] == "accepted"
            ]
            rejected = [
                row for row in ledger if row["learner"] == name and row["status"] == "quarantined"
            ]
            assert [row["update"] for row in opened] == list(range(1, len(opened) + 1))
            assert [row["update"] for row in accepted] == list(range(1, len(accepted) + 1))
            for row in [*accepted, *rejected]:
                guard()
                assert digest(row["checkpoint"]) == row["checkpoint_sha256"]
                assert row["cursor"] == row["update"]
                assert row["step_cpu"] <= execution["max_single_update_cpu"]
            record: dict[str, Any] = {
                "opened": len(opened),
                "accepted": len(accepted),
                "quarantined": len(rejected),
            }
            if name in result["results"]:
                measured = result["results"][name]
                assert len(accepted) == measured["selected_updates"]
                assert len(rejected) == 1 and len(opened) == len(accepted) + 1
                assert rejected[0]["update"] == len(opened)
                assert accepted[-1]["checkpoint"] == measured["checkpoint"]
                assert (
                    accepted[-1]["training_cpu"]
                    == measured["selected_training_cpu"]
                    <= execution["cpu_per_group"]
                )
                assert (
                    rejected[0]["training_cpu"]
                    == measured["spent_training_cpu"]
                    > execution["cpu_per_group"]
                )
                assert digest(recipe["initial"]) == recipe["initial_sha256"]
                model, optimizer, payload = load_learner(measured["checkpoint"])
                initial_model, initial_optimizer, initial = load_learner(recipe["initial"])
                assert payload["update"] == payload["cursor"] == len(accepted)
                assert payload["order"] == admission["order"]
                assert payload["data_digest"] == admission["data"]["data_digest"]
                assert value_digest(model.state_dict()) == measured["parameter_digest"]
                assert value_digest(initial_model.state_dict()) == recipe["parameter_digest"]
                assert value_digest(initial_optimizer.state_dict()) == recipe["optimizer_digest"]
                assert initial["update"] == initial["cursor"] == 0
                assert model.parameter_revision == initial_model.parameter_revision + len(accepted)
                before = max(int(value["step"]) for value in initial_optimizer.state.values())
                after = max(int(value["step"]) for value in optimizer.state.values())
                assert after == before + len(accepted)
                projection = getattr(model, "native_task_projection", None)
                projection_step = None
                if projection is not None:
                    assert (
                        initial_model.native_task_projection.weight not in initial_optimizer.state
                    )
                    projection_step = int(optimizer.state[projection.weight]["step"])
                    assert projection_step == len(accepted)
                    assert (
                        measured["first_update_credit"]["native_task_projection.weight"][
                            "max_parameter_change"
                        ]
                        > 0
                    )
                    assert "native_task_projection.weight" not in dict(
                        model.policy_named_parameters()
                    )
                for phase in ("initial", "final"):
                    for split in ("train", "dev"):
                        verify_score(measured[phase][split])
                for item in measured["controls"].values():
                    verify_score(item)
                    assert item["correct"] <= 180
                record.update(
                    {
                        "selected_cpu": measured["selected_training_cpu"],
                        "spent_cpu": measured["spent_training_cpu"],
                        "initial_adam_max_step": before,
                        "selected_adam_max_step": after,
                        "projection_adam_step": projection_step,
                        "train_accuracy": measured["final"]["train"]["accuracy"],
                        "dev_accuracy": measured["final"]["dev"]["accuracy"],
                        "dev_by_round": measured["dev_by_round"],
                        "controls": {
                            key: item["accuracy"] for key, item in measured["controls"].items()
                        },
                        "partial_input_bound_consistent": True,
                    }
                )
            records[name] = record
    complete = all(name in result["results"] for name in admission["groups"])
    if complete:
        assert sum(row["opened"] for row in records.values()) == result["learner_optimizer_steps"]
        assert all(
            row["opened"] == row["accepted"] + row["quarantined"] for row in records.values()
        )
        cpus = [row["selected_cpu"] for row in records.values()]
        spread = (max(cpus) - min(cpus)) / max(cpus)
        assert result["training_compute_matched"] == (
            spread <= execution["endpoint_compute_tolerance"]
        )
    else:
        spread = None
    guard()
    output = {
        "schema": "taiji-u2-task-goal-closeout-v1",
        "verdict": result["verdict"],
        "complete": complete,
        "groups": records,
        "endpoint_compute_spread": spread,
        "accepted_files_verified": sum(row["accepted"] for row in records.values()),
        "quarantined_files_verified": sum(row["quarantined"] for row in records.values()),
        "original_learner_optimizer_steps": result["learner_optimizer_steps"],
        "optimizer_steps": blocked.calls,
        "new_predictions": 0,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "artifact_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "evidence_sha256": {
            str(path.relative_to(ROOT))
            .replace("\\", "/"): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in (execution_path, admission_path, result_path, ledger_path)
        },
    }
    write_report(REPORT, output)
    print(
        json.dumps(
            {
                key: output[key]
                for key in ("verdict", "complete", "accepted_files_verified", "cpu_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    audit()
