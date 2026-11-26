"""Zero-update audit of shared-fusion experiment endpoints and real Adam ownership."""

# ruff: noqa: E402 -- include startup in the audit cost
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
from scripts.training.closeout_task_goal_learning_22_5 import verify_score
from scripts.training.joint_token_learner_22_5 import load_learner
from scripts.training.run_joint_learning_22_5 import ADMISSION, EXECUTION, digest, sources
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import value_digest
from taiji.u2_task_executor import writer_lease

CONTRACT = ROOT / "plans/manifests/taiji_u2_joint_closeout_contract_20261113.json"
RAW_REPORT = ROOT / "reports/taiji_u2_joint_learning_20261113.json"
REPORT = ROOT / "reports/taiji_u2_joint_learning_closeout_20261113.json"


def audit():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, pin in contract["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == pin
        )
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    result = json.loads(RAW_REPORT.read_text(encoding="utf-8"))
    assert result["execution_sha256"] == digest(EXECUTION)
    assert execution["admission_sha256"] == digest(ADMISSION)
    assert execution["source_pins"] == sources()
    root = Path(execution["run_root"])
    ledger_path = root / "ledger.jsonl"
    with writer_lease(root / "train.lock"):
        pass
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    records: dict[str, Any] = {}

    def guard():
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("joint_closeout_budget_stop")

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
                assert row["checkpoint_sha256"] == digest(row["checkpoint"])
                assert row["cursor"] == row["update"]
                assert row["step_cpu"] <= execution["max_single_update_cpu"]
            record: dict[str, Any] = {
                "opened": len(opened),
                "accepted": len(accepted),
                "quarantined": len(rejected),
            }
            if name in result["results"]:
                measured = result["results"][name]
                assert len(opened) == len(accepted) + 1 and len(rejected) == 1
                assert len(accepted) == measured["selected_updates"]
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
                assert rejected[0]["update"] == len(opened)
                model, optimizer, payload = load_learner(measured["checkpoint"])
                initial, inherited, first = load_learner(recipe["initial"])
                assert digest(recipe["initial"]) == recipe["initial_sha256"]
                assert payload["update"] == payload["cursor"] == len(accepted)
                assert payload["order"] == admission["order"]
                assert payload["data_digest"] == admission["data"]["data_digest"]
                assert first["update"] == first["cursor"] == 0
                assert value_digest(initial.state_dict()) == recipe["parameter_digest"]
                assert value_digest(inherited.state_dict()) == recipe["optimizer_digest"]
                assert value_digest(model.state_dict()) == measured["parameter_digest"]
                assert model.parameter_revision == initial.parameter_revision + len(accepted)
                initial_max = max(int(state["step"]) for state in inherited.state.values())
                selected_max = max(int(state["step"]) for state in optimizer.state.values())
                assert selected_max == initial_max + len(accepted)
                fusion_ages = {}
                for key, parameter in model.named_parameters():
                    if key.startswith("joint_fusion."):
                        assert dict(initial.named_parameters())[key] not in inherited.state
                        fusion_ages[key] = int(optimizer.state[parameter]["step"])
                        assert fusion_ages[key] == len(accepted)
                        assert measured["first_update_credit"][key]["max_parameter_change"] > 0
                        assert key not in dict(model.policy_named_parameters())
                assert len(fusion_ages) == (12 if name.startswith("joint-") else 0)
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
                        "initial_adam_max_step": initial_max,
                        "selected_adam_max_step": selected_max,
                        "fusion_adam_ages": fusion_ages,
                        "train_accuracy": measured["final"]["train"]["accuracy"],
                        "dev_accuracy": measured["final"]["dev"]["accuracy"],
                        "dev_by_round": measured["dev_by_round"],
                        "controls": {
                            key: item["accuracy"] for key, item in measured["controls"].items()
                        },
                    }
                )
            records[name] = record
    complete = all(name in result["results"] for name in admission["groups"])
    spread = None
    if complete:
        assert sum(item["opened"] for item in records.values()) == result["learner_optimizer_steps"]
        cpus = [item["selected_cpu"] for item in records.values()]
        spread = (max(cpus) - min(cpus)) / max(cpus)
        assert result["training_compute_matched"] == (
            spread <= execution["endpoint_compute_tolerance"]
        )
    guard()
    report = {
        "schema": "taiji-u2-joint-learning-closeout-v1",
        "verdict": result["verdict"],
        "complete": complete,
        "groups": records,
        "accepted_files_verified": sum(item["accepted"] for item in records.values()),
        "quarantined_files_verified": sum(item["quarantined"] for item in records.values()),
        "original_learner_optimizer_steps": result["learner_optimizer_steps"],
        "endpoint_compute_spread": spread,
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
            str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
            for path in (ADMISSION, EXECUTION, RAW_REPORT, ledger_path)
        },
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("verdict", "complete", "accepted_files_verified", "cpu_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    audit()
