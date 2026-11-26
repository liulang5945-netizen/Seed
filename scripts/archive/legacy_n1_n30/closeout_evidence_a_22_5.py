"""Read-only durable update and resource audit for the completed evidence experiment."""

# ruff: noqa: E402 -- include startup in measured cost
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import load_learner
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import atomic_load, value_digest


def closeout():
    torch.set_num_threads(1)
    output = ROOT / "reports/taiji_u2_evidence_A_closeout_20261112.json"
    if output.exists():
        raise FileExistsError(output)
    contract = json.loads(
        (ROOT / "plans/manifests/taiji_u2_p3_4_5_closeout_contract_20261112.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_path = ROOT / "plans/manifests/taiji_u2_evidence_A_execution_20261112.json"
    report_path = ROOT / "reports/taiji_u2_evidence_A_learning_20261112.json"
    recovered_report = report_path.with_name(report_path.stem + "_v2.json")
    if recovered_report.exists():
        report_path = recovered_report
        manifest_path = manifest_path.with_name(manifest_path.stem + "_v2.json")
    execution = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    root = Path(execution["run_root"])
    ledger = [
        json.loads(line)
        for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    opened = [row for row in ledger if row["status"] == "opened"]
    accepted = [row for row in ledger if row["status"] == "accepted"]
    assert len(accepted) == result["accepted_checkpoints"]
    assert len(opened) >= len(accepted) and result["optimizer_steps"] >= len(accepted)
    audits = {}
    with StepGuard() as guard:
        for arm in execution["arm_order"]:
            rows = [row for row in accepted if row["arm"] == arm]
            beginnings = [row for row in opened if row["arm"] == arm]
            assert [row["update"] for row in rows] == list(range(1, len(rows) + 1))
            for row in rows:
                if time.process_time() > contract["cpu_cap"]:
                    raise RuntimeError("evidence_A_closeout_budget_stop")
                assert row["cursor"] == 2 * row["update"]
                checkpoint = root / arm / f"update-{row['update']:06}.bin"
                assert (
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["checkpoint_sha256"]
                )
                claim = atomic_load(root / arm / f"claim-{row['update']:06}.bin")
                assert claim == {
                    "update": row["update"],
                    "cursor": 2 * (row["update"] - 1),
                    "arm": arm,
                }
            audit = {
                "opened": len(beginnings),
                "accepted": len(rows),
                "planned": execution["arms"][arm]["proposed_updates"],
            }
            if rows:
                last = root / arm / f"update-{rows[-1]['update']:06}.bin"
                model, optimizer, payload = load_learner(last)
                assert payload["update"] == len(rows) and payload["cursor"] == 2 * len(rows)
                assert payload["data_digest"] == execution["data"]["digest"]
                assert payload["order"] == execution["data"]["order"]
                assert optimizer.state and max(
                    int(state["step"]) for state in optimizer.state.values()
                ) == len(rows)
                parameter_digest = value_digest(model.state_dict())
                audit.update(
                    {
                        "terminal_parameter_digest": parameter_digest,
                        "adam_parameter_states": len(optimizer.state),
                        "cursor": payload["cursor"],
                        "checkpoint_bytes": last.stat().st_size,
                    }
                )
                if arm in result["arms"]:
                    complete = result["arms"][arm]
                    assert parameter_digest == complete["final_parameter_digest"]
                    assert complete["updates"] == len(rows) == audit["planned"]
                    assert complete["final"]["denominator"] == 32
                    assert complete["final"]["correct"] == sum(
                        row["correct"] for row in complete["final"]["rows"]
                    )
                    audit.update(
                        {
                            "initial_fit": complete["initial"]["accuracy"],
                            "final_fit": complete["final"]["accuracy"],
                            "training_cpu": complete["training_cpu"],
                            "diagnostic_accuracy": {
                                name: item["accuracy"]
                                for name, item in complete["diagnostics"].items()
                            },
                            "first_update_credit": complete["first_update_credit"],
                        }
                    )
            audits[arm] = audit
    complete_run = all(
        arm in result["arms"] and audits[arm]["accepted"] == audits[arm]["planned"]
        for arm in execution["arm_order"]
    )
    if complete_run:
        assert (
            result["optimizer_steps"]
            == len(accepted)
            == len(opened)
            == sum(row["planned"] for row in audits.values())
        )
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    cpu = time.process_time()
    assert cpu <= contract["cpu_cap"] and time.monotonic() - STARTED <= contract["wall_cap"]
    report = {
        "schema": "taiji-u2-evidence-A-closeout-v1",
        "verdict": result["verdict"],
        "durable_run_complete": complete_run,
        "arms": audits,
        "original_optimizer_steps": result["optimizer_steps"],
        "accepted_checkpoint_digests_verified": len(accepted),
        "ledger_sha256": hashlib.sha256((root / "ledger.jsonl").read_bytes()).hexdigest(),
        "run_artifact_bytes": total_bytes,
        "original_accounted_cpu_seconds": result["accounted_cpu_seconds"],
        "optimizer_steps": guard.calls,
        "fit_predictions": 0,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "evidence_sha256": {
            str(path.relative_to(ROOT))
            .replace("\\", "/"): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in (manifest_path, report_path, Path(__file__))
        },
    }
    write_report(output, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "complete": complete_run,
                "accepted": len(accepted),
                "cpu_seconds": cpu,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    closeout()
