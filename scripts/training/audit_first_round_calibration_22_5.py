"""Audit frozen calibration controls, proposal clocks and body-only export."""

# ruff: noqa: E402 -- include imports in audit resource accounting
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

from scripts.training import calibrate_content_credit_22_5 as content
from scripts.training import calibrate_first_round_22_5 as original
from scripts.training import calibrate_stable_credit_22_5 as stable
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_round_baseline_22_5 import make_batch
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import assert_values_equal, atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_calibration_audit_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_calibration_audit_20261113.json"
ACCEPTED = ROOT / "plans/manifests/taiji_u2_first_round_reference_20261113.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_score(item, partition):
    assert item["denominator"] == len(item["rows"]) == len(partition["episodes"]) == 180
    for row, episode in zip(item["rows"], partition["episodes"], strict=True):
        assert (
            row["episode_id"] == episode["id"] and row["target_hex"] == episode["targets"][0].hex()
        )
        raw = bytes.fromhex(row["output_hex"])
        try:
            raw.decode("utf-8", errors="strict")
            valid = True
        except UnicodeDecodeError:
            valid = False
        assert valid == row["valid_utf8"]
        assert row["correct"] == bool(row["eos"] and valid and raw == episode["targets"][0])
    assert item["correct"] == sum(row["correct"] for row in item["rows"])
    assert item["accuracy"] == item["correct"] / 180
    assert sum(value["denominator"] for value in item["by_question"].values()) == 180
    for key, value in item["by_question"].items():
        rows = [row for row in item["rows"] if row["question_hex"] == key]
        assert len(rows) == value["denominator"] and value["accuracy"] == sum(
            row["correct"] for row in rows
        ) / len(rows)


def audit():
    if REPORT.exists() or ACCEPTED.exists():
        raise FileExistsError("calibration_already_audited")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == contract["auditor_sha256"]
    original_contract = json.loads(original.CONTRACT.read_text(encoding="utf-8"))
    data = atomic_load(original_contract["data_path"])
    assert value_digest(data) == original_contract["data_digest"]
    batch = make_batch(data["train"])
    labels, classes, weights = content.labels_for(batch)
    records, identities = {}, {}
    with StepGuard() as blocked:
        for kind, module in (("byte", original), ("content", content), ("stable", stable)):
            report = json.loads(module.REPORT.read_text(encoding="utf-8"))
            study = json.loads(module.CONTRACT.read_text(encoding="utf-8"))
            assert report["contract_sha256"] == digest(module.CONTRACT)
            assert study["source_pins"] == module.sources()
            root = Path(report["run_root"])
            checkpoint = atomic_load(report["checkpoint"])
            assert digest(report["checkpoint"]) == report["checkpoint_sha256"]
            files = sorted(
                [*root.glob("update-*.bin"), *root.glob("attempt-*.bin"), root / "selected.bin"]
            )
            if (root / "overshoot.bin").exists():
                files.append(root / "overshoot.bin")
            for path in files:
                if time.process_time() > contract["cpu_cap"]:
                    raise RuntimeError("calibration_audit_budget_stop")
                payload = atomic_load(path)
                assert payload["data_digest"] == original_contract["data_digest"]
                assert payload["data_order"] == batch["episode_ids"]
                assert payload["examples_seen"] == 180 * payload["update"]
                assert payload["source_pins"] == (
                    original.sources() if kind == "byte" else content.sources()
                )
                if kind != "byte":
                    assert payload["labels"] == labels
                    assert_values_equal(payload["weights"], weights)
                calls = payload["optimizer_calls"] if kind == "stable" else payload["update"]
                assert len(payload["optimizer"]["state"]) == (
                    0 if calls == 0 else 21 if kind == "byte" else 23
                )
                if kind == "stable":
                    assert payload["update_rule_sources"] == stable.sources()
                    assert 0 <= payload["accepted_parameter_updates"] <= calls <= payload["update"]
                for state in payload["optimizer"]["state"].values():
                    assert int(state["step"]) == calls
                identities[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
            assert checkpoint["training_cpu"] <= study["training_cpu_cap"]
            assert checkpoint["update"] == report.get(
                "selected_updates", report.get("selected_attempts")
            )
            assert report["cpu_seconds"] <= study["cpu_cap"]
            assert report["wall_seconds"] <= study["wall_cap"]
            if kind == "stable":
                prefix = [
                    row for row in report["decisions"] if row["attempt"] <= checkpoint["update"]
                ]
                assert (
                    sum(row["accepted"] for row in prefix)
                    == checkpoint["accepted_parameter_updates"]
                    == report["selected_accepted_parameter_updates"]
                )
                assert len(report["decisions"]) == report["attempts"]
                for row in report["decisions"]:
                    if row["accepted"]:
                        assert row["scale"] in study["scales"] and row["slope"] < 0
                        assert (
                            row["after"]
                            <= row["before"]
                            + study["armijo"] * row["scale"] * row["slope"]
                            + study["tolerance"]
                        )
                assert (
                    atomic_load(root / "overshoot.bin")["training_cpu"] > study["training_cpu_cap"]
                )
            for split in ("train", "dev"):
                verify_score(report["final"][split], data[split])
            for value in report["controls"].values():
                verify_score(value, data["dev"])
                assert value["accuracy"] <= 0.5
            if kind == "byte":
                model, optimizer = original.restore(checkpoint, original_contract["data_digest"])
                measurement = original.measure(model, batch)
            else:
                model, head, optimizer = module.restore(checkpoint, study, batch)
                measurement = content.measure(model, head, batch, classes, weights)
                inference = atomic_load(root / "inference-only.bin")
                assert_values_equal(inference["parameters"], checkpoint["parameters"])
                exported, _ = original.restore(inference, original_contract["data_digest"])
                assert sum(p.numel() for p in exported.parameters()) == 65169
                assert "labels" not in inference and "head" not in inference
                identities[
                    str((root / "inference-only.bin").relative_to(ROOT)).replace("\\", "/")
                ] = digest(root / "inference-only.bin")
            recovered = atomic_load(root / "worker.bin")["result"]
            assert_values_equal(measurement, recovered["measurement"])
            assert_values_equal(report["final"]["train"], recovered["generated"])
            assert recovered["optimizer_digest"] == value_digest(optimizer.state_dict())
            records[kind] = {
                "verdict": report["verdict"],
                "files_verified": len(files),
                "update": checkpoint["update"],
                "train": report["final"]["train"]["accuracy"],
                "dev": report["final"]["dev"]["accuracy"],
                "data_digest": checkpoint["data_digest"],
            }
            identities[str(module.REPORT.relative_to(ROOT)).replace("\\", "/")] = digest(
                module.REPORT
            )
    cpu = time.process_time()
    assert cpu <= contract["cpu_cap"] and time.monotonic() - STARTED <= contract["wall_cap"]
    result = {
        "schema": "taiji-u2-calibration-audit-v1",
        "verdict": "passed_calibration_evidence_audit",
        "studies": records,
        "files_verified": sum(row["files_verified"] for row in records.values()),
        "evidence_sha256": identities,
        "optimizer_steps": blocked.calls,
        "teacher_gradient_rechecks": 3,
        "new_free_generation_evaluations": 0,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    write_report(REPORT, result)
    reference = json.loads(stable.REPORT.read_text(encoding="utf-8"))
    assert reference["verdict"] == "first_round_calibrated_stable_credit"
    body = Path(reference["run_root"]) / "inference-only.bin"
    accepted = {
        "schema": "taiji-u2-first-round-reference-v1",
        "status": "calibrated_first_round_reference",
        "role": "public-input first-round neural reference",
        "checkpoint": str(body),
        "sha256": digest(body),
        "parameters": 65169,
        "train_correct": reference["final"]["train"]["correct"],
        "dev_correct": reference["final"]["dev"]["correct"],
        "denominator_per_partition": reference["final"]["dev"]["denominator"],
        "native_parent_qualified": False,
        "data_digest": original_contract["data_digest"],
        "audit_report_sha256": digest(REPORT),
        "run_conditions": result["run_conditions"],
    }
    ACCEPTED.write_text(json.dumps(accepted, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "files_verified": result["files_verified"],
                "cpu_seconds": cpu,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    audit()
