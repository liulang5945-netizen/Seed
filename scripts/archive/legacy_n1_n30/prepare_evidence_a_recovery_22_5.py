"""Verify the accepted prefix and admit an engineering-only continuation amendment."""

# ruff: noqa: E402 -- include recovery preparation in the original CPU cap
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.evidence_a_learner_22_5 import load_learner
from scripts.training.first_a_support_22_5 import task_loss
from scripts.training.run_evidence_a_22_5 import MANIFEST, REPORT, digest, sources
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import atomic_load


def prepare():
    torch.set_num_threads(1)
    destination = MANIFEST.with_name(MANIFEST.stem + "_v2.json")
    if destination.exists():
        raise FileExistsError(destination)
    original = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["verdict"] == "engineering_failure"
    assert (
        report["arms"]["error"]
        == "AttributeError: 'EvidenceSubject' object has no attribute '_verification_cuts'"
    )
    assert report["optimizer_steps"] == report["accepted_checkpoints"] == 1000
    archive = ROOT / "scripts/archive/run_evidence_a_22_5_20261112_v1.py.txt"
    assert (
        hashlib.sha256(archive.read_text(encoding="utf-8-sig").encode()).hexdigest()
        == original["source_pins"]["scripts/training/run_evidence_a_22_5.py"]
    )
    for name, pin in original["source_pins"].items():
        if name != "scripts/training/run_evidence_a_22_5.py":
            assert (
                hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode()).hexdigest()
                == pin
            )
    root = Path(original["run_root"])
    ledger = [
        json.loads(line)
        for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    opens = [row for row in ledger if row["status"] == "opened"]
    accepted = [row for row in ledger if row["status"] == "accepted"]
    assert len(opens) == len(accepted) == 1000
    data = atomic_load(Path(original["data"]["run_root"]) / "data.bin")
    resumed = {}
    with StepGuard() as guard:
        for arm, count in (("native-old", 510), ("native-query", 490)):
            rows = [row for row in accepted if row["arm"] == arm]
            assert [row["update"] for row in rows] == list(range(1, count + 1))
            for row in rows:
                checkpoint = root / arm / f"update-{row['update']:06}.bin"
                assert digest(checkpoint) == row["checkpoint_sha256"]
                assert atomic_load(root / arm / f"claim-{row['update']:06}.bin") == {
                    "arm": arm,
                    "update": row["update"],
                    "cursor": 2 * (row["update"] - 1),
                }
            terminal, optimizer, payload = load_learner(root / arm / f"update-{count:06}.bin")
            assert payload["update"] == count and payload["cursor"] == count * 2
            assert optimizer.state
            if arm == "native-query":
                initial, _, _ = load_learner(original["arms"][arm]["initial"])
                first, _, _ = load_learner(root / arm / "update-000001.bin")
                batch = [data["train"][original["data"]["order"][i]] for i in range(2)]
                loss = task_loss(initial, batch)
                loss.backward()
                first_named = dict(first.named_parameters())
                credit = {
                    name: {
                        "gradient_norm": (
                            None if parameter.grad is None else float(parameter.grad.norm())
                        ),
                        "max_parameter_change": float(
                            (first_named[name].detach() - parameter.detach()).abs().max()
                        ),
                        "reconstructed_from_initial_and_accepted_step1": True,
                    }
                    for name, parameter in initial.named_parameters()
                    if name.startswith("evidence_reader.") or name == "encoder.rgb.weight"
                }
            else:
                credit = report["arms"][arm]["first_update_credit"]
            resumed[arm] = {
                "updates": count,
                "checkpoint_sha256": rows[-1]["checkpoint_sha256"],
                "training_cpu": rows[-1]["training_cpu"],
                "first_update_credit": credit,
            }
    contract = {
        **original,
        "schema": "taiji-u2-evidence-A-execution-v2",
        "source_pins": sources(),
        "resume_arms": resumed,
        "resume_ledger_sha256": digest(root / "ledger.jsonl"),
        "previous_report_sha256": digest(REPORT),
        "previous_optimizer_steps": 1000,
        "preparation_cpu_seconds": report["accounted_cpu_seconds"] + time.process_time(),
        "original_wall_origin_unix": REPORT.stat().st_mtime - report["wall_seconds"],
        "recovery_preparation_cpu_seconds": time.process_time(),
        "recovery_optimizer_steps": guard.calls,
        "amendment": "Default missing diagnostic-cut attribute to empty set; verified exact1000 accepted updates, preserve completed training and resume remaining4381. No change to model graph, optimizer, parents, data, doses, gates or original CPU/wall caps.",
        "prior_runner_archive": str(archive.relative_to(ROOT)).replace("\\", "/"),
    }
    assert contract["preparation_cpu_seconds"] < contract["cpu_cap"]
    destination.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "recovery_admitted",
                "accepted_updates": 1000,
                "remaining_updates": 4381,
                "accounted_cpu_before_resume": contract["preparation_cpu_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
