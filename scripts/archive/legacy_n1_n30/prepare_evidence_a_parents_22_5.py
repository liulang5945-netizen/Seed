"""Freeze research-parent checkpoints and exact next-batch recovery without updates."""

# ruff: noqa: E402 -- include startup in measured costs
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import (
    fresh_optimizer,
    load_learner,
    load_legacy_transformer,
    save_learner,
    source_pins,
)
from scripts.training.first_a_support_22_5 import generate_examples, task_loss
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_capacity_control import CapacitySubject
from taiji.u2_evidence_runtime import load_legacy_learner
from taiji.u2_evidence_subject import EvidenceSubject
from taiji.u2_history import (
    assert_values_equal,
    atomic_load,
    atomic_save,
    detached_tree,
    value_digest,
)
from taiji.u2_subject import U2Subject
from taiji.u2_task_executor import writer_lease

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_4_1_parent_contract_20261112.json"
MANIFEST = ROOT / "plans/manifests/taiji_u2_evidence_A_parents_20261112.json"
REPORT = ROOT / "reports/taiji_u2_evidence_A_parent_preflight_20261112.json"


def measurement(model, examples):
    before = value_digest(model.state_dict())
    named = dict(model.named_parameters())
    loss = task_loss(model, examples)
    values = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    gradients = dict(zip(named, detached_tree(values), strict=True))
    assert all(value is None or bool(torch.isfinite(value).all()) for value in gradients.values())
    assert value_digest(model.state_dict()) == before
    return {"loss": loss.detach(), "gradients": gradients}


def worker(jobs_path, destination):
    torch.set_num_threads(1)
    packet = atomic_load(jobs_path)
    rows = {}
    with StepGuard() as guard:
        for architecture, path in packet["paths"].items():
            model, optimizer, payload = load_learner(path)
            assert not optimizer.state and payload["update"] == payload["cursor"] == 0
            assert payload["data_digest"] == value_digest(packet["examples"])
            rows[architecture] = measurement(model, packet["examples"][:2])
    atomic_save(
        {
            "pid": os.getpid(),
            "rows": rows,
            "optimizer_steps": guard.calls,
            "cpu_seconds": time.process_time(),
        },
        destination,
    )


def verified_parent(arm, raw, closeout):
    path = Path(raw["arms"][arm]["checkpoint"])
    audit = closeout["checkpoint_audit"][arm]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["checkpoint_sha256"]
    return path, {
        "arm": arm,
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "base_parameter_digest": audit["final_parameter_digest"],
        "prior_P2_updates": audit["updates"],
        "capability_qualifications": [],
    }


def prepare():
    if MANIFEST.exists() or REPORT.exists():
        raise FileExistsError("evidence_A_parent_preflight_already_frozen")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    p2 = json.loads(
        (ROOT / "reports/taiji_u2_conditional_A_20261112.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (ROOT / "reports/taiji_u2_conditional_A_closeout_20261112.json").read_text(encoding="utf-8")
    )
    root = Path(get_external_path(f"data/experiments/taiji_u2/evidence_A_parents/{time.time_ns()}"))
    paths, rows, expected = {}, {}, {}
    with StepGuard() as guard, writer_lease(root / "prepare.lock"):
        native_path, native_lineage = verified_parent("native-conditional", p2, audit)
        transformer_path, transformer_lineage = verified_parent("transformer-token", p2, audit)
        base, _, native_verified = load_legacy_learner(native_path)
        transformer, transformer_verified = load_legacy_transformer(transformer_path)
        assert value_digest(base.state_dict()) == native_lineage["base_parameter_digest"]
        assert (
            value_digest(transformer.state_dict()) == transformer_lineage["base_parameter_digest"]
        )
        old, _ = U2Subject.from_bundle(base.export_bundle(base.initial("parent-copy")))
        old.parameter_revision += 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(contract["query_seed"])
            query = EvidenceSubject.from_parent(base)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(contract["capacity_seed"])
            capacity = CapacitySubject.from_parent(base)
        models = {
            "native-old": old,
            "native-query": query,
            "native-capacity": capacity,
            "transformer": transformer,
        }
        examples = generate_examples("price", contract["price_fixture_examples"])
        atomic_save(examples, root / "price-fixtures.bin")
        for architecture, model in models.items():
            if time.process_time() > contract["cpu_cap"]:
                raise RuntimeError("parent_preflight_budget_stop")
            if architecture.startswith("native-"):
                for name, value in base.state_dict().items():
                    assert torch.equal(value, model.state_dict()[name]), (architecture, name)
            optimizer = fresh_optimizer(model)
            assert not optimizer.state
            lineage = transformer_lineage if architecture == "transformer" else native_lineage
            path = root / architecture / "initial.bin"
            path.parent.mkdir(parents=True)
            digest = save_learner(
                path,
                model,
                optimizer,
                architecture,
                lineage,
                update=0,
                cursor=0,
                order=list(range(len(examples))),
                data_digest=value_digest(examples),
            )
            restored, other, payload = load_learner(path)
            assert_values_equal(model.state_dict(), restored.state_dict())
            assert_values_equal(optimizer.state_dict(), other.state_dict())
            measured = measurement(model, examples[:2])
            assert_values_equal(
                measured, measurement(restored, examples[:2]), "same_process_next_batch"
            )
            owner_norms = {
                name: float(value.norm())
                for name, value in measured["gradients"].items()
                if name.startswith("evidence_reader.") and value is not None
            }
            if architecture in {"native-query", "native-capacity"}:
                assert len(owner_norms) == len(list(model.evidence_reader.parameters()))
                assert all(value > 0 for value in owner_norms.values())
            paths[architecture] = str(path)
            expected[architecture] = measured
            rows[architecture] = {
                "checkpoint": str(path),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "payload_digest": digest,
                "parameter_digest": value_digest(model.state_dict()),
                "parameters": sum(p.numel() for p in model.parameters()),
                "parameter_tensors": len(list(model.parameters())),
                "revision": model.parameter_revision,
                "new_owner_gradient_norms": owner_norms,
                "lineage": lineage,
                "optimizer_state_entries": len(other.state),
                "source_pins": payload["source_pins"],
            }
        atomic_save({"paths": paths, "examples": examples}, root / "jobs.bin")
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__)),
                "--worker",
                str(root / "jobs.bin"),
                "--output",
                str(root / "worker.bin"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=240,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-6000:])
        recovered = atomic_load(root / "worker.bin")
        assert recovered["pid"] != os.getpid() and recovered["optimizer_steps"] == 0
        assert_values_equal(expected, recovered["rows"], "independent_next_batch")
    cpu = time.process_time() + recovered["cpu_seconds"]
    assert cpu <= contract["cpu_cap"] and time.monotonic() - STARTED <= contract["wall_cap"]
    manifest = {
        "schema": "taiji-u2-evidence-A-parents-v1",
        "scope": contract["scope"],
        "parents": rows,
        "native_parent_verification": native_verified,
        "transformer_parent_verification": transformer_verified,
        "source_pins": source_pins(),
        "price_fixture_digest": value_digest(examples),
        "price_fixtures": str(root / "price-fixtures.bin"),
        "run_root": str(root),
        "optimizer_steps": guard.calls,
        "fit_reads": 0,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": "taiji-u2-evidence-A-parent-preflight-v1",
        "verdict": "passed_parent_preflight_only",
        "parents": {
            name: {key: value for key, value in row.items() if key != "source_pins"}
            for name, row in rows.items()
        },
        "independent_process_cases": len(expected),
        "parameter_gradient_comparisons": sum(len(row["gradients"]) for row in expected.values()),
        "max_gradient_difference": 0.0,
        "optimizer_steps": guard.calls,
        "fit_reads": 0,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch": torch.__version__,
        },
        "verifier_sha256": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8-sig").encode()
        ).hexdigest(),
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
                "parameters": {name: row["parameters"] for name, row in rows.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
    else:
        prepare()
