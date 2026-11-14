"""P3.8.2b Native credit transfer: full-capsule checkpoint preflight.

Phase ``precheck`` performs ZERO optimizer steps and reads no dev data.  It
proves that a NativeLearnerCapsule built from the real P3.7 research parent can
be saved and restored in an independent process, and prices the 130 CPU cap
before the six Adam proposals are admitted.  Later phases add the proposals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training.joint_token_learner_22_5 import load_learner  # noqa: E402
from taiji.u2_credit_learner import (  # noqa: E402
    BODY_TENSORS,
    HEAD_TENSORS,
    REGISTRY_TENSORS,
    build_capsule,
    capsule_summary,
    headless_runtime,
    load_capsule,
    save_capsule,
    validate_capsule,
)
from taiji.u2_history import atomic_load, atomic_save, detached_tree, value_digest  # noqa: E402

PARENT = ROOT / (
    "data/experiments/taiji_u2/joint_learning/1794554480667378100/joint-query/update-000758.bin"
)
PARENT_SHA256 = "c8f2359b0bef3085bce88a0a0a8ee7eff5f66a0bc614d2bf52e7ef361638b017"
P37_REPORT = ROOT / "reports/taiji_u2_joint_learning_20261113.json"
REPORT = ROOT / "reports/taiji_u2_native_credit_transfer_precheck_20261114.json"
CPU_CAP = 130.0
WALL_CAP = 260.0
HEAD_SEED = 2026111311
PROPOSALS = 4
FAULTS = 2
MAX_ADAM_CALLS = PROPOSALS + FAULTS


def source_pins() -> dict[str, str]:
    names = (
        "scripts/training/verify_native_credit_transfer_22_5.py",
        "scripts/training/joint_token_learner_22_5.py",
        "scripts/training/credit_update_22_5.py",
        "taiji/u2_credit_update.py",
        "taiji/u2_credit_learner.py",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def build_from_parent() -> tuple[dict[str, Any], dict[str, Any]]:
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA256
    model, optimizer, payload = load_learner(PARENT)
    body_ids = [name for name, _ in model.named_parameters()]
    assert len(body_ids) == BODY_TENSORS
    assert payload["architecture"] == "native-joint-query"
    assert payload["update"] == payload["cursor"] == 758

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(HEAD_SEED)
        head = torch.nn.Linear(8, 4)
    head_state = detached_tree(head.state_dict())
    assert list(head_state) == ["weight", "bias"]
    assert head_state["weight"].shape == (4, 8)

    pins = dict(payload["source_pins"])
    pins.update(source_pins())
    capsule = build_capsule(
        native_runtime={
            "architecture": payload["architecture"],
            "model_bundle": payload["model"],
            "training": payload.get("training"),
            "torch_rng": payload["torch_rng"],
            "field_count": 44,
            "fact_history": [],
            "pending": [],
            "resources": {"cpu_seconds": 0.0},
        },
        learning_state={
            "body_moment": detached_tree(optimizer.state_dict())["state"],
            "content_head": head_state,
            "content_head_moment": {},
            "label_source": "not_read_in_precheck",
            "class_weights": [],
            "parameter_ancestor": value_digest(model.state_dict()),
            "proposal_clock": 0,
            "parameter_version": 0,
        },
        sampling={
            "episode_cursor": 0,
            "first_answer_prefix": "",
            "prefix_source": "not_sampled_in_precheck",
            "data_rights": {"train": False, "dev": False, "check": False, "final": False},
            "sampling_rng": torch.get_rng_state(),
        },
        proposal=None,
        identity={
            "factory_schema": "taiji-u2-joint-token-query-v1",
            "source_pins": pins,
            "parameter_registry": {
                "parameter_ids": body_ids + ["content_head.weight", "content_head.bias"],
                "owners": {"body": BODY_TENSORS, "content_head": HEAD_TENSORS},
            },
            "body_digest": value_digest(model.state_dict()),
        },
    )
    facts = {
        "architecture": payload["architecture"],
        "parent_update": payload["update"],
        "parent_cursor": payload["cursor"],
        "body_parameters": sum(item.numel() for item in model.parameters()),
        "body_state_dict_tensors": len(model.state_dict()),
        "body_named_parameters": len(body_ids),
        "body_moment_entries": len(optimizer.state),
        "head_parameters": sum(item.numel() for item in head.parameters()),
        "registry_tensors": REGISTRY_TENSORS,
        "parameter_digest": value_digest(model.state_dict()),
    }
    return capsule, facts


def worker(checkpoint: str, destination: str) -> None:
    torch.set_num_threads(1)
    capsule = load_capsule(checkpoint)
    atomic_save(
        {
            "capsule_digest": value_digest(capsule),
            "summary": capsule_summary(capsule),
            "headless_field_count": headless_runtime(capsule)["field_count"],
            "pid": __import__("os").getpid(),
        },
        destination,
    )


def independent_restore(capsule: dict[str, Any], root: Path) -> dict[str, Any]:
    checkpoint = root / "initial_capsule.bin"
    destination = root / "worker_receipt.bin"
    save_capsule(checkpoint, capsule)
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(checkpoint),
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=120)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    assert process.returncode == 0, stderr[-2000:] + stdout[-500:]
    receipt = atomic_load(destination)
    assert receipt["pid"] != __import__("os").getpid()
    assert receipt["capsule_digest"] == value_digest(capsule)
    assert receipt["summary"]["registry_tensors"] == REGISTRY_TENSORS
    assert receipt["headless_field_count"] == 44
    return receipt


def tamper_rejected(capsule: dict[str, Any], root: Path) -> bool:
    clean = root / "initial_capsule.bin"
    broken = root / "tampered_capsule.bin"
    raw = bytearray(clean.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    broken.write_bytes(raw)
    try:
        load_capsule(broken)
    except (ValueError, RuntimeError):
        return True
    return False


def price() -> dict[str, Any]:
    p37 = json.loads(P37_REPORT.read_text(encoding="utf-8"))
    total_cpu = float(p37["cpu_seconds"])
    calls = int(p37["learner_optimizer_steps"])
    assert calls > 0 and total_cpu > 0
    per_call = total_cpu / calls
    # worst case: each proposal re-evaluates the objective over the eight alpha
    # scales, so one call is charged as eight objective evaluations.
    worst_per_proposal = 8 * per_call
    estimate = (
        MAX_ADAM_CALLS * worst_per_proposal
        + 4 * (per_call * 4)  # two pre/post independent restore pairs
        + 8.0  # interpreter startup and source-pin hashing
    )
    return {
        "p37_total_cpu": total_cpu,
        "p37_learner_calls": calls,
        "p37_cpu_per_call": round(per_call, 6),
        "worst_case_per_proposal_cpu": round(worst_per_proposal, 6),
        "estimated_six_proposals_plus_recovery_cpu": round(estimate, 4),
        "cpu_cap": CPU_CAP,
        "wall_cap": WALL_CAP,
        "margin_cpu": round(CPU_CAP - estimate, 4),
        "basis": "P3.7 measured cpu_seconds / learner_optimizer_steps (both read from the frozen report), 8 alpha evaluations per proposal, 4 restore pairs, fixed startup allowance; two-question objective priced conservatively at 8x",
    }


def precheck() -> int:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    root = ROOT / "data" / "experiments" / "taiji_u2" / "credit_transfer_precheck"
    root.mkdir(parents=True, exist_ok=True)
    capsule, facts = build_from_parent()
    assert validate_capsule(capsule)["identity"]["parameter_registry"]["owners"]["body"] == 214
    receipt = independent_restore(capsule, root)
    rejected = tamper_rejected(capsule, root)
    assert rejected, "tampered capsule was accepted"
    pricing = price()
    cpu = time.process_time() - started_cpu
    wall = time.monotonic() - started_wall
    admitted = pricing["margin_cpu"] > 0 and cpu < CPU_CAP and wall < WALL_CAP
    verdict = "precheck_passed_admitted" if admitted else "not_admitted_for_budget"
    report = {
        "schema": "taiji-u2-native-credit-transfer-precheck-v1",
        "date": "2026-11-14",
        "phase": "precheck",
        "card": "22.5.P3.8.2b",
        "parent": {
            "path": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": PARENT_SHA256,
            **facts,
        },
        "checks": {
            "parent_sha_matches": True,
            "parent_loads_with_source_pins": True,
            "registry_216_tensors": facts["registry_tensors"] == 216,
            "formal_operator_byte_identical": hashlib.sha256(
                (ROOT / "scripts/training/credit_update_22_5.py").read_bytes()
            ).digest()
            == hashlib.sha256((ROOT / "taiji/u2_credit_update.py").read_bytes()).digest(),
            "initial_capsule_saved": True,
            "independent_process_restore_matches": receipt["capsule_digest"]
            == value_digest(capsule),
            "worker_pid_distinct": True,
            "headless_runtime_isolated": receipt["headless_field_count"] == 44,
            "tampered_capsule_rejected": rejected,
            "first_nonempty_moment_check": "deferred_to_first_proposal",
        },
        "pricing": pricing,
        "source_pins": source_pins(),
        "measured": {"cpu_seconds": round(cpu, 4), "wall_seconds": round(wall, 4)},
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "dev_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "new_task_data_reads": 0,
        },
        "verdict": verdict,
        "next_action": (
            "execute 4 normal + 2 fault Adam proposals under the frozen contract"
            if admitted
            else "write the budget gap and close the card"
        ),
    }
    temporary = REPORT.with_suffix(".json.writing")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(REPORT)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "cpu": round(cpu, 4),
                "wall": round(wall, 4),
                "margin_cpu": pricing["margin_cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if admitted else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--phase", default="precheck", choices=["precheck"])
    args = parser.parse_args()
    if args.worker:
        assert args.output
        worker(args.worker, args.output)
        return 0
    return precheck()


if __name__ == "__main__":
    sys.exit(main())
