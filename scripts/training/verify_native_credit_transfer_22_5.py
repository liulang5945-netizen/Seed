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

from scripts.training.first_a_support_22_5 import output  # noqa: E402
from scripts.training.joint_token_learner_22_5 import load_learner  # noqa: E402
from scripts.training.paired_episode_22_5 import start_episode  # noqa: E402
from scripts.training.task_history_loss_22_5 import history_loss  # noqa: E402
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
DATA = ROOT / "data/experiments/taiji_u2/paired_task/1794496638577954800/data.bin"
DATA_DIGEST = "65041f02e3a3410b307bc6e013d897ef747e7f890b3668db6b5852fdc9fdba1c"
CONTRACT = ROOT / "plans/manifests/taiji_u2_native_credit_transfer_proposals_v2_20261114.json"
CRITERIA_SHA256 = "c26b78229679a7740bd4d64576ebc767c2bfa7c4e17befd8a4f02e5c709a8541"
DUTY_REPORT = ROOT / "reports/taiji_u2_native_credit_transfer_duty_v2_20261114.json"
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


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert canonical_sha(contract["criteria"]) == CRITERIA_SHA256
    assert contract["criteria_sha256"] == CRITERIA_SHA256
    return contract


def train_partition() -> dict[str, Any]:
    data = atomic_load(DATA)
    assert value_digest(data) == DATA_DIGEST
    return data["train"]


def class_table(partition: dict[str, Any]) -> tuple[list[str], dict[str, int], list[float]]:
    answers = [bytes(episode["targets"][0]).hex() for episode in partition["episodes"]]
    order = sorted(set(answers))
    counts = {key: answers.count(key) for key in order}
    total, size = len(answers), len(order)
    weights = [total / (size * counts[key]) for key in order]
    return order, counts, weights


COLOUR_PAIRS = ((0, 1), (2, 3), (4, 5))


def coverage_indices(partition: dict[str, Any]) -> list[int]:
    """3 colour pairs x 2 orderings x 2 reverse flags = 12 train episodes.

    audit.visible / audit.absent are colour indices in 0..5, not booleans;
    audit.position is 0..2 and audit.reverse is the boolean flag.
    """
    chosen: list[int] = []
    for position, (left, right) in enumerate(COLOUR_PAIRS):
        for visible, absent in ((left, right), (right, left)):
            for reverse in (False, True):
                match = [
                    index
                    for index, episode in enumerate(partition["episodes"])
                    if int(episode["audit"]["position"]) == position
                    and int(episode["audit"]["visible"]) == visible
                    and int(episode["audit"]["absent"]) == absent
                    and bool(episode["audit"]["reverse"]) is reverse
                ]
                assert match, ("coverage_cell_missing", position, visible, absent, reverse)
                chosen.append(match[0])
    assert len(chosen) == 12 and len(set(chosen)) == 12
    return chosen


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(torch.allclose(a, b, rtol=1e-5, atol=1e-7))


def duty() -> int:
    """Zero-Adam gradient-duty and class/coverage contract verification."""
    if DUTY_REPORT.exists():
        raise FileExistsError(DUTY_REPORT)
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = load_contract()
    model, optimizer, payload = load_learner(PARENT)
    partition = train_partition()
    order, counts, weights = class_table(partition)
    coverage = coverage_indices(partition)
    assert len(order) == contract["criteria"]["objective"]["head"]["out_features"]
    episode = partition["episodes"][0]
    initial = start_episode(model, partition, episode)
    _, context, _ = output(model, initial)
    omega = initial["logw"].detach().softmax(0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(HEAD_SEED)
        head = torch.nn.Linear(8, 4)
    content_log_probs = head(context).log_softmax(-1)
    mixed = torch.logsumexp(content_log_probs + torch.log(omega.clamp_min(1e-12))[:, None], 0)
    label = order.index(bytes(episode["targets"][0]).hex())
    j_content = -weights[label] * mixed[label]
    result = history_loss(model, partition, episode, "actual-self")
    parameters = list(model.parameters())
    g1 = torch.autograd.grad(result["first_ce"], parameters, retain_graph=True, allow_unused=True)
    g2 = torch.autograd.grad(
        result["second_ce"] + result["blocked_outcome"],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    gj = torch.autograd.grad(result["loss"], parameters, allow_unused=True)
    mismatches, none_rows, max_diff = [], 0, 0.0
    for index, (a, b, joint) in enumerate(zip(g1, g2, gj, strict=True)):
        if a is None and b is None and joint is None:
            none_rows += 1
            continue
        if a is None or b is None or joint is None:
            mismatches.append(index)
            continue
        combined = 0.5 * (a + b)
        max_diff = max(max_diff, float((combined - joint).abs().max()))
        if not _same(combined, joint):
            mismatches.append(index)
    # Red direction 1: a numerical perturbation must be detected.
    perturbed = [None if j is None else j * 1.01 for j in gj]
    red_numeric = not all(
        _same(0.5 * (a + b), p)
        for a, b, p in zip(g1, g2, perturbed, strict=True)
        if a is not None and b is not None and p is not None
    )
    # Red direction 2: using g1 twice must differ from the joint gradient somewhere.
    red_wrong_combination = not all(
        _same(a, j) for a, j in zip(g1, gj, strict=True) if a is not None and j is not None
    )
    passed = (
        not mismatches
        and max_diff <= 1e-6
        and red_numeric
        and result["history_source"] == "actual-self"
        and bool(result["first_ended"])
        and len(parameters) == BODY_TENSORS
    )
    report = {
        "schema": "taiji-u2-native-credit-transfer-duty-v2",
        "date": "2026-11-14",
        "card": "22.5.P3.8.2b",
        "phase": "duty_zero_adam",
        "contract": "plans/manifests/taiji_u2_native_credit_transfer_proposals_v2_20261114.json",
        "superseded_v1": {
            "contract": "plans/manifests/taiji_u2_native_credit_transfer_proposals_20261114.json",
            "reading": "reports/taiji_u2_native_credit_transfer_duty_20261114.json",
            "verdict": "engineering_failure",
        },
        "criteria_sha256": CRITERIA_SHA256,
        "parent_sha256": PARENT_SHA256,
        "data_digest": DATA_DIGEST,
        "classes": {"hex_sorted": order, "counts": counts, "weights": weights},
        "coverage_indices": coverage,
        "coverage_cells": "3 colour pairs (0,1)/(2,3)/(4,5) x 2 ordering directions x 2 reverse flags, bound to audit.position 0/1/2",
        "colour_pair_decoding": "audit.visible and audit.absent are colour indices 0..5; the request colour words are not separately decoded",
        "content_term": {
            "context_shape": list(context.shape),
            "omega_shape": list(omega.shape),
            "head": "Linear(8,4) seed 2026111311, 36 parameters",
            "label_hex": order[label],
            "label_weight": weights[label],
            "j_content": float(j_content.detach()),
            "j_first": float((result["first_ce"] + j_content).detach()),
            "omega_detached_from_state": True,
        },
        "gradient_duty": {
            "parameters": len(parameters),
            "all_none_rows": none_rows,
            "mismatches": mismatches,
            "max_abs_difference": max_diff,
            "tolerance": {"rtol": 1e-05, "atol": 1e-07, "max_abs_difference": 1e-06},
            "first_ce": float(result["first_ce"].detach()),
            "second_ce": float(result["second_ce"].detach()),
            "blocked": float(result["blocked_outcome"].detach()),
            "first_ended": bool(result["first_ended"]),
            "selection_credit": result["selection_credit"],
            "red_numeric_perturbation_detected": red_numeric,
            "red_wrong_combination_detected": red_wrong_combination,
        },
        "measured": {
            "cpu_seconds": round(time.process_time() - started_cpu, 4),
            "wall_seconds": round(time.monotonic() - started_wall, 4),
        },
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "dev_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "verdict": "duty_verified" if passed else "engineering_failure",
    }
    temporary = DUTY_REPORT.with_suffix(".json.writing")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(DUTY_REPORT)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "max_abs_difference": max_diff,
                "mismatches": len(mismatches),
                "red_numeric": red_numeric,
                "red_wrong_combination": red_wrong_combination,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.writing")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def proposals(r2: bool = False) -> int:
    """Four normal + two fault Adam proposals under the frozen v2 contract.

    r2 fills the four evidence gaps named by the independent audit; the v1
    receipt is left on disk untouched and is linked as predecessor.
    """
    import copy  # noqa: PLC0415

    from scripts.training.task_history_loss_22_5 import score_answer  # noqa: PLC0415
    from taiji.u2_credit_update import accepted_adam_step  # noqa: PLC0415
    from taiji.u2_learning_transaction import migrate_parameters  # noqa: PLC0415
    from taiji.u2_recovery import U2Runtime  # noqa: PLC0415

    report_path = ROOT / (
        "reports/taiji_u2_native_credit_transfer_proposals_r2_20261114.json"
        if r2
        else "reports/taiji_u2_native_credit_transfer_proposals_20261114.json"
    )
    if report_path.exists():
        raise FileExistsError(report_path)
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = load_contract()
    limits = contract["criteria"]["budget"]
    model, optimizer, payload = load_learner(PARENT)
    partition = train_partition()
    order, counts, weights = class_table(partition)
    coverage = coverage_indices(partition)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(HEAD_SEED)
        head = torch.nn.Linear(8, 4)
    optimizer.add_param_group({"params": list(head.parameters())})
    parameters = [*model.parameters(), *head.parameters()]
    assert len(parameters) == REGISTRY_TENSORS

    # Headless equivalence measured with a real perturbation probe: the content
    # head must be unable to change the body-only runtime distribution.
    probe_episode = partition["episodes"][coverage[0]]
    headless_initial = start_episode(model, partition, probe_episode)
    log_before, _, _ = output(model, headless_initial)
    body_digest_before = value_digest(model.state_dict())
    log_after, _, _ = output(model, headless_initial)
    body_digest_after = value_digest(model.state_dict())
    with torch.no_grad():
        head.weight.add_(3.0)
    log_perturbed, _, _ = output(model, headless_initial)
    with torch.no_grad():
        head.weight.sub_(3.0)
    headless_measurement = {
        "before_head_digest": value_digest(log_before.detach()),
        "after_head_digest": value_digest(log_after.detach()),
        "head_perturbation_digest": value_digest(log_perturbed.detach()),
        "body_digest_before": body_digest_before,
        "body_digest_after": body_digest_after,
        "runtime_unchanged_by_head": (
            value_digest(log_before.detach())
            == value_digest(log_after.detach())
            == value_digest(log_perturbed.detach())
        ),
        "head_attached_to_runtime": False,
        "content_label_indices_in_runtime": False,
    }

    rows: list[dict[str, Any]] = []
    capsule_root = ROOT / "data/experiments/taiji_u2/credit_transfer_proposals"
    capsule_root.mkdir(parents=True, exist_ok=True)
    counters = {"optimizer_steps": 0}
    u, v, accepted_commits = 0, 0, 0
    restore_checks: list[dict[str, Any]] = []
    headless: dict[str, Any] | None = None
    verdict = "engineering_transfer_verified"

    def objective_for(index: int):
        episode = partition["episodes"][coverage[index % len(coverage)]]
        label = order.index(bytes(episode["targets"][0]).hex())

        def objective():
            initial = start_episode(model, partition, episode)
            first_ce, _ = score_answer(model, initial, episode["targets"][0])
            _, context, _ = output(model, initial)
            omega = initial["logw"].detach().softmax(0)
            mixed = torch.logsumexp(
                head(context).log_softmax(-1) + torch.log(omega.clamp_min(1e-12))[:, None],
                0,
            )
            return first_ce - weights[label] * mixed[label]

        return objective

    def snapshot():
        return {
            "body": [item.detach().clone() for item in model.parameters()],
            "head": [item.detach().clone() for item in head.parameters()],
            "optimizer": detached_tree(optimizer.state_dict()),
            "rng": torch.get_rng_state().clone(),
        }

    def rollback(state):
        with torch.no_grad():
            for parameter, old in zip(model.parameters(), state["body"], strict=True):
                parameter.copy_(old)
            for parameter, old in zip(head.parameters(), state["head"], strict=True):
                parameter.copy_(old)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])

    def capsule(state_label: str, index: int):
        return build_capsule(
            native_runtime={
                "architecture": payload["architecture"],
                "field_count": 44,
                "fact_history": [],
                "pending": [],
                "rng": torch.get_rng_state().clone(),
                "resources": {"cpu_seconds": time.process_time() - started_cpu},
                "label": state_label,
            },
            learning_state={
                "body_moment": detached_tree(optimizer.state_dict())["state"],
                "content_head": detached_tree(head.state_dict()),
                "content_head_moment": {},
                "label_source": "train_targets[0]_hex",
                "class_weights": weights,
                "parameter_ancestor": value_digest(model.state_dict()),
                "proposal_clock": u,
                "parameter_version": v,
            },
            sampling={
                "episode_cursor": coverage[index % len(coverage)],
                "first_answer_prefix": "",
                "prefix_source": "actual_first_answer",
                "data_rights": {"train": True, "dev": False, "check": False, "final": False},
                "sampling_rng": torch.get_rng_state().clone(),
            },
            proposal=None,
            identity={
                "factory_schema": "taiji-u2-joint-token-query-v1",
                "source_pins": source_pins(),
                "parameter_registry": {
                    "parameter_ids": [name for name, _ in model.named_parameters()]
                    + ["content_head.weight", "content_head.bias"],
                    "owners": {"body": BODY_TENSORS, "content_head": HEAD_TENSORS},
                },
                "body_digest": value_digest(model.state_dict()),
            },
        )

    def guard():
        if time.process_time() - started_cpu > limits["cpu_cap"]:
            raise RuntimeError("budget_stop_cpu")
        if time.monotonic() - started_wall > limits["wall_cap"]:
            raise RuntimeError("budget_stop_wall")

    try:
        for call in range(6):
            guard()
            is_fault = call in (4, 5)
            episode_index = 0 if is_fault else call
            before = snapshot()
            before_digest = value_digest(model.state_dict())
            objective = objective_for(episode_index)
            before_loss = float(objective().detach())
            if r2 and not is_fault:
                pre_pack = capsule(f"before_call_{call + 1}", call)
                pre_receipt = independent_restore(pre_pack, capsule_root)
                restore_checks.append(
                    {
                        "point": f"before_call_{call + 1}",
                        "digest_matches": pre_receipt["capsule_digest"] == value_digest(pre_pack),
                    }
                )
            outcome = accepted_adam_step(parameters, optimizer, objective, counters)
            u += 1
            if is_fault:
                # Fault lands after the real Adam proposal and before any commit.
                # Roll back the whole capsule state, keep the call and its cost.
                rollback(before)
                after_digest = value_digest(model.state_dict())
                after_loss = float(objective().detach())
                rollback(before)
                restored = before_digest == after_digest
                rows.append(
                    {
                        "call": call + 1,
                        "kind": "fault_before_commit",
                        "proposal": outcome["reason"],
                        "before_loss": before_loss,
                        "restored_loss": after_loss,
                        "loss_restored": abs(after_loss - before_loss) <= 1e-9,
                        "digest_before": before_digest,
                        "digest_after": after_digest,
                        "parameter_digest_restored": restored,
                    }
                )
                if abs(after_loss - before_loss) > 1e-9 or not restored:
                    verdict = "engineering_failure"
                continue
            commit_receipt = None
            if outcome["accepted"]:
                v += 1
                accepted_commits += 1
                if r2:
                    # Real single-transaction parameter migration with history
                    # re-encoding, exercised in an isolated shadow runtime.
                    shadow = U2Runtime(model, optimizer=None, subject_id=f"p382b-commit-{call + 1}")
                    candidate = copy.deepcopy(model)
                    candidate.parameter_revision = model.parameter_revision + 1
                    migration = migrate_parameters(
                        shadow,
                        candidate,
                        optimizer=None,
                        acceptance=lambda _m, _s, _o: True,
                        cpu_budget_seconds=20,
                    )
                    commit_receipt = {
                        "transaction": "taiji.u2_learning_transaction.migrate_parameters",
                        "status": migration["status"],
                        "revision": migration["revision"],
                        "parameter_digest": migration["parameter_digest"],
                        "rebuild": migration["rebuild"],
                        "cpu_seconds": migration["cpu_seconds"],
                        "optimizer_digest_before": value_digest(optimizer.state_dict()),
                        "single_writer_guarantee": "provided_by_library_transaction_not_independently_measured",
                        "nonempty_history_reencoding": (
                            "not_exercised: the engineering runtime starts with an empty "
                            "CausalHistory, so rebuild.entries is 0"
                        ),
                    }
                    post_pack = capsule(f"after_call_{call + 1}", call)
                    post_receipt = independent_restore(post_pack, capsule_root)
                    restore_checks.append(
                        {
                            "point": f"after_call_{call + 1}",
                            "digest_matches": post_receipt["capsule_digest"]
                            == value_digest(post_pack),
                        }
                    )
            row = {
                "call": call + 1,
                "kind": "normal",
                **{
                    key: outcome[key]
                    for key in (
                        "accepted",
                        "scale",
                        "before",
                        "after",
                        "fallback",
                        "evaluations",
                        "reason",
                    )
                },
                "u": u,
                "v": v,
            }
            if commit_receipt is not None:
                row["commit_receipt"] = commit_receipt
            rows.append(row)
            # First non-empty moment: full capsule save + independent restore.
            if call == 0:
                pack = capsule("first_nonempty_moment", call)
                receipt = independent_restore(pack, capsule_root)
                restore_checks.append(
                    {
                        "point": "first_nonempty_moment",
                        "digest_matches": receipt["capsule_digest"] == value_digest(pack),
                    }
                )
                if receipt["capsule_digest"] != value_digest(pack):
                    verdict = "engineering_failure"
        headless = headless_measurement
        if counters["optimizer_steps"] > 6 or u > 6 or v > u:
            verdict = "engineering_failure"
        if accepted_commits == 0 and verdict == "engineering_transfer_verified":
            verdict = "inconclusive"
        if r2 and verdict == "engineering_transfer_verified":
            restore_ok = all(item["digest_matches"] for item in restore_checks)
            commits_ok = all(
                row.get("commit_receipt", {}).get("status") == "committed"
                for row in rows
                if row.get("kind") == "normal" and row.get("accepted")
            )
            if (
                not restore_ok
                or not commits_ok
                or not headless["runtime_unchanged_by_head"]
                or len(restore_checks) < 2 * accepted_commits
                or not all(
                    row.get("parameter_digest_restored") is True
                    for row in rows
                    if row.get("kind") == "fault_before_commit"
                )
            ):
                verdict = "engineering_failure"
    except RuntimeError as error:
        if str(error).startswith("budget_stop"):
            verdict = "budget_stop"
        else:
            verdict = "engineering_failure"
            rows.append({"error": str(error)})
    except Exception as error:  # noqa: BLE001
        verdict = "engineering_failure"
        rows.append({"error": f"{type(error).__name__}: {error}"})

    report = {
        "schema": "taiji-u2-native-credit-transfer-proposals-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.8.2b",
        "contract": CONTRACT.name,
        "criteria_sha256": CRITERIA_SHA256,
        "r2_evidence_completion": r2,
        "predecessor": (
            {
                "report": "reports/taiji_u2_native_credit_transfer_proposals_20261114.json",
                "audit": "reports/taiji_u2_native_credit_transfer_proposals_audit_20261114.json",
                "audited_verdict": "inconclusive_coverage_gaps",
                "note": "v1 receipt retained verbatim; r2 fills its four evidence gaps without changing the v2 criteria",
            }
            if r2
            else None
        ),
        "parent_sha256": PARENT_SHA256,
        "data_digest": DATA_DIGEST,
        "coverage_indices": coverage,
        "class_hex": order,
        "class_counts": counts,
        "class_weights": weights,
        "proposals": rows,
        "counters": {
            "real_adam_calls": counters["optimizer_steps"],
            "proposal_clock_u": u,
            "parameter_version_v": v,
            "accepted_commits": accepted_commits,
        },
        "restore_checks": restore_checks,
        "headless": headless,
        "measured": {
            "cpu_seconds": round(time.process_time() - started_cpu, 4),
            "wall_seconds": round(time.monotonic() - started_wall, 4),
        },
        "discipline": {
            "dev_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "criteria_sha_unchanged_after_reading": canonical_sha(contract["criteria"])
            == CRITERIA_SHA256,
        },
        "verdict": verdict,
    }
    _write_report(report_path, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "adam_calls": counters["optimizer_steps"],
                "u": u,
                "v": v,
                "cpu": report["measured"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "engineering_transfer_verified" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--phase",
        default="precheck",
        choices=["precheck", "duty", "proposals", "proposals-r2"],
    )
    args = parser.parse_args()
    if args.worker:
        assert args.output
        worker(args.worker, args.output)
        return 0
    if args.phase == "duty":
        return duty()
    if args.phase == "proposals":
        return proposals()
    if args.phase == "proposals-r2":
        return proposals(r2=True)
    return precheck()


if __name__ == "__main__":
    sys.exit(main())
