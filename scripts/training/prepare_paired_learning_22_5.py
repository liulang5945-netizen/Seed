"""Price inherited paired learners and validate source-bound empty episode checkpoints."""

# ruff: noqa: E402 -- include startup in the preparation resource receipt
from __future__ import annotations

import hashlib
import json
import shutil
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
from scripts.training.paired_compute_22_5 import shadow_optimizer
from scripts.training.paired_episode_22_5 import episode_loss
from seed_platform.paths import get_external_path
from taiji.u2_history import assert_values_equal, atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_5_4_admission_contract_20261113.json"
MANIFEST = ROOT / "plans/manifests/taiji_u2_paired_learning_admission_20261113.json"
REPORT = ROOT / "reports/taiji_u2_paired_learning_admission_20261113.json"


def sources():
    result = source_pins()
    for name in (
        "paired_world_22_5.py",
        "paired_episode_22_5.py",
        "paired_compute_22_5.py",
        "prepare_paired_learning_22_5.py",
    ):
        path = ROOT / "scripts/training" / name
        result[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(
            path.read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
    return result


def prepare():
    if MANIFEST.exists() or REPORT.exists():
        raise FileExistsError("paired_learning_already_prepared")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    compute = json.loads(
        (ROOT / "plans/manifests/taiji_u2_paired_compute_accepted_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    task = json.loads((ROOT / compute["data_manifest"]).read_text(encoding="utf-8"))
    design = json.loads(
        (ROOT / "plans/manifests/taiji_u2_p3_5_2_learning_contract_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    data = atomic_load(task["data_path"])
    assert value_digest(data) == task["data_digest"] == compute["data_digest"]
    order = torch.randperm(
        180, generator=torch.Generator().manual_seed(contract["order_seed"])
    ).tolist()
    root = Path(get_external_path(f"data/experiments/taiji_u2/paired_learning/{time.time_ns()}"))
    rows = {}
    for name, parent in compute["parents"].items():
        assert (
            hashlib.sha256(Path(parent["checkpoint"]).read_bytes()).hexdigest() == parent["sha256"]
        )
        model, optimizer, payload = load_learner(parent["checkpoint"])
        before = value_digest(model.state_dict()), value_digest(optimizer.state_dict())
        initial = root / f"{name}-initial.bin"
        lineage = {
            **payload["lineage"],
            "paired_task_parent_sha256": parent["sha256"],
            "old_task_updates": payload["update"],
            "optimizer_policy": "inherit real moments and counters",
        }
        save_learner(
            initial,
            model,
            optimizer,
            name,
            lineage,
            update=0,
            cursor=0,
            order=order,
            data_digest=task["data_digest"],
        )
        restored, other, loaded = load_learner(initial)
        assert loaded["update"] == loaded["cursor"] == 0 and optimizer.state and other.state
        assert_values_equal(optimizer.state_dict(), other.state_dict())
        episode = data["train"]["episodes"][order[0]]
        first_loss = episode_loss(model, data["train"], episode)
        next_loss = episode_loss(restored, data["train"], episode)
        first = torch.autograd.grad(first_loss, list(model.parameters()), allow_unused=True)
        second = torch.autograd.grad(next_loss, list(restored.parameters()), allow_unused=True)
        assert_values_equal(first, second, "paired-inherited-checkpoint-next-gradient")
        assert_values_equal(first_loss, next_loss, "paired-inherited-checkpoint-next-loss")
        temporary, shadow = shadow_optimizer(model, optimizer)
        samples = []
        for index in contract["price_episode_indices"]:
            started = time.process_time()
            model.zero_grad(set_to_none=True)
            loss = episode_loss(model, data["train"], data["train"]["episodes"][index])
            loss.backward()
            for source, target in zip(model.parameters(), temporary, strict=True):
                target.grad = None if source.grad is None else source.grad.detach().clone()
            shadow.step()
            samples.append(time.process_time() - started)
        assert before == (value_digest(model.state_dict()), value_digest(optimizer.state_dict()))
        last_step_bytes = (
            initial.stat().st_size
            + 2 * sum(p.numel() * p.element_size() for p in model.parameters())
            + 1024 * len(list(model.parameters()))
        )
        median = statistics.median(samples)
        assert median * contract["maximum_updates"] > contract["train_cpu_per_learner"]
        rows[name] = {
            "initial": str(initial),
            "initial_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
            "parameter_digest": before[0],
            "optimizer_digest": before[1],
            "price_samples": samples,
            "median_training_cpu": median,
            "estimated_updates": int(contract["train_cpu_per_learner"] / median),
            "estimated_checkpoint_bytes": last_step_bytes,
            "inherited_adam_states": len(optimizer.state),
            "next_loss_and_gradient_exact": True,
        }
    maximum_artifacts = sum(
        row["estimated_checkpoint_bytes"] * (contract["maximum_updates"] + 1)
        for row in rows.values()
    )
    free = shutil.disk_usage(root).free
    assert free > maximum_artifacts * 1.25
    cpu = time.process_time()
    assert cpu < contract["cpu_cap"] and time.monotonic() - STARTED < contract["wall_cap"]
    manifest = {
        "schema": "taiji-u2-paired-learning-admission-v1",
        "status": "prepared_pending_runner",
        "run_root": str(root),
        "learners": rows,
        "data": task,
        "order": order,
        "source_pins": sources(),
        "contract": contract,
        "design": design,
        "checkpoint_estimated_max_bytes": maximum_artifacts,
        "disk_free": free,
        "learner_optimizer_steps": 0,
        "shadow_optimizer_steps": 16,
        "model_task_evaluations": 0,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": cpu,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        **manifest,
        "schema": "taiji-u2-paired-learning-admission-report-v1",
        "verdict": "passed_inherited_checkpoint_and_real_adam_price",
        "optimizer_steps": 16,
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {
            "device": "cpu",
            "threads": 1,
            "dtype": "float32",
            "torch": torch.__version__,
        },
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cpu_seconds": cpu,
                "prices": {name: row["median_training_cpu"] for name, row in rows.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
