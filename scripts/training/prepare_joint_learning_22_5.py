"""Price four actual-history arms after shared-token subject recovery acceptance."""

# ruff: noqa: E402 -- account for import costs
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import source_pins
from scripts.training.joint_token_learner_22_5 import EXTRA, load_learner, load_parent, save_learner
from scripts.training.paired_compute_22_5 import shadow_optimizer
from scripts.training.task_history_loss_22_5 import history_loss
from seed_platform.paths import get_external_path
from taiji.u2_history import assert_values_equal, atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_joint_learning_price_contract_20261113.json"
ADMISSION = ROOT / "plans/manifests/taiji_u2_joint_learning_admission_20261113.json"
REPORT = ROOT / "reports/taiji_u2_joint_learning_price_20261113.json"


def sources():
    pins = source_pins(EXTRA)
    for name in (
        "prepare_joint_learning_22_5.py",
        "task_history_loss_22_5.py",
        "paired_episode_22_5.py",
        "paired_world_22_5.py",
        "paired_compute_22_5.py",
    ):
        relative = f"scripts/training/{name}"
        pins[relative] = hashlib.sha256(
            (ROOT / relative).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
    return pins


def gradient(model, partition, episode):
    loss = history_loss(model, partition, episode, "actual-self")["loss"]
    values = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    return loss.detach(), [None if value is None else value.detach() for value in values]


def prepare():
    if REPORT.exists() or ADMISSION.exists():
        raise FileExistsError("joint_learning_already_priced")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    engineering = json.loads(
        (ROOT / "plans/manifests/taiji_u2_joint_subject_accepted_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    assert engineering["status"] == "shared_task_credit_updates_and_recovery_verified"
    task = json.loads(
        (ROOT / "plans/manifests/taiji_u2_joint_subject_contract_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    data = atomic_load(task["data_path"])
    assert value_digest(data) == task["data_digest"]
    order = torch.randperm(
        180, generator=torch.Generator().manual_seed(contract["order_seed"])
    ).tolist()
    root = Path(get_external_path(f"data/experiments/taiji_u2/joint_learning/{time.time_ns()}"))
    rows = {}
    shadow_steps = 0
    for name, recipe in contract["groups"].items():
        if name in {"native-base", "transformer"}:
            parent_name = "D" if name == "native-base" else "TF-actual"
            model, optimizer, payload = load_parent(parent_name)
            model.parameter_revision += 1
            source_sha = task["parents"][parent_name]["sha256"]
        else:
            parent = engineering["parents"][recipe["mode"]]
            assert (
                hashlib.sha256(Path(parent["checkpoint"]).read_bytes()).hexdigest()
                == parent["sha256"]
            )
            model, optimizer, payload = load_learner(parent["checkpoint"])
            source_sha = parent["sha256"]
        before = value_digest(model.state_dict()), value_digest(optimizer.state_dict())
        initial = root / f"{name}-initial.bin"
        lineage = {
            **payload["lineage"],
            "source_checkpoint_sha256": source_sha,
            "group": name,
            "history_source": "actual-self",
        }
        save_learner(
            initial,
            model,
            optimizer,
            recipe["architecture"],
            lineage,
            update=0,
            cursor=0,
            order=order,
            data_digest=task["data_digest"],
        )
        restored, restored_opt, _ = load_learner(initial)
        assert_values_equal(optimizer.state_dict(), restored_opt.state_dict())
        assert_values_equal(
            gradient(model, data["train"], data["train"]["episodes"][order[0]]),
            gradient(restored, data["train"], data["train"]["episodes"][order[0]]),
            "joint-learning.next-gradient",
        )
        temporary, shadow = shadow_optimizer(model, optimizer)
        prices, blocked = [], 0
        for index in contract["price_indices"]:
            started = time.process_time()
            model.zero_grad(set_to_none=True)
            result = history_loss(
                model, data["train"], data["train"]["episodes"][index], "actual-self"
            )
            result["loss"].backward()
            for source, target in zip(model.parameters(), temporary, strict=True):
                target.grad = None if source.grad is None else source.grad.detach().clone()
            shadow.step()
            shadow_steps += 1
            prices.append(time.process_time() - started)
            blocked += not result["first_ended"]
        assert before == (value_digest(model.state_dict()), value_digest(optimizer.state_dict()))
        assert statistics.median(prices) * contract["max_updates"] > contract["cpu_per_group"]
        rows[name] = {
            **recipe,
            "history": "actual-self",
            "initial": str(initial),
            "initial_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
            "parameter_digest": before[0],
            "optimizer_digest": before[1],
            "price_samples": prices,
            "median_training_cpu": statistics.median(prices),
            "blocked_price_episodes": blocked,
            "next_gradient_exact": True,
        }
    assert rows["joint-query"]["parameter_digest"] == rows["joint-goal"]["parameter_digest"]
    assert rows["joint-query"]["optimizer_digest"] == rows["joint-goal"]["optimizer_digest"]
    assert shadow_steps == contract["shadow_optimizer_steps"]
    cpu = time.process_time()
    assert cpu < contract["cpu_cap"] and time.monotonic() - STARTED < contract["wall_cap"]
    admission = {
        "schema": "taiji-u2-joint-learning-admission-v1",
        "groups": rows,
        "contract": contract,
        "data": {"data_path": task["data_path"], "data_digest": task["data_digest"]},
        "order": order,
        "run_root": str(root),
        "source_pins": sources(),
        "shadow_optimizer_steps": shadow_steps,
        "learner_optimizer_steps": 0,
        "model_accuracy_evaluations": 0,
        "check_reads": 0,
        "final_reads": 0,
        "cpu_seconds": cpu,
    }
    ADMISSION.write_text(json.dumps(admission, indent=2) + "\n", encoding="utf-8")
    write_report(
        REPORT,
        {
            **admission,
            "schema": "taiji-u2-joint-learning-price-v1",
            "verdict": "passed_inherited_joint_learning_price",
            "optimizer_steps": shadow_steps,
            "wall_seconds": time.monotonic() - STARTED,
            "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        },
    )
    print(
        json.dumps(
            {
                "verdict": "passed_inherited_joint_learning_price",
                "cpu_seconds": cpu,
                "prices": {name: value["median_training_cpu"] for name, value in rows.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
