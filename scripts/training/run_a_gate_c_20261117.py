"""22.5.0 A-gate training under the Plan C FLOPs-matched re-freeze.

The U2 arm learns the first-A position task for 256 updates per seed. Train/fit
data are development-visible; the 192 development-check worlds are evaluated by a
fixed worker that reads the check labels internally and releases only aggregate
counts. Check exposure is 1/2; final is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    evaluate,
    generate_examples,
    load_learner,
    make_model,
    save_learner,
    task_loss,
)
from taiji.u2_history import assert_values_equal, value_digest  # noqa: E402
from taiji.u2_task_episode import NAMES, locate_public, position_word  # noqa: E402
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

PARTITION = ROOT / "reports/taiji_u2_partition_22_4_3_20261117.json"
REPORT = ROOT / "reports/taiji_u2_22_5_a_gate_c_20261117.json"
SEEDS = (20261110, 20261111, 20261112)
UPDATES = 256
BATCH = 2
CPU_CAP = 1200.0
WALL_CAP = 2400.0
CHECK_SEEDS = 192
A_GATE_MIN = 0.80
GOAL = "依据当前可核验观测，用自然中文回答请求。".encode()
LABELS = ("左边", "中间", "右边", "未知")
RGB_COLORS = (
    (220, 45, 45),
    (45, 90, 220),
    (45, 190, 80),
    (230, 190, 40),
    (170, 60, 200),
    (40, 190, 195),
)


def _check_seeds() -> list[int]:
    payload = json.loads(PARTITION.read_text(encoding="utf-8"))
    seeds = payload["manifest"]["splits"]["development-check"]["seeds"]
    if len(seeds) != CHECK_SEEDS:
        raise ValueError("check_split_size_mismatch")
    return [int(seed) for seed in seeds]


def _check_examples() -> list[dict]:
    examples = []
    for index, seed in enumerate(_check_seeds()):
        world = PhysicalWorld.reset(seed, {"body_count": 3})
        result = world.elapse(f"a-check-{index}")
        bodies = world.snapshot()["bodies"]
        names = [NAMES[RGB_COLORS.index(tuple(body["color"]))] for body in bodies]
        if index % 4 == 0:
            absent = [name for name in NAMES if name not in names]
            entity = absent[seed % len(absent)]
            target = "未知"
        else:
            start = (index // 4) % len(bodies)
            order = list(range(len(bodies)))
            order = order[start:] + order[:start]
            chosen = None
            for body_index in order:
                local = locate_public(result["raw_media"]["rgb"], names[body_index])
                word = "未知" if local is None else position_word(local)
                actual = position_word(bodies[body_index]["position"][0])
                if word == actual:
                    chosen = (names[body_index], actual)
                    break
            if chosen is None:
                absent = [name for name in NAMES if name not in names]
                entity = absent[seed % len(absent)]
                target = "未知"
            else:
                entity, target = chosen
        local = locate_public(result["raw_media"]["rgb"], entity)
        oracle = "未知" if local is None else position_word(local)
        if oracle != target:
            raise ValueError("check_example_oracle_mismatch")
        examples.append(
            {
                "id": f"check-{index}",
                "public": {
                    "goal": GOAL,
                    "request": f"{entity}现在在哪里？".encode(),
                    "observations": result["observations"],
                    "coordinates": result["coordinates"],
                    "outcome": result["execution_outcome"],
                    "action": torch.zeros(15),
                    "t_start": result["t_start"],
                    "t_end": result["t_end"],
                },
                "target": target.encode(),
                "audit": {"seed": seed, "entity": entity},
            }
        )
    return examples


def _gradients(model, batch):
    model.zero_grad(set_to_none=True)
    loss = task_loss(model, batch)
    loss.backward()
    measured = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    if not all(value is None or bool(torch.isfinite(value).all()) for value in measured.values()):
        raise ValueError("nonfinite_first_A_gradient")
    return float(loss.detach()), measured


def _train(model, optimizer, examples, order, updates: int) -> float:
    cursor = 0
    last = 0.0
    for _ in range(updates):
        indices = [order[(cursor + offset) % len(order)] for offset in range(BATCH)]
        batch = [examples[index] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last = float(loss.detach())
        cursor = (cursor + BATCH) % len(order)
    return last


def _check_worker(checkpoint: Path, out: Path) -> int:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    model, _, _ = load_learner(checkpoint)
    examples = _check_examples()
    result = evaluate(model, examples)
    aggregate = {
        "correct": result["correct"],
        "denominator": result["denominator"],
        "accuracy": result["accuracy"],
        "valid_utf8": sum(int(row["valid_utf8"]) for row in result["rows"]),
        "eos": sum(int(row["eos"]) for row in result["rows"]),
        "per_label": {},
        "self_cpu": round(time.process_time() - started_cpu, 6),
        "self_wall": round(time.perf_counter() - started_wall, 6),
    }
    for row in result["rows"]:
        bucket = aggregate["per_label"].setdefault(row["target"], {"correct": 0, "denominator": 0})
        bucket["denominator"] += 1
        bucket["correct"] += int(row["correct"])
    out.write_bytes(json.dumps(aggregate, ensure_ascii=False, indent=2).encode() + b"\n")
    return 0


def _seed_run(seed: int, train, fit, price, directory: Path) -> dict:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    model = make_model("native", seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    order = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(seed + 20261209)
    ).tolist()
    initial = directory / f"seed-{seed}-initial.bin"
    final = directory / f"seed-{seed}-final.bin"
    save_learner(initial, model, optimizer, "native", 0, 0, order, value_digest(train))
    restored, restored_optimizer, _ = load_learner(initial)
    first_loss, first_grads = _gradients(model, price[:2])
    second_loss, second_grads = _gradients(restored, price[:2])
    if first_loss != second_loss:
        raise ValueError("initial_checkpoint_loss_mismatch")
    assert_values_equal(first_grads, second_grads, "a_gate.initial_gradients")
    preflight_equal = True
    training_started = time.process_time()
    last_loss = _train(model, optimizer, train, order, UPDATES)
    training_cpu = time.process_time() - training_started
    fit_result = evaluate(model, fit)
    save_learner(
        final, model, optimizer, "native", UPDATES, UPDATES * BATCH, order, value_digest(train)
    )
    recovered, _, _ = load_learner(final)
    recovered_fit = evaluate(recovered, fit)
    recovery_equal = value_digest(fit_result["rows"]) == value_digest(recovered_fit["rows"])
    check_out = directory / f"seed-{seed}-check.json"
    command = [
        sys.executable,
        "-m",
        "scripts.training.run_a_gate_c_20261117",
        "--check-worker",
        "--checkpoint",
        str(final),
        "--check-out",
        str(check_out),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"check worker failed: {completed.stderr[-2000:]}")
    check = json.loads(check_out.read_text(encoding="utf-8"))
    gate = (
        preflight_equal
        and recovery_equal
        and check["accuracy"] >= A_GATE_MIN
        and check["valid_utf8"] == check["denominator"]
        and check["eos"] == check["denominator"]
    )
    return {
        "seed": seed,
        "preflight_equal": preflight_equal,
        "last_train_loss": round(last_loss, 6),
        "training_cpu": round(training_cpu, 4),
        "fit_correct": fit_result["correct"],
        "fit_denominator": fit_result["denominator"],
        "fit_accuracy": fit_result["accuracy"],
        "recovery_equal": recovery_equal,
        "check_correct": check["correct"],
        "check_denominator": check["denominator"],
        "check_accuracy": check["accuracy"],
        "check_valid_utf8": check["valid_utf8"],
        "check_eos": check["eos"],
        "check_per_label": check["per_label"],
        "check_worker_cpu": check["self_cpu"],
        "check_worker_wall": check["self_wall"],
        "gate": gate,
        "initial_checkpoint": str(initial.relative_to(ROOT)),
        "final_checkpoint": str(final.relative_to(ROOT)),
        "io_cpu": round(time.process_time() - started_cpu - training_cpu, 4),
        "io_wall": round(time.perf_counter() - started_wall, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-worker", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check-out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.check_worker:
        if args.checkpoint is None or args.check_out is None:
            raise ValueError("check worker requires checkpoint and check-out")
        return _check_worker(args.checkpoint, args.check_out)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "output" / f"a_gate_{run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    train = generate_examples("train", 64)
    fit = generate_examples("fit", 32)
    price = generate_examples("price", 8)
    seeds = SEEDS[:1] if args.smoke else SEEDS
    records = [_seed_run(seed, train, fit, price, directory) for seed in seeds]
    child_cpu = sum(row["check_worker_cpu"] for row in records)
    child_wall = sum(row["check_worker_wall"] for row in records)
    cpu = round(time.process_time() - started_cpu + child_cpu, 4)
    wall = round(time.perf_counter() - started_wall + child_wall, 4)
    accepted = all(row["gate"] for row in records) and len(records) == len(SEEDS)
    if cpu > CPU_CAP or wall > WALL_CAP:
        verdict = "budget_stop"
    elif not accepted:
        verdict = "a_parent_not_met"
    else:
        verdict = "a_parent_accepted"
    report = {
        "schema": "taiji-u2-22-5-a-gate-c-v1",
        "card": "22.5.0.A",
        "run_id": run_id,
        "verdict": verdict,
        "authorization": {
            "owner_approval_ref": "chat:2026-11-17#approve-refreeze-and-A-gate",
            "training_authorized": True,
            "scope": "U2 A-parent gate under Plan C FLOPs-matched re-freeze",
        },
        "conditions": {
            "updates_per_seed": UPDATES,
            "batch": BATCH,
            "seeds": list(seeds),
            "optimizer": {"type": "Adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-08},
            "check_split": CHECK_SEEDS,
            "a_gate_min": A_GATE_MIN,
            "a_task_rights": {
                "source": "sensor",
                "feedback_writes": 0,
                "oracle_writes": 0,
                "external_help": 0,
                "goal_rewrites": 0,
                "note": "read-only A task; only sensory assimilation and request binding",
            },
        },
        "records": records,
        "cost": {"cpu": cpu, "wall": wall, "cpu_cap": CPU_CAP, "wall_cap": WALL_CAP},
        "check_exposure": {
            "used": 1,
            "remaining": 1,
            "allocated_22_5": 2,
            "labels_returned_to_selector": False,
            "aggregate_only": True,
        },
        "final_read": False,
        "discipline": {
            "optimizer_steps": UPDATES * len(seeds),
            "project_data_reads": 0,
            "final_reads": 0,
        },
        "source_pins": {
            "scripts/training/run_a_gate_c_20261117.py": hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
            "scripts/training/first_a_support_22_5.py": hashlib.sha256(
                (ROOT / "scripts/training/first_a_support_22_5.py")
                .read_text(encoding="utf-8-sig")
                .encode()
            ).hexdigest(),
        },
    }
    if args.smoke:
        print(
            json.dumps({"smoke": True, "verdict": verdict, "records": records}, ensure_ascii=False)
        )
        return 0
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "A gate verdict="
        + verdict
        + " seeds="
        + str(len(records))
        + " fit="
        + str([round(row["fit_accuracy"], 4) for row in records])
        + " check="
        + str([round(row["check_accuracy"], 4) for row in records])
        + " cpu="
        + str(cpu)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
