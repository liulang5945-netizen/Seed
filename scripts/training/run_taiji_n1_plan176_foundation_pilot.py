"""Fixed F0/F1 ordering and workspace comparison, from the uncalibrated N175 parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import torch

from scripts.training.audit_taiji_n1_plan176_learning_paths import CORPUS, PARENT, digest
from scripts.training.audit_taiji_n1_plan176_learning_paths import PARENT_REPORT as V3_PARENT_REPORT
from scripts.training.build_taiji_n1_plan176_foundation_corpus import ROOT, as_language_episode
from scripts.training.calibrate_taiji_n1_plan176_foundation import fresh_restore
from scripts.training.migrate_taiji_n1_plan176_v3_to_v4 import OUTPUT as V4_PARENT
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3
from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4
from taiji.persistence import atomic_save

ARMS = ("flat_v3", "curriculum_v3", "curriculum_v4")
COPIES = 16
CPU_GUARD = 1500
CHECKPOINTS = ROOT / "data/checkpoints/n1_plan176_foundation_pilot_20261026"
SELECTION = ROOT / "reports/taiji_n1_plan176_pilot_selection_20261026.json"
SPEC = ROOT / "plans/reference/SPEC-N1-176_F0_F1_pilot_20261026.md"


def report_path(arm: str) -> Path:
    return ROOT / f"reports/taiji_n1_plan176_foundation_pilot_20261026_{arm}.json"


def progress_path(arm: str) -> Path:
    return ROOT / f"reports/taiji_n1_plan176_foundation_pilot_20261026_{arm}_progress.json"


def schedule(rows: list[dict], arm: str) -> list[list[int]]:
    if arm not in ARMS or any(r["curriculum_level"] not in {"F0", "F1"} for r in rows):
        raise ValueError("pilot supports only its three arms and F0/F1")
    f0 = [i for i, r in enumerate(rows) if r["curriculum_level"] == "F0"]
    f1 = [i for i, r in enumerate(rows) if r["curriculum_level"] == "F1"]
    if len(f0) != 320 or len(f1) != 320:
        raise ValueError("frozen F0/F1 training pool differs")
    rng = random.Random(176)
    if arm == "flat_v3":
        order = list(range(len(rows))) * COPIES
        rng.shuffle(order)
    else:
        first, second = f0 * 8, f0 * 8 + f1 * COPIES
        rng.shuffle(first)
        rng.shuffle(second)
        order = first + second
    return [order[i : i + 4] for i in range(0, len(order), 4)]


def atomic_json(path: Path, payload: dict) -> None:
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    arm = args.arm
    folder, output, progress = CHECKPOINTS / arm, report_path(arm), progress_path(arm)
    if folder.exists() or output.exists() or progress.exists():
        raise FileExistsError("preserve the existing run; recover its durable cursor explicitly")
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    child_cpu = 0.0
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    train = ROOT / corpus["output_directory"] / "train.jsonl"
    if digest(train) != corpus["split_sha256"]["train"]:
        raise ValueError("frozen foundation train file differs")
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    rows = sorted(
        [r for r in rows if r["curriculum_level"] in {"F0", "F1"}], key=lambda r: r["episode_id"]
    )
    episodes = [as_language_episode(r) for r in rows]
    batches = schedule(rows, arm)
    order_digest = hashlib.sha256(
        json.dumps([[rows[i]["episode_id"] for i in b] for b in batches]).encode()
    ).hexdigest()
    version = 4 if arm == "curriculum_v4" else 3
    trainer_type = ConditionalLanguageTrainerV4 if version == 4 else ConditionalLanguageTrainerV3
    parent = V4_PARENT if version == 4 else PARENT
    parent_receipt_path = (
        ROOT / "reports/taiji_n1_plan176_v4_parent_migration_20261026.json"
        if version == 4
        else V3_PARENT_REPORT
    )
    parent_receipt = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
    expected_parent_sha = parent_receipt[
        "checkpoint_sha256" if version == 4 else "final_checkpoint_sha256"
    ]
    if digest(parent) != expected_parent_sha:
        raise ValueError("fixed uncalibrated parent lineage differs")
    trainer, state, _ = trainer_type.from_checkpoint(torch.load(parent, weights_only=False))
    initial_steps, initial_exposures = trainer.steps, trainer.target_byte_exposures
    sources = {
        "spec": digest(SPEC),
        "selection": digest(SELECTION),
        "runner": digest(Path(__file__)),
        "candidate": digest(ROOT / f"taiji/conditional_language_v{version}.py"),
        "restore_helper": digest(
            ROOT / "scripts/training/calibrate_taiji_n1_plan176_foundation.py"
        ),
    }
    folder.mkdir(parents=True)
    zero = folder / "zero.pt"
    atomic_save(
        trainer.checkpoint(state, cursor={"next_batch": 0, "order_sha256": order_digest}), zero
    )
    restored = fresh_restore(zero, version)
    child_cpu += restored["cpu_seconds"]
    trainer, _, _ = trainer_type.from_checkpoint(torch.load(zero, weights_only=False))
    ledger = {
        "schema": "taiji-n176-foundation-pilot-v1",
        "status": "running",
        "arm": arm,
        "pid": os.getpid(),
        "version": version,
        "parent_checkpoint": str(parent.relative_to(ROOT)),
        "parent_checkpoint_sha256": digest(parent),
        "sources_sha256": sources,
        "train_corpus_sha256": digest(train),
        "order_sha256": order_digest,
        "planned_updates": len(batches),
        "planned_target_byte_exposures": sum(len(e.target_bytes) for e in episodes) * COPIES,
        "completed_updates": 0,
        "initial_steps": initial_steps,
        "initial_target_byte_exposures": initial_exposures,
        "zero_fresh_restore": restored,
        "sealed_text_read": 0,
        "n175_key_opened": False,
        "can_promote": False,
        "cpu_guard_seconds": CPU_GUARD,
        "durable_checkpoints": [],
    }
    atomic_json(progress, ledger)
    weighted_loss, exposures = 0.0, 0
    completed = 0
    for index, indices in enumerate(batches):
        if time.process_time() - started_cpu + child_cpu > CPU_GUARD - 30 or (
            time.monotonic() - started_wall > 6000
        ):
            ledger["status"] = "partial_resource_guard"
            break
        batch = [(episodes[i].prompt_bytes, episodes[i].target_bytes) for i in indices]
        result = trainer.update(batch)
        byte_count = sum(len(target) for _, target in batch)
        exposures += byte_count
        weighted_loss += result["loss_nats"] * byte_count
        completed = index + 1
        if completed % 256 == 0 or completed in {640, len(batches)}:
            state = trainer.model.new_state(
                scope_id="n176-pilot", session_id=arm, native_state_ref=f"n176:{arm}:{completed}"
            )
            path = folder / f"batch_{completed:05d}.pt"
            atomic_save(
                trainer.checkpoint(
                    state,
                    cursor={
                        "next_batch": completed,
                        "order_sha256": order_digest,
                        "train_corpus_sha256": digest(train),
                        "arm": arm,
                    },
                ),
                path,
            )
            ledger["durable_checkpoints"].append(
                {
                    "next_batch": completed,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest(path),
                }
            )
        if completed % 64 == 0 or completed == len(batches):
            ledger.update(
                {
                    "completed_updates": completed,
                    "target_byte_exposures": exposures,
                    "mean_loss_nats": weighted_loss / exposures,
                    "process_cpu_seconds": time.process_time() - started_cpu,
                    "child_process_cpu_seconds": child_cpu,
                    "wall_seconds": time.monotonic() - started_wall,
                }
            )
            atomic_json(progress, ledger)
    if completed == len(batches):
        path = ROOT / ledger["durable_checkpoints"][-1]["path"]
        final_restore = fresh_restore(path, version)
        child_cpu += final_restore["cpu_seconds"]
        if (
            digest(Path(__file__)) != sources["runner"]
            or digest(ROOT / f"taiji/conditional_language_v{version}.py") != sources["candidate"]
            or trainer.steps != initial_steps + len(batches)
            or exposures != ledger["planned_target_byte_exposures"]
        ):
            raise ValueError("source, update counter or fixed exposure differs at endpoint")
        ledger.update(
            {
                "status": "complete",
                "final_checkpoint": str(path.relative_to(ROOT)),
                "final_checkpoint_sha256": digest(path),
                "final_fresh_restore": final_restore,
            }
        )
    ledger.update(
        {
            "completed_updates": completed,
            "target_byte_exposures": exposures,
            "process_cpu_seconds": time.process_time() - started_cpu,
            "child_process_cpu_seconds": child_cpu,
            "wall_seconds": time.monotonic() - started_wall,
        }
    )
    atomic_json(progress, ledger)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(ledger, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: ledger[key]
                for key in ("arm", "status", "completed_updates", "process_cpu_seconds")
            }
        )
    )


if __name__ == "__main__":
    main()
