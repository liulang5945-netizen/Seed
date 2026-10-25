"""Restore and round-trip the latest durable epoch checkpoint for every seed."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest
from scripts.training.run_taiji_n1_plan175_v3_scale import checkpoint_path, progress_path
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3

SEEDS = (169, 170, 171)
MIGRATION = ROOT / "reports/taiji_n1_plan175_v3_parent_migration_20261026.json"
OUTPUT = ROOT / "reports/taiji_n1_plan175_v3_checkpoint_restore_epoch01_20261026.json"


def verify_seed(seed: int, epoch: int) -> dict:
    progress = json.loads(progress_path(seed).read_text(encoding="utf-8"))
    if progress.get("last_durable_epoch", 0) < epoch:
        raise ValueError(f"seed {seed} has not durably saved epoch {epoch}")
    history = [row for row in progress.get("epoch_history", []) if row.get("epoch") <= epoch]
    if len(history) != epoch or history[-1]["epoch"] != epoch:
        raise ValueError(f"seed {seed} progress lacks complete metrics through epoch {epoch}")
    path = checkpoint_path(seed, epoch)
    epoch_receipt = next((row["checkpoint"] for row in history if row["epoch"] == epoch), None)
    if epoch_receipt is None or digest(path) != epoch_receipt["sha256"]:
        raise ValueError(f"seed {seed} durable checkpoint SHA differs from progress")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trainer, state, cursor = ConditionalLanguageTrainerV3.from_checkpoint(payload)
    expected_steps = 5000 + sum(row["updates"] for row in history)
    expected_exposures = 2159160 + sum(row["target_bytes"] for row in history)
    if (
        trainer.steps != expected_steps
        or trainer.target_byte_exposures != expected_exposures
        or cursor.get("seed") != seed
        or cursor.get("epoch") != epoch
        or state.to_payload().get("format") != "taiji-task-state-v3"
    ):
        raise ValueError(f"seed {seed} epoch {epoch} restored counters/cursor differ")
    roundtrip = trainer.checkpoint(state, cursor=cursor)
    if roundtrip["checkpoint_digest"] != payload["checkpoint_digest"]:
        raise ValueError(f"seed {seed} epoch {epoch} checkpoint round-trip differs")
    return {
        "seed": seed,
        "epoch": epoch,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": digest(path),
        "checkpoint_digest": payload["checkpoint_digest"],
        "steps": trainer.steps,
        "target_byte_exposures": trainer.target_byte_exposures,
        "cursor": cursor,
        "roundtrip": "passed",
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(1)
    started = time.process_time()
    results = [verify_seed(seed, 1) for seed in SEEDS]
    payload = {
        "schema": "taiji-n1-plan175-v3-durable-checkpoint-restore-v1",
        "status": "all_seed_epoch1_fresh_process_restore_and_roundtrip_passed",
        "verifier_sha256": digest(Path(__file__)),
        "runner_sha256": digest(ROOT / "scripts/training/run_taiji_n1_plan175_v3_scale.py"),
        "candidate_sha256": digest(ROOT / "taiji/conditional_language_v3.py"),
        "migration_report_sha256": digest(MIGRATION),
        "training_updates": 0,
        "heldout_text_read": 0,
        "process_cpu_seconds": time.process_time() - started,
        "seeds": results,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "seeds": len(results),
                "process_cpu_seconds": payload["process_cpu_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
