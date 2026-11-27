"""Run the fixed three-seed, train-only N1-175 V3 full-answer package."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest
from scripts.training.calibrate_taiji_n1_plan175_v3_train_only import load_train_episodes
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3

SEEDS = (169, 170, 171)
EPOCHS = 40
BATCH_SIZE = 4
MAX_CPU_SECONDS_PER_SEED = 300_000
MAX_WALL_SECONDS_PER_SEED = 500_000
MIGRATION = ROOT / "reports/taiji_n1_plan175_v3_parent_migration_20261026.json"
CORPUS_REPORT = ROOT / "reports/taiji_n1_plan175_v3_train_corpus_v2_20261026.json"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan175_v3_scale_20261026"
REPORT_ROOT = ROOT / "reports"


def bucket(target_bytes: int) -> int:
    for edge in (128, 512, 1024, 1536, 2048):
        if target_bytes <= edge:
            return edge
    raise ValueError("target exceeds frozen V3 training bound")


def epoch_batches(episodes, *, seed: int, epoch: int):
    rng = random.Random(seed * 1_000_003 + epoch)
    buckets = collections.defaultdict(list)
    for candidate, episode in episodes:
        buckets[bucket(len(episode.target_bytes))].append((candidate, episode))
    batches = []
    for edge in sorted(buckets):
        group = buckets[edge]
        rng.shuffle(group)
        batches.extend(
            group[index : index + BATCH_SIZE] for index in range(0, len(group), BATCH_SIZE)
        )
    rng.shuffle(batches)
    return batches


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def progress_path(seed: int) -> Path:
    return REPORT_ROOT / f"taiji_n1_plan175_v3_scale_20261026_tf_seed{seed}_progress.json"


def report_path(seed: int) -> Path:
    return REPORT_ROOT / f"taiji_n1_plan175_v3_scale_20261026_tf_seed{seed}.json"


def checkpoint_path(seed: int, epoch: int) -> Path:
    return CHECKPOINT_ROOT / f"tf_seed{seed}" / f"epoch_{epoch:02d}.pt"


def verify_fresh_process(path: Path, expected_digest: str, seed: int, epoch: int) -> float:
    code = (
        "import json,torch,sys,time; started=time.process_time(); "
        "from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3; "
        "p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "t,s,c=ConditionalLanguageTrainerV3.from_checkpoint(p); "
        "assert p['checkpoint_digest']==sys.argv[2]; "
        "assert c['seed']==int(sys.argv[3]) and c['epoch']==int(sys.argv[4]); "
        "assert s.to_payload()['format']=='taiji-task-state-v3'; "
        "print(json.dumps({'digest':p['checkpoint_digest'],'cpu_seconds':time.process_time()-started}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), expected_digest, str(seed), str(epoch)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        check=True,
        text=True,
    )
    restored = json.loads(result.stdout)
    if restored["digest"] != expected_digest:
        raise ValueError("final checkpoint differs after fresh-process restoration")
    return float(restored["cpu_seconds"])


def run_seed(seed: int) -> dict[str, Any]:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    progress = progress_path(seed)
    report = report_path(seed)
    if report.exists():
        raise FileExistsError(report)
    torch.set_num_threads(1)
    episodes, data = load_train_episodes()
    corpus = json.loads(CORPUS_REPORT.read_text(encoding="utf-8"))
    if (
        corpus.get("status") != "train_only_structural_corpus_audited"
        or corpus.get("partition_sha256") != data["partition_sha256"]
        or corpus.get("episode_count") != len(episodes)
    ):
        raise ValueError("frozen V3 train corpus differs from its audit receipt")
    migration = json.loads(MIGRATION.read_text(encoding="utf-8"))
    if migration.get("status") != "three_seed_parent_migrated_no_training":
        raise ValueError("three migrated N174 TF parent checkpoints are required")
    lineage = next((item for item in migration["parents"] if item["seed"] == seed), None)
    if lineage is None:
        raise ValueError(f"no migrated parent for seed {seed}")

    completed = 0
    epoch_rows = []
    available = sorted(
        (
            [path for path in (CHECKPOINT_ROOT / f"tf_seed{seed}").glob("epoch_*.pt")]
            if (CHECKPOINT_ROOT / f"tf_seed{seed}").exists()
            else []
        ),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    expected_parent = Path(lineage["v3_checkpoint"])
    if available:
        last = available[-1]
        payload = torch.load(last, map_location="cpu", weights_only=False)
        trainer, state, cursor = ConditionalLanguageTrainerV3.from_checkpoint(payload)
        completed = int(cursor["epoch"])
        if completed != int(last.stem.split("_")[-1]) or cursor.get("seed") != seed:
            raise ValueError("latest durable epoch checkpoint cursor differs")
        if cursor.get("dataset_manifest_sha256") != corpus["candidate_manifest_sha256"]:
            raise ValueError("durable checkpoint belongs to a different data manifest")
        if trainer.lineage.get("parent_checkpoint_sha256") != lineage["v3_checkpoint_sha256"]:
            raise ValueError("durable checkpoint has a different N174 parent lineage")
        parent_sha = lineage["v3_checkpoint_sha256"]
        parent_checkpoint = str(expected_parent)
    else:
        parent_path = ROOT / expected_parent
        parent_sha = digest(parent_path)
        if parent_sha != lineage["v3_checkpoint_sha256"]:
            raise ValueError("migrated V3 starting checkpoint SHA differs from its receipt")
        payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        trainer, state, cursor = ConditionalLanguageTrainerV3.from_checkpoint(payload)
        if trainer.steps != 5000 or trainer.model.config.position_extension_seed != seed + 1:
            raise ValueError("N175 seed does not match its frozen N174 V3 parent")
        parent_checkpoint = str(expected_parent)

    if progress.exists():
        prior_progress = json.loads(progress.read_text(encoding="utf-8"))
        if (
            prior_progress.get("seed") != seed
            or prior_progress.get("dataset_manifest_sha256") != corpus["candidate_manifest_sha256"]
        ):
            raise ValueError("progress record differs from current seed/data contract")
        epoch_rows = list(prior_progress.get("epoch_history", []))
    for epoch in range(completed + 1, EPOCHS + 1):
        elapsed_cpu = time.process_time() - started_cpu
        elapsed_wall = time.monotonic() - started_wall
        previous_cpu = [row.get("process_cpu_seconds", 0.0) for row in epoch_rows]
        previous_wall = [row.get("wall_seconds", 0.0) for row in epoch_rows]
        reserve_cpu = (
            max(10_000.0, 2 * sum(previous_cpu[-5:]) / len(previous_cpu[-5:]))
            if previous_cpu
            else 10_000.0
        )
        reserve_wall = (
            max(20_000.0, 2 * sum(previous_wall[-5:]) / len(previous_wall[-5:]))
            if previous_wall
            else 20_000.0
        )
        if (
            elapsed_cpu + reserve_cpu > MAX_CPU_SECONDS_PER_SEED
            or elapsed_wall + reserve_wall > MAX_WALL_SECONDS_PER_SEED
        ):
            break
        epoch_started_cpu, epoch_started_wall = time.process_time(), time.monotonic()
        batches = epoch_batches(episodes, seed=seed, epoch=epoch)
        ordered_ids = [candidate["target_id"] for batch in batches for candidate, _ in batch]
        if len(ordered_ids) != len(episodes) or len(set(ordered_ids)) != len(episodes):
            raise ValueError("epoch batching dropped or duplicated a frozen train episode")
        order_digest = hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest()
        weighted_loss, epoch_bytes, update_count = 0.0, 0, 0
        for batch_index, batch in enumerate(batches, 1):
            examples = [(episode.prompt_bytes, episode.target_bytes) for _, episode in batch]
            batch_bytes = sum(len(target) for _, target in examples)
            metrics = trainer.update(examples)
            if metrics["target_byte_exposures"] != trainer.target_byte_exposures:
                raise ValueError("V3 target exposure counter differs from optimizer result")
            weighted_loss += metrics["loss_nats"] * batch_bytes
            epoch_bytes += batch_bytes
            update_count += 1
            if batch_index % 16 == 0 or batch_index == len(batches):
                atomic_json(
                    progress,
                    {
                        "schema": "taiji-n1-plan175-v3-scale-progress-v1",
                        "status": "running",
                        "seed": seed,
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches_in_epoch": len(batches),
                        "last_durable_epoch": completed,
                        "epoch_history": epoch_rows,
                        "updates_this_epoch": update_count,
                        "target_bytes_this_epoch": epoch_bytes,
                        "process_cpu_seconds": time.process_time() - started_cpu,
                        "wall_seconds": time.monotonic() - started_wall,
                        "parent_checkpoint_sha256": parent_sha,
                        "dataset_manifest_sha256": corpus["candidate_manifest_sha256"],
                    },
                )
        if epoch_bytes != corpus["complete_target_bytes_per_epoch"]:
            raise ValueError("epoch target byte exposure differs from frozen corpus total")
        completed = epoch
        cursor = {
            "seed": seed,
            "epoch": epoch,
            "batch": 0,
            "epoch_order_sha256": order_digest,
            "dataset_manifest_sha256": corpus["candidate_manifest_sha256"],
        }
        state = trainer.model.new_state(
            scope_id="n1-175",
            session_id=f"seed-{seed}",
            native_state_ref=f"n1-175:seed-{seed}:epoch-{epoch}",
        )
        durable_path = checkpoint_path(seed, epoch)
        trainer.save(durable_path, state, cursor=cursor)
        saved = torch.load(durable_path, map_location="cpu", weights_only=False)
        checkpoint_receipt = {
            "epoch": epoch,
            "path": str(durable_path.relative_to(ROOT)),
            "sha256": digest(durable_path),
            "checkpoint_digest": saved["checkpoint_digest"],
            "steps": trainer.steps,
            "target_byte_exposures": trainer.target_byte_exposures,
        }
        epoch_rows.append(
            {
                "epoch": epoch,
                "order_sha256": order_digest,
                "updates": update_count,
                "target_bytes": epoch_bytes,
                "mean_loss_nats": weighted_loss / epoch_bytes,
                "process_cpu_seconds": time.process_time() - epoch_started_cpu,
                "wall_seconds": time.monotonic() - epoch_started_wall,
                "checkpoint": checkpoint_receipt,
            }
        )
        atomic_json(
            progress,
            {
                "schema": "taiji-n1-plan175-v3-scale-progress-v1",
                "status": "running" if epoch < EPOCHS else "training_complete_awaiting_restore",
                "seed": seed,
                "epoch": epoch,
                "batch": 0,
                "last_durable_epoch": epoch,
                "process_cpu_seconds": time.process_time() - started_cpu,
                "wall_seconds": time.monotonic() - started_wall,
                "parent_checkpoint_sha256": parent_sha,
                "dataset_manifest_sha256": corpus["candidate_manifest_sha256"],
                "latest_checkpoint": checkpoint_receipt,
                "epoch_history": epoch_rows,
            },
        )

    status = "complete" if completed == EPOCHS else "partial_resource_guard"
    final_path = checkpoint_path(seed, completed) if completed else (ROOT / parent_checkpoint)
    final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
    final_digest = final_payload["checkpoint_digest"]
    restore_cpu = 0.0
    if status == "complete":
        restore_cpu = verify_fresh_process(final_path, final_digest, seed, EPOCHS)
    result = {
        "schema": "taiji-n1-plan175-v3-scale-run-v1",
        "status": status,
        "run": f"tf_seed{seed}",
        "seed": seed,
        "epochs_complete": completed,
        "epochs_planned": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_episode_count": len(episodes),
        "train_family_count": corpus["family_count"],
        "target_bytes_per_epoch": corpus["complete_target_bytes_per_epoch"],
        "parent_v3_checkpoint": parent_checkpoint,
        "parent_v3_checkpoint_sha256": parent_sha,
        "parent_v2_checkpoint_sha256": lineage["parent_checkpoint_sha256"],
        "dataset_partition_sha256": corpus["partition_sha256"],
        "dataset_manifest_sha256": corpus["candidate_manifest_sha256"],
        "candidate_source_sha256": digest(ROOT / "taiji/conditional_language_v3.py"),
        "runner_source_sha256": digest(Path(__file__)),
        "epoch_results": epoch_rows,
        "final_checkpoint": str(final_path.relative_to(ROOT)),
        "final_checkpoint_sha256": digest(final_path),
        "final_checkpoint_digest": final_digest,
        "fresh_process_restore": "passed" if status == "complete" else "not_final_yet",
        "fresh_process_restore_cpu_seconds": restore_cpu,
        "process_cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.monotonic() - started_wall,
        "cpu_guard_seconds": MAX_CPU_SECONDS_PER_SEED,
        "heldout_text_parsed": 0,
        "development_reads": 0,
        "can_promote": False,
    }
    if status == "complete":
        with report.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    atomic_json(progress, {**result, "progress_status": status})
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "seed",
                    "epochs_complete",
                    "process_cpu_seconds",
                    "wall_seconds",
                    "final_checkpoint_sha256",
                    "can_promote",
                )
            }
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    run_seed(args.seed)


if __name__ == "__main__":
    main()
